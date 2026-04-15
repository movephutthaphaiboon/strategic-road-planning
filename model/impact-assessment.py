#!/usr/bin/env python3
"""
impact-assessment.py — Impact assessment for mine-to-port road scenarios.

Takes a generated path GeoPackage (output of path-generator) and computes
impact metrics for the proposed road network.

Metrics implemented:
  [1] Road classification — km of paved / unpaved / to_be_built

Metrics to be added:
  [ ] Road construction cost
  [ ] Mining capacity
  [ ] Increased deforestation
  [ ] Increased forest fragmentation

Approach for road classification:
  Rather than vector overlay (slow, needs buffer tolerance), both the path
  network and the existing roads are rasterized to the same reference grid.
  Pixel-by-pixel comparison is then exact and cheap.

    road_raster : 0 = no road, 1 = paved, 2 = unpaved
    path_raster : 0 = no path, 1 = path pixel

  Overlay → count path pixels by road class → multiply by pixel size in km.

Usage:
    python impact-assessment.py <paths_gpkg>
    python impact-assessment.py <paths_gpkg> --downsample 3

Example:
    python impact-assessment.py model/results/least-cost-paths/late_stage__kribi__base__protected_areas__ds1.gpkg
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR  = Path(__file__).parent.parent
ROADS_FP  = BASE_DIR / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"
FRICTION_FP = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"

ROAD_TYPE_COL = "liu_surface"

# =============================================================================
# REFERENCE GRID
# =============================================================================

def load_reference_grid(friction_fp: Path, downsample: int = 1):
    """
    Load the reference raster grid (transform, shape, CRS) from the friction raster.
    The same grid is used to rasterize both roads and paths, guaranteeing alignment.

    downsample : reduce resolution to save memory.
                 1 = native ~90 m, 5 = ~450 m, 10 = ~900 m
    """
    with rasterio.open(friction_fp) as src:
        orig_h, orig_w = src.height, src.width
        new_h = max(1, orig_h // downsample)
        new_w = max(1, orig_w // downsample)

        transform = rasterio.transform.Affine(
            src.transform.a * (orig_w / new_w),
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e * (orig_h / new_h),
            src.transform.f,
        )
        crs = src.crs

    print(f"  Reference grid: {new_h}×{new_w} px  (downsample={downsample}×, CRS={crs})")
    return (new_h, new_w), transform, crs


def pixel_size_km(transform) -> float:
    """
    Average pixel step size in km (geodesic) at the raster's centre.
    Used to convert pixel counts to kilometres.
    Accounts for the fact that degrees-of-longitude shrink toward the poles.
    """
    geod  = Geod(ellps="WGS84")
    lon0  = transform.c + transform.a  * 0.5
    lat0  = transform.f + transform.e  * 0.5
    _, _, dist_x = geod.inv(lon0, lat0, lon0 + transform.a,  lat0)
    _, _, dist_y = geod.inv(lon0, lat0, lon0, lat0 + abs(transform.e))
    return (abs(dist_x) + abs(dist_y)) / 2 / 1000   # average, km


# =============================================================================
# RASTERIZATION HELPERS
# =============================================================================

def rasterize_roads(roads_fp: Path, shape: tuple, transform, crs) -> np.ndarray:
    """
    Burn road network into a uint8 raster on the reference grid.

    Values:
        0 = no road
        1 = paved
        2 = unpaved

    Paved roads are burned last so they take priority over unpaved
    where both share the same pixel.
    """
    roads = gpd.read_file(roads_fp, columns=[ROAD_TYPE_COL, "geometry"])
    roads = roads[roads.geometry.notna()]
    if roads.crs != crs:
        roads = roads.to_crs(crs)

    arr = np.zeros(shape, dtype=np.uint8)

    for value, surface in [(2, "unpaved"), (1, "paved")]:
        subset = roads[roads[ROAD_TYPE_COL] == surface]
        if subset.empty:
            continue
        burned = rasterize(
            [(geom, value) for geom in subset.geometry],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,   # capture thin road pixels
            merge_alg=rasterio.enums.MergeAlg.replace,
        )
        arr = np.where(burned > 0, burned, arr)

    paved_px   = int((arr == 1).sum())
    unpaved_px = int((arr == 2).sum())
    print(f"  Road raster: {paved_px:,} paved px, {unpaved_px:,} unpaved px")
    return arr


def rasterize_paths(paths_gdf: gpd.GeoDataFrame, shape: tuple, transform, crs) -> np.ndarray:
    """
    Merge all mine-to-port paths and burn into a binary uint8 raster.

    Merging removes overlapping segments where multiple mines share a route,
    so each pixel is counted only once regardless of how many mines use it.

    Values:
        0 = no path
        1 = path pixel
    """
    paths = paths_gdf[paths_gdf.geometry.notna()].copy()
    if paths.crs != crs:
        paths = paths.to_crs(crs)

    merged_geom = unary_union(paths.geometry)

    arr = rasterize(
        [(merged_geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    path_px = int((arr == 1).sum())
    print(f"  Path raster: {path_px:,} path pixel(s)")
    return arr


# =============================================================================
# ROAD RASTER SMOOTHING
# =============================================================================

def dilate_road_raster(road_raster: np.ndarray, pixels: int = 2) -> np.ndarray:
    """
    Expand each road class outward by `pixels` pixels to fill rasterization
    gaps and absorb small path-to-road alignment offsets.

    Paved is applied last so it always takes priority over unpaved where
    both classes overlap after dilation.

    pixels : dilation radius in pixels.
             At 90 m resolution: 2 px = 180 m, 3 px = 270 m.
    """
    from scipy.ndimage import binary_dilation

    struct = np.ones((3, 3), dtype=bool)   # 8-connected neighbourhood
    result = np.zeros_like(road_raster)

    unpaved_expanded = binary_dilation(road_raster == 2, structure=struct, iterations=pixels)
    result[unpaved_expanded] = 2

    paved_expanded = binary_dilation(road_raster == 1, structure=struct, iterations=pixels)
    result[paved_expanded] = 1             # paved overwrites unpaved

    return result


# =============================================================================
# METRIC 1 — Road classification
# =============================================================================

def metric_road_classification(path_raster: np.ndarray,
                                road_raster: np.ndarray,
                                px_km: float) -> tuple:
    """
    Overlay path and road rasters to classify each path pixel.

    Parameters
    ----------
    path_raster : binary array (1 = path pixel)
    road_raster : road type array (0 = none, 1 = paved, 2 = unpaved)
    px_km       : average pixel size in km (from pixel_size_km())

    Returns
    -------
    (metrics_dict, classified_raster)

    metrics_dict keys : paved_km, unpaved_km, to_be_built_km, total_km
    classified_raster : uint8 array, same shape as input rasters
        0 = not a path pixel
        1 = paved       (path follows existing paved road)
        2 = unpaved     (path follows existing unpaved road → upgrade)
        3 = to_be_built (path has no existing road → build new)
    """
    path_mask = path_raster == 1
    road_vals = road_raster[path_mask]      # road type at each path pixel

    n_paved        = int((road_vals == 1).sum())
    n_unpaved      = int((road_vals == 2).sum())
    n_to_be_built  = int((road_vals == 0).sum())
    n_total        = n_paved + n_unpaved + n_to_be_built

    # Build classified raster: 0=background, 1=paved, 2=unpaved, 3=to_be_built
    classified = np.zeros(path_raster.shape, dtype=np.uint8)
    classified[path_mask & (road_raster == 1)] = 1   # paved
    classified[path_mask & (road_raster == 2)] = 2   # unpaved
    classified[path_mask & (road_raster == 0)] = 3   # to_be_built

    metrics = {
        "paved_km":        round(n_paved       * px_km, 2),
        "unpaved_km":      round(n_unpaved     * px_km, 2),
        "to_be_built_km":  round(n_to_be_built * px_km, 2),
        "total_km":        round(n_total       * px_km, 2),
    }
    return metrics, classified


def save_classified_raster(classified: np.ndarray, transform, crs,
                           out_fp: Path):
    """
    Save the classified path raster as a GeoTIFF.

    Pixel values:
        0 = background (not a path)
        1 = paved
        2 = unpaved
        3 = to_be_built
    """
    with rasterio.open(
        out_fp, "w",
        driver="GTiff",
        height=classified.shape[0],
        width=classified.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=0,
    ) as dst:
        dst.write(classified, 1)
    print(f"  Classified raster saved → {out_fp}")


def print_road_classification(result: dict):
    total = result["total_km"]
    def pct(v): return 100 * v / total if total > 0 else 0

    print(f"\n  Road classification:")
    print(f"  {'─'*48}")
    print(f"  {'Paved  (maintain)':<25}: {result['paved_km']:>8.1f} km  ({pct(result['paved_km']):5.1f}%)")
    print(f"  {'Unpaved (upgrade)':<25}: {result['unpaved_km']:>8.1f} km  ({pct(result['unpaved_km']):5.1f}%)")
    print(f"  {'New road (build)':<25}: {result['to_be_built_km']:>8.1f} km  ({pct(result['to_be_built_km']):5.1f}%)")
    print(f"  {'─'*48}")
    print(f"  {'Total':<25}: {result['total_km']:>8.1f} km")


# =============================================================================
# FUTURE METRICS (stubs — implement here as the project grows)
# =============================================================================

# def metric_construction_cost(path_raster, road_classification, cost_table) -> dict:
#     """Compute total construction cost from road classification + cost table."""
#     ...

# def metric_mining_capacity(paths_gdf, mines_gdf) -> dict:
#     """Estimate mining capacity unlocked by new road connectivity."""
#     ...

# def metric_deforestation(path_raster, forest_raster, px_km) -> dict:
#     """Estimate forest area lost along new road corridors."""
#     ...

# def metric_forest_fragmentation(path_raster, forest_raster) -> dict:
#     """Compute forest fragmentation indices along new road corridors."""
#     ...


# =============================================================================
# MAIN
# =============================================================================

def run_assessment(paths_fp: Path, downsample: int = 1) -> pd.DataFrame:
    """
    Run all implemented impact metrics for a given path scenario.

    Returns a one-row DataFrame with all metric columns.
    """
    print("=" * 62)
    print("  IMPACT ASSESSMENT")
    print(f"  Scenario: {paths_fp.stem}")
    print("=" * 62)

    # ── Load paths ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading generated paths …")
    paths_gdf = gpd.read_file(paths_fp)
    paths_gdf = paths_gdf[paths_gdf.geometry.notna()]
    print(f"  {len(paths_gdf)} path(s) loaded")

    # ── Reference grid ────────────────────────────────────────────────────────
    print(f"\n[2/4] Building reference grid (downsample={downsample}×) …")
    shape, transform, crs = load_reference_grid(FRICTION_FP, downsample)
    px_km = pixel_size_km(transform)
    print(f"  Pixel size: ~{px_km*1000:.0f} m  ({px_km:.4f} km)")

    # ── Rasterize roads and paths ─────────────────────────────────────────────
    print("\n[3/4] Rasterizing roads and paths …")
    road_raster = rasterize_roads(ROADS_FP, shape, transform, crs)
    road_raster = dilate_road_raster(road_raster, pixels=2)
    path_raster = rasterize_paths(paths_gdf, shape, transform, crs)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results/impact-assessment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics …")

    # Metric 1: road classification
    road_class, classified_raster = metric_road_classification(path_raster, road_raster, px_km)
    print_road_classification(road_class)
    save_classified_raster(classified_raster, transform, crs,
                           out_dir / f"classified_{paths_fp.stem}.tif")

    # [Future metrics called here]

    # ── Compile results ───────────────────────────────────────────────────────
    results = {"scenario": paths_fp.stem, **road_class}
    # results.update(metric_construction_cost(...))
    # results.update(metric_deforestation(...))

    results_df = pd.DataFrame([results])
    out_fp = out_dir / f"assessment_{paths_fp.stem}.csv"
    results_df.to_csv(out_fp, index=False)
    print(f"\n  Saved → {out_fp}")

    print("\nDone.")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Impact assessment for mine-to-port road scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths_gpkg", type=Path,
                        help="Generated path GeoPackage (output of path-generator)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Reference grid downsample factor (default: 1 = native ~90 m). "
                             "Increase to reduce memory usage.")
    args = parser.parse_args()

    if not args.paths_gpkg.exists():
        sys.exit(f"Error: '{args.paths_gpkg}' not found.")

    run_assessment(args.paths_gpkg, downsample=args.downsample)


if __name__ == "__main__":
    main()
