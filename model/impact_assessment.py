#!/usr/bin/env python3
"""
impact-assessment.py — Impact assessment for mine-to-port road scenarios.

Takes a generated path GeoPackage (output of path-generator) and computes
impact metrics for the proposed road network.

Metrics implemented:
  [1] Road classification    — km of paved / unpaved / to_be_built
  [2] Construction cost      — USD for upgrade and new-build segments
  [3] Deforestation risk     — expected forest cleared using Damania et al. (2018) distance-decay
  [4] Forest fragmentation   — new forest patches created by action roads

Metrics to be added:
  [ ] Mining capacity

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
import time
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
# CONFIGURATION  ← edit here to control what gets computed and saved
# =============================================================================

# --- Metrics to run ----------------------------------------------------------
RUN_ROAD_CLASSIFICATION = True   # paved / unpaved / new-build km
RUN_CONSTRUCTION_COST   = True   # USD cost for upgrade and new-build
RUN_DEFORESTATION       = True   # expected forest cleared using Damania et al. (2018) distance-decay
RUN_FRAGMENTATION       = True   # forest patch count and fragmentation index (slow)

# --- Deforestation settings (Damania et al. 2018) ----------------------------
DAMANIA_CURVE       = "good"   # road quality curve: "good", "fair", "poor", "very_poor"
DAMANIA_MAX_DIST_KM = 10.0     # maximum distance to apply the clearing decay (km)
SAVE_DEFOR_TIFS     = True     # save loss-pct and remaining-treecover rasters (float32, 0–100 scale)

# --- Fragmentation settings (forest_patch_detection) -------------------------
FRAGMENTATION_MIN_PATCH_HA = 1.0    # minimum patch size counted (ha)
FOREST_THRESHOLD           = 10     # minimum canopy cover (%) — FAO definition
BASELINE_FRAG_GPKG = (
    Path(__file__).parent.parent
    / "data/output/forest-patch-detection"
    / "no_action__forest_patches__thresh10__minpatch1ha__adm2__roads_paved.gpkg"
)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR          = Path(__file__).parent.parent
ROADS_FP          = BASE_DIR / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"
FRICTION_FP       = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
TREECOVER_FP      = BASE_DIR / "data/output/processed-hansen-treecover/cameroon_treecover2024_30m.tif"
DAMANIA_LOOKUP_FP = BASE_DIR / "data/input/damania-deforestation-lookup/damania_pct_cleared.csv"

ROAD_TYPE_COL = "liu_surface"

# --- Load Damania lookup at module level (interpolated at runtime) ------------
_damania_df   = pd.read_csv(DAMANIA_LOOKUP_FP)
_damania_dist = _damania_df["distance_km"].values.astype(np.float32)
_damania_pct  = (_damania_df[f"pct_cleared_{DAMANIA_CURVE}"].values / 100.0).astype(np.float32)

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


def save_classified_lines(paths_gdf: gpd.GeoDataFrame,
                          classified_raster: np.ndarray,
                          transform,
                          crs,
                          friction: np.ndarray,
                          out_dir: Path,
                          scenario_stem: str):
    """
    Split mine-to-port paths into line segments by road action class and
    save as a GeoPackage.

    Points are sampled at sub-pixel intervals along each path geometry,
    classified by sampling the classified_raster, then grouped into
    contiguous segments of the same action class (upgrade / build).
    Paved segments are discarded — only action roads are written.

    Output attributes per segment:
        mine_id, mine_name, closest_port, action_needed,
        length_km, construction_cost_usd, scenario
    """
    from shapely.geometry import LineString

    geod = Geod(ellps="WGS84")
    sample_interval_m = 45   # sub-pixel (~half the 90 m pixel size)

    cls_h, cls_w = classified_raster.shape
    fri_h, fri_w = friction.shape

    paths = paths_gdf[paths_gdf.geometry.notna()].copy()
    if paths.crs != crs:
        paths = paths.to_crs(crs)

    records = []

    for _, row in paths.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        mine_id   = row.get("mine_id",      "")
        mine_name = row.get("mine_name",     "")
        port      = row.get("closest_port",  "")

        # Dense point sampling along the path geometry
        path_length_m = abs(geod.geometry_length(geom))
        n_steps = max(2, int(path_length_m / sample_interval_m))
        pts = [geom.interpolate(i / n_steps, normalized=True) for i in range(n_steps + 1)]

        # Sample classified raster at each point
        pt_classes = []
        pt_coords  = []
        for pt in pts:
            r, c = rasterio.transform.rowcol(transform, pt.x, pt.y)
            cls = int(classified_raster[r, c]) if (0 <= r < cls_h and 0 <= c < cls_w) else 0
            pt_classes.append(cls)
            pt_coords.append((pt.x, pt.y))

        # Group consecutive points of the same action class (2=upgrade, 3=build)
        i = 0
        while i < len(pt_classes):
            cls = pt_classes[i]
            if cls not in (2, 3):
                i += 1
                continue

            j = i + 1
            while j < len(pt_classes) and pt_classes[j] == cls:
                j += 1

            seg_coords = pt_coords[i:j]
            if len(seg_coords) < 2:
                i = j
                continue

            line = LineString(seg_coords)
            seg_length_km = abs(geod.geometry_length(line)) / 1000

            # Sum friction over unique pixels — avoids double-counting pixels
            # that contain multiple sample points
            seen_px = set()
            cost = 0.0
            for x, y in seg_coords:
                r, c = rasterio.transform.rowcol(transform, x, y)
                if (r, c) not in seen_px and 0 <= r < fri_h and 0 <= c < fri_w:
                    seen_px.add((r, c))
                    v = friction[r, c]
                    if np.isfinite(v):
                        cost += float(v)

            records.append({
                "mine_id":               mine_id,
                "mine_name":             mine_name,
                "closest_port":          port,
                "action_needed":         "upgrade" if cls == 2 else "build",
                "length_km":             round(seg_length_km, 3),
                "construction_cost_usd": round(cost, 2),
                "scenario":              scenario_stem,
                "geometry":              line,
            })
            i = j

    if not records:
        print("  No upgrade or build segments found.")
        return

    out_gdf = gpd.GeoDataFrame(records, crs=crs)
    out_fp = out_dir / f"{scenario_stem}__action_roads.gpkg"
    out_gdf.to_file(out_fp, driver="GPKG")
    print(f"  Action roads → {out_fp}  ({len(out_gdf)} segments)")


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
# METRIC 2 — Construction cost
# =============================================================================

def load_friction_values(friction_fp: Path, shape: tuple, transform,
                         crs) -> np.ndarray:
    """
    Read the friction raster (USD/pixel construction cost) resampled to the
    reference grid.  Returns a float32 array with NaN for nodata pixels.
    """
    with rasterio.open(friction_fp) as src:
        friction = np.full(shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=friction,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.enums.Resampling.nearest,
        )
    n_valid = int(np.isfinite(friction).sum())
    print(f"  Friction raster: {n_valid:,} valid pixels loaded")
    return friction


def metric_construction_cost(classified_raster: np.ndarray,
                              friction: np.ndarray) -> dict:
    """
    Sum friction pixel values (USD/pixel) for each road action class.

    - upgrade_cost_usd   : sum over pixels where path follows unpaved road
                           (needs surfacing / upgrading)
    - new_build_cost_usd : sum over pixels where no road exists
                           (needs to be built from scratch)
    - total_cost_usd     : upgrade + new build

    Paved pixels are excluded — road already meets standard, cost = 0.

    Parameters
    ----------
    classified_raster : uint8 array (0=bg, 1=paved, 2=unpaved, 3=to_be_built)
    friction          : float32 array of construction cost in USD/pixel
    """
    upgrade_mask   = classified_raster == 2
    new_build_mask = classified_raster == 3

    upgrade_cost   = float(np.nansum(friction[upgrade_mask]))
    new_build_cost = float(np.nansum(friction[new_build_mask]))
    total_cost     = upgrade_cost + new_build_cost

    return {
        "upgrade_cost_usd":   round(upgrade_cost,   2),
        "new_build_cost_usd": round(new_build_cost, 2),
        "total_cost_usd":     round(total_cost,     2),
    }


def print_construction_cost(result: dict):
    print(f"\n  Construction cost:")
    print(f"  {'─'*48}")
    print(f"  {'Upgrade (unpaved → paved)':<25}: ${result['upgrade_cost_usd']:>15,.0f}")
    print(f"  {'New build':<25}: ${result['new_build_cost_usd']:>15,.0f}")
    print(f"  {'─'*48}")
    print(f"  {'Total':<25}: ${result['total_cost_usd']:>15,.0f}")

# =============================================================================
# METRIC 3 — Deforestation risk
# =============================================================================

def metric_deforestation(classified_raster_fp: Path,
                         out_dir: Path = None,
                         scenario_stem: str = None) -> dict:
    """
    Estimate expected forest cleared using the Damania et al. (2018) distance-decay.

    For each 30 m pixel, the expected clearing equals:
        canopy_frac × pct_cleared(distance_to_nearest_action_road) × pixel_area_km²

    where pct_cleared is linearly interpolated from the Damania lookup table
    (empirical % of land cleared at a given distance from roads of known quality,
    derived from cross-sectional data across Sub-Saharan Africa).

    EDT runs at 90 m on a cropped subregion (low memory), then the resulting
    % surfaces are upsampled to 30 m with bilinear resampling before counting.
    Build takes priority over upgrade where both zones overlap.

    Returns keys:
        defor_upgrade_km2  — expected km² cleared near roads to be upgraded
        defor_build_km2    — expected km² cleared near roads to be built
        defor_action_km2   — combined (upgrade + build)
    """
    import math
    from scipy.ndimage import distance_transform_edt
    from rasterio.warp import reproject as warp_reproject, Resampling as WarpResampling
    from rasterio.transform import Affine

    print(f"  Loading classified raster: {classified_raster_fp.name}")
    with rasterio.open(classified_raster_fp) as cls_src:
        cls_arr       = cls_src.read(1)
        cls_transform = cls_src.transform
        cls_crs       = cls_src.crs

    action_mask = (cls_arr == 2) | (cls_arr == 3)
    if not action_mask.any():
        print("  WARNING: no upgrade or build pixels found — skipping deforestation.")
        return {}

    # Pixel size of the classified raster (~90 m)
    lat_cls    = cls_transform.f + (cls_arr.shape[0] / 2) * cls_transform.e
    cls_px_m_x = abs(cls_transform.a) * 111_320 * math.cos(math.radians(lat_cls))
    cls_px_m_y = abs(cls_transform.e) * 111_320
    cls_px_size_m = (cls_px_m_x + cls_px_m_y) / 2

    # ── Crop cls_arr to a tight subregion at 90 m ─────────────────────────────
    # Pad by enough pixels to cover the largest buffer distance.
    # This keeps the EDT array small — the full classified raster is only read
    # once and then immediately cropped, so memory stays bounded.
    rows, cols = np.where(action_mask)
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    px_pad = math.ceil((DAMANIA_MAX_DIST_KM * 1000 + 200) / cls_px_size_m)
    r0 = max(0, r_min - px_pad);  r1 = min(cls_arr.shape[0], r_max + px_pad + 1)
    c0 = max(0, c_min - px_pad);  c1 = min(cls_arr.shape[1], c_max + px_pad + 1)

    cls_sub = cls_arr[r0:r1, c0:c1]
    cls_sub_transform = Affine(
        cls_transform.a, 0, cls_transform.c + c0 * cls_transform.a,
        0, cls_transform.e, cls_transform.f + r0 * cls_transform.e,
    )
    del cls_arr   # free the full raster — only the subregion is needed from here

    print(f"  Classified subregion: {cls_sub.shape[0]:,} rows × {cls_sub.shape[1]:,} cols "
          f"(~{cls_sub.shape[0]*cls_px_size_m/1000:.0f} km × "
          f"{cls_sub.shape[1]*cls_px_size_m/1000:.0f} km at ~{cls_px_size_m:.0f} m)")

    # ── Resample treecover to 90 m classified subregion grid ─────────────────
    # Reading the 30 m treecover at full resolution and upsampling pct arrays
    # to 30 m creates ~2.7 GB float32 arrays for large scenarios.  Instead,
    # resample the treecover down to the existing 90 m grid — all computation
    # stays small (~9× fewer pixels) with negligible loss in accuracy.
    print(f"  Resampling treecover to 90 m grid from {TREECOVER_FP.name} ...")
    tc_90 = np.zeros(cls_sub.shape, dtype=np.float32)
    with rasterio.open(TREECOVER_FP) as tc_src:
        tc_crs = tc_src.crs
        warp_reproject(
            source=rasterio.band(tc_src, 1),
            destination=tc_90,
            src_transform=tc_src.transform, src_crs=tc_crs,
            dst_transform=cls_sub_transform, dst_crs=cls_crs,
            resampling=WarpResampling.average,
        )
    # Values > 100 are water (Hansen code 200) or nodata (255) — zero out for
    # calculation (no canopy contribution) but remember the mask so the saved
    # TIFs use NaN for those pixels (prevents overlay from inflating forest area)
    tc_90_valid = tc_90 <= 100
    tc_90 = np.where(tc_90_valid, tc_90, 0.0).astype(np.float32)

    # Pixel area at 90 m
    px_area_ha  = cls_px_m_x * cls_px_m_y / 10_000
    canopy_frac = tc_90 / 100.0
    print(f"  Total canopy-weighted forest area in subregion: "
          f"{float(canopy_frac.sum() * px_area_ha):,.0f} ha")

    # ── Distance transforms at 90 m (small array) ─────────────────────────────
    # Scipy distance_transform_edt allocates float64 internally — running on the
    # 90 m subregion keeps peak usage ~0.9 GB instead of 7+ GB at 30 m.
    # .astype(float32) frees the float64 intermediate immediately.
    _inf_sub = np.full(cls_sub.shape, np.inf, dtype=np.float32)

    mask_upgrade = cls_sub == 2
    mask_build   = cls_sub == 3

    dist_upgrade_90 = (distance_transform_edt(~mask_upgrade).astype(np.float32) * cls_px_size_m
                       if mask_upgrade.any() else _inf_sub.copy())
    dist_build_90   = (distance_transform_edt(~mask_build).astype(np.float32) * cls_px_size_m
                       if mask_build.any() else _inf_sub)

    # ── Damania distance-decay: expected forest cleared ───────────────────────
    # For each 90 m pixel, look up the % of land cleared at that distance from
    # the nearest action road (Damania et al. 2018, "Good" quality curve).
    # Build takes priority where upgrade and build zones overlap.
    dist_upgrade_km = dist_upgrade_90 / 1000.0
    dist_build_km   = dist_build_90   / 1000.0

    pct_upgrade = np.where(
        dist_upgrade_km <= DAMANIA_MAX_DIST_KM,
        np.interp(dist_upgrade_km, _damania_dist, _damania_pct), 0.0,
    ).astype(np.float32)
    pct_build = np.where(
        dist_build_km <= DAMANIA_MAX_DIST_KM,
        np.interp(dist_build_km, _damania_dist, _damania_pct), 0.0,
    ).astype(np.float32)

    # Each pixel is influenced only by its nearest action road type (no double-counting)
    pct_upgrade = np.where(dist_build_km  < dist_upgrade_km, 0.0, pct_upgrade).astype(np.float32)
    pct_build   = np.where(dist_upgrade_km <= dist_build_km, 0.0, pct_build).astype(np.float32)

    # Expected cleared forest per pixel = canopy_frac × pct_cleared × px_area_km2
    cleared_upgrade = canopy_frac * pct_upgrade * px_area_ha
    cleared_build   = canopy_frac * pct_build   * px_area_ha
    cleared_action  = cleared_upgrade + cleared_build

    results = {
        "defor_upgrade_ha": round(float(cleared_upgrade.sum()), 2),
        "defor_build_ha":   round(float(cleared_build.sum()),   2),
        "defor_action_ha":  round(float(cleared_action.sum()),  2),
    }

    if SAVE_DEFOR_TIFS:
        if out_dir is None:
            raise ValueError("out_dir must be provided when SAVE_DEFOR_TIFS=True")
        stem = scenario_stem or classified_raster_fp.stem

        # Combined clearing probability per pixel (upgrade and build are mutually exclusive)
        pct_cleared_90 = (pct_upgrade + pct_build).astype(np.float32)

        # loss_pct: subregion only (NaN for water/nodata), same grid as cls_sub
        _nan32 = np.float32("nan")
        loss_pct_90 = np.where(tc_90_valid, tc_90 * pct_cleared_90, _nan32).astype(np.float32)
        with rasterio.open(out_dir / f"{stem}__defor_loss_pct.tif", "w",
                           driver="GTiff", dtype="float32", nodata=float("nan"),
                           width=cls_sub.shape[1], height=cls_sub.shape[0], count=1,
                           crs=cls_crs, transform=cls_sub_transform, compress="lzw") as dst:
            dst.write(loss_pct_90, 1)
        print(f"  Saved → {stem}__defor_loss_pct.tif")

        # remaining_treecover: full-country 90 m (treecover's own grid) so that
        # metric_forest_fragmentation can pass it directly to run_patch_detection
        # without any reprojection or overlay step.
        TARGET_M = 90.0
        with rasterio.open(TREECOVER_FP) as tc_src:
            px_m_fc = abs(tc_src.transform.a) * 111_320
            fc_scale = max(1, round(TARGET_M / px_m_fc))
            fc_h = tc_src.height // fc_scale
            fc_w = tc_src.width  // fc_scale
            tc_raw_fc = tc_src.read(1, out_shape=(fc_h, fc_w), resampling=Resampling.average)
            tc_fc_transform = tc_src.transform * Affine.scale(fc_scale, fc_scale)
            tc_fc_crs = tc_src.crs

        tc_valid_fc = tc_raw_fc <= 100   # water (200) and nodata (255) → invalid
        tc_90_fc    = np.where(tc_valid_fc, tc_raw_fc.astype(np.float32), 0.0)

        # Reproject pct_cleared from cls_sub grid onto the full-country 90 m grid
        pct_cleared_fc = np.zeros((fc_h, fc_w), dtype=np.float32)
        warp_reproject(
            source=pct_cleared_90, destination=pct_cleared_fc,
            src_transform=cls_sub_transform, src_crs=cls_crs,
            dst_transform=tc_fc_transform, dst_crs=tc_fc_crs,
            resampling=WarpResampling.bilinear,
        )
        pct_cleared_fc = np.clip(pct_cleared_fc, 0.0, 1.0)

        remaining_fc = np.where(tc_valid_fc,
                                tc_90_fc * (1.0 - pct_cleared_fc),
                                _nan32).astype(np.float32)
        with rasterio.open(out_dir / f"{stem}__defor_remaining_treecover.tif", "w",
                           driver="GTiff", dtype="float32", nodata=float("nan"),
                           width=fc_w, height=fc_h, count=1,
                           crs=tc_fc_crs, transform=tc_fc_transform, compress="lzw") as dst:
            dst.write(remaining_fc, 1)
        print(f"  Saved → {stem}__defor_remaining_treecover.tif  "
              f"({fc_h}×{fc_w} px, full country)")

        # Total remaining forest: Σ pixels [remaining_pct / 100 × px_area_ha]
        # Clip to Cameroon boundary so pixels in neighbouring countries are excluded.
        import math
        from rasterio.features import geometry_mask as _geometry_mask
        _cmr_adm0 = BASE_DIR / "data/input/cmr_admin_boundaries_humdata/cmr_admin0_em.shp"
        _cmr_gdf  = gpd.read_file(_cmr_adm0)
        if _cmr_gdf.crs.to_epsg() != 4326:
            _cmr_gdf = _cmr_gdf.to_crs("EPSG:4326")
        _outside_cmr = _geometry_mask(
            [g.__geo_interface__ for g in _cmr_gdf.geometry],
            transform=tc_fc_transform, invert=False, out_shape=(fc_h, fc_w),
        )
        remaining_fc_cmr = np.where(_outside_cmr, np.nan, remaining_fc)

        center_lat_fc  = tc_fc_transform.f + (fc_h / 2) * tc_fc_transform.e
        px_m_x_fc      = abs(tc_fc_transform.a) * 111_320 * math.cos(math.radians(center_lat_fc))
        px_m_y_fc      = abs(tc_fc_transform.e) * 111_320
        px_area_ha_fc  = px_m_x_fc * px_m_y_fc / 10_000
        total_remaining_forest_ha = float(np.nansum(remaining_fc_cmr)) / 100.0 * px_area_ha_fc
        results["total_remaining_forest_ha"] = round(total_remaining_forest_ha, 2)

    return results


def print_deforestation(result: dict):
    up  = result.get("defor_upgrade_ha", 0)
    bld = result.get("defor_build_ha",   0)
    act = result.get("defor_action_ha",  0)
    print(f"\n  Deforestation risk — expected forest cleared (Damania et al. 2018, {DAMANIA_CURVE} road, 0–{DAMANIA_MAX_DIST_KM:.0f} km decay):")
    print(f"  {'─'*58}")
    print(f"  {'':>12}  {'Upgrade (ha)':>14}  {'New build (ha)':>14}  {'Combined (ha)':>14}")
    print(f"  {'─'*58}")
    print(f"  {'Expected':>12}  {up:>14,.1f}  {bld:>14,.1f}  {act:>14,.1f}")
    print(f"  {'─'*58}")
    print(f"  Formula: Σ_pixels [ canopy_frac × pct_cleared(distance) × pixel_area_ha ]")


# def metric_mining_capacity(paths_gdf, mines_gdf) -> dict:
#     """Estimate mining capacity unlocked by new road connectivity."""
#     ...


# =============================================================================
# METRIC 4 — Forest fragmentation (via forest_patch_detection)
# =============================================================================

# Patch size bins — (column_key, condition on patch_ha array)
_PSD_BINS = [
    ("gt1000kha",    lambda h: h > 1_000_000),
    ("100k_1000kha", lambda h: (h >= 100_000) & (h <= 1_000_000)),
    ("10k_100kha",   lambda h: (h >= 10_000)  & (h <  100_000)),
    ("1k_10kha",     lambda h: (h >= 1_000)   & (h <  10_000)),
    ("100_1kha",     lambda h: (h >= 100)      & (h <  1_000)),
    ("10_100ha",     lambda h: (h >= 10)       & (h <  100)),
    ("1_10ha",       lambda h: (h >= 1)        & (h <  10)),
    ("lt1ha",        lambda h:  h < 1),
]

_PSD_LABELS = {
    "gt1000kha":    "> 1,000,000 ha",
    "100k_1000kha": "100,000–1,000,000 ha",
    "10k_100kha":   "10,000–100,000 ha",
    "1k_10kha":     "1,000–10,000 ha",
    "100_1kha":     "100–1,000 ha",
    "10_100ha":     "10–100 ha",
    "1_10ha":       "1–10 ha",
    "lt1ha":        "< 1 ha",
}


def _patch_size_distribution(tif_path: Path) -> dict:
    """Compute patch size distribution from a patch-ID raster (uint32, 0=background).

    Returns flat dict with keys psd_{bin}_n (count) and psd_{bin}_ha (total area).
    """
    with rasterio.open(tif_path) as src:
        arr  = src.read(1)
        tf   = src.transform
        lat_c   = tf.f + (src.height / 2) * tf.e
        pix_w   = abs(tf.a) * 111_320 * np.cos(np.radians(lat_c))
        pix_h   = abs(tf.e) * 111_320
        pixel_ha = (pix_w * pix_h) / 10_000

    _, px_counts = np.unique(arr[arr > 0], return_counts=True)
    patch_ha = px_counts * pixel_ha

    result = {}
    for key, condition in _PSD_BINS:
        mask = condition(patch_ha)
        result[f"psd_{key}_n"]  = int(mask.sum())
        result[f"psd_{key}_ha"] = round(float(patch_ha[mask].sum()), 2)
    return result


def metric_forest_fragmentation(scenario_stem: str,
                                 remaining_tif: Path,
                                 cls_sub: np.ndarray,
                                 cls_sub_transform,
                                 cls_crs,
                                 out_dir: Path) -> dict:
    """
    Estimate forest fragmentation caused by action roads using run_patch_detection.

    remaining_tif is the full-country 90 m treecover produced by metric_deforestation
    (same grid as the original treecover resampled to 90 m). It is passed directly to
    run_patch_detection — no overlay or reprojection needed.

    Returns a dict with frag_* keys (action state, baseline, and deltas).
    """
    from forest_patch_detection import PatchConfig, run_patch_detection

    action_mask = (cls_sub == 2) | (cls_sub == 3)
    if not action_mask.any():
        print("  No action roads — fragmentation metric skipped.")
        return {}

    if not BASELINE_FRAG_GPKG.exists():
        print(f"  Warning: baseline gpkg not found: {BASELINE_FRAG_GPKG}")
        print(f"  Run forest-patch-detection-run__base.py first.")
        return {}

    # Use remaining_tif (full-country, treecover's 90 m grid) if available,
    # otherwise fall back to the original treecover (no deforestation effect).
    if remaining_tif.exists():
        action_treecover = remaining_tif
        print(f"  Using full-country remaining treecover: {remaining_tif.name}")
    else:
        action_treecover = TREECOVER_FP
        print(f"  Warning: {remaining_tif.name} not found — using original treecover")

    # Burn action road geometries explicitly as barriers (in addition to treecover
    # reduction from remaining_tif) so that road pixels create hard patch boundaries
    action_roads_gpkg = out_dir / f"{scenario_stem}__action_roads.gpkg"
    extra_burn = action_roads_gpkg if action_roads_gpkg.exists() else None
    if extra_burn is None:
        print(f"  Note: {scenario_stem}__action_roads.gpkg not found — "
              f"road barriers not burned (enable RUN_CONSTRUCTION_COST to generate it)")

    # Run patch detection on the action scenario treecover
    action_cfg = PatchConfig(
        treecover_path=action_treecover,
        prefix=scenario_stem,
        output_dir=out_dir,
        forest_threshold=FOREST_THRESHOLD,
        min_patch_size_ha=FRAGMENTATION_MIN_PATCH_HA,
        admin_level=2,
        burn_paved_roads=True,
        burn_unpaved_roads=False,
        extra_burn_gpkg=extra_burn,
    )
    print("\n  Running patch detection for action scenario …")
    action_tif, action_gpkg = run_patch_detection(action_cfg)

    # 5. Aggregate country-level stats from a patch-detection GeoPackage
    def _aggregate(gpkg_path: Path) -> dict:
        gdf = gpd.read_file(gpkg_path)
        total_forest_ha  = float(gdf["forest_ha"].sum())
        total_area_ha    = float(gdf["area_ha"].sum())
        total_n_patches  = int(gdf["n_patches"].sum())
        total_pct_forest = round(100.0 * total_forest_ha / total_area_ha, 2) if total_area_ha > 0 else 0.0
        w = gdf["n_patches"].values.astype(float)
        mean_patch_ha = float(np.average(gdf["mean_patch_ha"].values, weights=w)) if w.sum() > 0 else 0.0
        frag_index = (total_n_patches - 1) / (total_forest_ha - 1) if total_forest_ha > 1 else 0.0
        return {
            "forest_ha":     round(total_forest_ha, 2),
            "pct_forest":    total_pct_forest,
            "n_patches":     total_n_patches,
            "mean_patch_ha": round(mean_patch_ha, 2),
            "frag_index":    round(frag_index, 8),
        }

    baseline = _aggregate(BASELINE_FRAG_GPKG)
    action   = _aggregate(action_gpkg)

    # Patch size distributions from TIF rasters
    baseline_tif = BASELINE_FRAG_GPKG.with_suffix(".tif")
    if baseline_tif.exists():
        print("  Computing patch size distributions …")
        baseline_psd = _patch_size_distribution(baseline_tif)
        action_psd   = _patch_size_distribution(action_tif)
    else:
        print(f"  Warning: baseline TIF not found ({baseline_tif.name}) — skipping PSD")
        baseline_psd = None
        action_psd   = None

    # 6. Print comparison table
    print(f"\n  Forest fragmentation (paved-road baseline, {FRAGMENTATION_MIN_PATCH_HA:.0f} ha min patch):")
    print(f"  {'─'*72}")
    print(f"  {'Metric':<26}  {'Baseline':>12}  {'Action':>12}  {'Delta':>12}")
    print(f"  {'─'*72}")
    for key, label in [
        ("forest_ha",     "Forest (ha)"),
        ("pct_forest",    "Forest (%)"),
        ("n_patches",     "N patches"),
        ("mean_patch_ha", "Mean patch (ha)"),
        ("frag_index",    "Frag index"),
    ]:
        b, a = baseline[key], action[key]
        d = a - b
        if key == "n_patches":
            print(f"  {label:<26}  {b:>12,}  {a:>12,}  {d:>+12,}")
        elif key == "frag_index":
            print(f"  {label:<26}  {b:>12.6f}  {a:>12.6f}  {d:>+12.6f}")
        else:
            print(f"  {label:<26}  {b:>12.1f}  {a:>12.1f}  {d:>+12.1f}")
    print(f"  {'─'*72}")

    if baseline_psd is not None:
        print(f"\n  Patch size distribution:")
        print(f"  {'─'*80}")
        print(f"  {'Size class':<22}  {'Base N':>8}  {'Act N':>8}  {'ΔN':>8}  "
              f"{'Base ha':>12}  {'Act ha':>12}  {'Δha':>12}")
        print(f"  {'─'*80}")
        for key, _ in _PSD_BINS:
            bn = baseline_psd[f"psd_{key}_n"]
            an = action_psd[f"psd_{key}_n"]
            bh = baseline_psd[f"psd_{key}_ha"]
            ah = action_psd[f"psd_{key}_ha"]
            print(f"  {_PSD_LABELS[key]:<22}  {bn:>8,}  {an:>8,}  {an-bn:>+8,}  "
                  f"  {bh:>10,.0f}  {ah:>10,.0f}  {ah-bh:>+10,.0f}")
        print(f"  {'─'*80}")

    ret = {
        "frag_forest_ha":              action["forest_ha"],
        "frag_pct_forest":             action["pct_forest"],
        "frag_n_patches":              action["n_patches"],
        "frag_mean_patch_ha":          action["mean_patch_ha"],
        "frag_index":                  action["frag_index"],
        "frag_baseline_forest_ha":     baseline["forest_ha"],
        "frag_baseline_pct_forest":    baseline["pct_forest"],
        "frag_baseline_n_patches":     baseline["n_patches"],
        "frag_baseline_mean_patch_ha": baseline["mean_patch_ha"],
        "frag_baseline_frag_index":    baseline["frag_index"],
        "frag_delta_forest_ha":        round(action["forest_ha"]     - baseline["forest_ha"],     2),
        "frag_delta_pct_forest":       round(action["pct_forest"]    - baseline["pct_forest"],    4),
        "frag_delta_n_patches":        action["n_patches"]           - baseline["n_patches"],
        "frag_delta_mean_patch_ha":    round(action["mean_patch_ha"] - baseline["mean_patch_ha"], 2),
        "frag_delta_frag_index":       round(action["frag_index"]    - baseline["frag_index"],    8),
    }

    if baseline_psd is not None:
        for key, _ in _PSD_BINS:
            ret[f"frag_patch_size_dis_{key}_n"]          = action_psd[f"psd_{key}_n"]
            ret[f"frag_patch_size_dis_{key}_ha"]         = action_psd[f"psd_{key}_ha"]
            ret[f"frag_patch_size_dis_baseline_{key}_n"] = baseline_psd[f"psd_{key}_n"]
            ret[f"frag_patch_size_dis_baseline_{key}_ha"]= baseline_psd[f"psd_{key}_ha"]
            ret[f"frag_patch_size_dis_delta_{key}_n"]    = action_psd[f"psd_{key}_n"]  - baseline_psd[f"psd_{key}_n"]
            ret[f"frag_patch_size_dis_delta_{key}_ha"]   = round(action_psd[f"psd_{key}_ha"] - baseline_psd[f"psd_{key}_ha"], 2)

    return ret



# =============================================================================
# MAIN
# =============================================================================

def run_assessment(paths_fp: Path, downsample: int = 1,
                   out_dir: Path = None) -> pd.DataFrame:
    """
    Run all implemented impact metrics for a given path scenario.

    out_dir : directory where all output files are written.
              Defaults to results/impact-assessment/ next to this script.
              In batch mode, callers pass results/impact-assessment/<folder_name>/.

    Returns a one-row DataFrame with all metric columns.
    """
    _t0 = time.perf_counter()
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
    if out_dir is None:
        out_dir = Path(__file__).parent / "results/impact-assessment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics …")
    if not TREECOVER_FP.exists():
        print(f"  WARNING: treecover raster not found at {TREECOVER_FP}")
        print(f"  Run notebook/01_clean-forests.py first to generate it.")

    # Metric 1: road classification
    road_class, classified_raster = metric_road_classification(path_raster, road_raster, px_km)
    print_road_classification(road_class)
    classified_fp = out_dir / f"{paths_fp.stem}__classified.tif"
    save_classified_raster(classified_raster, transform, crs, classified_fp)

    # Metric 2: construction cost
    cost_metrics = {}
    if RUN_CONSTRUCTION_COST:
        friction = load_friction_values(FRICTION_FP, shape, transform, crs)
        cost_metrics = metric_construction_cost(classified_raster, friction)
        print_construction_cost(cost_metrics)
        save_classified_lines(paths_gdf, classified_raster, transform, crs,
                              friction, out_dir, paths_fp.stem)

    # Metric 3: deforestation risk
    defor_metrics = {}
    if RUN_DEFORESTATION:
        defor_metrics = metric_deforestation(
            classified_fp,
            out_dir=out_dir,
            scenario_stem=paths_fp.stem,
        )
        if defor_metrics:
            print_deforestation(defor_metrics)

    # Metric 4: forest fragmentation
    frag_metrics = {}
    if RUN_FRAGMENTATION:
        remaining_tif = out_dir / f"{paths_fp.stem}__defor_remaining_treecover.tif"
        frag_metrics = metric_forest_fragmentation(
            scenario_stem=paths_fp.stem,
            remaining_tif=remaining_tif,
            cls_sub=classified_raster,
            cls_sub_transform=transform,
            cls_crs=crs,
            out_dir=out_dir,
        )

    # ── Compile results ───────────────────────────────────────────────────────
    results = {"scenario": paths_fp.stem, **road_class, **cost_metrics}
    results.update(defor_metrics)
    results.update(frag_metrics)

    results_df = pd.DataFrame([results])
    out_fp = out_dir / f"{paths_fp.stem}__assessment.csv"
    results_df.to_csv(out_fp, index=False)
    print(f"\n  Saved → {out_fp}")

    elapsed = time.perf_counter() - _t0
    print(f"\nDone. Total time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Impact assessment for mine-to-port road scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single file:  python impact_assessment.py results/least-cost-paths/experiment_00/late_stage__kribi__base__no_mask__ds1.gpkg\n"
            "Batch folder: python impact_assessment.py experiment_00\n"
            "              (finds all .gpkg files under results/least-cost-paths/experiment_00/\n"
            "               and writes outputs to results/impact-assessment/experiment_00/)"
        ),
    )
    parser.add_argument("input", type=Path,
                        help=".gpkg file path, or folder name under results/least-cost-paths/")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Reference grid downsample factor (default: 1 = native ~90 m).")
    args = parser.parse_args()

    _model_dir  = Path(__file__).parent
    _paths_root = _model_dir / "results/least-cost-paths"
    _out_root   = _model_dir / "results/impact-assessment"

    inp = args.input

    if inp.suffix == ".gpkg":
        # ── Single-file mode (original behaviour) ────────────────────────────
        if not inp.exists():
            sys.exit(f"Error: '{inp}' not found.")
        run_assessment(inp, downsample=args.downsample)
    else:
        # ── Batch-folder mode ────────────────────────────────────────────────
        # Accept both a bare folder name ("experiment_00") and a full path.
        folder_name = inp.name if inp.is_absolute() or inp.parent != Path(".") else str(inp)
        paths_dir   = _paths_root / folder_name
        if not paths_dir.exists():
            sys.exit(f"Error: folder '{paths_dir}' not found.")
        gpkg_files = sorted(paths_dir.glob("*.gpkg"))
        if not gpkg_files:
            sys.exit(f"Error: no .gpkg files found in '{paths_dir}'.")
        out_dir = _out_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch mode: {len(gpkg_files)} scenario(s) in '{folder_name}' → {out_dir}")
        for i, gpkg_fp in enumerate(gpkg_files, 1):
            print(f"\n{'━'*62}")
            print(f"  [{i}/{len(gpkg_files)}] {gpkg_fp.name}")
            print(f"{'━'*62}")
            run_assessment(gpkg_fp, downsample=args.downsample, out_dir=out_dir)


if __name__ == "__main__":
    main()
