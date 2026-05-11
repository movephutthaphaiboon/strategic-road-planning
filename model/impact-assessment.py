#!/usr/bin/env python3
"""
impact-assessment.py — Impact assessment for mine-to-port road scenarios.

Takes a generated path GeoPackage (output of path-generator) and computes
impact metrics for the proposed road network.

Metrics implemented:
  [1] Road classification    — km of paved / unpaved / to_be_built
  [2] Construction cost      — USD for upgrade and new-build segments
  [3] Deforestation risk     — ha of forest within buffer zones of action roads
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

BASE_DIR       = Path(__file__).parent.parent
ROADS_FP       = BASE_DIR / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"
FRICTION_FP    = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
TREECOVER_FP   = BASE_DIR / "data/output/processed-hansen-treecover/cameroon_treecover2024_30m.tif"

ROAD_TYPE_COL      = "liu_surface"
FOREST_THRESHOLD    = 10    # minimum canopy cover (%) to count as forest (FAO-aligned)
MIN_FOREST_PATCH_HA = 1.0   # minimum patch size (ha) to count as a distinct forest patch
BUFFER_DISTANCES_M  = [0, 100, 250, 500, 750, 1000]

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
    out_fp = out_dir / f"action_roads_{scenario_stem}.gpkg"
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
                         save_buffer_rasters: bool = False,
                         out_dir: Path = None,
                         scenario_stem: str = None) -> dict:
    """
    Estimate the area of forest at risk of deforestation from road actions.

    Reads the classified raster produced by save_classified_raster() and the
    30 m Hansen tree-cover layer.  For pixels classified as 'unpaved' (upgrade)
    or 'to_be_built' (new construction), counts how much forest falls within
    each of several buffer distances from those roads.

    Buffer distances: 0 m (road footprint only), 100, 250, 500, 750, 1000 m.

    No double-counting: rasterize_paths() already calls unary_union() before
    burning, so shared segments across multiple mines are merged into one
    geometry.  The distance transform then runs on a binary mask — each pixel
    is counted once regardless of how many mines use that corridor.

    Strategy
    --------
    1. Load the classified raster and the treecover window that covers the
       action-road bounding box plus 1 km padding.
    2. Reproject the classified raster to the 30 m treecover pixel grid
       (nearest-neighbour, preserving class values).
    3. Compute one Euclidean distance transform per road class (upgrade / build).
       The combined (action) distance is np.minimum of the two — no third EDT.
    4. Threshold at each buffer distance → upsample to 30 m → weighted sum → ha.

    Optional raster output (save_buffer_rasters=True)
    -------------------------------------------------
    Saves one GeoTIFF per buffer distance to out_dir, named:
        defor_buffer_{buf_m}m_{scenario_stem}.tif
    Pixel values match the classified raster convention:
        0 = not in any action-road buffer
        2 = within buffer of a road to be upgraded
        3 = within buffer of a road to be built  (overwrites 2 where both overlap)

    Returns keys (per class × buffer):
        defor_upgrade_Xm_ha  — forest within X m of roads to be upgraded
        defor_build_Xm_ha    — forest within X m of roads to be built
        defor_action_Xm_ha   — combined (upgrade + build)
    """
    import math
    from scipy.ndimage import distance_transform_edt
    from rasterio.warp import reproject as warp_reproject, Resampling as WarpResampling
    from rasterio.windows import from_bounds as window_from_bounds
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

    px_pad = math.ceil((BUFFER_DISTANCES_M[-1] + 200) / cls_px_size_m)
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

    # ── Read 30 m treecover for the same geographic extent ────────────────────
    # Derive bounds from cls_sub's transform so the windows align.
    sub_left   = cls_sub_transform.c
    sub_top    = cls_sub_transform.f
    sub_right  = sub_left + cls_sub.shape[1] * cls_sub_transform.a
    sub_bottom = sub_top  + cls_sub.shape[0] * cls_sub_transform.e

    print(f"  Reading treecover window from {TREECOVER_FP.name} ...")
    with rasterio.open(TREECOVER_FP) as tc_src:
        tc_crs = tc_src.crs

        if cls_crs != tc_crs:
            from pyproj import Transformer
            sub_left, sub_bottom, sub_right, sub_top = Transformer.from_crs(
                cls_crs, tc_crs, always_xy=True
            ).transform_bounds(sub_left, sub_bottom, sub_right, sub_top)

        b          = tc_src.bounds
        sub_left   = max(sub_left,   b.left);   sub_right  = min(sub_right,  b.right)
        sub_top    = min(sub_top,    b.top);    sub_bottom = max(sub_bottom, b.bottom)

        tc_window    = window_from_bounds(sub_left, sub_bottom, sub_right, sub_top, tc_src.transform)
        tc_window    = tc_window.round_lengths().round_offsets()
        tc_arr       = tc_src.read(1, window=tc_window)
        tc_transform = tc_src.window_transform(tc_window)

    tc_shape = tc_arr.shape
    print(f"  Treecover window: {tc_shape[0]:,} rows × {tc_shape[1]:,} cols "
          f"(~{tc_shape[0]*30/1000:.0f} km × {tc_shape[1]*30/1000:.0f} km at 30 m)")

    # Pixel area at 30 m (used for ha counts)
    lat_tc     = tc_transform.f + (tc_shape[0] / 2) * tc_transform.e
    px_m_x     = abs(tc_transform.a) * 111_320 * math.cos(math.radians(lat_tc))
    px_m_y     = abs(tc_transform.e) * 111_320
    px_area_ha = px_m_x * px_m_y / 10_000

    # Canopy fraction (0.0–1.0) — used to weight each pixel's contribution.
    # A pixel with 80% cover counts as 0.8 × px_area_ha of forest.
    canopy_frac = tc_arr.astype(np.float32) / 100.0
    print(f"  Total canopy-weighted forest area in window: "
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

    # ── Per-buffer: upsample 90 m mask → 30 m, count forest, save raster ──────
    # Build the combined uint8 mask at 90 m for each buffer distance, upsample
    # to the 30 m treecover grid (nearest-neighbour), then count.  buf_30m is
    # reused each iteration to avoid accumulating large arrays.
    if save_buffer_rasters:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save_buffer_rasters=True")
        stem = scenario_stem or classified_raster_fp.stem
        buf_profile = {
            "driver": "GTiff", "dtype": np.uint8, "nodata": 0,
            "width": tc_shape[1], "height": tc_shape[0], "count": 1,
            "crs": tc_crs, "transform": tc_transform, "compress": "lzw",
        }
        print(f"  Saving buffer rasters to {out_dir} ...")

    results = {}
    buf_30m = np.zeros(tc_shape, dtype=np.uint8)   # reused each iteration

    for buf_m in BUFFER_DISTANCES_M:
        # 90 m buffer mask: build takes priority over upgrade where they overlap
        buf_90m = np.zeros(cls_sub.shape, dtype=np.uint8)
        buf_90m[(dist_upgrade_90 <= buf_m) & (dist_build_90 > buf_m)] = 2
        buf_90m[ dist_build_90   <= buf_m]                             = 3

        # Upsample to 30 m treecover grid
        buf_30m[:] = 0
        warp_reproject(
            source=buf_90m, destination=buf_30m,
            src_transform=cls_sub_transform, src_crs=cls_crs,
            dst_transform=tc_transform,      dst_crs=tc_crs,
            resampling=WarpResampling.nearest,
            src_nodata=0, dst_nodata=0,
        )

        # Weighted forest area: each pixel contributes (canopy % / 100) × px_area_ha
        results[f"defor_upgrade_{buf_m}m_ha"] = round(float(((buf_30m == 2) * canopy_frac).sum()) * px_area_ha, 2)
        results[f"defor_build_{buf_m}m_ha"]   = round(float(((buf_30m == 3) * canopy_frac).sum()) * px_area_ha, 2)
        results[f"defor_action_{buf_m}m_ha"]  = round(float(((buf_30m >  0) * canopy_frac).sum()) * px_area_ha, 2)

        if save_buffer_rasters:
            out_fp = out_dir / f"defor_buffer_{buf_m}m_{stem}.tif"
            with rasterio.open(out_fp, "w", **buf_profile) as dst:
                dst.write(buf_30m, 1)
            print(f"    {out_fp.name}")

    return results


def print_deforestation(result: dict):
    print(f"\n  Deforestation risk — forest area within buffer of action roads:")
    print(f"  {'─'*58}")
    print(f"  {'Buffer':>10}  {'Upgrade (ha)':>14}  {'New build (ha)':>14}  {'Combined (ha)':>14}")
    print(f"  {'─'*58}")
    for d in BUFFER_DISTANCES_M:
        label = "direct" if d == 0 else f"{d} m"
        up  = result.get(f"defor_upgrade_{d}m_ha", 0)
        bld = result.get(f"defor_build_{d}m_ha",   0)
        act = result.get(f"defor_action_{d}m_ha",  0)
        print(f"  {label:>10}  {up:>14,.1f}  {bld:>14,.1f}  {act:>14,.1f}")
    print(f"  {'─'*58}")
    print(f"  Values are canopy-cover-weighted forest area (canopy % / 100 × pixel area)")


# def metric_mining_capacity(paths_gdf, mines_gdf) -> dict:
#     """Estimate mining capacity unlocked by new road connectivity."""
#     ...


# =============================================================================
# METRIC 4 — Forest fragmentation
# =============================================================================

def metric_forest_fragmentation(classified_raster_fp: Path,
                                 road_raster: np.ndarray,
                                 window_km: float = 100.0,
                                 out_dir: Path | None = None,
                                 scenario_stem: str = "") -> dict:
    """
    Estimate the increase in number of forest patches caused by action roads.

    Approach: sliding non-overlapping windows across the treecover raster.
    For each window that contains at least one action road pixel:
      1. Build a binary forest mask (canopy ≥ FOREST_THRESHOLD %).
      2. Count connected forest patches BEFORE burning the roads.
      3. Burn action road pixels (values 2 and 3) to non-forest.
      4. Count patches AFTER.
      5. Accumulate the difference.

    Non-overlapping windows avoid double-counting patches that span window
    boundaries.  Windows with no action roads are skipped for efficiency.

    Returns
    -------
    fragmentation_new_patches      : total additional patches across all windows
    fragmentation_windows_processed: number of windows that contained roads
    """
    import math
    from rasterio.transform import Affine
    from scipy.ndimage import label as ndimage_label
    from rasterio.warp import reproject as warp_reproject, Resampling as WarpResampling
    from rasterio.windows import Window, from_bounds as window_from_bounds

    struct_8 = np.ones((3, 3), dtype=int)   # 8-connectivity for forest patches

    print(f"  Loading classified raster: {classified_raster_fp.name}")
    with rasterio.open(classified_raster_fp) as cls_src:
        cls_arr       = cls_src.read(1)
        cls_transform = cls_src.transform
        cls_crs       = cls_src.crs

    action_mask = (cls_arr == 2) | (cls_arr == 3)
    if not action_mask.any():
        print("  No action roads — fragmentation metric skipped.")
        return {}

    # Window size in treecover pixels (~30 m)
    with rasterio.open(TREECOVER_FP) as tc_src:
        tc_transform = tc_src.transform
        tc_crs       = tc_src.crs
        tc_height    = tc_src.height
        tc_width     = tc_src.width

    lat_c        = tc_transform.f + (tc_height / 2) * tc_transform.e
    px_size_m    = abs(tc_transform.a) * 111_320 * math.cos(math.radians(lat_c))
    win_px       = round(window_km * 1_000 / px_size_m)
    px_area_km2  = (px_size_m ** 2) / 1_000_000
    min_patch_px = max(1, round((MIN_FOREST_PATCH_HA / 100) / px_area_km2))

    print(f"  Window: {window_km:.0f} km = {win_px} × {win_px} px  |  "
          f"forest ≥ {FOREST_THRESHOLD}%  |  min patch ≥ {MIN_FOREST_PATCH_HA} ha ({min_patch_px} px)")

    n_windows              = 0
    n_windows_action_roads = 0
    total_patches_before   = 0
    total_patches_after    = 0

    # Output grids: one cell per window
    grid_rows = math.ceil(tc_height / win_px)
    grid_cols = math.ceil(tc_width  / win_px)
    _nan = float("nan")
    grid_index_before  = np.full((grid_rows, grid_cols), _nan, dtype=np.float32)
    grid_index_after   = np.full((grid_rows, grid_cols), _nan, dtype=np.float32)
    grid_patches_before = np.full((grid_rows, grid_cols), -1,  dtype=np.int32)
    grid_patches_after  = np.full((grid_rows, grid_cols), -1,  dtype=np.int32)

    with rasterio.open(TREECOVER_FP) as tc_src:
        for row_off in range(0, tc_height, win_px):
            for col_off in range(0, tc_width, win_px):
                grid_r = row_off // win_px
                grid_c = col_off // win_px

                win_h = min(win_px, tc_height - row_off)
                win_w = min(win_px, tc_width  - col_off)
                window        = Window(col_off, row_off, win_w, win_h)
                win_transform = tc_src.window_transform(window)

                # Quick overlap check using the 90 m classified raster
                w_left   = win_transform.c
                w_top    = win_transform.f
                w_right  = w_left + win_w * win_transform.a
                w_bottom = w_top  + win_h * win_transform.e

                cls_win = window_from_bounds(w_left, w_bottom, w_right, w_top,
                                             cls_transform)
                cls_win = cls_win.round_lengths().round_offsets()
                r0 = max(0, int(cls_win.row_off))
                r1 = min(cls_arr.shape[0], r0 + int(cls_win.height))
                c0 = max(0, int(cls_win.col_off))
                c1 = min(cls_arr.shape[1], c0 + int(cls_win.width))

                if r1 <= r0 or c1 <= c0:
                    continue

                has_action = ((cls_arr[r0:r1, c0:c1] == 2) |
                              (cls_arr[r0:r1, c0:c1] == 3)).any()

                # Read treecover — forest extent is the shared denominator
                tc_win    = tc_src.read(1, window=window)
                forest    = (tc_win >= FOREST_THRESHOLD).astype(np.uint8)
                forest_px = int(forest.sum())
                if forest_px < 2:
                    continue

                # Reproject both rasters onto this window's 30 m grid
                cls_repr  = np.zeros((win_h, win_w), dtype=np.uint8)
                road_repr = np.zeros((win_h, win_w), dtype=np.uint8)
                warp_reproject(
                    source=cls_arr, destination=cls_repr,
                    src_transform=cls_transform, src_crs=cls_crs,
                    dst_transform=win_transform, dst_crs=tc_crs,
                    resampling=WarpResampling.nearest,
                    src_nodata=0, dst_nodata=0,
                )
                warp_reproject(
                    source=road_raster, destination=road_repr,
                    src_transform=cls_transform, src_crs=cls_crs,
                    dst_transform=win_transform, dst_crs=tc_crs,
                    resampling=WarpResampling.nearest,
                    src_nodata=0, dst_nodata=0,
                )

                # Before: existing road network burned
                forest_before = forest.copy()
                forest_before[road_repr > 0] = 0
                labeled_before, _ = ndimage_label(forest_before, structure=struct_8)
                if labeled_before.max() > 0:
                    sizes_before = np.bincount(labeled_before.ravel())
                    n_before = int((sizes_before[1:] >= min_patch_px).sum())
                else:
                    n_before = 0

                # After: additionally burn action roads
                forest_after = forest_before.copy()
                forest_after[(cls_repr == 2) | (cls_repr == 3)] = 0
                labeled_after, _ = ndimage_label(forest_after, structure=struct_8)
                if labeled_after.max() > 0:
                    sizes_after = np.bincount(labeled_after.ravel())
                    n_after = int((sizes_after[1:] >= min_patch_px).sum())
                else:
                    n_after = 0

                forest_km2 = forest_px * px_area_km2
                denom = max(forest_km2 - 1, 1e-9)
                grid_index_before [grid_r, grid_c] = (n_before - 1) / denom
                grid_index_after  [grid_r, grid_c] = (n_after  - 1) / denom
                grid_patches_before[grid_r, grid_c] = n_before
                grid_patches_after [grid_r, grid_c] = n_after

                total_patches_before += n_before
                total_patches_after  += n_after
                n_windows += 1
                if has_action:
                    n_windows_action_roads += 1

    # Country-level index: sum patches across all windows / total forest pixels
    # (approximated as sum-of-window patches over sum-of-window forest extents)
    # Re-read full treecover for exact country totals
    with rasterio.open(TREECOVER_FP) as tc_src:
        # Read in blocks to avoid loading the full raster at once
        total_forest_px = 0
        for _, block_win in tc_src.block_windows(1):
            blk = tc_src.read(1, window=block_win)
            total_forest_px += int((blk >= FOREST_THRESHOLD).sum())

    total_forest_km2     = total_forest_px * px_area_km2
    country_index_before = (total_patches_before - 1) / max(total_forest_km2 - 1, 1e-9)
    country_index_after  = (total_patches_after  - 1) / max(total_forest_km2 - 1, 1e-9)

    # Save index_before and index_after grids (one pixel = one window)
    if out_dir is not None:
        win_label      = f"{int(window_km)}km"
        grid_transform = Affine(
            win_px * tc_transform.a, 0.0, tc_transform.c,
            0.0, win_px * tc_transform.e, tc_transform.f,
        )
        grid_common = dict(
            driver="GTiff", count=1, width=grid_cols, height=grid_rows,
            crs=tc_crs, transform=grid_transform, compress="lzw",
        )
        files = [
            (f"fragmentation_index_before_{win_label}_{scenario_stem}.tif",   "float32", _nan,  grid_index_before),
            (f"fragmentation_index_after_{win_label}_{scenario_stem}.tif",    "float32", _nan,  grid_index_after),
            (f"fragmentation_patches_before_{win_label}_{scenario_stem}.tif", "int32",   -1,    grid_patches_before),
            (f"fragmentation_patches_after_{win_label}_{scenario_stem}.tif",  "int32",   -1,    grid_patches_after),
        ]
        for fname, dtype, nodata, arr in files:
            fp = out_dir / fname
            with rasterio.open(fp, "w", dtype=dtype, nodata=nodata, **grid_common) as dst:
                dst.write(arr, 1)
            print(f"  Saved → {fp.name}")

    print(f"  Windows processed: {n_windows}  |  with action roads: {n_windows_action_roads}")
    win_label = f"{int(window_km)}km"
    return {
        f"fragmentation_patches_before_{win_label}":              total_patches_before,
        f"fragmentation_patches_after_{win_label}":               total_patches_after,
        f"fragmentation_index_before_{win_label}":                round(country_index_before, 8),
        f"fragmentation_index_after_{win_label}":                 round(country_index_after,  8),
        f"fragmentation_n_windows_with_action_roads_{win_label}": n_windows_action_roads,
    }


def print_forest_fragmentation(results: dict):
    window_sizes = [10, 50, 100, 150]
    print(f"\n  Forest fragmentation (sensitivity test):")
    print(f"  {'─'*80}")
    print(f"  {'Window':>10}  {'Patches before':>16}  {'Patches after':>15}  {'Index before':>13}  {'Index after':>12}")
    print(f"  {'─'*80}")
    for w in window_sizes:
        label   = f"{w} km"
        pb      = results.get(f"fragmentation_patches_before_{w}km", "—")
        pa      = results.get(f"fragmentation_patches_after_{w}km",  "—")
        ib      = results.get(f"fragmentation_index_before_{w}km",   "—")
        ia      = results.get(f"fragmentation_index_after_{w}km",    "—")
        pb_str  = f"{pb:,}"      if isinstance(pb, int)   else pb
        pa_str  = f"{pa:,}"      if isinstance(pa, int)   else pa
        ib_str  = f"{ib:.6f}"   if isinstance(ib, float) else ib
        ia_str  = f"{ia:.6f}"   if isinstance(ia, float) else ia
        print(f"  {label:>10}  {pb_str:>16}  {pa_str:>15}  {ib_str:>13}  {ia_str:>12}")
    print(f"  {'─'*80}")
    print(f"  (non-overlapping windows, 8-connectivity, forest ≥ {FOREST_THRESHOLD}% canopy, "
          f"min patch ≥ {MIN_FOREST_PATCH_HA} ha, index unit: patches per km²)")


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
    if not TREECOVER_FP.exists():
        print(f"  WARNING: treecover raster not found at {TREECOVER_FP}")
        print(f"  Run notebook/01_clean-forests.py first to generate it.")

    # Metric 1: road classification
    road_class, classified_raster = metric_road_classification(path_raster, road_raster, px_km)
    print_road_classification(road_class)
    save_classified_raster(classified_raster, transform, crs,
                           out_dir / f"classified_{paths_fp.stem}.tif")

    # Metric 2: construction cost
    friction = load_friction_values(FRICTION_FP, shape, transform, crs)
    cost_metrics = metric_construction_cost(classified_raster, friction)
    print_construction_cost(cost_metrics)

    # Action roads vector file
    save_classified_lines(paths_gdf, classified_raster, transform, crs,
                          friction, out_dir, paths_fp.stem)

    # Metric 3: deforestation risk
    classified_fp = out_dir / f"classified_{paths_fp.stem}.tif"
    defor_metrics = metric_deforestation(
        classified_fp,
        save_buffer_rasters=True,
        out_dir=out_dir,
        scenario_stem=paths_fp.stem,
    )
    if defor_metrics:
        print_deforestation(defor_metrics)

    # Metric 4: forest fragmentation (sensitivity test across window sizes)
    frag_metrics = {}
    for win_km in [10, 50, 100, 150]:
        frag_metrics.update(metric_forest_fragmentation(
            classified_fp, road_raster=road_raster, window_km=win_km,
            out_dir=out_dir, scenario_stem=paths_fp.stem,
        ))
    if frag_metrics:
        print_forest_fragmentation(frag_metrics)

    # ── Compile results ───────────────────────────────────────────────────────
    results = {"scenario": paths_fp.stem, **road_class, **cost_metrics}
    results.update(defor_metrics)
    results.update(frag_metrics)

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
