"""
Forest patch detection — core module.

Exposes run_patch_detection(config) which produces:
  <output_path>.tif   — country-level patch-ID raster (uint32, 0 = background)
  <output_path>.gpkg  — admin boundary vectors with fragmentation stats

Called from forest-patch-detection-run.py (or impact-assessment.py).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from scipy.ndimage import label as ndimage_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_ROADS = _REPO_ROOT / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data/output/forest-patch-detection"
_ADMIN_DIR = _REPO_ROOT / "data/input/cmr_admin_boundaries_humdata"

_ADMIN_NAME_COL = {0: "adm0_name", 1: "adm1_name", 2: "adm2_name1", 3: "adm3_name1"}
_STRUCT_8 = np.ones((3, 3), dtype=int)  # 8-connectivity


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PatchConfig:
    treecover_path: Path
    prefix: str = "no_action"          # scenario label prepended to output filenames
    output_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT_DIR)
    forest_threshold: int = 10          # % tree cover to classify as forest
    min_patch_size_ha: float = 1.0      # patches smaller than this are discarded
    admin_level: int = 2                # 0=country, 1=region, 2=department, 3=arrondissement
    burn_paved_roads: bool = False      # treat paved roads as non-forest barriers
    burn_unpaved_roads: bool = False    # treat unpaved roads as non-forest barriers
    road_path: Path = field(default_factory=lambda: _DEFAULT_ROADS)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _load_treecover_90m(treecover_path: Path):
    """Read treecover at 90 m (3× downsample from 30 m) using average resampling."""
    with rasterio.open(treecover_path) as src:
        scale = 3
        out_h = src.height // scale
        out_w = src.width // scale
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
            masked=True,
        )
        out_transform = src.transform * Affine.scale(scale, scale)
        crs = src.crs
    return arr, out_transform, crs


def _compute_pixel_area_ha(transform: Affine) -> float:
    """Geodetically-accurate pixel area in hectares (WGS84)."""
    geod = Geod(ellps="WGS84")
    cx = transform.c + transform.a * 0.5
    cy = transform.f + transform.e * 0.5
    _, _, dx = geod.inv(cx, cy, cx + transform.a, cy)
    _, _, dy = geod.inv(cx, cy, cx, cy + abs(transform.e))
    return abs(dx) * abs(dy) / 10_000  # m² → ha


def _make_forest_binary(arr: np.ma.MaskedArray, threshold: int):
    """Return bool forest array and valid-pixel mask."""
    valid = ~arr.mask if np.ma.is_masked(arr) else np.ones(arr.shape, dtype=bool)
    forest = valid & (arr.filled(0) >= threshold)
    return forest.astype(bool), valid.astype(bool)


def _burn_roads(forest: np.ndarray, valid: np.ndarray,
                transform: Affine, crs, road_path: Path, road_types: list) -> tuple:
    """
    Set road pixels to non-forest (and non-valid) in both arrays.
    road_types: liu_surface values to burn, e.g. ['paved'] or ['paved', 'unpaved'].
    """
    roads = gpd.read_file(road_path, layer=0)
    roads = roads[roads["liu_surface"].isin(road_types)].copy()
    if roads.empty:
        print(f"  Warning: no roads found for surface types {road_types}")
        return forest, valid

    if roads.crs != crs:
        roads = roads.to_crs(crs)

    road_raster = rasterize(
        [(geom, 1) for geom in roads.geometry if geom is not None],
        out_shape=forest.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    road_mask = road_raster == 1
    n_burned = int(road_mask.sum())
    print(f"  Burned {n_burned:,} road pixels ({n_burned * _compute_pixel_area_ha(transform):.0f} ha)")
    return forest & ~road_mask, valid & ~road_mask


def _label_patches(forest: np.ndarray, min_patch_px: int) -> np.ndarray:
    """
    Label connected forest patches (8-connectivity). Patches smaller than
    min_patch_px are removed. Returns a compact uint32 label array.
    """
    labeled, _ = ndimage_label(forest, structure=_STRUCT_8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    too_small = (sizes > 0) & (sizes < min_patch_px)
    if too_small.any():
        labeled[too_small[labeled]] = 0

    unique_ids = np.unique(labeled)
    unique_ids = unique_ids[unique_ids != 0]
    remap = np.zeros(labeled.max() + 1, dtype=np.uint32)
    remap[unique_ids] = np.arange(1, len(unique_ids) + 1, dtype=np.uint32)
    return remap[labeled].astype(np.uint32)


def _save_patch_raster(labeled: np.ndarray, transform: Affine, crs, out_path: Path):
    profile = {
        "driver": "GTiff", "dtype": "uint32", "nodata": 0, "count": 1,
        "height": labeled.shape[0], "width": labeled.shape[1],
        "crs": crs, "transform": transform,
        "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labeled, 1)
    print(f"  Saved patch raster → {out_path.name}")


def _stats_for_unit(geom, forest: np.ndarray, valid: np.ndarray,
                    transform: Affine, pixel_ha: float, min_patch_px: int) -> dict:
    """Fragmentation stats for one admin unit."""
    outside = geometry_mask([geom.__geo_interface__], transform=transform,
                            invert=False, out_shape=forest.shape)
    inside = ~outside
    area_px = int(valid[inside].sum())
    forest_clip = forest & inside
    forest_px = int(forest_clip.sum())

    if forest_px == 0:
        return dict(area_ha=round(area_px * pixel_ha, 2), forest_ha=0.0, pct_forest=0.0,
                    n_patches=0, largest_patch_ha=0.0, mean_patch_ha=0.0, frag_index=0.0)

    local_labeled, _ = ndimage_label(forest_clip, structure=_STRUCT_8)
    sizes = np.bincount(local_labeled.ravel())[1:]
    valid_sizes = sizes[sizes >= min_patch_px]

    area_ha = area_px * pixel_ha
    forest_ha = forest_px * pixel_ha
    n_patches = int(valid_sizes.size)
    largest_ha = float(valid_sizes.max() * pixel_ha) if n_patches else 0.0
    mean_ha = float(valid_sizes.mean() * pixel_ha) if n_patches else 0.0
    frag = (n_patches - 1) / (forest_ha - 1) if forest_ha > 1 and n_patches > 0 else 0.0

    return dict(
        area_ha=round(area_ha, 2),
        forest_ha=round(forest_ha, 2),
        pct_forest=round(100 * forest_ha / area_ha, 2) if area_ha > 0 else 0.0,
        n_patches=n_patches,
        largest_patch_ha=round(largest_ha, 2),
        mean_patch_ha=round(mean_ha, 2),
        frag_index=round(frag, 6),
    )


def _compute_admin_stats(gdf: gpd.GeoDataFrame, forest: np.ndarray, valid: np.ndarray,
                         transform: Affine, pixel_ha: float, min_patch_px: int,
                         admin_level: int) -> gpd.GeoDataFrame:
    name_col = _ADMIN_NAME_COL[admin_level]
    rows = []
    total = len(gdf)
    for i, (_, row) in enumerate(gdf.iterrows(), 1):
        print(f"  [{i}/{total}] {row.get(name_col, '?')}", end="\r")
        rows.append(_stats_for_unit(row.geometry, forest, valid, transform, pixel_ha, min_patch_px))
    print()
    return gdf.join(gpd.GeoDataFrame(rows, index=gdf.index))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_patch_detection(config: PatchConfig) -> tuple:
    """
    Run forest patch detection for the given config.
    Returns (tif_path, gpkg_path).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    road_types = []
    if config.burn_paved_roads:
        road_types.append("paved")
    if config.burn_unpaved_roads:
        road_types.append("unpaved")

    roads_suffix = ("__roads_" + "_".join(road_types)) if road_types else ""
    stem = (f"{config.prefix}__forest_patches"
            f"__thresh{config.forest_threshold}"
            f"__minpatch{config.min_patch_size_ha:.0f}ha"
            f"__adm{config.admin_level}"
            f"{roads_suffix}")
    tif_out = config.output_dir / f"{stem}.tif"
    gpkg_out = config.output_dir / f"{stem}.gpkg"

    print(f"\n{'='*60}")
    print(f"Config: {stem}")
    print(f"{'='*60}")

    print("Loading treecover at 90 m …")
    arr, transform, crs = _load_treecover_90m(config.treecover_path)
    pixel_ha = _compute_pixel_area_ha(transform)
    min_patch_px = max(1, math.ceil(config.min_patch_size_ha / pixel_ha))
    print(f"  Pixel area: {pixel_ha:.4f} ha  |  min patch: {min_patch_px} px "
          f"({config.min_patch_size_ha} ha)")

    print(f"Creating forest binary (threshold ≥ {config.forest_threshold}%) …")
    forest, valid = _make_forest_binary(arr, config.forest_threshold)
    print(f"  Forest cover: {100 * forest.sum() / valid.sum():.1f}% of valid area")

    if road_types:
        print(f"Burning roads ({', '.join(road_types)}) …")
        forest, valid = _burn_roads(forest, valid, transform, crs, config.road_path, road_types)
        print(f"  Forest cover after roads: {100 * forest.sum() / valid.sum():.1f}% of valid area")

    print("Labeling country-level forest patches …")
    labeled = _label_patches(forest, min_patch_px)
    print(f"  {int(labeled.max())} patches after filtering < {config.min_patch_size_ha} ha")
    _save_patch_raster(labeled, transform, crs, tif_out)

    admin_path = _ADMIN_DIR / f"cmr_admin{config.admin_level}.shp"
    print(f"Loading admin boundaries: {admin_path.name} …")
    gdf = gpd.read_file(admin_path)

    print(f"Computing stats per admin unit ({len(gdf)} units) …")
    gdf_stats = _compute_admin_stats(gdf, forest, valid, transform, pixel_ha,
                                     min_patch_px, config.admin_level)
    gdf_stats.to_file(gpkg_out, driver="GPKG")
    print(f"  Saved stats → {gpkg_out.name}")

    name_col = _ADMIN_NAME_COL[config.admin_level]
    cols = [name_col, "pct_forest", "n_patches", "mean_patch_ha", "frag_index"]
    print(f"\nTop 5 by forest cover:")
    print(gdf_stats[cols].sort_values("pct_forest", ascending=False).head(5).to_string(index=False))

    return tif_out, gpkg_out
