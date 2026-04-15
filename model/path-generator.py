#!/usr/bin/env python3
"""
path-generator.py — Least-cost path engine (model only, no configuration).

This file contains all core logic. Do not edit for experiments.
Configure and run experiments from path-generator-run.py instead.

Public API
----------
run_experiment(config) → gpd.GeoDataFrame
    Run one complete least-cost path experiment and save results.
    See ExperimentConfig for all parameters.
"""

import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

NO_GO_COST = 1e9   # cost assigned to masked / nodata pixels


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    All parameters that define one experiment run.

    Parameters
    ----------
    name          : Short label used in output filenames and log messages.
    mine_ids      : List of mine ID strings to process (e.g. ["cmr_001", "cmr_005"]).
    ports         : Dict of {port_name: (latitude, longitude)}.
    friction_fp   : Path to the friction/cost raster (.tif, EPSG:4326).
    mines_fp      : Path to the mines CSV (must have ID, LATITUDE, LONGITUDE columns).
    mask_fp       : Path to no-go zone vector (.gpkg/.shp), or None for no mask.
    output_dir    : Directory where result GeoPackages are written.
    downsample    : Integer downsampling factor applied to the friction raster.
                    1 = native resolution (~90 m), 5 = ~450 m, 10 = ~900 m.
    """
    name         : str
    mine_ids     : list
    ports        : dict                   # {name: (lat, lon)}
    friction_fp  : Path
    mines_fp     : Path
    mask_fp      : Path | None
    output_dir   : Path
    downsample   : int = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mines(fp: Path, mine_ids: list) -> gpd.GeoDataFrame:
    """Load mine CSV, filter to mine_ids, return GeoDataFrame in EPSG:4326."""
    df = pd.read_csv(fp)
    if mine_ids:
        df = df[df["ID"].isin(mine_ids)].copy()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )
    print(f"  Mines: {len(gdf)} loaded (filtered from {len(pd.read_csv(fp))} total)")
    return gdf


def load_friction(fp: Path, downsample: int = 1):
    """
    Read friction raster band 1, optionally downsampled to save memory.

    Returns
    -------
    arr       : float32 ndarray (H, W)
    transform : Affine transform matching arr
    nodata    : scalar or None
    crs       : rasterio CRS
    """
    with rasterio.open(fp) as src:
        orig_h, orig_w = src.height, src.width
        new_h = max(1, orig_h // downsample)
        new_w = max(1, orig_w // downsample)

        arr = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=Resampling.average,
        ).astype(np.float32)

        transform = rasterio.transform.Affine(
            src.transform.a * (orig_w / new_w),
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e * (orig_h / new_h),
            src.transform.f,
        )
        nodata = src.nodata
        crs    = src.crs

    mem_mb = arr.nbytes / 1e6
    print(f"  Friction: {orig_h}×{orig_w} → {new_h}×{new_w}  ({mem_mb:.1f} MB)")
    return arr, transform, nodata, crs


def load_mask(fp: Path, target_crs) -> gpd.GeoDataFrame:
    """Load no-go zone polygons, reprojecting if needed. Returns empty GDF if fp is None."""
    if fp is None:
        print("  Mask: none (no spatial restriction applied)")
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(fp)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    print(f"  Mask: {len(gdf)} polygon(s) from {Path(fp).name}")
    return gdf


def rasterize_mask(mask_gdf: gpd.GeoDataFrame, shape: tuple, transform) -> np.ndarray:
    """Burn mask polygons into a uint8 raster matching the friction grid."""
    if mask_gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    shapes = [(g, 1) for g in mask_gdf.geometry if g is not None]
    arr = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    pct = 100.0 * arr.sum() / arr.size
    print(f"  Mask rasterized: {pct:.1f}% of pixels marked no-go")
    return arr


def build_cost_surface(friction: np.ndarray, mask: np.ndarray, nodata) -> np.ndarray:
    """Combine friction and mask into the final cost array (float32)."""
    cost = friction.copy()
    if nodata is not None:
        cost[cost == nodata] = NO_GO_COST
    cost[np.isnan(cost)] = NO_GO_COST
    cost[cost <= 0]       = NO_GO_COST
    cost[mask == 1]       = NO_GO_COST
    return cost


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def latlon_to_rc(lat: float, lon: float, transform) -> tuple:
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


def clamp_rc(row: int, col: int, shape: tuple) -> tuple:
    return (max(0, min(row, shape[0] - 1)),
            max(0, min(col, shape[1] - 1)))


def rc_path_to_linestring(path: list, transform) -> LineString:
    if len(path) < 2:
        return None
    coords = [
        (transform.c + (c + 0.5) * transform.a,
         transform.f + (r + 0.5) * transform.e)
        for r, c in path
    ]
    return LineString(coords)


# =============================================================================
# LEAST-COST PATH — one port at a time
# =============================================================================

def run_port(port_name: str, port_latlon: tuple,
             mines_gdf: gpd.GeoDataFrame,
             cost_arr: np.ndarray,
             transform) -> list:
    """
    Run MCP_Geometric once from the port, then traceback every mine.
    Returns a list of record dicts (one per mine).
    """
    port_lat, port_lon = port_latlon
    port_rc = clamp_rc(*latlon_to_rc(port_lat, port_lon, transform), cost_arr.shape)

    if cost_arr[port_rc] >= NO_GO_COST * 0.5:
        print(f"  WARNING: port '{port_name}' pixel is in a no-go/nodata zone. "
              f"Adjust port coordinates if paths look wrong.")

    print(f"  MCP sweep from '{port_name}' (pixel {port_rc}) …")
    mcp = MCP_Geometric(cost_arr)
    cumulative_costs, _ = mcp.find_costs([port_rc])
    print(f"  Sweep done. Tracing {len(mines_gdf)} mine(s) …")

    records = []
    for _, mine in mines_gdf.iterrows():
        mine_id   = mine["ID"]
        mine_name = mine.get("PROP_NAME", "")

        # Skip mines pre-flagged as inside protected areas
        if str(mine.get("MINE_IN_PROTECTED_AREA", "False")).strip().lower() == "true":
            print(f"    [{mine_id}] SKIP — mine flagged in protected area")
            records.append(_record(mine_id, mine_name, port_name,
                                   cost=np.inf, skipped=True, geom=None))
            continue

        mine_rc = clamp_rc(*latlon_to_rc(mine["LATITUDE"], mine["LONGITUDE"], transform),
                           cost_arr.shape)

        if cost_arr[mine_rc] >= NO_GO_COST * 0.5:
            print(f"    [{mine_id}] SKIP — mine pixel in no-go/nodata zone")
            records.append(_record(mine_id, mine_name, port_name,
                                   cost=np.inf, skipped=True, geom=None))
            continue

        total_cost = float(cumulative_costs[mine_rc[0], mine_rc[1]])

        if not np.isfinite(total_cost) or total_cost >= NO_GO_COST * 0.5:
            print(f"    [{mine_id}] No viable path to {port_name}")
            records.append(_record(mine_id, mine_name, port_name,
                                   cost=np.inf, skipped=False, geom=None))
            continue

        path = mcp.traceback(mine_rc)
        geom = rc_path_to_linestring(path, transform)
        print(f"    [{mine_id}] → {port_name}:  cost={total_cost:.3e},  nodes={len(path)}")
        records.append(_record(mine_id, mine_name, port_name,
                               cost=total_cost, skipped=False, geom=geom))

    del mcp, cumulative_costs
    gc.collect()
    return records


def _record(mine_id, mine_name, port, cost, skipped, geom) -> dict:
    return {
        "mine_id":    mine_id,
        "mine_name":  mine_name,
        "port":       port,
        "total_cost": cost,
        "skipped":    skipped,
        "geometry":   geom,
    }


# =============================================================================
# OUTPUT ASSEMBLY
# =============================================================================

def _path_length_km(geom) -> float:
    """Geodesic length of a LineString (EPSG:4326) in kilometres."""
    if geom is None:
        return np.nan
    geod = Geod(ellps="WGS84")
    return geod.geometry_length(geom) / 1000.0


def select_cheapest_port(records: list, crs) -> gpd.GeoDataFrame:
    """Keep the cheapest port per mine from all mine×port records."""
    df = pd.DataFrame(records)
    idx_min = df.groupby("mine_id")["total_cost"].idxmin()
    best = df.loc[idx_min].copy().reset_index(drop=True)
    best = best.rename(columns={"port": "closest_port"})
    gdf = gpd.GeoDataFrame(best, geometry="geometry", crs=crs)
    gdf["path_km"] = gdf["geometry"].apply(_path_length_km)
    cols = ["mine_id", "mine_name", "closest_port", "total_cost", "path_km", "skipped", "geometry"]
    return gdf[cols]


def save_results(result_gdf: gpd.GeoDataFrame,
                 all_records: list,
                 crs,
                 experiment_name: str,
                 output_dir: Path,
                 multi_port: bool):
    """Write result GeoPackage(s) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = result_gdf[result_gdf.geometry.notna()].copy()
    out_best = output_dir / f"{experiment_name}.gpkg"
    if not valid.empty:
        valid.to_file(out_best, driver="GPKG")
        print(f"\n  Saved {len(valid)} path(s)  →  {out_best}")
    else:
        print("\n  No valid paths — output file not written.")

    if multi_port:
        all_gdf = gpd.GeoDataFrame(pd.DataFrame(all_records), geometry="geometry", crs=crs)
        out_all = output_dir / f"{experiment_name}__all_ports.gpkg"
        all_gdf[all_gdf.geometry.notna()].to_file(out_all, driver="GPKG")
        print(f"  All-port comparison  →  {out_all}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_experiment(cfg: ExperimentConfig) -> gpd.GeoDataFrame:
    """
    Run one complete least-cost path experiment.

    Parameters
    ----------
    cfg : ExperimentConfig

    Returns
    -------
    GeoDataFrame with columns:
        mine_id, mine_name, closest_port, total_cost, path_km, skipped, geometry
    """
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Experiment: {cfg.name}")
    print(f"  Mines: {len(cfg.mine_ids)}  |  Ports: {list(cfg.ports.keys())}")
    print(f"  Friction: {Path(cfg.friction_fp).name}  |  Downsample: {cfg.downsample}×")
    print(f"  Mask: {Path(cfg.mask_fp).name if cfg.mask_fp else 'none'}")
    print(sep)

    # 1. Mines
    print("\n[1/5] Loading mines …")
    mines_gdf = load_mines(cfg.mines_fp, cfg.mine_ids)

    # 2. Friction
    print(f"\n[2/5] Loading friction raster (downsample={cfg.downsample}×) …")
    friction, transform, nodata, crs = load_friction(cfg.friction_fp, cfg.downsample)

    # 3. Mask
    print("\n[3/5] Loading spatial mask …")
    mask_gdf = load_mask(cfg.mask_fp, crs)
    mask_arr = rasterize_mask(mask_gdf, friction.shape, transform)

    # 4. Cost surface
    print("\n[4/5] Building cost surface …")
    cost_arr = build_cost_surface(friction, mask_arr, nodata)
    del friction, mask_arr
    gc.collect()
    print(f"  Ready: {cost_arr.shape}, dtype={cost_arr.dtype}")

    # 5. LCP per port
    print(f"\n[5/5] Running LCP for {len(cfg.ports)} port(s) …")
    all_records = []
    for port_name, port_latlon in cfg.ports.items():
        print(f"\n  ── Port: {port_name}")
        records = run_port(port_name, port_latlon, mines_gdf, cost_arr, transform)
        all_records.extend(records)

    # Assemble
    result_gdf = select_cheapest_port(all_records, crs)

    print("\n  Summary:")
    print(result_gdf[["mine_id", "mine_name", "closest_port", "total_cost", "path_km"]]
          .to_string(index=False))

    save_results(result_gdf, all_records, crs,
                 experiment_name=cfg.name,
                 output_dir=cfg.output_dir,
                 multi_port=len(cfg.ports) > 1)

    print(f"\n  ✓ Experiment '{cfg.name}' complete.")
    return result_gdf
