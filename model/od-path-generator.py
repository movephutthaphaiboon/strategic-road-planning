#!/usr/bin/env python3
"""
od-path-generator.py — Least-cost path between arbitrary origin-destination pairs.

Finds the cheapest route between any two coordinates using the same friction
raster and MCP_Geometric algorithm as the main path generator. Useful for:
  - Comparing model routes vs. alternative routes
  - Debugging the friction surface along specific corridors
  - Testing any custom O-D pair independently of the mine/port setup

Input:
    OD_PAIRS list below — each entry is:
        (label, origin_lat, origin_lon, dest_lat, dest_lon)

Output:
    GeoPackage with one row per pair:
        pair_id, label, origin_lat, origin_lon, dest_lat, dest_lon,
        total_cost, path_km, geometry

Usage:
    python od-path-generator.py
    python od-path-generator.py --downsample 3
    python od-path-generator.py --output my_paths.gpkg
"""

import argparse
import gc
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
from rasterio.transform import rowcol
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = Path(__file__).parent.parent
FRICTION_FP = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
MASK_FP     = BASE_DIR / "data/output/processed-protected-areas/cmr-protected-areas.gpkg"
OUTPUT_DIR  = BASE_DIR / "model/results/od-paths"

# Set to MASK_FP to apply spatial restriction, or None for no mask
MASK        = None

DEFAULT_DOWNSAMPLE = 1
NO_GO_COST         = 1e9

# ── O-D pairs ─────────────────────────────────────────────────────────────────
# Format: (label, origin_lat, origin_lon, dest_lat, dest_lon)
# Add as many pairs as you need.

KRIBI = (2.940594, 9.910192)

OD_PAIRS = [
    # ("label",  origin_lat,  origin_lon,  dest_lat,    dest_lon)
    # ("central_route_01",  6.46290, 12.62098,  KRIBI[0], KRIBI[1]),
    # ("central_route_02",  6.25500, 12.53400,  KRIBI[0], KRIBI[1]),
    # ("central_route_03",  6.08720, 12.46670,  KRIBI[0], KRIBI[1]),
    # ("central_route_04",  5.83220, 12.27325,  KRIBI[0], KRIBI[1]),
    # ("central_route_05",  5.68649, 12.35225,  KRIBI[0], KRIBI[1]),
    # ("central_route_06",  5.12056, 12.10328,  KRIBI[0], KRIBI[1]),
    # ("central_route_07",  4.57881, 11.63381,  KRIBI[0], KRIBI[1]),
    # ("central_route_08",  4.45805, 11.62480,  KRIBI[0], KRIBI[1]),
    ("central_route_09",  5.0236449,12.0036389,  KRIBI[0], KRIBI[1]), # tipping point!
    ("central_route_10",  5.019848,12.003718,  KRIBI[0], KRIBI[1]),
]


# =============================================================================
# CORE FUNCTIONS  (shared with path-generator.py)
# =============================================================================

def load_friction(fp: Path, downsample: int = 1):
    with rasterio.open(fp) as src:
        oh, ow = src.height, src.width
        nh = max(1, oh // downsample)
        nw = max(1, ow // downsample)
        arr = src.read(1, out_shape=(nh, nw),
                       resampling=Resampling.nearest).astype(np.float32)
        transform = rasterio.transform.Affine(
            src.transform.a * (ow / nw), src.transform.b, src.transform.c,
            src.transform.d, src.transform.e * (oh / nh), src.transform.f,
        )
        nodata = src.nodata
        crs    = src.crs
    mem_mb = arr.nbytes / 1e6
    print(f"  Friction: {oh}×{ow} → {nh}×{nw}  ({mem_mb:.1f} MB)")
    return arr, transform, nodata, crs


def load_and_rasterize_mask(mask_fp, shape, transform, crs):
    if mask_fp is None:
        print("  Mask: none")
        return np.zeros(shape, dtype=np.uint8)
    gdf = gpd.read_file(mask_fp)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(g, 1) for g in gdf.geometry if g is not None]
    arr = rasterize(shapes, out_shape=shape, transform=transform,
                    fill=0, dtype=np.uint8, all_touched=False)
    pct = 100.0 * arr.sum() / arr.size
    print(f"  Mask: {pct:.1f}% of pixels marked no-go")
    return arr


def build_cost_surface(friction, mask, nodata):
    cost = friction.copy()
    if nodata is not None:
        cost[cost == nodata] = NO_GO_COST
    cost[np.isnan(cost)] = NO_GO_COST
    cost[cost <= 0]       = NO_GO_COST
    cost[mask == 1]       = NO_GO_COST
    return cost


def latlon_to_rc(lat, lon, transform, shape):
    r, c = rowcol(transform, lon, lat)
    return (max(0, min(int(r), shape[0] - 1)),
            max(0, min(int(c), shape[1] - 1)))


def rc_path_to_linestring(path, transform):
    if len(path) < 2:
        return None
    coords = [(transform.c + (c + 0.5) * transform.a,
               transform.f + (r + 0.5) * transform.e)
              for r, c in path]
    return LineString(coords)


def path_length_km(geom) -> float:
    if geom is None:
        return np.nan
    geod = Geod(ellps="WGS84")
    return geod.geometry_length(geom) / 1000.0


# =============================================================================
# O-D PATH COMPUTATION
# =============================================================================

def compute_od_path(label, origin_lat, origin_lon, dest_lat, dest_lon,
                    cost_arr, transform):
    """
    Find least-cost path from origin to destination.
    Runs MCP from origin (single sweep), then tracebacks to destination.
    Returns a record dict.
    """
    origin_rc = latlon_to_rc(origin_lat, origin_lon, transform, cost_arr.shape)
    dest_rc   = latlon_to_rc(dest_lat,   dest_lon,   transform, cost_arr.shape)

    def _record(cost, geom, note=""):
        return {
            "label":      label,
            "origin_lat": origin_lat, "origin_lon": origin_lon,
            "dest_lat":   dest_lat,   "dest_lon":   dest_lon,
            "total_cost": cost,
            "path_km":    path_length_km(geom),
            "note":       note,
            "geometry":   geom,
        }

    # Check origin / destination are not in no-go zones
    if cost_arr[origin_rc] >= NO_GO_COST * 0.5:
        print(f"  [{label}] SKIP — origin pixel is in no-go/nodata zone")
        return _record(np.inf, None, "origin in no-go zone")
    if cost_arr[dest_rc] >= NO_GO_COST * 0.5:
        print(f"  [{label}] SKIP — destination pixel is in no-go/nodata zone")
        return _record(np.inf, None, "destination in no-go zone")

    mcp = MCP_Geometric(cost_arr)
    cumulative_costs, _ = mcp.find_costs([origin_rc], [dest_rc])

    total_cost = float(cumulative_costs[dest_rc[0], dest_rc[1]])
    if not np.isfinite(total_cost) or total_cost >= NO_GO_COST * 0.5:
        print(f"  [{label}] No viable path found")
        del mcp, cumulative_costs
        gc.collect()
        return _record(np.inf, None, "no viable path")

    path = mcp.traceback(dest_rc)
    geom = rc_path_to_linestring(path, transform)
    km   = path_length_km(geom)
    print(f"  [{label}]  cost={total_cost:.3e}  path={km:.1f} km  nodes={len(path)}")

    del mcp, cumulative_costs
    gc.collect()
    return _record(total_cost, geom)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Least-cost path between arbitrary O-D coordinate pairs"
    )
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE,
                        help=f"Raster downsample factor (default: {DEFAULT_DOWNSAMPLE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GeoPackage filename (default: od_paths_ds<N>.gpkg)")
    args = parser.parse_args()

    if not OD_PAIRS:
        sys.exit("No O-D pairs defined. Add entries to OD_PAIRS in the config section.")

    print("=" * 62)
    print("  O-D LEAST-COST PATH GENERATOR")
    print("=" * 62)
    print(f"  Pairs: {len(OD_PAIRS)}  |  Downsample: {args.downsample}×")

    # ── Load friction ─────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading friction raster …")
    friction, transform, nodata, crs = load_friction(FRICTION_FP, args.downsample)

    # ── Load mask ─────────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading mask …")
    mask_arr = load_and_rasterize_mask(MASK, friction.shape, transform, crs)
    cost_arr = build_cost_surface(friction, mask_arr, nodata)
    del friction, mask_arr
    gc.collect()
    print(f"  Cost surface ready: {cost_arr.shape}")

    # ── Compute paths ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Computing {len(OD_PAIRS)} path(s) …")
    records = []
    for i, pair in enumerate(OD_PAIRS):
        label, o_lat, o_lon, d_lat, d_lon = pair
        print(f"\n  Pair {i+1}/{len(OD_PAIRS)}: {label}")
        print(f"    Origin:      ({o_lat}, {o_lon})")
        print(f"    Destination: ({d_lat}, {d_lon})")
        rec = compute_od_path(label, o_lat, o_lon, d_lat, d_lon, cost_arr, transform)
        records.append(rec)

    # ── Save ──────────────────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.insert(0, "pair_id", [f"od_{i+1:03d}" for i in range(len(records))])

    print("\n  Summary:")
    print(gdf[["pair_id", "label", "total_cost", "path_km"]].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f"od_paths_ds{args.downsample}.gpkg"
    out_fp   = OUTPUT_DIR / out_name
    valid    = gdf[gdf.geometry.notna()]
    if not valid.empty:
        valid.to_file(out_fp, driver="GPKG")
        print(f"\n  Saved {len(valid)} path(s) → {out_fp}")
    else:
        print("\n  No valid paths to save.")

    print("\nDone.")
    return gdf


if __name__ == "__main__":
    main()
