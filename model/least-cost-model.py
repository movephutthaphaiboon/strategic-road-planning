#!/usr/bin/env python3
"""
Least-cost path model for mine-to-port road generation in Cameroon.

For each mine, computes the cheapest construction path to each configured port
using a friction/cost raster, respecting spatial no-go zones (protected areas).
Selects the cheapest port per mine and outputs a GeoPackage with path geometries.

Algorithm:
    MCP_Geometric (scikit-image) is run ONCE per port, spreading from the port
    across the entire cost raster. Each mine then cheaply reads its cumulative
    cost and traces back its path — O(1) per mine after the single sweep.

Memory strategy:
    - Uses the Cameroon-clipped friction raster (smaller than the merged tile)
    - Downsamples at runtime (--downsample, default=5 → ~450m resolution)
    - Processes one port at a time, releasing MCP memory between ports
    - Rasterizes the vector mask at runtime to match the downsampled raster

Usage:
    python least-cost-model.py                        # all ports, default settings
    python least-cost-model.py --port Kribi           # single port
    python least-cost-model.py --downsample 10        # more aggressive memory saving
    python least-cost-model.py --port Kribi --downsample 3  # higher resolution
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
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# =============================================================================
# CONFIGURATION — edit paths and ports here
# =============================================================================

BASE_DIR = Path(__file__).parent.parent  # strategic-road-planning/

MINES_FP    = BASE_DIR / "data/output/cmr-mine-locations/all_mines_with_id.csv"
FRICTION_FP = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
MASK_FP     = BASE_DIR / "data/output/processed-protected-areas/cmr-protected-areas.gpkg"
OUTPUT_DIR  = BASE_DIR / "model/results/least-cost-paths"

# Mine IDs to process — set to None or empty list to process all mines, or specify a subset for testing.
selected_mine_ids = ["cmr_027"]


# Port locations: {name: (latitude, longitude)}
PORTS = {
    "Kribi":  (2.940594, 9.910192),
    # "Douala": (4.0511,  9.7679),  # uncomment to add more ports
}

# Default downsampling factor for the friction raster.
# Higher value → less RAM, lower spatial resolution of paths.
#   factor 1  → native ~90 m   (~600 MB MCP arrays)
#   factor 5  → ~450 m  (~25 MB cost arr, ~100 MB MCP)  ← default
#   factor 10 → ~900 m  (~6 MB cost arr,  ~25 MB MCP)
DEFAULT_DOWNSAMPLE = 5

# Cells inside no-go zones and nodata cells receive this cost so paths avoid them.
# Set very high but finite (not inf) so MCP_Geometric can still propagate past
# them if the geometry forces it (e.g. mine/port on a border).
NO_GO_COST = 1e9

# =============================================================================
# DATA LOADING
# =============================================================================

def load_mines(fp: Path) -> gpd.GeoDataFrame:
    """Load mine CSV → GeoDataFrame in EPSG:4326."""
    df = pd.read_csv(fp)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )
    print(f"  {len(gdf)} mines loaded from {fp.name}")
    return gdf


def load_friction(fp: Path, downsample: int = 1):
    """
    Open friction raster and read band 1, optionally at reduced resolution.

    Returns
    -------
    arr       : float32 ndarray  (H, W)
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

        # Scale transform to match the downsampled shape
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
    print(f"  Friction: {orig_h}×{orig_w} → {new_h}×{new_w}  "
          f"({mem_mb:.1f} MB, dtype=float32)")
    return arr, transform, nodata, crs


def load_mask(fp: Path, target_crs) -> gpd.GeoDataFrame:
    """Load no-go zone polygons, reprojecting to target CRS if needed."""
    gdf = gpd.read_file(fp)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    print(f"  Mask: {len(gdf)} polygon(s) from {fp.name}")
    return gdf


def rasterize_mask(mask_gdf: gpd.GeoDataFrame, shape: tuple, transform) -> np.ndarray:
    """Burn mask polygons into a uint8 array matching the friction grid."""
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
    """
    Merge friction values with the no-go mask into the final cost array.
    Nodata, NaN, zero/negative, and masked cells all become NO_GO_COST.
    """
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
    """Convert (lat, lon) → (row, col) in the raster grid."""
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


def clamp_rc(row: int, col: int, shape: tuple) -> tuple:
    """Clamp (row, col) to valid array bounds."""
    return (max(0, min(row, shape[0] - 1)),
            max(0, min(col, shape[1] - 1)))


def rc_path_to_linestring(path: list, transform) -> LineString:
    """
    Convert a list of (row, col) indices to a LineString.
    Each point is placed at the pixel centre in the raster's CRS.
    """
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
             transform,
             mask_gdf: gpd.GeoDataFrame) -> list:
    """
    Run MCP_Geometric once from the port, then traceback every mine.

    Returns a list of record dicts, one per mine.
    """
    port_lat, port_lon = port_latlon
    port_rc = clamp_rc(*latlon_to_rc(port_lat, port_lon, transform), cost_arr.shape)

    # Warn if port lands in a no-go zone (unlikely but possible near boundaries)
    if cost_arr[port_rc] >= NO_GO_COST * 0.5:
        print(f"  WARNING: port '{port_name}' pixel falls in a no-go/nodata zone. "
              f"Consider adjusting port coordinates.")

    print(f"  Running MCP from '{port_name}' (pixel {port_rc})…")
    mcp = MCP_Geometric(cost_arr)
    cumulative_costs, _ = mcp.find_costs([port_rc])   # full raster sweep
    print(f"  MCP sweep done. Tracing paths for {len(mines_gdf)} mines…")

    records = []
    for _, mine in mines_gdf.iterrows():
        mine_id   = mine["ID"]
        mine_name = mine.get("PROP_NAME", "")

        # Pre-filter: skip mines flagged as inside protected areas
        if str(mine.get("MINE_IN_PROTECTED_AREA", "False")).strip().lower() == "true":
            print(f"    [{mine_id}] SKIP — mine flagged in protected area")
            records.append(_record(mine_id, mine_name, port_name,
                                   cost=np.inf, skipped=True, geom=None))
            continue

        mine_rc = clamp_rc(*latlon_to_rc(mine["LATITUDE"], mine["LONGITUDE"], transform),
                           cost_arr.shape)

        # Also check if mine pixel itself is in a no-go zone
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

        path = mcp.traceback(mine_rc)           # list of (row, col) from mine → port
        geom = rc_path_to_linestring(path, transform)
        print(f"    [{mine_id}] → {port_name}:  cost={total_cost:.3e},  "
              f"path_nodes={len(path)}")
        records.append(_record(mine_id, mine_name, port_name,
                                cost=total_cost, skipped=False, geom=geom))

    del mcp, cumulative_costs
    gc.collect()
    return records


def _record(mine_id, mine_name, port, cost, skipped, geom) -> dict:
    return {
        "mine_id":   mine_id,
        "mine_name": mine_name,
        "port":      port,
        "total_cost": cost,
        "skipped":   skipped, # True if pre-filtered (e.g. in protected area), False if attempted but no path found
        "geometry":  geom,
    }

# =============================================================================
# OUTPUT ASSEMBLY
# =============================================================================

def path_length_km(geom) -> float:
    """
    Compute geodesic length of a LineString (EPSG:4326) in kilometres.
    Uses pyproj.Geod so the result is accurate regardless of latitude.
    Returns NaN if geometry is None.
    """
    if geom is None:
        return np.nan
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    length_m = geod.geometry_length(geom)
    return length_m / 1000.0


def select_cheapest_port(records: list, crs) -> gpd.GeoDataFrame:
    """
    From all mine×port records, keep the cheapest (lowest total_cost) port
    per mine. Returns a GeoDataFrame with columns:
        mine_id, mine_name, closest_port, total_cost, path_km, skipped, geometry
    """
    df = pd.DataFrame(records)
    idx_min = df.groupby("mine_id")["total_cost"].idxmin()
    best = df.loc[idx_min].copy().reset_index(drop=True)
    best = best.rename(columns={"port": "closest_port"})
    gdf = gpd.GeoDataFrame(best, geometry="geometry", crs=crs)
    gdf["path_km"] = gdf["geometry"].apply(path_length_km)
    cols = ["mine_id", "mine_name", "closest_port", "total_cost", "path_km", "skipped", "geometry"]
    return gdf[cols]


def save_results(result_gdf: gpd.GeoDataFrame,
                 all_records: list,
                 crs,
                 multi_port: bool,
                 downsample: int = DEFAULT_DOWNSAMPLE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Best-port-per-mine output
    valid = result_gdf[result_gdf.geometry.notna()].copy()
    out_best = OUTPUT_DIR / f"mine_to_port_paths_ds{downsample}.gpkg"
    if not valid.empty:
        valid.to_file(out_best, driver="GPKG")
        print(f"\n  Saved {len(valid)} paths  →  {out_best}")
    else:
        print("\n  No valid paths found — output file not written.")

    # All port comparisons (only when multiple ports were evaluated)
    if multi_port:
        all_df  = pd.DataFrame(all_records)
        all_gdf = gpd.GeoDataFrame(all_df, geometry="geometry", crs=crs)
        out_all = OUTPUT_DIR / "mine_to_port_all_paths.gpkg"
        all_gdf[all_gdf.geometry.notna()].to_file(out_all, driver="GPKG")
        print(f"  Saved all-port comparison  →  {out_all}")

# =============================================================================
# MAIN
# =============================================================================

def main(downsample: int = DEFAULT_DOWNSAMPLE, port_filter: str = None):
    print("=" * 62)
    print("  LEAST-COST PATH MODEL — Mine-to-Port Roads, Cameroon")
    print("=" * 62)

    # Resolve which ports to run
    ports_to_run = PORTS
    if port_filter:
        if port_filter not in PORTS:
            sys.exit(f"Error: port '{port_filter}' not in PORTS config. "
                     f"Available: {list(PORTS.keys())}")
        ports_to_run = {port_filter: PORTS[port_filter]}

    # ── 1. Load mines ────────────────────────────────────────────────────────
    print("\n[1/5] Loading mines …")
    all_mines = load_mines(MINES_FP)
    mines_gdf = all_mines[all_mines["ID"].isin(selected_mine_ids)]

    # ── 2. Load & downsample friction raster ─────────────────────────────────
    print(f"\n[2/5] Loading friction raster (downsample={downsample}×) …")
    friction, transform, nodata, crs = load_friction(FRICTION_FP, downsample)

    # ── 3. Load & rasterize spatial mask ─────────────────────────────────────
    print("\n[3/5] Loading spatial mask …")
    mask_gdf  = load_mask(MASK_FP, crs)
    mask_arr  = rasterize_mask(mask_gdf, friction.shape, transform)

    # ── 4. Build combined cost surface ───────────────────────────────────────
    print("\n[4/5] Building cost surface …")
    cost_arr = build_cost_surface(friction, mask_arr, nodata)
    del friction, mask_arr
    gc.collect()
    print(f"  Cost surface ready: {cost_arr.shape}, dtype={cost_arr.dtype}")

    # ── 5. Run LCP for each port ──────────────────────────────────────────────
    print(f"\n[5/5] Running LCP for {len(ports_to_run)} port(s): "
          f"{list(ports_to_run.keys())} …")

    all_records = []
    for port_name, port_latlon in ports_to_run.items():
        print(f"\n  ── Port: {port_name} ──────────────────────────────────")
        records = run_port(port_name, port_latlon, mines_gdf,
                           cost_arr, transform, mask_gdf)
        all_records.extend(records)

    # ── Assemble & save results ───────────────────────────────────────────────
    print("\n  Selecting cheapest port per mine …")
    result_gdf = select_cheapest_port(all_records, crs)

    print("\n  Summary:")
    print(result_gdf[["mine_id", "mine_name", "closest_port", "total_cost", "path_km"]]
          .to_string(index=False))

    save_results(result_gdf, all_records, crs, multi_port=len(PORTS) > 1, downsample=downsample)

    print("\nDone.")
    return result_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Least-cost path model: mines → ports, Cameroon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--downsample", type=int, default=DEFAULT_DOWNSAMPLE,
        help=(f"Raster downsampling factor (default: {DEFAULT_DOWNSAMPLE}). "
              f"Higher → less RAM, coarser path resolution. "
              f"factor 5 → ~450 m; factor 10 → ~900 m."),
    )
    parser.add_argument(
        "--port", type=str, default=None,
        metavar="PORT_NAME",
        help="Process a single named port only (default: all ports in PORTS config).",
    )
    args = parser.parse_args()
    main(downsample=args.downsample, port_filter=args.port)
