#!/usr/bin/env python3
"""
route-cost-comparison.py

Compares the construction cost of:
  A) The model's least-cost path (from the generated GeoPackage)
  B) A manually defined alternative route (via user-specified waypoints)

Both routes are traced through the friction raster pixel-by-pixel.
Cost is computed the same way MCP_Geometric does:
    cost_step = (cost[pixel_i] + cost[pixel_i+1]) / 2  × pixel_distance
    where pixel_distance = 1.0 for cardinal moves, √2 for diagonal moves.

Usage:
    python route-cost-comparison.py <paths_gpkg> [--mine MINE_ID]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from skimage.draw import line as bresenham

BASE_DIR    = Path(__file__).parent.parent
FRICTION_FP = BASE_DIR / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
MINES_FP    = BASE_DIR / "data/output/cmr-mine-locations/all_mines_with_id.csv"

KRIBI = (2.940594, 9.910192)   # (lat, lon)

# ── Alternative route waypoints (lat, lon) — edit these ──────────────────────
ALTERNATIVE_WAYPOINTS = [
    # mine location is prepended automatically
    (6.46290, 12.62098),
    (5.74588, 12.40578),
    (4.71667, 11.67994),
    KRIBI,
]

DOWNSAMPLE     = 1    # use native resolution for accurate cost
SNAP_BUFFER_PX = 20   # snap waypoints to cheapest pixel within ±20 pixels (~1800 m at ds=1)


# =============================================================================

def load_friction(fp, downsample=1):
    with rasterio.open(fp) as src:
        oh, ow = src.height, src.width
        nh, nw = max(1, oh // downsample), max(1, ow // downsample)
        arr = src.read(1, out_shape=(nh, nw),
                       resampling=Resampling.nearest).astype(np.float32)
        transform = rasterio.transform.Affine(
            src.transform.a * (ow / nw), src.transform.b, src.transform.c,
            src.transform.d, src.transform.e * (oh / nh), src.transform.f,
        )
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr <= 0] = np.nan
    return arr, transform


def latlon_to_rc(lat, lon, transform, shape):
    r, c = rowcol(transform, lon, lat)
    r = max(0, min(int(r), shape[0] - 1))
    c = max(0, min(int(c), shape[1] - 1))
    return r, c


def snap_to_cheapest(lat, lon, friction_arr, transform, buffer_px=20):
    """
    Snap a (lat, lon) coordinate to the cheapest (lowest friction) pixel
    within a square search window of ±buffer_px pixels.

    This corrects for imprecise waypoint placement — the cheapest pixel
    within the buffer is most likely on an existing road.
    """
    r0, c0 = latlon_to_rc(lat, lon, transform, friction_arr.shape)

    r_min = max(0, r0 - buffer_px)
    r_max = min(friction_arr.shape[0] - 1, r0 + buffer_px)
    c_min = max(0, c0 - buffer_px)
    c_max = min(friction_arr.shape[1] - 1, c0 + buffer_px)

    window = friction_arr[r_min:r_max+1, c_min:c_max+1]
    flat_idx = np.nanargmin(window)
    dr, dc = np.unravel_index(flat_idx, window.shape)

    r_snapped = r_min + dr
    c_snapped = c_min + dc
    snapped_cost = friction_arr[r_snapped, c_snapped]

    # Convert snapped pixel back to lat/lon for reporting
    lon_s = transform.c + (c_snapped + 0.5) * transform.a
    lat_s = transform.f + (r_snapped + 0.5) * transform.e

    offset_px = np.sqrt((r_snapped - r0)**2 + (c_snapped - c0)**2)
    print(f"    ({lat:.5f}, {lon:.5f}) → snapped {offset_px:.1f} px → "
          f"({lat_s:.5f}, {lon_s:.5f})  cost={snapped_cost:.4f}")

    return r_snapped, c_snapped


def trace_waypoints(waypoints_latlon, transform, shape):
    """
    Convert a list of (lat, lon) waypoints to a pixel sequence using
    Bresenham's line between each consecutive pair.
    Returns list of (row, col) tuples.
    """
    pixels = []
    for i in range(len(waypoints_latlon) - 1):
        r0, c0 = latlon_to_rc(*waypoints_latlon[i],     transform, shape)
        r1, c1 = latlon_to_rc(*waypoints_latlon[i + 1], transform, shape)
        rr, cc = bresenham(r0, c0, r1, c1)
        # Avoid duplicating junction pixels between segments
        segment = list(zip(rr.tolist(), cc.tolist()))
        if pixels and segment:
            segment = segment[1:]
        pixels.extend(segment)
    return pixels


def mcp_cost_along_pixels(pixels, friction_arr):
    """
    Compute accumulated cost along a pixel sequence using the same formula
    as MCP_Geometric:
        cost_step = (cost[i] + cost[i+1]) / 2 × distance
        distance  = 1.0 for cardinal, √2 for diagonal moves
    Skips steps where either pixel is NaN (nodata).
    """
    total = 0.0
    n_skipped = 0
    for i in range(len(pixels) - 1):
        r0, c0 = pixels[i]
        r1, c1 = pixels[i + 1]
        v0 = friction_arr[r0, c0]
        v1 = friction_arr[r1, c1]
        if np.isnan(v0) or np.isnan(v1):
            n_skipped += 1
            continue
        dist = np.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
        total += (v0 + v1) / 2.0 * dist
    return total, n_skipped


def pixel_path_length_km(pixels, transform):
    """Approximate path length in km by summing pixel step distances."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lon0 = transform.c + transform.a * 0.5
    lat0 = transform.f + transform.e * 0.5
    _, _, dx = geod.inv(lon0, lat0, lon0 + transform.a,  lat0)
    _, _, dy = geod.inv(lon0, lat0, lon0, lat0 + abs(transform.e))
    px_card = (abs(dx) + abs(dy)) / 2 / 1000   # km, cardinal
    px_diag = px_card * np.sqrt(2)              # km, diagonal

    total_km = 0.0
    for i in range(len(pixels) - 1):
        r0, c0 = pixels[i]
        r1, c1 = pixels[i + 1]
        is_diag = (r0 != r1 and c0 != c1)
        total_km += px_diag if is_diag else px_card
    return total_km


def nearest_mine(lat, lon):
    mines = pd.read_csv(MINES_FP)
    mines["dist"] = np.sqrt((mines["LATITUDE"] - lat)**2 + (mines["LONGITUDE"] - lon)**2)
    return mines.loc[mines["dist"].idxmin()]


# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare route costs")
    parser.add_argument("paths_gpkg", type=Path,
                        help="Generated path GeoPackage (output of path-generator)")
    parser.add_argument("--mine", type=str, default=None,
                        help="Mine ID to compare (default: nearest mine to first waypoint)")
    args = parser.parse_args()

    if not args.paths_gpkg.exists():
        sys.exit(f"File not found: {args.paths_gpkg}")

    # ── Find mine ─────────────────────────────────────────────────────────────
    if args.mine:
        mines = pd.read_csv(MINES_FP)
        mine_row = mines[mines["ID"] == args.mine].iloc[0]
    else:
        mine_row = nearest_mine(*ALTERNATIVE_WAYPOINTS[0])

    mine_id   = mine_row["ID"]
    mine_name = mine_row["PROP_NAME"]
    mine_lat  = mine_row["LATITUDE"]
    mine_lon  = mine_row["LONGITUDE"]
    print(f"Mine: {mine_id} — {mine_name}  ({mine_lat:.4f}°N, {mine_lon:.4f}°E)")

    # ── Load friction ─────────────────────────────────────────────────────────
    print(f"\nLoading friction raster (downsample={DOWNSAMPLE}×) …")
    friction, transform = load_friction(FRICTION_FP, DOWNSAMPLE)
    print(f"  Shape: {friction.shape}")

    # ── Route A: model's least-cost path ─────────────────────────────────────
    print(f"\nLoading model path from {args.paths_gpkg.name} …")
    paths_gdf = gpd.read_file(args.paths_gpkg)
    mine_path = paths_gdf[paths_gdf["mine_id"] == mine_id]
    if mine_path.empty:
        sys.exit(f"Mine {mine_id} not found in {args.paths_gpkg.name}. "
                 f"Available: {paths_gdf['mine_id'].tolist()}")

    model_geom = mine_path.iloc[0].geometry
    model_total_cost = float(mine_path.iloc[0]["total_cost"])

    # Trace model path on raster for length calculation
    if model_geom and not model_geom.is_empty:
        if model_geom.geom_type == "LineString":
            coords = list(model_geom.coords)
        else:
            coords = [c for part in model_geom.geoms for c in part.coords]
        model_pixels = [latlon_to_rc(lat, lon, transform, friction.shape)
                        for lon, lat in coords]
        model_km = pixel_path_length_km(model_pixels, transform)
    else:
        model_km = float(mine_path.iloc[0].get("path_km", np.nan))

    # ── Route B: alternative waypoint route ──────────────────────────────────
    alt_waypoints = [(mine_lat, mine_lon)] + ALTERNATIVE_WAYPOINTS
    print(f"\nSnapping {len(alt_waypoints)} waypoints to nearest cheap pixel "
          f"(buffer=±{SNAP_BUFFER_PX} px = ±{SNAP_BUFFER_PX * DOWNSAMPLE * 90:.0f} m) …")
    snapped_pixels = [snap_to_cheapest(lat, lon, friction, transform, SNAP_BUFFER_PX)
                      for lat, lon in alt_waypoints]

    print(f"Tracing alternative route …")
    alt_pixels = []
    for i in range(len(snapped_pixels) - 1):
        r0, c0 = snapped_pixels[i]
        r1, c1 = snapped_pixels[i + 1]
        rr, cc = bresenham(r0, c0, r1, c1)
        segment = list(zip(rr.tolist(), cc.tolist()))
        if alt_pixels and segment:
            segment = segment[1:]
        alt_pixels.extend(segment)
    alt_cost, alt_skipped = mcp_cost_along_pixels(alt_pixels, friction)
    alt_km = pixel_path_length_km(alt_pixels, transform)
    print(f"  {len(alt_pixels):,} pixels traced  ({alt_skipped} nodata pixels skipped)")

    # ── Print comparison ──────────────────────────────────────────────────────
    sep = "─" * 52
    print(f"\n{'='*52}")
    print(f"  Route cost comparison: {mine_id} ({mine_name}) → Kribi")
    print(f"{'='*52}")
    print(f"  {'Route':<28} {'Length (km)':>12} {'Total cost':>12}")
    print(sep)
    print(f"  {'Model (least-cost path)':<28} {model_km:>12.1f} {model_total_cost:>12.3e}")
    print(f"  {'Alternative (central)':<28} {alt_km:>12.1f} {alt_cost:>12.3e}")
    print(sep)

    if alt_cost > model_total_cost:
        ratio = alt_cost / model_total_cost
        print(f"\n  ✓ Model route is {ratio:.1f}× CHEAPER than the alternative,")
        print(f"    despite being {alt_km - model_km:+.0f} km longer in distance.")
    else:
        ratio = model_total_cost / alt_cost
        print(f"\n  Alternative is {ratio:.1f}× cheaper than the model route.")
        print(f"    This may indicate a suboptimal model result for this mine.")

    print()


if __name__ == "__main__":
    main()
