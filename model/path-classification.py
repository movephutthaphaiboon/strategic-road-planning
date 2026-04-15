#!/usr/bin/env python3
"""
path-classification.py — Classify generated mine-to-port paths against the
existing road network.

For each segment of the generated path network, determines whether it:
  - follows an existing PAVED road      → should be maintained / kept
  - follows an existing UNPAVED road    → should be upgraded
  - has no existing road underneath     → needs to be built from scratch

Steps:
  1. Merge all mine paths into one unified network (dissolve overlaps)
  2. Reproject to the road network's projected CRS (metres)
  3. Clip roads to path extent (memory optimization)
  4. Build paved / unpaved buffers; classify path by intersection
  5. Compute km per class; save classified GeoPackage

Usage:
    python path-classification.py <paths_gpkg>
    python path-classification.py <paths_gpkg> --buffer 200 --roads <roads_gpkg>

Example:
    python path-classification.py results/least-cost-paths/late_stage__kribi__base__protected_areas__ds1.gpkg
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import (GeometryCollection, LineString,
                               MultiLineString, MultiPolygon, Polygon)
from shapely.ops import linemerge, unary_union

warnings.filterwarnings("ignore")

# =============================================================================
# DEFAULTS
# =============================================================================

BASE_DIR  = Path(__file__).parent.parent
ROADS_FP  = BASE_DIR / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"

# Buffer around existing roads (metres) used to detect overlap with generated paths.
# Should be ~1.5–2× the raster pixel size used to generate paths.
#   ds=1  (~90 m pixel)  → use ~150 m
#   ds=5  (~450 m pixel) → use ~700 m
DEFAULT_BUFFER_M = 150

ROAD_TYPE_COL = "liu_surface"   # column in road network with 'paved' / 'unpaved'

# =============================================================================
# HELPERS
# =============================================================================

def extract_lines(geom):
    """
    Return only the LineString/MultiLineString parts of a geometry.
    Intersection / difference operations can return GeometryCollections
    that mix lines and points; we only want the lines.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (LineString, MultiLineString)):
        return geom
    if isinstance(geom, GeometryCollection):
        lines = [g for g in geom.geoms
                 if isinstance(g, (LineString, MultiLineString)) and not g.is_empty]
        if not lines:
            return None
        return unary_union(lines)
    return None


def geom_length_km(geom) -> float:
    """Length of a geometry in km (CRS must be in metres)."""
    if geom is None or geom.is_empty:
        return 0.0
    return geom.length / 1000.0


# =============================================================================
# MAIN STEPS
# =============================================================================

def load_paths(fp: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(fp)
    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    print(f"  Paths: {len(valid)} valid route(s) loaded from {fp.name}")
    return valid


def merge_path_network(paths_gdf: gpd.GeoDataFrame):
    """
    Union all mine-to-port paths into one network, removing duplicated segments
    where multiple mines share the same route section.
    Returns a single (Multi)LineString.
    """
    unioned = unary_union(paths_gdf.geometry)
    merged  = linemerge(unioned)          # stitch touching lines end-to-end
    total_km = geom_length_km(merged)
    print(f"  Merged network: {merged.geom_type}, {total_km:.1f} km total")
    return merged


def load_roads(fp: Path, clip_geom, target_crs) -> gpd.GeoDataFrame:
    """
    Load road network, reproject to target_crs, and clip to clip_geom extent
    to avoid processing the full national dataset (saves significant RAM/time).
    """
    gdf = gpd.read_file(fp, columns=[ROAD_TYPE_COL, "geometry"])
    gdf = gdf[gdf.geometry.notna()].copy()

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Clip to path bounding box + margin to limit the dataset size
    clip_bounds = clip_geom.bounds          # (minx, miny, maxx, maxy)
    margin = 2000                           # 2 km margin around path extent
    bbox = (clip_bounds[0] - margin, clip_bounds[1] - margin,
            clip_bounds[2] + margin, clip_bounds[3] + margin)
    bbox_poly = gpd.GeoSeries([Polygon([
        (bbox[0], bbox[1]), (bbox[2], bbox[1]),
        (bbox[2], bbox[3]), (bbox[0], bbox[3]),
    ])], crs=target_crs).iloc[0]
    gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()

    counts = gdf[ROAD_TYPE_COL].value_counts().to_dict()
    print(f"  Roads (clipped to path extent): {len(gdf)} segments  {counts}")
    return gdf


def build_road_buffers(roads_gdf: gpd.GeoDataFrame, buffer_m: float):
    """
    Dissolve roads by surface type and buffer each.
    Returns (paved_buffer_polygon, unpaved_buffer_polygon).
    Dissolving first (292k → 2 geometries) makes buffering tractable.
    """
    paved_geom   = unary_union(roads_gdf.loc[roads_gdf[ROAD_TYPE_COL] == "paved",   "geometry"])
    unpaved_geom = unary_union(roads_gdf.loc[roads_gdf[ROAD_TYPE_COL] == "unpaved", "geometry"])

    paved_buf   = paved_geom.buffer(buffer_m)   if not paved_geom.is_empty   else Polygon()
    unpaved_buf = unpaved_geom.buffer(buffer_m) if not unpaved_geom.is_empty else Polygon()

    print(f"  Buffers built: paved={buffer_m}m, unpaved={buffer_m}m")
    return paved_buf, unpaved_buf


def classify(merged_path, paved_buf, unpaved_buf) -> dict:
    """
    Split merged_path into three non-overlapping classes.
    Paved takes priority over unpaved where both buffers overlap.
    """
    paved_part   = extract_lines(merged_path.intersection(paved_buf))
    unpaved_part = extract_lines(merged_path.intersection(unpaved_buf).difference(paved_buf))
    new_part     = extract_lines(merged_path.difference(paved_buf).difference(unpaved_buf))

    return {
        "paved":        paved_part,
        "unpaved":      unpaved_part,
        "to_be_built":  new_part,
    }


def build_output_gdf(parts: dict, crs) -> gpd.GeoDataFrame:
    """Explode classified geometry parts into a flat GeoDataFrame."""
    rows = []
    for road_type, geom in parts.items():
        if geom is None or geom.is_empty:
            continue
        geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            if not g.is_empty:
                rows.append({
                    "road_type": road_type,
                    "length_km": round(g.length / 1000, 4),
                    "geometry":  g,
                })
    return gpd.GeoDataFrame(rows, crs=crs)


def print_summary(result_gdf: gpd.GeoDataFrame, buffer_m: float):
    summary = result_gdf.groupby("road_type")["length_km"].sum().round(2)
    total   = summary.sum()
    print(f"\n  Road classification summary (buffer = {buffer_m} m):")
    print(f"  {'─'*45}")
    for road_type in ["paved", "unpaved", "to_be_built"]:
        km  = summary.get(road_type, 0.0)
        pct = 100 * km / total if total > 0 else 0
        label = {"paved": "Paved (maintain)", "unpaved": "Unpaved (upgrade)",
                 "to_be_built": "New road (build)"}[road_type]
        print(f"  {label:<25}: {km:>8.1f} km  ({pct:5.1f}%)")
    print(f"  {'─'*45}")
    print(f"  {'Total':<25}: {total:>8.1f} km")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify mine-to-port paths against existing road network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths_gpkg", type=Path,
                        help="Input paths GeoPackage (output of path-generator)")
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER_M,
                        metavar="METRES",
                        help=f"Buffer around roads for overlap detection "
                             f"(default: {DEFAULT_BUFFER_M} m). "
                             f"Use ~1.5× raster pixel size: ds=1→150, ds=5→700.")
    parser.add_argument("--roads", type=Path, default=ROADS_FP,
                        help="Road network GeoPackage (default: project road layer)")
    args = parser.parse_args()

    if not args.paths_gpkg.exists():
        sys.exit(f"Error: '{args.paths_gpkg}' not found.")
    if not args.roads.exists():
        sys.exit(f"Error: road network not found at '{args.roads}'.")

    print("=" * 62)
    print("  PATH CLASSIFICATION")
    print("=" * 62)

    # 1. Load and merge generated paths (in EPSG:4326)
    print("\n[1/4] Loading and merging generated paths …")
    paths_gdf    = load_paths(args.paths_gpkg)
    merged_path  = merge_path_network(paths_gdf)

    # 2. Reproject merged path to road CRS (projected, metres)
    road_crs = gpd.read_file(args.roads, rows=1).crs
    print(f"\n[2/4] Reprojecting to road CRS: {road_crs} …")
    merged_proj = (gpd.GeoSeries([merged_path], crs=paths_gdf.crs)
                   .to_crs(road_crs)
                   .iloc[0])

    # 3. Load roads clipped to path extent
    print("\n[3/4] Loading road network …")
    roads_gdf = load_roads(args.roads, merged_proj, road_crs)

    # 4. Buffer, classify, summarise
    print(f"\n[4/4] Classifying (buffer={args.buffer} m) …")
    paved_buf, unpaved_buf = build_road_buffers(roads_gdf, args.buffer)
    parts      = classify(merged_proj, paved_buf, unpaved_buf)
    result_gdf = build_output_gdf(parts, road_crs)

    print_summary(result_gdf, args.buffer)

    # Save — reproject back to EPSG:4326 for consistency with other outputs
    out_fp = args.paths_gpkg.parent / f"classified_{args.paths_gpkg.stem}.gpkg"
    result_gdf.to_crs("EPSG:4326").to_file(out_fp, driver="GPKG")
    print(f"\n  Saved → {out_fp}")
    print("\nDone.")


if __name__ == "__main__":
    main()
