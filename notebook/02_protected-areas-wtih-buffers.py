"""
Build buffered versions of the Cameroon protected areas GeoPackage.

For each buffer size in buffer_sizes_km, the polygons are expanded by that
distance (in kilometres) and saved as a separate GeoPackage under
data/output/processed-protected-areas/with-buffer/.

Projection strategy: reproject to EPSG:32632 (UTM Zone 32N, metric) for
accurate km buffers, then reproject back to EPSG:4326 for output.
"""

from pathlib import Path

import geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INPUT_GPKG = ROOT / "data/output/processed-protected-areas/cmr-protected-areas.gpkg"
OUTPUT_DIR = ROOT / "data/output/processed-protected-areas/with-buffer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
buffer_sizes_km = [3, 5, 10]
METRIC_CRS = "EPSG:32632"   # UTM Zone 32N — covers Cameroon
OUTPUT_CRS = "EPSG:4326"

# ── Load ───────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(INPUT_GPKG)
print(f"Loaded {len(gdf)} protected areas (CRS: {gdf.crs})")

gdf_metric = gdf.to_crs(METRIC_CRS)

# ── Buffer & save ──────────────────────────────────────────────────────────────
for km in buffer_sizes_km:
    metres = km * 1_000
    buffered = gdf_metric.copy()
    buffered["geometry"] = gdf_metric.buffer(metres)
    buffered = buffered.to_crs(OUTPUT_CRS)

    out_path = OUTPUT_DIR / f"cmr-protected-areas__buffer{km}km.gpkg"
    buffered.to_file(out_path, driver="GPKG")
    print(f"Saved {km} km buffer → {out_path.relative_to(ROOT)}")

print("Done.")
