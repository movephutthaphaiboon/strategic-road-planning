"""
Forest cover layer for Cameroon (clipped to buffered boundary).

Memory-efficient approach — no full mosaic is ever held in RAM:
  1. Snap the output extent to the Hansen pixel grid (buffer bounding box).
  2. Create an empty output GeoTIFF at that extent.
  3. For each tile pair, read only the window that overlaps the output,
     zero out tree cover where lossyear > 0, and write to the output file.
  4. Apply the buffer polygon as a vector mask block-by-block.
"""

import math
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
HANSEN_DIR = BASE / "data/input/global-forest-change-hansen"
BUFFER_PATH = BASE / "data/input/cmr_admin_boundaries_humdata/cmr-buffer.gpkg"
OUT_DIR = BASE / "data/output/processed-hansen-treecover"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Collect tile pairs
# ---------------------------------------------------------------------------
treecover_files = sorted(HANSEN_DIR.glob("*_treecover2000_*.tif"))
if not treecover_files:
    raise FileNotFoundError(f"No treecover2000 tiles found in {HANSEN_DIR}")

print(f"Found {len(treecover_files)} treecover2000 tiles:")
for f in treecover_files:
    print(f"  {f.name}")

# ---------------------------------------------------------------------------
# Load Cameroon buffer (reproject to WGS84 to match Hansen CRS)
# ---------------------------------------------------------------------------
buffer_gdf = gpd.read_file(BUFFER_PATH)
if buffer_gdf.crs is None or buffer_gdf.crs.to_epsg() != 4326:
    buffer_gdf = buffer_gdf.to_crs(epsg=4326)
buffer_geom = [geom.__geo_interface__ for geom in buffer_gdf.geometry]
buf_minx, buf_miny, buf_maxx, buf_maxy = buffer_gdf.total_bounds

# ---------------------------------------------------------------------------
# Derive output grid from the first tile's pixel size.
# All Hansen tiles share the same pixel grid (corners aligned to integer
# degrees), so we snap the output extent to that grid using integer arithmetic.
# ---------------------------------------------------------------------------
with rasterio.open(treecover_files[0]) as ref:
    res_x = ref.transform.a        # positive  (~0.00025 °)
    res_y = ref.transform.e        # negative  (~-0.00025 °)
    crs = ref.crs
    nodata = int(ref.nodata) if ref.nodata is not None else 255
    dtype = ref.dtypes[0]
    origin_x = ref.transform.c     # tile left  edge (integer °)
    origin_y = ref.transform.f     # tile top   edge (integer °)

# Column / row offsets in the tile's pixel grid
col_start = math.floor((buf_minx - origin_x) / res_x)
col_end   = math.ceil( (buf_maxx - origin_x) / res_x)
row_start = math.floor((buf_maxy - origin_y) / res_y)   # res_y < 0
row_end   = math.ceil( (buf_miny - origin_y) / res_y)

out_width  = col_end  - col_start
out_height = row_end  - row_start
out_left   = origin_x + col_start * res_x
out_top    = origin_y + row_start * res_y
out_transform = Affine(res_x, 0.0, out_left, 0.0, res_y, out_top)

print(f"\nOutput grid: {out_height} rows × {out_width} cols")
print(f"  Extent: {out_left:.4f}°E  {out_top:.4f}°N  →  "
      f"{out_left + out_width*res_x:.4f}°E  {out_top + out_height*res_y:.4f}°N")

out_profile = {
    "driver":    "GTiff",
    "dtype":     dtype,
    "nodata":    nodata,
    "width":     out_width,
    "height":    out_height,
    "count":     1,
    "crs":       crs,
    "transform": out_transform,
    "compress":  "lzw",
    "tiled":     True,
    "blockxsize": 512,
    "blockysize": 512,
}

out_path = OUT_DIR / "cameroon_treecover2024_30m.tif"

# ---------------------------------------------------------------------------
# Write tiles one at a time (no full mosaic in memory)
# ---------------------------------------------------------------------------
with rasterio.open(out_path, "w", **out_profile) as out_ds:

    for tc_path in treecover_files:
        ly_path = Path(str(tc_path).replace("_treecover2000_", "_lossyear_"))
        if not ly_path.exists():
            print(f"\nWARNING: lossyear tile not found for {tc_path.name}, skipping.")
            continue

        tile_label = tc_path.stem.split("_treecover2000_")[1]
        print(f"\nProcessing {tile_label}...")

        with rasterio.open(tc_path) as tc_src:
            tc_origin_x = tc_src.transform.c
            tc_origin_y = tc_src.transform.f
            tc_width    = tc_src.width
            tc_height   = tc_src.height

            # Tile extents in the shared pixel grid (integer offsets)
            tile_col0 = round((tc_origin_x - origin_x) / res_x)
            tile_row0 = round((tc_origin_y - origin_y) / res_y)

            # Intersection of tile and output, in shared-grid coordinates
            inter_col_start = max(col_start, tile_col0)
            inter_col_end   = min(col_end,   tile_col0 + tc_width)
            inter_row_start = max(row_start, tile_row0)
            inter_row_end   = min(row_end,   tile_row0 + tc_height)

            if inter_col_start >= inter_col_end or inter_row_start >= inter_row_end:
                print(f"  No overlap with output extent — skipping.")
                continue

            inter_w = inter_col_end - inter_col_start
            inter_h = inter_row_end - inter_row_start

            # Window in the source tile
            src_window = Window(
                col_off = inter_col_start - tile_col0,
                row_off = inter_row_start - tile_row0,
                width   = inter_w,
                height  = inter_h,
            )
            # Window in the output file
            out_window = Window(
                col_off = inter_col_start - col_start,
                row_off = inter_row_start - row_start,
                width   = inter_w,
                height  = inter_h,
            )

            tc_data = tc_src.read(1, window=src_window)

        with rasterio.open(ly_path) as ly_src:
            ly_data = ly_src.read(1, window=src_window)

        # Zero tree cover where loss occurred (lossyear > 0 → loss in 2001-2024)
        tc_data[ly_data > 0] = 0

        out_ds.write(tc_data, 1, window=out_window)
        print(f"  Written: {inter_h} rows × {inter_w} cols → output window "
              f"(row {out_window.row_off}, col {out_window.col_off})")

# ---------------------------------------------------------------------------
# Apply vector mask block-by-block (set pixels outside buffer to nodata)
# ---------------------------------------------------------------------------
print("\nApplying Cameroon buffer mask (block-by-block)...")

with rasterio.open(out_path, "r+") as ds:
    for _, window in ds.block_windows(1):
        data = ds.read(1, window=window)
        win_transform = ds.window_transform(window)

        outside = geometry_mask(
            buffer_geom,
            transform=win_transform,
            out_shape=(window.height, window.width),
            invert=False,   # True = outside the buffer
        )
        data[outside] = nodata
        ds.write(data, 1, window=window)

print(f"\nDone. Output saved to:\n  {out_path}")
print(f"  Shape: {out_height} rows × {out_width} cols")
print(f"  Resolution: {res_x:.7f} ° (~30 m)")
