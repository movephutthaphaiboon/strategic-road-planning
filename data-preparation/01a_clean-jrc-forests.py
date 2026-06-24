"""
Forest cover layer for Cameroon from JRC Global Forest Cover 2020 (clipped to buffered boundary).

Memory-efficient approach — no full mosaic is ever held in RAM:
  1. Snap the output extent to a 0.00025° grid (matching Hansen ~30 m resolution).
  2. Create an empty output GeoTIFF at that extent filled with nodata.
  3. For each tile, read the overlapping window and resample on-the-fly to the
     output resolution using nearest-neighbour, then write to the output file.
  4. Apply the buffer polygon as a vector mask block-by-block.

JRC GFC 2020 pixel values: 1 = forest, 0 = non-forest.
Pixels outside the Cameroon buffer are set to nodata (255).
"""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
JRC_DIR = BASE / "data/input/global-forest-cover-2020-forobs"
BUFFER_PATH = BASE / "data/input/cmr_admin_boundaries_humdata/cmr-buffer.gpkg"
OUT_DIR = BASE / "data/output/processed-jrc-treecover"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODATA = 255
# Target resolution matching Hansen output (~30 m)
TARGET_RES_X =  0.00025
TARGET_RES_Y = -0.00025

# ---------------------------------------------------------------------------
# Collect tiles
# ---------------------------------------------------------------------------
tile_files = sorted(JRC_DIR.glob("JRC_GFC2020_V3_*.tif"))
if not tile_files:
    raise FileNotFoundError(f"No JRC GFC 2020 tiles found in {JRC_DIR}")

print(f"Found {len(tile_files)} JRC tiles:")
for f in tile_files:
    print(f"  {f.name}")

# ---------------------------------------------------------------------------
# Load Cameroon buffer (reproject to WGS84 to match JRC CRS)
# ---------------------------------------------------------------------------
buffer_gdf = gpd.read_file(BUFFER_PATH)
if buffer_gdf.crs is None or buffer_gdf.crs.to_epsg() != 4326:
    buffer_gdf = buffer_gdf.to_crs(epsg=4326)
buffer_geom = [geom.__geo_interface__ for geom in buffer_gdf.geometry]
buf_minx, buf_miny, buf_maxx, buf_maxy = buffer_gdf.total_bounds

print(f"\nCameroon buffer extent:")
print(f"  {buf_minx:.4f}°E  {buf_miny:.4f}°N  →  {buf_maxx:.4f}°E  {buf_maxy:.4f}°N")

# ---------------------------------------------------------------------------
# Read native JRC resolution and CRS from the first tile.
# Use an integer-degree point as the shared origin — both the JRC grid
# (1/12000°) and the target grid (1/4000° = 0.00025°) align exactly at
# integer degrees, so snapping works cleanly.
# ---------------------------------------------------------------------------
with rasterio.open(tile_files[0]) as ref:
    jrc_res_x = ref.transform.a    # ~8.333e-05 °
    jrc_res_y = ref.transform.e    # ~-8.333e-05 °
    crs = ref.crs
    dtype = ref.dtypes[0]
    origin_x = ref.transform.c    # integer ° (tile left edge)
    origin_y = ref.transform.f    # integer ° (tile top  edge)

print(f"\nJRC native resolution: {jrc_res_x:.4e} ° (~10 m)")
print(f"Output  resolution:    {TARGET_RES_X:.4e} ° (~30 m, matching Hansen)")

# ---------------------------------------------------------------------------
# Snap output extent to the target pixel grid
# ---------------------------------------------------------------------------
col_start = math.floor((buf_minx - origin_x) / TARGET_RES_X)
col_end   = math.ceil( (buf_maxx - origin_x) / TARGET_RES_X)
row_start = math.floor((buf_maxy - origin_y) / TARGET_RES_Y)   # TARGET_RES_Y < 0
row_end   = math.ceil( (buf_miny - origin_y) / TARGET_RES_Y)

out_width  = col_end  - col_start
out_height = row_end  - row_start
out_left   = origin_x + col_start * TARGET_RES_X
out_top    = origin_y + row_start * TARGET_RES_Y
out_transform = Affine(TARGET_RES_X, 0.0, out_left, 0.0, TARGET_RES_Y, out_top)

# Geographic bounds of the output extent (for intersection math)
out_right = out_left + out_width  * TARGET_RES_X
out_bot   = out_top  + out_height * TARGET_RES_Y   # smaller latitude

print(f"\nOutput grid: {out_height} rows × {out_width} cols")
print(f"  Extent: {out_left:.4f}°E  {out_top:.4f}°N  →  "
      f"{out_right:.4f}°E  {out_bot:.4f}°N")

out_profile = {
    "driver":    "GTiff",
    "dtype":     dtype,
    "nodata":    NODATA,
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

out_path = OUT_DIR / "cameroon_jrc_forest2020_30m.tif"

# ---------------------------------------------------------------------------
# Write tiles one at a time, resampling to target resolution on the fly
# ---------------------------------------------------------------------------
with rasterio.open(out_path, "w", **out_profile) as out_ds:

    # Initialise with nodata so unwritten areas aren't garbage
    for _, init_window in out_ds.block_windows(1):
        block_data = np.full(
            (init_window.height, init_window.width), NODATA, dtype=dtype
        )
        out_ds.write(block_data, 1, window=init_window)

    for tile_path in tile_files:
        tile_label = tile_path.stem
        print(f"\nProcessing {tile_label}...")

        with rasterio.open(tile_path) as tile_src:
            tc_origin_x = tile_src.transform.c
            tc_origin_y = tile_src.transform.f
            tc_width    = tile_src.width
            tc_height   = tile_src.height

        # Tile geographic bounds
        tile_left  = tc_origin_x
        tile_top   = tc_origin_y
        tile_right = tc_origin_x + tc_width  * jrc_res_x
        tile_bot   = tc_origin_y + tc_height * jrc_res_y   # smaller lat

        # Intersection in geographic coordinates
        inter_left  = max(out_left,  tile_left)
        inter_right = min(out_right, tile_right)
        inter_top   = min(out_top,   tile_top)   # smaller value = further north
        inter_bot   = max(out_bot,   tile_bot)

        if inter_left >= inter_right or inter_bot >= inter_top:
            print("  No overlap with output extent — skipping.")
            continue

        # Source window in JRC native pixels
        src_col_off = round((inter_left - tile_left) / jrc_res_x)
        src_row_off = round((inter_top  - tile_top)  / jrc_res_y)
        src_w       = round((inter_right - inter_left) / jrc_res_x)
        src_h       = round((inter_bot   - inter_top)  / jrc_res_y)

        # Output window in target-resolution pixels
        out_col_off = round((inter_left - out_left) / TARGET_RES_X)
        out_row_off = round((inter_top  - out_top)  / TARGET_RES_Y)
        out_w       = round((inter_right - inter_left) / TARGET_RES_X)
        out_h       = round((inter_bot   - inter_top)  / TARGET_RES_Y)

        src_window = Window(src_col_off, src_row_off, src_w, src_h)
        out_window = Window(out_col_off, out_row_off, out_w, out_h)

        with rasterio.open(tile_path) as tile_src:
            # Read at native res, resample to output size in one call
            tile_data = tile_src.read(
                1,
                window=src_window,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            )

        out_ds.write(tile_data, 1, window=out_window)
        print(f"  Written: {out_h} rows × {out_w} cols → output window "
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
            invert=False,   # False = outside the buffer → True in mask
        )
        data[outside] = NODATA
        ds.write(data, 1, window=window)

print(f"\nDone. Output saved to:\n  {out_path}")
print(f"  Shape: {out_height} rows × {out_width} cols")
print(f"  Resolution: {TARGET_RES_X:.5f} ° (~30 m, matching Hansen)")
print(f"  Pixel values: 1=forest, 0=non-forest, {NODATA}=outside buffer")
