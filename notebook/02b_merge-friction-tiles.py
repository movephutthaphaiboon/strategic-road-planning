"""
Merge individual 3x3-degree friction tiles into a single seamless raster,
then clip and mask to the Cameroon national boundary.

Root cause of tile seams: each tile is processed and downscaled independently,
so the resulting pixel grids differ by a sub-pixel offset at shared edges.
rasterio.merge snaps all tiles to one common aligned grid, eliminating the gaps.

Outputs:
  1. A VRT (Virtual Raster) — instant, no data copy, open directly in QGIS.
  2. A merged GeoTIFF — single file on a common pixel grid.
  3. A clipped GeoTIFF — merged raster masked to Cameroon boundary (NaN outside).
"""

import glob
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

# ---- paths ----
TILE_DIR   = "../data/output/cmr-construction-cost-friction-90m"
OUT_VRT    = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m.vrt"
OUT_TIF    = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_merged.tif"
OUT_CLIPPED = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
BOUNDARY_SHP = "../data/input/cmr_admin_boundaries_humdata/cmr_admin0_em.shp"

# ---- collect tiles ----
tile_files = sorted(glob.glob(os.path.join(TILE_DIR, "construction_cost_friction_*.tif")))
print(f"Found {len(tile_files)} tiles:")
for f in tile_files:
    print(f"  {os.path.basename(f)}")

assert len(tile_files) > 0, "No tiles found — check TILE_DIR path"

# ============================================================
# Step 1: VRT — lightweight virtual mosaic (open in QGIS now)
# ============================================================
# gdalbuildvrt via rasterio's GDAL bindings
from osgeo import gdal

vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
vrt_ds = gdal.BuildVRT(OUT_VRT, tile_files, options=vrt_options)
vrt_ds.FlushCache()
vrt_ds = None
print(f"\nVRT saved → {OUT_VRT}")
print("  Open this in QGIS for a quick seamless preview (no data duplicated).")

# ============================================================
# Step 2: Pad tile edges, then merge onto a common pixel grid
# ============================================================
# Each tile is processed independently and downscaled, so adjacent tiles
# end up with a sub-pixel grid offset at their shared boundary — visible
# as a thin gap or stripe in QGIS.
#
# Fix: extend each tile by N_PAD pixels at its south and east edges by
# repeating the edge row/column. This gives rasterio.merge overlapping
# data at every seam so it always has a value to place there.
#
# Tile ordering (sorted = south-to-north, west-to-east) combined with
# method='first' means each tile's real interior data wins at the seam;
# the padded extension of the neighbouring tile only fills any leftover gap.

N_PAD = 5   # pixels to extend (~450 m at 90 m resolution; well beyond 1-pixel offset)

print(f"\nPadding {len(tile_files)} tiles by {N_PAD} pixels at south and east edges...")
padded_memfiles = []
padded_sources  = []
first_meta      = None

for tile_file in tile_files:
    with rasterio.open(tile_file) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan})
        if first_meta is None:
            first_meta = meta.copy()

        # The processing script writes 0 as the fill value; convert to NaN
        # so merge() treats those pixels as transparent.
        data[data == 0] = np.nan

        # --- south edge padding ---
        # In raster convention row 0 = north, last row = south.
        # Repeating the last N rows extends the tile south without inventing values.
        data = np.concatenate([data, data[:, -N_PAD:, :]], axis=1)

        # --- east edge padding ---
        data = np.concatenate([data, data[:, :, -N_PAD:]], axis=2)

        meta.update({"height": data.shape[1], "width": data.shape[2]})

        mf = MemoryFile()
        with mf.open(**meta) as m:
            m.write(data)
        padded_memfiles.append(mf)
        padded_sources.append(mf.open())   # keep open for merge()

mosaic, transform = merge(
    padded_sources,
    resampling=Resampling.nearest,
    method="first",   # real interior data (placed first) wins over padded extensions
    nodata=np.nan,
)

for src in padded_sources:
    src.close()
for mf in padded_memfiles:
    mf.close()

out_meta = first_meta.copy()
out_meta.update({
    "driver":     "GTiff",
    "height":     mosaic.shape[1],
    "width":      mosaic.shape[2],
    "transform":  transform,
    "nodata":     np.nan,
    "dtype":      "float32",
    "compress":   "lzw",
    "tiled":      True,
    "blockxsize": 512,
    "blockysize": 512,
})

with rasterio.open(OUT_TIF, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f"\nMerged GeoTIFF saved → {OUT_TIF}")
print(f"  Shape : {mosaic.shape[1]} rows × {mosaic.shape[2]} cols")
print(f"  Dtype : {mosaic.dtype}")

# ============================================================
# Step 3: Clip to Cameroon boundary — mask outside to NaN
# ============================================================
# Read the boundary shapefile and reproject to match the raster CRS (EPSG:4326).
cmr = gpd.read_file(BOUNDARY_SHP)
print(f"\nBoundary shapefile CRS : {cmr.crs}")

with rasterio.open(OUT_TIF) as src:
    raster_crs = src.crs
    if cmr.crs != raster_crs:
        cmr = cmr.to_crs(raster_crs)
        print(f"  Reprojected boundary → {raster_crs}")

    # Extract geometries as GeoJSON-like dicts for rasterio
    shapes = [geom.__geo_interface__ for geom in cmr.geometry]

    # crop=True trims the raster extent to the bounding box of the shapes.
    # All pixels outside the boundary polygons are set to nodata (np.nan here).
    clipped, clipped_transform = rio_mask(
        src,
        shapes,
        crop=True,
        nodata=np.nan,
        all_touched=True,   # only pixels whose centre falls inside the boundary
    )

    clipped_meta = src.meta.copy()
    clipped_meta.update({
        "driver":     "GTiff",
        "height":     clipped.shape[1],
        "width":      clipped.shape[2],
        "transform":  clipped_transform,
        "nodata":     np.nan,
        "dtype":      "float32",
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 512,
        "blockysize": 512,
    })

# Cast to float32 so NaN is a valid nodata value
clipped = clipped.astype(np.float32)

with rasterio.open(OUT_CLIPPED, "w", **clipped_meta) as dest:
    dest.write(clipped)

print(f"\nClipped GeoTIFF saved → {OUT_CLIPPED}")
print(f"  Shape  : {clipped.shape[1]} rows × {clipped.shape[2]} cols")
print(f"  Nodata : NaN (pixels outside Cameroon boundary)")
print("\nDone.")
