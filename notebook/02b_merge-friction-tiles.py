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
from rasterio.mask import mask as rio_mask

# ---- paths ----
TILE_DIR   = "../data/output/cmr-construction-cost-friction-90m"
OUT_VRT    = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m.vrt"
OUT_TIF    = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_merged.tif"
OUT_CLIPPED = "../data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
BOUNDARY_SHP = "../data/input/cmr_admin_boundaries_humdata/cmr-buffer.gpkg"

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
# Step 2: Pad each tile by 1 pixel on south and east edges, then merge
# ============================================================
# Adjacent tiles have a 1-pixel gap at shared edges due to sub-pixel grid
# offsets from independent downscaling. Repeating the last row/column by
# exactly 1 pixel fills the gap without duplicating data visibly.

from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

N_PAD = 1

print(f"\nPadding {len(tile_files)} tiles by {N_PAD} pixel on south and east edges ...")
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

        data[data == 0] = np.nan   # fill value → transparent for merge

        data = np.concatenate([data, data[:, -N_PAD:, :]], axis=1)   # south
        data = np.concatenate([data, data[:, :, -N_PAD:]], axis=2)   # east
        meta.update({"height": data.shape[1], "width": data.shape[2]})

        mf = MemoryFile()
        with mf.open(**meta) as m:
            m.write(data)
        padded_memfiles.append(mf)
        padded_sources.append(mf.open())

mosaic, transform = merge(
    padded_sources,
    resampling=Resampling.nearest,
    method="first",
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
