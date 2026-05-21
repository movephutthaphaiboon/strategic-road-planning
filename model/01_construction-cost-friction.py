import geopandas as gpd
import pandas as pd
import rioxarray as rio
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from fuzzywuzzy import process
from rasterio import features
import datetime

# ====================================================================

def computePercentageSlope(dem):
    """compute percentage slope

    Parameters
    ----------
    dem: object containing the dem

    Returns
    -------
    array containing the percentage slope
    """
    dx = dem.differentiate('x')
    dy = dem.differentiate('y')
    # the scaling factor of 111120 converts degrees to metres.
    # this is a good approximation near the equator
    slope = (dx * dx + dy * dy) ** 0.5 * 100 / 111120
    return slope


def slopespeed(slopes, TI):
    """Calculate road construction cost via earthwork volume model (EWV)"""

    RW = 8
    def _model(x, y):
        a = 0.2 * y**(0.57) - 1
        b = 1.06 * y**(0.060)
        C = 747.37 * y**(0.23)
        d = 1.27 * y**(0.22) - 1
        EWV = C * (1) * RW**(a) * (1 + np.exp(b * (x/100 - d)))
        return EWV

    return xr.apply_ufunc(
        _model,
        slopes,
        TI,
        dask='parallelized',
        output_dtypes=[np.float32]
    )


def readLandcoverSpeedMap(fname, landcover='Code',
                          speed='cost_km',
                          dropNaN=True, scale=None):
    """construct landcover to cost map"""
    costs = pd.read_csv(fname)
    if dropNaN:
        costs.dropna(inplace=True)
    lc = np.array(costs[landcover])
    s = np.array(costs[speed])
    if scale:
        s = (np.round(s * scale)).astype(np.int64)
    return (lc, s)


def readRoadSpeedMap(fname, road='fclass',
                     speed='cost_km',
                     dropNaN=True):
    """construct road type to cost map"""
    costs = pd.read_csv(fname, index_col=road)
    if dropNaN:
        costs = costs[costs[speed].notna()]
    return costs[speed]


# FIX: use `roads` parameter instead of `roads_all` global.
# FIX: rewritten as explicit list comprehension — clearer and avoids lazy iterator issues.
# FIX: early exit with warning when no roads match the map keys (helps diagnose fuzzy-match failures).
def rasterizeRoads(roads, landcover, road_speed_map):
    """rasterize roads

    Parameters
    ----------
    roads: roads GeoDataFrame
    landcover: xarray used as raster template
    road_speed_map: dict mapping fclass value to rasterized value

    Returns
    -------
    numpy array containing the rasterized values
    """
    shapes = [
        (row.geometry, road_speed_map[row.fclass])
        for _, row in roads.iterrows()
        if row.fclass in road_speed_map
    ]

    if not shapes:
        print("  WARNING: rasterizeRoads — no road features matched road_speed_map keys.")
        print(f"  road_speed_map keys: {list(road_speed_map.keys())}")
        print(f"  sample fclass values: {roads['fclass'].unique()[:5]}")
        return np.zeros(landcover.rio.shape, dtype=np.float32)

    speedsurface = features.rasterize(
        shapes,
        out_shape=landcover.rio.shape,
        transform=landcover.rio.transform(),
        all_touched=True,
        fill=0,
        dtype=np.float32)

    return speedsurface


# FIX: copy road_speed_map to avoid mutating the caller's Series in place
#      (original modifies .index directly which is a side effect).
def rasterizeAllRoads(roads, landcover, road_speed_map, maxspeed=True):
    """rasterize all roads

    Parameters
    ----------
    roads: roads GeoDataFrame
    landcover: xarray used as raster template
    road_speed_map: pandas Series (fclass → value)
    maxspeed: when True, pixels with multiple roads keep the highest value

    Returns
    -------
    xarray containing the rasterized surface
    """
    road_speed_map = road_speed_map.copy()  # FIX: don't mutate the original

    road_types = roads['fclass'].unique()
    idx = road_speed_map.index.to_list()
    for i, rt in enumerate(idx):
        matched_rt = process.extractOne(rt, road_types)[0]
        print(f'  fuzzy match: "{rt}" → "{matched_rt}"')
        idx[i] = matched_rt
    road_speed_map.index = idx

    if maxspeed:
        rcost = rasterizeAllRoadsMax(roads, landcover, road_speed_map)
    else:
        rcost = rasterizeRoads(roads, landcover, road_speed_map.to_dict())

    # replace fill value (0) with NaN so xr.where can distinguish road vs non-road
    rcost = np.where(rcost == 0, np.nan, rcost)

    speedsurface = xr.zeros_like(landcover, dtype=np.float32)
    speedsurface.values[:, :] = rcost[:, :]

    return speedsurface


# FIX: use landcover.rio.shape (2D tuple) instead of landcover.shape[1:],
#      which gives a 1D tuple when the DataArray has been squeezed to 2D.
def rasterizeAllRoadsMax(roads, landcover, road_speed_map):
    """rasterize all roads, keeping the highest value per pixel

    Parameters
    ----------
    roads: roads GeoDataFrame
    landcover: xarray used as raster template
    road_speed_map: pandas Series (fclass → value)

    Returns
    -------
    numpy array containing the rasterized surface
    """
    speedsurface = np.zeros(landcover.rio.shape, dtype=np.float32)  # FIX: was shape[1:]

    for speed in road_speed_map.unique():
        selected_roads = road_speed_map[road_speed_map == speed]
        rcost = rasterizeRoads(roads, landcover, selected_roads.to_dict())
        speedsurface = np.maximum(speedsurface, rcost)

    return speedsurface


def speed_to_cost(speed):
    """convert cost-per-km surface to total cost per pixel

    Parameters
    ----------
    speed: DataArray with values in mUSD/km

    Returns
    -------
    DataArray with values in USD per pixel
    """
    # pixel_size_m * cost_mUSD/km * 1e6 (mUSD→USD) / 1000 (km→m) = USD/pixel
    return abs(speed.rio.resolution()[0]) * 111120 * speed * 1e6 / (1000)


def clip_box(rio_orig, rio_clip):
    return rio_orig.rio.clip_box(
        minx=rio_clip.x.min().values,
        miny=rio_clip.y.min().values,
        maxx=rio_clip.x.max().values,
        maxy=rio_clip.y.max().values)


# FIX: added resampling parameter.
#      Use Resampling.bilinear for continuous data (DEM, cost surfaces).
#      Use Resampling.max for categorical data (landcover).
def downscaling(rio_orig, downscale_factor, resampling=Resampling.bilinear):
    """Downscale a rioxarray DataArray by integer factor.

    Parameters
    ----------
    rio_orig: input DataArray
    downscale_factor: int, e.g. 3 reduces 30m → 90m
    resampling: rasterio Resampling method
                bilinear for continuous (DEM), max for categorical (landcover)
    """
    new_width = int(round(rio_orig.rio.width / downscale_factor))
    new_height = int(round(rio_orig.rio.height / downscale_factor))
    return rio_orig.rio.reproject(
        dst_crs=rio_orig.rio.crs,
        shape=(new_height, new_width),
        resampling=resampling)


def TI_index(x):
    if x/100 <= 0.05:
        TI = 0 + 40 * x/100
    elif x/100 > 0.05 and x/100 <= 0.1:
        TI = 2 + 40 * ((x/100 - 0.05))
    elif x/100 > 0.1 and x/100 <= 0.3:
        TI = 4 + 10 * ((x/100 - 0.1))
    elif x/100 > 0.3 and x/100 <= 0.65:
        TI = 6 + 5.71 * ((x/100 - 0.3))
    elif x/100 > 0.65:
        TI = 8 + 5.71 * ((x/100 - 0.65))
    else:
        TI = np.nan
    return TI

# ====================================================================

# --- Configuration ---
# Increase DEM_DOWNSCALE_FACTOR to lower resolution and speed up processing.
#   1 = native ~30 m (original)
#   3 = ~90 m  (9x fewer pixels, recommended for testing)
#  10 = ~300 m (100x fewer pixels, very fast)
DEM_DOWNSCALE_FACTOR = 3

# all_tiles = ["N03E009", "N03E012"]

all_tiles = [ "N00E006", "N00E009", "N00E012", "N00E015",
             "N03E006", "N03E009", "N03E012", "N03E015",
             "N06E006", "N06E009", "N06E012", "N06E015",
             "N09E006", "N09E009", "N09E012", "N09E015",
             "N12E006", "N12E009", "N12E012", "N12E015"]

for selected_tile in all_tiles:
    print(f"\n=== {selected_tile}  {datetime.datetime.now()} ===")

    # ---- DEM ----
    dem = rio.open_rasterio(
        f"../data/input/DEM/COP30/COP30_{selected_tile}.tif",
        masked=True, chunks={"x": 512, "y": 512})
    if len(dem.shape) == 3 and dem.shape[0] == 1:
        dem = dem.squeeze("band")

    # FIX: downscale DEM early so all subsequent rasters are computed at lower resolution.
    #      bilinear is appropriate for continuous elevation data.
    if DEM_DOWNSCALE_FACTOR > 1:
        dem = downscaling(dem, DEM_DOWNSCALE_FACTOR, resampling=Resampling.bilinear)
        print(f"  DEM downscaled → {dem.rio.shape}  (factor {DEM_DOWNSCALE_FACTOR})")

    # ---- slope / terrain cost ----
    dem_slope = computePercentageSlope(dem=dem)
    # dem_slope = xr.where(dem_slope >= 100, np.nan, dem_slope)  # removed: allow steep slopes (>=45°) at high cost

    dem_TI = xr.apply_ufunc(
        TI_index,
        dem_slope,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    dem_cost_slope = slopespeed(dem_slope, dem_TI)

    dem_cost = dem_cost_slope / dem_cost_slope.mean()
    dem_cost = 0.6 + dem_cost * 0.6
    dem_cost = dem_cost.rio.write_crs(4326)

    # ---- cost map parameters ----
    mode = 'driving'
    baseline_cost = 0.5  # 0.5 mUSD/km

    speed_map_OSM = readRoadSpeedMap(
        '../data/input/Functions/OSM_road_cost_' + mode + '-surface-based.csv',
        road='fclass', speed='cost_km')
    speed_map_lc = readLandcoverSpeedMap(
        '../data/input/Functions/land_cover_cost_' + mode + '.csv',
        landcover='Code', speed='cost_km')

    speed_map_OSM = baseline_cost * speed_map_OSM

    # ---- road network ----
    roads_all = gpd.read_file(
        '../data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg'
    )[['osm_id', 'highway', 'geometry', 'liu_surface']].rename(columns={'liu_surface': 'fclass'})

    # FIX: reproject roads to match raster CRS before rasterizing.
    #      features.rasterize uses raw coordinates — a CRS mismatch silently produces wrong results.
    print(f"  Roads input CRS: {roads_all.crs}")
    roads_all = roads_all.to_crs(epsg=4326)
    print(f"  Roads reprojected CRS: {roads_all.crs}")

    # Filter to roads with a recognised surface label.
    # Rows where liu_surface is NaN or not in the speed map produce 0 in rasterization,
    # which becomes NaN and falls through to the expensive landcover cost — wrong.
    valid_fclass = set(speed_map_OSM.index.tolist())
    n_before = len(roads_all)
    roads_all = roads_all[roads_all['fclass'].isin(valid_fclass)].copy()
    n_after = len(roads_all)
    print(f"  Roads after fclass filter: {n_after:,} / {n_before:,} "
          f"(dropped {n_before - n_after:,} rows without paved/unpaved label)")

    # ---- landcover ----
    landcover = rio.open_rasterio(
        "../data/input/Landcover/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E000/"
        "ESA_WorldCover_10m_2021_V200_" + selected_tile + "_Map.tif",
        masked=True, chunks={"x": 512, "y": 512})
    if len(landcover.shape) == 3 and landcover.shape[0] == 1:
        landcover = landcover.squeeze("band")

    landcover = clip_box(rio_orig=landcover, rio_clip=dem_cost)

    lc_codes, lc_costs = speed_map_lc
    lc_map = dict(zip(lc_codes.tolist(), lc_costs.tolist()))

    # Downscale the raw categorical landcover FIRST (uint8 = 8x smaller than float32),
    # then apply cost mapping on the already-small downscaled array.
    # rioxarray's reproject() forces .values (full numpy load) regardless of dask backing,
    # so doing the reproject on uint8 before converting to float32 avoids the OOM error.
    lc_downscale_factor = 3 * DEM_DOWNSCALE_FACTOR
    landcover_ds = downscaling(landcover, lc_downscale_factor, resampling=Resampling.max)
    print(f"  Landcover downscaled → {landcover_ds.rio.shape}  (factor {lc_downscale_factor})")

    lc_vals = landcover_ds.values  # small numpy array after downscaling
    out = np.full(lc_vals.shape, np.nan, dtype=np.float32)
    for code, cost in lc_map.items():
        out[lc_vals == code] = cost

    speedsurface_lc = landcover_ds.copy(data=out).astype(np.float32)
    speedsurface_lc = baseline_cost * speedsurface_lc

    # ---- rasterize roads ----
    # Roads are stored as 1/cost (inverted) so np.maximum in rasterizeAllRoadsMax
    # correctly picks the best (cheapest) road where segments overlap.
    # The inversion is undone in the xr.where step below.
    speed_map_OSM_reverse = 1 / speed_map_OSM
    speedsurface_OSM = rasterizeAllRoads(roads_all, speedsurface_lc, speed_map_OSM_reverse)
    speedsurface_OSM = speedsurface_OSM.rio.write_crs(4326)

    # DIAGNOSTIC: check how many road pixels were actually rasterized
    _road_vals = speedsurface_OSM.values
    _n_road_px = int(np.isfinite(_road_vals).sum() - np.isnan(_road_vals).sum())
    _n_road_px = int((~np.isnan(_road_vals)).sum())
    print(f"  Road pixels rasterized : {_n_road_px:,}  "
          f"(out of {_road_vals.size:,} total pixels, "
          f"{100*_n_road_px/_road_vals.size:.2f}%)")
    if _n_road_px > 0:
        _road_only = _road_vals[~np.isnan(_road_vals)]
        print(f"  speedsurface_OSM range : {_road_only.min():.4f} – {_road_only.max():.4f}  "
              f"(inverted costs; larger = cheaper road)")

    # ---- align DEM cost to road/landcover grid ----
    dem_cost_project = clip_box(rio_orig=dem_cost, rio_clip=speedsurface_OSM)
    dem_cost_project = dem_cost_project.rio.reproject_match(speedsurface_OSM)
    dem_cost_project['x'] = speedsurface_OSM['x']
    dem_cost_project['y'] = speedsurface_OSM['y']

    # ---- combine layers ----
    # Where roads exist → use road cost (terrain-independent: road is already built).
    # Elsewhere → use land cover cost scaled by slope (terrain matters for new construction).
    speedsurface_no_road = speedsurface_lc * dem_cost_project
    speedsurface = xr.where(speedsurface_OSM.notnull(), 1 / speedsurface_OSM, speedsurface_no_road)

    # DIAGNOSTIC: check final cost surface at road vs non-road pixels
    _cost_vals = speed_to_cost(speedsurface).values
    _road_mask = ~np.isnan(_road_vals)
    if _road_mask.any():
        print(f"  costsurface at road px : min={_cost_vals[_road_mask].min():.2f}  "
              f"max={_cost_vals[_road_mask].max():.2f}  "
              f"mean={_cost_vals[_road_mask].mean():.2f}")
    _nonroad_mask = np.isnan(_road_vals) & np.isfinite(_cost_vals)
    if _nonroad_mask.any():
        print(f"  costsurface non-road   : min={_cost_vals[_nonroad_mask].min():.2f}  "
              f"max={_cost_vals[_nonroad_mask].max():.2f}  "
              f"mean={_cost_vals[_nonroad_mask].mean():.2f}")

    # Convert mUSD/km → USD/pixel
    costsurface = speed_to_cost(speedsurface)
    costsurface = costsurface.rio.write_crs(4326)

    out_path = f"../data/output/cmr-construction-cost-friction-90m/construction_cost_friction_{selected_tile}_90m.tif"
    costsurface.compute().rio.to_raster(out_path)
    print(f"  Saved → {out_path}")
    print(f"  Done: {selected_tile}  {datetime.datetime.now()}")
