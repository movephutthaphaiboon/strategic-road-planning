import geopandas as gpd
import pandas as pd
import rioxarray as rio
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from fuzzywuzzy import process
from rasterio import features
import fiona
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
    """Function to calculate road construction cost """

    RW = 8
    def _model(x, y):

        a = 0.2 * y**(0.57) - 1
        b = 1.06 * y**(0.060)
        C = 747.37 * y**(0.23)
        d = 1.27 * y**(0.22) - 1

        EWV = C * (1) * RW**(a) * (1+ np.exp(b * (x/100 - d)))

        return EWV
    

    return xr.apply_ufunc(_model, slopes, TI)



def computeSlopeImpact(dem):
    """Function to calculate slope impact from DEM

    Parameters
    ----------
    dem: object holding digital elevation model

    Returns
    -------
    array of slope impact
    """

    slopes = computePercentageSlope(dem)

    # mask out slopes above 45 degree, ie 100%
    slopes = xr.where(slopes >= 100, np.nan, slopes)

    # take the mean speed for both upwards and downwards slope.
    # relative to going along the flat (0 slope)

    slopes = 0.5 * (slopespeed(slopes) + slopespeed(-slopes)) / slopespeed(0)

    return slopes

def readLandcoverSpeedMap(fname, landcover='Code',
                          speed='cost_km',
                          dropNaN=True, scale=None):
    """construct landcover to speed map
    Parameters
    ----------
    fname: name of csv file containing landcover to speed map
           the first row is assumed to contain column names
    landcover: the name of the column containing the landcover types
           default: 'Code'
    speed: the name of the column containing the speeds
           default: 'Walking Speed (km/h)'
    dropNaN: whether landtypes with NaN values should be dropped
           default: True
    scale: when set speed values will be scaled and turned into integers

    Returns
    -------
    tuple of two arrays containing the landcover type and the associated speed
    """
    costs = pd.read_csv(fname)
    if dropNaN:
        costs.dropna(inplace=True)
    lc = np.array(costs[landcover])
    s = np.array(costs[speed])
    if scale:
        s = (np.round(s * scale)).astype(np.int64)
    return (lc, s)


def applyLandcoverSpeedMap(landcover, speedmap):
    """convert a landcover surface to a speed surface using a map

    Parameters
    ----------
    landcover:  a 2D xarray containg the landcover
    map: a tuple with two arrays containing the landcover type and
         associated speed

    Returns
    -------
    an xarray containing the speed surface
    """

    landcover_types, speed_values = speedmap

    speedsurface = xr.zeros_like(landcover, dtype=np.float32)
    speedsurface.values[:] = np.nan

    # consider only pixels with interesting data
    mask = np.in1d(landcover.values, landcover_types)
    # create index into landcover_types
    idx = np.searchsorted(np.sort(landcover_types), landcover.values.ravel()[mask]) # sort landcover types - KM
    # assign speed values
    speedsurface.values.ravel()[mask] = speed_values[idx]

    return speedsurface

def readRoadSpeedMap(fname, road='fclass',
                     speed='cost_km',
                     dropNaN=True):
    """construct road type to speed map

    Parameters
    ----------
    fname: name of csv file containing road to speed map
           the first row is assumed to contain column names
    road: the name of the column containing the road types
           default: 'Feature_Class'
    speed: the name of the column containing the speeds
           default: 'Walking_Speed'
    dropNaN: whether road with NaN values should be dropped
           default: True

    Returns
    -------
    a pandas series containing speeds
    """

    costs = pd.read_csv(fname, index_col=road)
    if dropNaN:
        costs = costs[costs[speed].notna()]
    return costs[speed]

def rasterizeRoads(roads, landcover, road_speed_map):
    """rasterize roads

    Parameters
    ----------
    roads: roads vector layer
    landcover: xarry used for creating empty array
    road_speed_map: dictionary mapping road type to travel speed

    Returns
    -------
    an xarray containing the speed surface
    """

    # construct filter selecting all roads of a particular type
    #filtered = filter(lambda f: f['properties']['tag'] in road_speed_map,
                      #roads)
    
    filtered = filter(lambda row: row.fclass in road_speed_map,
                      (row for _, row in roads_all.iterrows()))

    speedsurface = features.rasterize(
        ((f['geometry'],
          road_speed_map[f['fclass']]) for f in filtered),
        out_shape=landcover.rio.shape,
        transform=landcover.rio.transform(),
        all_touched=True)

    return speedsurface


def rasterizeAllRoads(roads, landcover, road_speed_map, maxspeed=True):
    """rasterize all roads

    Parameters
    ----------
    roads: roads vector layer
    landcover: xarry used for creating empty array
    road_speed_map: pandas series containing speeds
    maxspeed: when set to False road types are not ordered and slower
              road speeds might override faster speeds

    Returns
    -------
    an xarray containing the speed surface
    """

    # extract road types from road shapefile
    road_types = roads['fclass'].unique()

    # modify index so that it matches the road types in the vector layer
    idx = road_speed_map.index.to_list()
    for i, rt in enumerate(idx):
        matched_rt = process.extractOne(rt, road_types)[0]
        print(f'using matched road type {matched_rt} for {rt}')
        idx[i] = matched_rt
    road_speed_map.index = idx

    if maxspeed:
        rcost = rasterizeAllRoadsMax(roads, landcover, road_speed_map)
    else:
        rcost = rasterizeRoads(roads, landcover, road_speed_map.to_dict())

    # replace fill values with nans
    rcost = np.where(rcost == 0, np.nan, rcost)

    speedsurface = xr.zeros_like(landcover, dtype=np.float32)
    speedsurface.values[:, :] = rcost[:, :]

    return speedsurface


def rasterizeAllRoadsMax(roads, landcover, road_speed_map):
    """rasterize all roads

    This version takes the largest speed when a pixel contains
    multiple roads

    Parameters
    ----------
    roads: roads vector layer
    landcover: xarry used for creating empty array
    road_speed_map: dictionary mapping road type to travel speed

    Returns
    -------
    a numpy arrray containing the speed surface
    """

    speedsurface = np.zeros(landcover.shape[1:], dtype=np.float32)

    # loop over unique speed values to group roads by speed
    for speed in road_speed_map.unique():
        # select all roads with that speed
        selected_roads = road_speed_map[road_speed_map == speed]
        rcost = rasterizeRoads(roads, landcover, selected_roads.to_dict())
        
        # take minimum values
        speedsurface = np.maximum(speedsurface, rcost)

    return speedsurface

def speed_to_cost(speed):
    """
    convert speed surface to cost surface

    Parameters
    ----------
    speed: object containing the speed surface
          in km/h


    Return
    ------
    cost surface
    """

    # apply child impact factor and convert to m/s
    
    # compute the costsurface, ie time.
    # the factor 111120 converts degree to m close to the equator
    return abs(speed.rio.resolution()[0]) * 111120 * speed*1e6/(1000)


def clip_box(rio_orig, rio_clip):
    return rio_orig.rio.clip_box(minx=rio_clip.x.min().values,
                                        miny=rio_clip.y.min().values,
                                        maxx=rio_clip.x.max().values,
                                        maxy=rio_clip.y.max().values,)

def downscaling(rio_orig, downscale_factor):
    new_width = int(round(rio_orig.rio.width / downscale_factor))
    new_height = int(round(rio_orig.rio.height / downscale_factor))

    # Downscale the data
    return rio_orig.rio.reproject(
        dst_crs=rio_orig.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.max,)

def TI_index(x):
    if x/100 <= 0.05:
        TI = 0 + 40 * x/100
    elif x/100 > 0.05 and x/100 <=0.1:
        TI = 2 + 40 * ((x/100-0.05))
    elif x/100 > 0.1 and x/100 <=0.3:
        TI = 4 + 10 * ((x/100-0.1))
    elif x/100 > 0.3 and x/100 <= 0.65:
        TI = 6 + 5.71 * ((x/100 - 0.3))
    elif x/100 > 0.65:
        TI = 8 + 5.71 * ((x/100 - 0.65))
    else:
        TI = np.nan
        
    return TI

# ====================================================================

# ### test the process for one tile

# ### read DEM ###
# selected_tile = "N00E015"
# dem = rio.open_rasterio(f"../data/input/DEM/COP30/COP30_{selected_tile}.tif", masked=True)[0,:,:]

# ### compute slope ###
# dem_slope = computePercentageSlope(dem = dem)

# # mask out slopes above 45 degree, ie 100%
# dem_slope = xr.where(dem_slope >= 100, np.nan, dem_slope)

# ### apply speed function ###
# dem_TI = xr.apply_ufunc(TI_index, dem_slope, vectorize=True)

# dem_cost_slope = slopespeed(dem_slope, dem_TI)

# ### scale costs ##
# dem_cost = dem_cost_slope/dem_cost_slope.mean()
# dem_cost = 0.6 + dem_cost*0.6
# dem_cost = dem_cost.rio.write_crs(4326)

# ### 
# mode = 'driving'

# baseline_cost = 0.5 ### 0.5 mUSD/km

# ### read speed conversation for OSM and landcover ##
# speed_map_OSM = readRoadSpeedMap('../data/input/Functions/OSM_road_cost_'+mode+'.csv',road='fclass',speed='cost_km')
# speed_map_lc = readLandcoverSpeedMap('../data/input/Functions/land_cover_cost_'+mode+'.csv',landcover='Code',speed='cost_km')

# ## convert ##
# speed_map_OSM = baseline_cost * speed_map_OSM

# ### read Road_network ###
# roads_all = gpd.read_file('../data/input/Road_network/hotosm_cmr_roads_lines_shp/hotosm_cmr_roads_lines_shp.shp')[['osm_id','highway','geometry']].rename(columns = {'highway':'fclass'})

# =================================================================

### run the whole process for all tiles ###

# all_tiles = [ #"N00E006", "N00E009", "N00E012", "N00E015",
#              "N03E006", "N03E009", "N03E012", "N03E015",
#              "N06E006", "N06E009", "N06E012", "N06E015",
#              "N09E006", "N09E009", "N09E012", "N09E015",
#              "N12E006", "N12E009", "N12E012", "N12E015"]

all_tiles = ["N09E012"]

for selected_tile in all_tiles:
    print(selected_tile, datetime.datetime.now())
    ### read DEM ###
    dem = rio.open_rasterio(f"../data/input/DEM/COP30/COP30_{selected_tile}.tif", masked=True)[0,:,:]

    ### compute slope ###
    dem_slope = computePercentageSlope(dem = dem)

    # mask out slopes above 45 degree, ie 100%
    dem_slope = xr.where(dem_slope >= 100, np.nan, dem_slope)

    ### apply speed function ###
    dem_TI = xr.apply_ufunc(TI_index, dem_slope, vectorize=True)

    dem_cost_slope = slopespeed(dem_slope, dem_TI)

    ### scale costs ##
    dem_cost = dem_cost_slope/dem_cost_slope.mean()
    dem_cost = 0.6 + dem_cost*0.6
    dem_cost = dem_cost.rio.write_crs(4326)

    ### 
    mode = 'driving'

    baseline_cost = 0.5 ### 0.5 mUSD/km

    ### read speed conversation for OSM and landcover ##
    speed_map_OSM = readRoadSpeedMap('../data/input/Functions/OSM_road_cost_'+mode+'.csv',road='fclass',speed='cost_km')
    speed_map_lc = readLandcoverSpeedMap('../data/input/Functions/land_cover_cost_'+mode+'.csv',landcover='Code',speed='cost_km')

    ## convert ##
    speed_map_OSM = baseline_cost * speed_map_OSM

    ### read Road_network ###
    roads_all = gpd.read_file('../data/input/Road_network/hotosm_cmr_roads_lines_shp/hotosm_cmr_roads_lines_shp.shp')[['osm_id','highway','geometry']].rename(columns = {'highway':'fclass'})

    ### read landcover ###
    landcover = rio.open_rasterio("../data/input/Landcover/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E000/ESA_WorldCover_10m_2021_V200_"+selected_tile+"_Map.tif", masked=True)[0,:,:]
    
    ### clip to DEM ###
    landcover = clip_box(rio_orig = landcover, rio_clip = dem_cost)

    ### apply cost factor to land cover ###
    # Memory-safe landcover -> cost mapping (chunked)
    lc_codes, lc_costs = speed_map_lc
    lc_map = dict(zip(lc_codes.tolist(), lc_costs.tolist()))

    speedsurface_lc = xr.full_like(landcover, np.nan, dtype=np.float32)

    chunk_rows = 1024
    for r0 in range(0, landcover.shape[0], chunk_rows):
        r1 = min(r0 + chunk_rows, landcover.shape[0])

        block = np.asarray(landcover.values[r0:r1, :])
        block_out = np.full(block.shape, np.nan, dtype=np.float32)

        for code, cost in lc_map.items():
            block_out[block == code] = cost

        speedsurface_lc.values[r0:r1, :] = block_out
    # BEFORE CHUNKING => MemoryError: Unable to allocate 9.65 GiB for an array with shape (1295784009,) and data type float64

    ### upscale from 10 to 30m ####
    speedsurface_lc = downscaling(rio_orig = speedsurface_lc, downscale_factor = 3)
    
    ### scale costs ##
    speedsurface_lc = baseline_cost * speedsurface_lc

    ### run OSM rastize process ##
    speed_map_OSM_reverse = 1/speed_map_OSM
    speedsurface_OSM = rasterizeAllRoads(roads_all, speedsurface_lc, speed_map_OSM_reverse)
    speedsurface_OSM = speedsurface_OSM.rio.write_crs(4326) ### crs


    ## DEM clip ##
    dem_cost_project = clip_box(rio_orig = dem_cost, rio_clip = speedsurface_OSM)

    ### reproject ##
    dem_cost_project = dem_cost_project.rio.reproject_match(speedsurface_OSM)
    dem_cost_project['x'], dem_cost_project['y'] = speedsurface_OSM['x'], speedsurface_OSM['y']


    ### combine road and landcover ##
    speedsurface = xr.where(speedsurface_OSM.notnull(), 1/speedsurface_OSM, speedsurface_lc)

    ### make speed ##
    costsurface = speed_to_cost(speedsurface * dem_cost_project)
    #costsurface = xr.where(costsurface == 0, np.nan, costsurface)
    costsurface = costsurface.rio.write_crs(4326)
    costsurface.rio.to_raster("../data/output/construction_cost_friction_"+selected_tile+'.tif')
    
