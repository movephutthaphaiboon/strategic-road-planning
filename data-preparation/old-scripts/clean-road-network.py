import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# --- Load ---
filepath = "../data/output/processed_roads_official/"
gdf = gpd.read_file(filepath + "merged_osm_heigit_liu.gpkg")

# --- Step 1: Deduplicate by osm_id (prefer HeiGit source) ---
# Assign source priority: Liu=0 (highest for surface), HeiGit=1, OSM=2
def get_source(row):
    if pd.notna(row.get('pred_class')):   return 'liu'
    if pd.notna(row.get('osm_surface_class')): return 'heigit'
    return 'osm'

gdf['_source'] = gdf.apply(get_source, axis=1)
src_rank = {'liu': 0, 'heigit': 1, 'osm': 2}
gdf['_rank'] = gdf['_source'].map(src_rank)

# Keep best-ranked row per osm_id
gdf = (gdf[gdf['osm_id'].notna()]
       .sort_values('_rank')
       .drop_duplicates(subset='osm_id', keep='first')
       .pipe(lambda d: pd.concat([d, gdf[gdf['osm_id'].isna()]])))

# --- Step 2: Deduplicate by exact geometry ---
gdf = gdf.to_crs(epsg=32632)   # UTM zone 32N – metric CRS for Cameroon
gdf['_geom_wkt'] = gdf.geometry.apply(lambda g: g.wkt)
gdf = (gdf.sort_values('_rank')
          .drop_duplicates(subset='_geom_wkt', keep='first')
          .drop(columns='_geom_wkt'))

# --- Step 3: Buffer-based near-duplicate detection (THE KEY STEP) ---
BUFFER_M = 8          # 8-metre buffer half-width
OVERLAP_THRESH = 0.6  # 60% IoU = near-duplicate

gdf = gdf.reset_index(drop=True)
buffered = gdf.geometry.buffer(BUFFER_M)

# Spatial index for efficient pair finding
sindex = gdf.sindex
to_drop = set()

for i, buf_i in enumerate(buffered):
    if i in to_drop:
        continue
    candidates = list(sindex.intersection(buf_i.bounds))
    for j in candidates:
        if j <= i or j in to_drop:
            continue
        buf_j = buffered.iloc[j]
        intersection = buf_i.intersection(buf_j).area
        union = buf_i.union(buf_j).area
        if union == 0:
            continue
        iou = intersection / union
        if iou >= OVERLAP_THRESH:
            # Keep the higher-priority source
            drop_idx = j if gdf.iloc[i]['_rank'] <= gdf.iloc[j]['_rank'] else i
            to_drop.add(drop_idx)

gdf = gdf.drop(index=list(to_drop)).reset_index(drop=True)

# --- Step 4: Resolve paved/unpaved conflicts using source priority ---
def resolve_surface(row):
    # liu_surface takes top priority
    if pd.notna(row.get('liu_surface')) and row['liu_surface'] in ('paved', 'unpaved'):
        return row['liu_surface']
    # fall back to HeiGit combined surface
    heigit_surf = row.get('combined_surface_DL_priority') or row.get('combined_surface_osm_priority')
    if pd.notna(heigit_surf):
        return 'paved' if 'pav' in str(heigit_surf).lower() else 'unpaved'
    # fall back to OSM surface
    osm_surf = row.get('surface', '')
    paved_types = {'asphalt', 'concrete', 'paved', 'sett', 'cobblestone'}
    if str(osm_surf).lower() in paved_types:
        return 'paved'
    return 'unpaved'

gdf['final_surface'] = gdf.apply(resolve_surface, axis=1)

# --- Export ---
gdf.drop(columns=['_source', '_rank']).to_file(filepath + "clean_roads.gpkg", driver="GPKG")
print(f"Clean dataset: {len(gdf)} road segments")