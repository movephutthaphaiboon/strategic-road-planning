"""
Aggregate all per-scenario assessment CSVs into one combined summary file and
append PA-spatial deforestation columns computed from the TIF rasters.

Run once (or re-run after new impact assessment results):
    python 05_impact-assessment-result-summary.py

Output:
    model/results/impact-assessment-summary/impact_assessment_combined.csv
    - All columns from the individual *__assessment.csv files
    - Extra metadata columns: scenario, mining, friction, mask
    - Extra spatial columns:
        defor_inside_pa_ha        total deforestation (direct + indirect) inside PA core
        defor_outside_pa_ha       total deforestation outside PA core
        defor_direct_inside_pa_ha road-footprint deforestation inside PA core
          (action-road pixels, values 2+3 in classified.tif, clipped to PA mask)
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject, Resampling

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
RESULTS_DIR  = BASE / 'results/impact-assessment/impact_assessment_final'
PA_CORE_FP   = BASE / '../data/output/processed-protected-areas/with-buffer/cmr-protected-areas__buffer0km.gpkg'
OUT_DIR      = BASE / 'results/impact-assessment-summary'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_stem(stem: str) -> tuple[str, str, str]:
    """Return (mining, friction, mask) from a scenario stem."""
    mining, rest = stem.split('__kribi__')
    rest   = rest.replace('__ds1', '')
    tokens = rest.split('__')
    return mining, '__'.join(tokens[:-1]), tokens[-1]


# ── Step 1: Load and combine all assessment CSVs ──────────────────────────────
csv_fps = sorted(RESULTS_DIR.glob('*__assessment.csv'))
print(f'Found {len(csv_fps)} assessment CSVs ...')

dfs = []
for fp in csv_fps:
    stem = fp.stem.replace('__assessment', '')
    mining, friction, mask = parse_stem(stem)
    df = pd.read_csv(fp)
    df['scenario'] = stem
    df['mining']   = mining
    df['friction'] = friction
    df['mask']     = mask
    # Move metadata columns to the front
    meta = ['scenario', 'mining', 'friction', 'mask']
    df   = df[meta + [c for c in df.columns if c not in meta]]
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f'  Combined: {len(combined)} rows x {combined.shape[1]} columns')


# ── Step 2: Compute deforestation inside / outside PA core ────────────────────
# Rasterizes the PA core boundary (buffer 0 km) onto each defor_loss_pct TIF
# grid. The PA mask is cached after the first TIF (all share the same grid).

print('\nLoading PA core boundary ...')
pa_core = gpd.read_file(PA_CORE_FP)
print(f'  {len(pa_core)} polygons, CRS: {pa_core.crs}')

_pa_mask_cache: dict = {}

def get_pa_mask(shape: tuple, transform, crs_epsg: int) -> np.ndarray:
    key = (shape, transform)
    if key not in _pa_mask_cache:
        pa_proj    = pa_core.to_crs(crs_epsg)
        geom_pairs = [(geom, 1) for geom in pa_proj.geometry if geom is not None]
        _pa_mask_cache[key] = rio_rasterize(
            geom_pairs, out_shape=shape, transform=transform, fill=0, dtype=np.uint8
        )
    return _pa_mask_cache[key]


tif_fps = sorted(RESULTS_DIR.glob('*__defor_loss_pct.tif'))
print(f'Processing {len(tif_fps)} defor_loss_pct + classified TIF pairs ...')

pa_records = []
for tif_fp in tif_fps:
    stem = tif_fp.stem.replace('__defor_loss_pct', '')
    mining, friction, mask = parse_stem(stem)

    clf_fp = RESULTS_DIR / f'{stem}__classified.tif'

    with rasterio.open(tif_fp) as src:
        arr      = src.read(1).astype(np.float32)
        fp_shape = (src.height, src.width)
        fp_xform = src.transform
        fp_epsg  = src.crs.to_epsg()
        cx   = abs(fp_xform.a)
        cy   = abs(fp_xform.e)
        lat0 = (src.bounds.bottom + src.bounds.top) / 2
        px_ha = (cx * 111320 * math.cos(math.radians(lat0))) * (cy * 111320) / 10_000

    arr_frac = np.nan_to_num(arr, nan=0.0) / 100.0  # 0-100 pct → 0-1 fraction
    pa_mask  = get_pa_mask(fp_shape, fp_xform, fp_epsg)

    # Direct deforestation: action-road pixels only (upgrade=2, new-build=3)
    # Resample classified TIF to match defor_loss_pct grid (nearest-neighbour).
    with rasterio.open(clf_fp) as src:
        clf_resampled = np.zeros(fp_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=clf_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=fp_xform,
            dst_crs=rasterio.CRS.from_epsg(fp_epsg),
            resampling=Resampling.nearest,
        )
    action_mask = np.isin(clf_resampled, [2, 3])

    pa_records.append({
        'scenario':                stem,
        'defor_inside_pa_ha':      float(arr_frac[pa_mask == 1].sum()) * px_ha,
        'defor_outside_pa_ha':     float(arr_frac[pa_mask == 0].sum()) * px_ha,
        'defor_direct_inside_pa_ha': float(arr_frac[action_mask & (pa_mask == 1)].sum()) * px_ha,
    })
    print(f'  {stem}: '
          f'inside={pa_records[-1]["defor_inside_pa_ha"]:.1f} ha  '
          f'direct_inside={pa_records[-1]["defor_direct_inside_pa_ha"]:.2f} ha')

pa_df = pd.DataFrame(pa_records)


# ── Step 3: Merge and save ─────────────────────────────────────────────────────
combined = combined.merge(pa_df, on='scenario', how='left')
combined = combined.sort_values(['mining', 'friction', 'mask']).reset_index(drop=True)

out_fp = OUT_DIR / 'impact_assessment_combined.csv'
combined.to_csv(out_fp, index=False)
print(f'\nSaved {len(combined)} rows x {combined.shape[1]} columns → {out_fp}')
print(combined[['scenario', 'mining', 'friction', 'mask',
                'construction_cost_usd', 'defor_action_ha',
                'defor_inside_pa_ha', 'defor_outside_pa_ha',
                'defor_direct_inside_pa_ha']].head(4).to_string())
