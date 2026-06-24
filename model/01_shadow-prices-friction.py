"""
Shadow-price friction layers for Cameroon road planning experiments.

Adds carbon and biodiversity damage costs (shadow prices) to the
construction-cost friction raster. Produces 7 output layers:

  1. construction_only              — baseline construction cost only
  2. carbon_low                     — + carbon damage at $5/tCO₂e (VCM)
  3. carbon_central                 — + carbon damage at $40/tCO₂e (UN-REDD+ cost, Paris floor by 2030)
  4. carbon_high                    — + carbon damage at $90/tCO₂e (EU ETS 2023 annual average)
  5. biodiversity                   — + biodiversity damage only (no carbon price)
  6. carbon_low__biodiversity       — layer 2 + biodiversity damage
  7. carbon_central__biodiversity   — layer 3 + biodiversity damage
  8. carbon_high__biodiversity      — layer 4 + biodiversity damage
  9. carbon_300                     — + carbon damage at $300/tCO₂e (EIB shadow price ~€250/t by 2030)
 10. carbon_930                     — + carbon damage at $930/tCO₂e (EIB shadow price ~€800/t by 2050)
 11. carbon_1200                    — + carbon damage at $1,200/tCO₂e (extreme stress-test)
 12. carbon_1400                    — + carbon damage at $1,400/tCO₂e (extreme stress-test)

Shadow prices are zeroed on existing road pixels: the road network is rasterized
and used as a mask so that pixels where infrastructure already exists (forest already
cleared) receive no carbon or biodiversity penalty. This avoids penalizing route
reuse of existing roads through the BII landscape signal (9 km resolution cannot
distinguish road pixels from surrounding habitat).

Per-pixel damage formulas (USD/pixel):

  carbon_damage    = pixel_area_ha × (treecover / 100) × CARBON_DENSITY × carbon_price
  biodiversity_dmg = pixel_area_ha × BII × BIODIVERSITY_BASE

Carbon density reference (550 tCO₂e/ha):
  Saatchi et al. (2011) PNAS 108(24), Table 1 — Cameroon carbon density by canopy threshold:
    10% threshold: 129 Mg C/ha → 129 × 3.667 = 473 tCO₂e/ha
    25% threshold: 142 Mg C/ha → 142 × 3.667 = 521 tCO₂e/ha
    30% threshold: 151 Mg C/ha → 151 × 3.667 = 554 tCO₂e/ha
  Cameroon mean treecover ~70% → denser forest → 30% canopy threshold most representative → 554 tCO₂e/ha.
  IPCC AFOLU Table 4.7 (tropical rain forest, Africa): 310 tDM/ha × 0.47 (carbon fraction, Table 2.2)
    × 3.667 (44/12, C→CO₂) = 534 tCO₂e/ha. Both sources converge around 534–554 tCO₂e/ha.
  550 tCO₂e/ha adopted as a round figure consistent with both references.

Carbon price references:
  Low  ($5)  — Voluntary Carbon Market (VCM) spot price range
  Central ($40) — midpoint of $30–50/tCO₂ high-quality REDD+ implementation cost; minimum required
                  by 2030 to meet Paris Agreement goals (UN-REDD Programme 2022)
  High ($90) — EU ETS annual average 2023 (relevant as EU is primary funder of Cameroon infrastructure)
  Very high ($300)  — EIB institutional shadow carbon price of €250/tCO₂ by 2030 (≈ $300 USD at ~1.16
                      EUR/USD), applied to all EIB-financed infrastructure project appraisals. Also
                      consistent with IEA Net Zero by 2050 scenario advanced-economy price of
                      $250/tCO₂ by 2030 and EPA Social Cost of Carbon (2023 final report, 2% discount
                      rate): $190/tCO₂.
  Extreme ($930)    — EIB institutional shadow carbon price of €800/tCO₂ by 2050 (≈ $930 USD at ~1.16
                      EUR/USD). Extreme upper-bound stress-test tier.

Biodiversity base reference:
  de Groot et al. (2012) Ecosystem Services 1:50–61 — precise tropical forest value: $5,264/ha/year.
  Road construction causes permanent forest loss, so the annual flow is capitalized to NPV:
    BIODIVERSITY_BASE = 5264 × (1 - (1 + 0.05)^-30) / 0.05 ≈ $80,920/ha
  Discount rate 5% — consistent with World Bank/AfDB infrastructure appraisal for African projects.
  Time horizon 30 years — standard road infrastructure lifetime.
  Scaled by BII (NHM v2.1.1, 2020) so degraded pixels receive proportionally less weight.

Intermediate files written to output dir (prefix _):
  _treecover_90m.tif  — Hansen treecover resampled to 90m, 0–1 float32
  _bii_90m.tif        — NHM BII 2020 resampled to 90m, 0–1 float32
"""

import numpy as np
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CARBON_DENSITY        = 550   # tCO₂e/ha — Saatchi et al. (2011) Table 1, Cameroon 30% canopy threshold
CARBON_PRICE_LOW      =   5   # USD/tCO₂e — voluntary carbon market
CARBON_PRICE_CENTRAL  =  40   # USD/tCO₂e — UN-REDD high-quality REDD+ cost / Paris floor by 2030
CARBON_PRICE_HIGH     =  90   # USD/tCO₂e — EU ETS annual average 2023
CARBON_PRICE_VERY_HIGH  =  300  # USD/tCO₂e — EIB shadow price €250/t by 2030 (≈ $300)
CARBON_PRICE_EXTREME    =  930  # USD/tCO₂e — EIB shadow price €800/t by 2050 (≈ $930)
CARBON_PRICE_EXTREME2   = 1200  # USD/tCO₂e — extreme stress-test tier
CARBON_PRICE_EXTREME3   = 1400  # USD/tCO₂e — extreme stress-test tier

# Biodiversity: annual value capitalized to NPV for permanent forest loss
ANNUAL_BIO_VALUE  = 5264   # USD/ha/year — de Groot et al. (2012), tropical forest
DISCOUNT_RATE     = 0.05   # 5% — World Bank/AfDB infrastructure appraisal rate
TIME_HORIZON      = 30     # years — standard road infrastructure lifetime
BIODIVERSITY_BASE = ANNUAL_BIO_VALUE * (1 - (1 + DISCOUNT_RATE) ** -TIME_HORIZON) / DISCOUNT_RATE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]

FRICTION_PATH  = BASE / "data/output/cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif"
TREECOVER_PATH = BASE / "data/output/processed-hansen-treecover/cameroon_treecover2024_30m.tif"
BII_PATH       = BASE / "data/input/biodiversity-intactness-index-nhm/bii-2020_v2-1-1.tif"
ROADS_PATH     = BASE / "data/output/processed_roads_official/cleaned_merged_osm_heigit_liu-ver3-midterm.gpkg"

OUT_DIR = BASE / "data/output/cmr-cost-friction-90m-for-experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TC_90M_PATH  = OUT_DIR / "_treecover_90m.tif"
BII_90M_PATH = OUT_DIR / "_bii_90m.tif"

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------
LAYERS = [
    {"name": "construction_only",            "carbon_price": None,                 "include_bio": False},
    {"name": "carbon_low",                   "carbon_price": CARBON_PRICE_LOW,     "include_bio": False},
    {"name": "carbon_central",               "carbon_price": CARBON_PRICE_CENTRAL, "include_bio": False},
    {"name": "carbon_high",                  "carbon_price": CARBON_PRICE_HIGH,    "include_bio": False},
    {"name": "biodiversity",                 "carbon_price": None,                 "include_bio": True},
    {"name": "carbon_low__biodiversity",     "carbon_price": CARBON_PRICE_LOW,     "include_bio": True},
    {"name": "carbon_central__biodiversity", "carbon_price": CARBON_PRICE_CENTRAL, "include_bio": True},
    {"name": "carbon_high__biodiversity",    "carbon_price": CARBON_PRICE_HIGH,    "include_bio": True},
    # Stress-test tiers: EIB shadow carbon prices and extreme sensitivity
    {"name": "carbon_300",  "carbon_price": CARBON_PRICE_VERY_HIGH, "include_bio": False},
    {"name": "carbon_930",  "carbon_price": CARBON_PRICE_EXTREME,   "include_bio": False},
    {"name": "carbon_1200", "carbon_price": CARBON_PRICE_EXTREME2,  "include_bio": False},
    {"name": "carbon_1400", "carbon_price": CARBON_PRICE_EXTREME3,  "include_bio": False},
]

# ---------------------------------------------------------------------------
# Reference grid from construction cost raster
# ---------------------------------------------------------------------------
with rasterio.open(FRICTION_PATH) as src:
    ref_profile   = src.profile.copy()
    ref_transform = src.transform
    ref_crs       = src.crs
    ref_shape     = (src.height, src.width)

# ---------------------------------------------------------------------------
# Existing road mask — shadow prices are zero on road pixels
# ---------------------------------------------------------------------------
# Road construction cost script (01_construction-cost-friction.py) assigns road
# pixels a low terrain-independent cost via speedsurface_OSM. Shadow prices
# represent NEW environmental damage, so pixels where infrastructure already
# exists (forest already cleared) must receive zero penalty. The BII raster
# at ~9 km cannot distinguish road pixels from surrounding habitat, making this
# mask essential for the biodiversity component.
print("Building existing road mask...")
roads_gdf = gpd.read_file(ROADS_PATH, columns=["geometry"]).to_crs(ref_crs)
road_shapes = [(geom, 1) for geom in roads_gdf.geometry if geom is not None]
existing_road = rio_rasterize(
    road_shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0,
    dtype=np.uint8,
    all_touched=True,
).astype(bool)   # True = existing road pixel → shadow price = 0
print(f"  Road pixels: {existing_road.sum():,} of {existing_road.size:,} "
      f"({100 * existing_road.sum() / existing_road.size:.2f}%)")

# ---------------------------------------------------------------------------
# Step 1: Resample input rasters to the 90m reference grid
# ---------------------------------------------------------------------------
def resample_to_ref(src_path: Path, out_path: Path, resampling: Resampling,
                    divide_by: float = 1.0) -> None:
    """Reproject src_path to the reference 90m grid, normalize, and save."""
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping.")
        return

    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", nodata=np.nan, count=1,
                       compress="lzw", tiled=True, blockxsize=512, blockysize=512)

    dest = np.full(ref_shape, np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    dest /= divide_by
    np.clip(dest, 0.0, 1.0, out=dest)   # guard against interpolation overshoot

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(dest, 1)

    print(f"  Written: {out_path.name}")


print("Step 1a: Resampling treecover 30m → 90m (average)...")
resample_to_ref(TREECOVER_PATH, TC_90M_PATH, Resampling.average, divide_by=100.0)

print("Step 1b: Resampling BII 9km → 90m (bilinear)...")
resample_to_ref(BII_PATH, BII_90M_PATH, Resampling.bilinear, divide_by=100.0)

# ---------------------------------------------------------------------------
# Step 2: Load all rasters into memory
# ---------------------------------------------------------------------------
print("\nStep 2: Loading rasters...")

with rasterio.open(FRICTION_PATH) as src:
    friction = src.read(1).astype(np.float32)

with rasterio.open(TC_90M_PATH) as src:
    treecover = src.read(1)   # 0–1 float32

with rasterio.open(BII_90M_PATH) as src:
    bii = src.read(1)         # 0–1 float32

nodata_mask = np.isnan(friction)

# Treat NaN in auxiliary layers as 0 (no damage where data is absent)
tc  = np.where(np.isnan(treecover), 0.0, treecover)
bii = np.where(np.isnan(bii),       0.0, bii)

# ---------------------------------------------------------------------------
# Pixel area in hectares — varies with latitude (WGS84)
# ---------------------------------------------------------------------------
res_x_deg = abs(ref_transform.a)
res_y_deg = abs(ref_transform.e)
row_lats  = ref_transform.f + (np.arange(ref_shape[0]) + 0.5) * ref_transform.e
pixel_area_ha = (
    res_x_deg * 111_320 * np.cos(np.deg2rad(row_lats))   # width in m
    * res_y_deg * 110_540                                  # height in m
    / 10_000                                               # m² → ha
)
pixel_area_2d = pixel_area_ha[:, np.newaxis]               # broadcast over columns

# ---------------------------------------------------------------------------
# Precompute damage components (USD/pixel, arrays over full raster)
# ---------------------------------------------------------------------------
# carbon_base × carbon_price = carbon damage per pixel
carbon_base = (pixel_area_2d * CARBON_DENSITY * tc).astype(np.float32)

# biodiversity damage is fixed across carbon tiers
bio_damage  = (pixel_area_2d * BIODIVERSITY_BASE * bii).astype(np.float32)

# ---------------------------------------------------------------------------
# Step 3: Write output layers
# ---------------------------------------------------------------------------
out_profile = ref_profile.copy()
out_profile.update(dtype="float32", nodata=np.nan,
                   compress="lzw", tiled=True, blockxsize=512, blockysize=512)

print("\nStep 3: Writing output layers...")

valid = ~nodata_mask

for layer in LAYERS:
    out_path = OUT_DIR / f"friction__{layer['name']}.tif"

    if layer["carbon_price"] is None:
        shadow = np.zeros(ref_shape, dtype=np.float32)
    else:
        shadow = carbon_base * layer["carbon_price"]

    if layer["include_bio"]:
        shadow = shadow + bio_damage

    shadow[existing_road] = 0.0   # no new damage where road already exists

    result = (friction + shadow).astype(np.float32)
    result[nodata_mask] = np.nan

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(result, 1)

    mean_shadow   = float(shadow[valid].mean())
    mean_friction = float(friction[valid].mean())
    print(f"  friction__{layer['name']}.tif"
          f"  (mean shadow ${mean_shadow:,.0f}/px, mean total ${mean_friction + mean_shadow:,.0f}/px)")

print(f"\nDone. {len(LAYERS)} layers saved to:\n  {OUT_DIR}")
