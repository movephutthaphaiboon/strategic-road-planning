#!/usr/bin/env python3
"""
per-mine-outcomes.py
====================
Creates per-mine breakdowns of road investment and deforestation for the
experiment_04_full120runs impact assessment results.

Outputs (to model/results/per-mine-summary/):
    per_mine_action_needed.csv    - upgrade / build / no_action / out_of_scope / inaccessible
    per_mine_length_km.csv        - action-road km per mine
    per_mine_cost_usd.csv         - construction cost USD per mine
    per_mine_defor_direct_ha.csv  - road-footprint deforestation per mine
    per_mine_defor_total_ha.csv   - total deforestation (direct + Damania indirect) per mine

Deforestation methodology mirrors impact_assessment.py:
    For each mine's road segments, the script rasterizes them onto the 90 m
    classified raster grid, crops to a subregion with DAMANIA_MAX_DIST_KM padding,
    runs a distance transform, and applies the Damania pct_cleared_good curve.

Run from: strategic-road-planning/
    python model/per-mine-outcomes.py
"""

import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject as warp_reproject, Resampling as WarpResampling
from scipy.ndimage import distance_transform_edt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent   # strategic-road-planning/
RESULTS_DIR  = BASE_DIR / "model/results/impact-assessment/experiment_04_full120runs"
OUT_DIR      = BASE_DIR / "model/results/per-mine-summary"
MINES_FP     = BASE_DIR / "data/output/cmr-mine-locations/all_mines_with_id.csv"
DAMANIA_FP   = BASE_DIR / "data/input/damania-deforestation-lookup/damania_pct_cleared.csv"
TREECOVER_FP = BASE_DIR / "data/output/processed-hansen-treecover/cameroon_treecover2024_30m.tif"

DAMANIA_MAX_DIST_KM = 10.0
DAMANIA_CURVE       = "good"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Reference data ─────────────────────────────────────────────────────
mines_df = pd.read_csv(MINES_FP)

LATE_AND_EARLY_IDS = frozenset(
    mines_df[mines_df["DEV_STAGE_AGGREGATED_SNL"].isin(["Late-stage", "Early-stage"])]["ID"]
)
LATE_IDS  = frozenset(
    mines_df[mines_df["DEV_STAGE_AGGREGATED_SNL"] == "Late-stage"]["ID"]
)
EARLY_IDS = LATE_AND_EARLY_IDS - LATE_IDS

ROW_MINE_IDS  = sorted(LATE_AND_EARLY_IDS)
mine_name_map = mines_df.set_index("ID")["PROP_NAME"].to_dict()

print(f"Mines — late+early: {len(LATE_AND_EARLY_IDS)}, late-only: {len(LATE_IDS)}")

# ── Step 2: Damania lookup ────────────────────────────────────────────────────
damania_df   = pd.read_csv(DAMANIA_FP)
damania_dist = damania_df["distance_km"].values.astype(np.float32)
damania_pct  = (damania_df[f"pct_cleared_{DAMANIA_CURVE}"].values / 100.0).astype(np.float32)

# ── Step 3: n_mines_no_path per scenario ──────────────────────────────────────
no_path_map: dict[str, int] = {}
for fp in RESULTS_DIR.glob("*__assessment.csv"):
    stem = fp.stem.replace("__assessment", "")
    no_path_map[stem] = int(pd.read_csv(fp)["n_mines_no_path"].iloc[0])

# ── Step 4: Load treecover at 90 m (once, reuse across scenarios) ─────────────
# Use the first classified raster as the reference 90 m grid.
sample_cls_fp = sorted(RESULTS_DIR.glob("*__classified.tif"))[0]
with rasterio.open(sample_cls_fp) as ref:
    ref_transform = ref.transform
    ref_shape     = (ref.height, ref.width)
    ref_crs       = ref.crs

print(f"Loading treecover at 90 m onto reference grid {ref_shape} ...")
tc_90_ref = np.zeros(ref_shape, dtype=np.float32)
with rasterio.open(TREECOVER_FP) as tc_src:
    warp_reproject(
        source=rasterio.band(tc_src, 1),
        destination=tc_90_ref,
        src_transform=tc_src.transform, src_crs=tc_src.crs,
        dst_transform=ref_transform,    dst_crs=ref_crs,
        resampling=WarpResampling.average,
    )
# Invalid pixels (water=200, nodata=255) → 0 canopy
tc_90_ref = np.where(tc_90_ref <= 100, tc_90_ref, 0.0).astype(np.float32)
canopy_ref = tc_90_ref / 100.0  # 0–1 fraction
print(f"  Treecover ready: {canopy_ref.shape}, "
      f"mean canopy fraction = {canopy_ref.mean():.3f}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def px_size_m(transform, shape) -> float:
    """Average pixel size in metres."""
    lat  = transform.f + (shape[0] / 2) * transform.e
    px_x = abs(transform.a) * 111_320 * math.cos(math.radians(lat))
    px_y = abs(transform.e) * 111_320
    return (px_x + px_y) / 2.0


def subregion_slice(road_mask: np.ndarray, pad_px: int, shape: tuple):
    """Return (r0, r1, c0, c1) bounding box with padding, or None if mask is empty."""
    rows, cols = np.where(road_mask)
    if len(rows) == 0:
        return None
    r0 = max(0, int(rows.min()) - pad_px)
    r1 = min(shape[0], int(rows.max()) + pad_px + 1)
    c0 = max(0, int(cols.min()) - pad_px)
    c1 = min(shape[1], int(cols.max()) + pad_px + 1)
    return r0, r1, c0, c1


def parse_stem(stem: str):
    """Parse mining, friction, mask from scenario stem."""
    mining, rest = stem.split("__kribi__")
    rest         = rest.replace("__ds1", "")
    tokens       = rest.split("__")
    return mining, "__".join(tokens[:-1]), tokens[-1]


def same_grid(t1, s1, t2, s2) -> bool:
    """Check if two raster grids share the same transform and shape."""
    return (s1 == s2 and
            abs(t1.a - t2.a) < 1e-9 and abs(t1.e - t2.e) < 1e-9 and
            abs(t1.c - t2.c) < 1e-9 and abs(t1.f - t2.f) < 1e-9)


# ── Step 5: Main loop ─────────────────────────────────────────────────────────
action_fps = sorted(RESULTS_DIR.glob("*__action_roads.gpkg"))
stems      = [fp.stem.replace("__action_roads", "") for fp in action_fps]

# Result containers: {mine_id: {stem: value}}
res_action = {m: {} for m in ROW_MINE_IDS}
res_km     = {m: {} for m in ROW_MINE_IDS}
res_cost   = {m: {} for m in ROW_MINE_IDS}
res_direct = {m: {} for m in ROW_MINE_IDS}
res_total  = {m: {} for m in ROW_MINE_IDS}

t0 = time.time()

for i, (act_fp, stem) in enumerate(zip(action_fps, stems), 1):
    mining, friction, mask = parse_stem(stem)
    scope_ids = LATE_AND_EARLY_IDS if "late_and_early" in mining else LATE_IDS
    n_no_path = no_path_map.get(stem, 0)

    elapsed = time.time() - t0
    eta_s   = (elapsed / i) * (len(action_fps) - i) if i > 1 else 0
    print(f"[{i:3d}/{len(action_fps)}] {stem}  "
          f"(elapsed {elapsed/60:.1f} min, ETA {eta_s/60:.1f} min)")

    # Load action roads with geometry
    action_gdf = gpd.read_file(act_fp)

    # Load classified raster
    cls_fp = RESULTS_DIR / f"{stem}__classified.tif"
    with rasterio.open(cls_fp) as cls_src:
        cls_transform = cls_src.transform
        cls_shape     = (cls_src.height, cls_src.width)
        cls_crs       = cls_src.crs

    cls_px_m    = px_size_m(cls_transform, cls_shape)
    cls_px_area = cls_px_m * cls_px_m / 10_000        # ha per pixel
    pad_px      = math.ceil((DAMANIA_MAX_DIST_KM * 1000 + 200) / cls_px_m)

    # Treecover on this classified grid (usually same as reference; fallback otherwise)
    if same_grid(cls_transform, cls_shape, ref_transform, ref_shape):
        canopy_cls = canopy_ref
    else:
        _tc = np.zeros(cls_shape, dtype=np.float32)
        with rasterio.open(TREECOVER_FP) as tc_src:
            warp_reproject(
                source=rasterio.band(tc_src, 1), destination=_tc,
                src_transform=tc_src.transform, src_crs=tc_src.crs,
                dst_transform=cls_transform,      dst_crs=cls_crs,
                resampling=WarpResampling.average,
            )
        canopy_cls = np.where(_tc <= 100, _tc, 0.0).astype(np.float32) / 100.0

    # Build per-mine road lookup from action roads GeoPackage
    mine_data: dict = {}
    for _, row in action_gdf.iterrows():
        mid = row["mine_id"]
        if mid not in mine_data:
            mine_data[mid] = {
                "action": row["action_needed"],
                "km":     0.0,
                "cost":   0.0,
                "geoms":  [],
            }
        mine_data[mid]["km"]   += float(row["length_km"])
        mine_data[mid]["cost"] += float(row["construction_cost_usd"])
        if row.geometry is not None:
            mine_data[mid]["geoms"].append(row.geometry)

    # Per-mine computation
    for mine_id in ROW_MINE_IDS:

        # ── Out of scope ──────────────────────────────────────────────────────
        if mine_id not in scope_ids:
            res_action[mine_id][stem] = "out_of_scope"
            res_km[mine_id][stem]     = float("nan")
            res_cost[mine_id][stem]   = float("nan")
            res_direct[mine_id][stem] = float("nan")
            res_total[mine_id][stem]  = float("nan")
            continue

        # ── In scope but no action roads ──────────────────────────────────────
        if mine_id not in mine_data:
            # If n_no_path == 0: all in-scope mines not in action_roads have paved access.
            # If n_no_path > 0: some mines are inaccessible. We conservatively mark
            # early-stage mines as inaccessible (they are more remote) and late-stage
            # mines as no_action (they already have paved access in most scenarios).
            inaccessible = (n_no_path > 0) and (mine_id in EARLY_IDS)
            status = "inaccessible" if inaccessible else "no_action"
            nan_or_zero = float("nan") if inaccessible else 0.0
            res_action[mine_id][stem] = status
            res_km[mine_id][stem]     = nan_or_zero
            res_cost[mine_id][stem]   = nan_or_zero
            res_direct[mine_id][stem] = nan_or_zero
            res_total[mine_id][stem]  = nan_or_zero
            continue

        # ── Mine has action roads ─────────────────────────────────────────────
        md = mine_data[mine_id]
        res_action[mine_id][stem] = md["action"]
        res_km[mine_id][stem]     = round(md["km"],   4)
        res_cost[mine_id][stem]   = round(md["cost"], 2)

        if not md["geoms"]:
            res_direct[mine_id][stem] = 0.0
            res_total[mine_id][stem]  = 0.0
            continue

        # Rasterize mine road segments onto classified grid
        mine_road_px = rio_rasterize(
            [(g, 1) for g in md["geoms"]],
            out_shape=cls_shape, transform=cls_transform,
            fill=0, dtype=np.uint8,
        ).astype(bool)

        # Crop to tight subregion + padding
        slc = subregion_slice(mine_road_px, pad_px, cls_shape)
        if slc is None:
            res_direct[mine_id][stem] = 0.0
            res_total[mine_id][stem]  = 0.0
            continue

        r0, r1, c0, c1 = slc
        sub_road   = mine_road_px[r0:r1, c0:c1]
        sub_canopy = canopy_cls[r0:r1, c0:c1]

        # Direct deforestation: canopy cleared by road footprint pixels
        defor_direct = float(np.sum(sub_canopy[sub_road])) * cls_px_area

        # Distance transform on small subregion → Damania decay
        dist_m  = distance_transform_edt(~sub_road).astype(np.float32) * cls_px_m
        dist_km = dist_m / 1000.0

        pct_clr = np.interp(dist_km, damania_dist, damania_pct).astype(np.float32)
        pct_clr[dist_km > DAMANIA_MAX_DIST_KM] = 0.0

        defor_total = float(np.sum(sub_canopy * pct_clr)) * cls_px_area

        res_direct[mine_id][stem] = round(defor_direct, 4)
        res_total[mine_id][stem]  = round(defor_total,  4)

print(f"\nAll scenarios processed in {(time.time()-t0)/60:.1f} min")

# ── Step 6: Build and save DataFrames ─────────────────────────────────────────

def build_df(data_dict: dict, mine_ids: list, name_map: dict, col_stems: list) -> pd.DataFrame:
    df = pd.DataFrame(data_dict).T.reindex(mine_ids)
    df.index.name = "mine_id"
    df.insert(0, "mine_name", [name_map.get(m, m) for m in mine_ids])
    valid_stems = [s for s in col_stems if s in df.columns]
    return df[["mine_name"] + valid_stems]


for csv_name, data in [
    ("per_mine_action_needed",   res_action),
    ("per_mine_length_km",       res_km),
    ("per_mine_cost_usd",        res_cost),
    ("per_mine_defor_direct_ha", res_direct),
    ("per_mine_defor_total_ha",  res_total),
]:
    df     = build_df(data, ROW_MINE_IDS, mine_name_map, stems)
    out_fp = OUT_DIR / f"{csv_name}.csv"
    df.to_csv(out_fp)
    print(f"Saved: {out_fp.name}  ({df.shape[0]} mines × {df.shape[1]-1} scenarios)")

print("\nDone.")
