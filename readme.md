# Impact Assessment Model for Mine-Access Road Planning in Cameroon

Evaluates road network scenarios connecting mining sites to export ports in Cameroon across three impact dimensions: construction cost, deforestation risk, and forest fragmentation.

---

## Model Structure

![Model structure](figures/model-structure.png)

**Stage 1 — Least-cost path generation** (`model/path_generator.py`)

Finds the optimal route from each mine to its nearest port using a cost-surface algorithm (`MCP_Geometric`). Routes follow existing roads where possible and cut new paths only when necessary. Output: GeoPackages in `model/results/least-cost-paths/`.

**Stage 2 — Impact assessment** (`model/impact_assessment.py`)

Classifies every path pixel as paved / upgrade / new-build, then computes construction cost, deforestation risk (Damania et al. 2018 distance-decay), and forest fragmentation (patch detection at admin2 level). Output: CSV + geospatial files in `model/results/impact-assessment/`.

---

## How to Run

All commands run from `strategic-road-planning/model/`.

### Step 0 — Data preparation (run once, in order)

See the [Inputs](#inputs) section for the full list of preparation scripts.

### Step 1 — Generate least-cost paths

Edit the experiment matrix in `02_path-generator-run.py`, then:

```bash
python 02_path-generator-run.py            # run all experiments
python 02_path-generator-run.py --dry-run  # preview without running
```

Output: `.gpkg` files in `results/least-cost-paths/<experiment_folder>/`

### Step 2 — Generate baseline forest patches

Required once before running impact assessment (produces the no-action fragmentation baseline).

```bash
python 03_forest-patch-detection-run__base.py
python 03_forest-patch-detection-run__base.py --dry-run
```

Output: `data/output/forest-patch-detection/no_action__forest_patches__thresh10__minpatch1ha__adm2__roads_paved.{tif,gpkg}`

### Step 3 — Run impact assessment

**Option A — Batch runner** (recommended): edit `EXPERIMENTS` in `04-impact-assessment-run.py`, then:

```bash
python 04-impact-assessment-run.py
```

**Option B — Single folder via CLI:**

```bash
python impact_assessment.py experiment_00
python impact_assessment.py experiment_00 --downsample 5   # faster, lower memory
```

Finds all `.gpkg` files in `results/least-cost-paths/experiment_00/` and writes outputs to `results/impact-assessment/experiment_00/`.

**Option C — Single scenario:**

```bash
python impact_assessment.py results/least-cost-paths/experiment_00/late_stage__kribi__base__no_mask__ds1.gpkg
```

---

## Inputs

### Data preparation (run once, in order)

| Step | Script | Output |
|---|---|---|
| 1 | `notebook/00_get-DEM.ipynb` | SRTM 90 m DEM tiles |
| 2 | `notebook/00_get-OSM.ipynb` | Road network (OSM, HeiGIT, Liu 2025) |
| 3 | `notebook/01_clean-roads.ipynb` | Merged road GeoPackage (`paved` / `unpaved`) |
| 4 | `notebook/02a_model-construction-cost-friction.py` | Per-tile cost rasters (USD/pixel) |
| 5 | `notebook/02b_merge-friction-tiles.py` | Merged friction raster (90 m, Cameroon extent) |
| 6 | `notebook/01a_clean-jrc-forests.py` | JRC 2020 forest mask (30 m) |
| 7 | `notebook/01b_clean-hansen-forests.py` | `cameroon_treecover2024_30m.tif` |

### Input datasets (`data/input/`)

| Dataset | Source | Used for |
|---|---|---|
| Mine locations (operational) | Ahmed et al. | Built mine footprint |
| Mine locations (planned) | S&P | Planned mine locations |
| Port coordinates | Hardcoded (Kribi, Douala) | Destination points |
| Road network | OSM / HeiGIT (2024) / Liu et al. (2025) | Upgrade vs. new-build classification |
| SRTM DEM (90 m) | NASA/USGS | Slope → construction cost |
| ESA WorldCover 2021 (10 m) | ESA | Land cover → base cost per km |
| Construction cost parameters | Literature (EWV model) | Cost lookup tables |
| Hansen Global Forest Change v1.12 | Hansen et al. (2013) | Treecover 2000 + loss 2001–2024 |
| Damania et al. (2018) decay curve | Damania et al. | Deforestation distance-decay lookup |
| JRC Global Forest Cover 2020 v3 | JRC | Cross-check forest mask |
| WDPA Protected Areas (Mar 2026) | UNEP-WCMC / IUCN | Spatial mask option |
| Biodiversity Intactness Index | NHM / PREDICTS | Available; not used in current metrics |
| Intact Forest Landscapes v2025 | IFL | Available; not used in current metrics |
| Cameroon admin boundary | HDX | Clipping extent |

### Scenario dimensions

| Dimension | Options |
|---|---|
| Mining scope | `late_stage`, `late_and_early_stage` |
| Port | `kribi`, `kribi_douala` |
| Friction | `base` |
| Mask | `no_mask`, `protected_areas` |
| Downsample | `ds1` (~90 m), `ds5`, `ds10` |

Scenario name format: `{mining_scope}__{port}__{friction}__{mask}__ds{n}`

---

## Outputs

All outputs are written to `model/results/impact-assessment/<experiment_folder>/`.

### `{scenario}__assessment.csv`

One row per scenario.

**Road classification**

| Column | Unit | Description |
|---|---|---|
| `paved_km` | km | Along existing paved roads — no action needed |
| `unpaved_km` | km | Existing unpaved roads to be upgraded |
| `to_be_built_km` | km | New road construction required |
| `total_km` | km | Total path length across all mines |

**Construction cost**

| Column | Unit | Description |
|---|---|---|
| `upgrade_cost_usd` | USD | Cost to upgrade unpaved → paved |
| `new_build_cost_usd` | USD | Cost to construct new segments |
| `total_cost_usd` | USD | `upgrade + new_build` |

**Deforestation risk** — Damania et al. (2018) distance-decay, "good" road quality curve, 0–10 km; canopy-weighted (Hansen ≥10% = forest); build takes priority where upgrade and build zones overlap

| Column | Unit | Description |
|---|---|---|
| `defor_upgrade_ha` | ha | Expected forest cleared near upgrade roads |
| `defor_build_ha` | ha | Expected forest cleared near new-build roads |
| `defor_action_ha` | ha | Combined (`upgrade + build`) |
| `total_remaining_forest_ha` | ha | Remaining forest across Cameroon after expected clearing |

**Forest fragmentation** — admin2-level patch detection; paved roads burned as barriers; min patch 1 ha; Hansen ≥10% threshold

| Column | Unit | Description |
|---|---|---|
| `frag_forest_ha` | ha | Total forest area (action scenario) |
| `frag_pct_forest` | % | Forest cover (action) |
| `frag_n_patches` | count | Number of patches ≥ 1 ha (action) |
| `frag_mean_patch_ha` | ha | Patch-count-weighted mean patch size (action) |
| `frag_index` | patches/ha | Fragmentation index (action) |
| `frag_baseline_*` | — | Same metrics for no-action baseline |
| `frag_delta_*` | — | Action minus baseline |
| `frag_patch_size_dis_{bin}_n` | count | Patch count in size bin (action); bins: `lt1ha`, `1_10ha`, `10_100ha`, `100_1kha`, `1k_10kha`, `10k_100kha`, `100k_1000kha`, `gt1000kha` |
| `frag_patch_size_dis_{bin}_ha` | ha | Total area in size bin (action) |
| `frag_patch_size_dis_baseline_{bin}_*` | — | Same for baseline |
| `frag_patch_size_dis_delta_{bin}_*` | — | Action minus baseline |

### `{scenario}__action_roads.gpkg`

Vector lines of upgrade and new-build segments.

| Attribute | Description |
|---|---|
| `mine_id`, `mine_name` | Mine identifier |
| `closest_port` | Destination port |
| `action_needed` | `upgrade` or `build` |
| `length_km` | Segment length |
| `construction_cost_usd` | Segment-level cost (for spatial inspection; use CSV for totals) |

### `{scenario}__classified.tif`

90 m raster: `0` = not a path pixel · `1` = paved · `2` = upgrade · `3` = new build

### `{scenario}__defor_loss_pct.tif`

90 m float32 raster (subregion around action roads): expected % treecover loss per pixel. NaN = water / nodata.

### `{scenario}__defor_remaining_treecover.tif`

90 m float32 raster (full Cameroon): treecover % remaining after expected clearing. NaN = water / nodata.

### `{scenario}__forest_patches__thresh10__minpatch1ha__adm2__roads_paved.{tif,gpkg}`

Patch-ID raster and admin2 fragmentation statistics for the action scenario.

---

<!-- ## References

- Damania et al. (2018). *World Bank Research Report.*
- Hansen et al. (2013). *Science*, 342, 850–853.
- Siqueira-Gay et al. (2022). *Nature Sustainability*, 5, 853–860.
- Sonter et al. (2017). *Nature Communications*, 8, 1013. -->
