# Impact Assessment Model for Mine-Access Road Planning in Cameroon

Evaluates road network scenarios connecting mining sites to export ports in Cameroon across three impact dimensions: construction cost, deforestation risk, and forest fragmentation.

---

## Model Structure

![Model structure](figures/model-structure.png)

**Stage 1 — Least-cost path generation** (`model/path-generator.py`)

Finds the optimal route from each mine to its nearest port using a cost-surface algorithm (`MCP_Geometric`). Routes follow existing roads where possible and cut new paths only when necessary. Output: GeoPackages in `model/results/least-cost-paths/`.

**Stage 2 — Impact assessment** (`model/impact-assessment.py`)

Classifies every path pixel as paved / upgrade / new-build, then computes construction cost, deforestation risk, and forest fragmentation. Output: CSV + geospatial files in `model/results/impact-assessment/`.

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
| Mine locations (operational) | Ahmed et al. Sub-Saharan Africa mine DB | Origin points |
| Mine locations (planned) | GFW Mining Permits | Early-stage scenario |
| Port coordinates | Hardcoded (Kribi, Douala) | Destination points |
| Road network | OSM / HeiGIT (2024) / Liu et al. (2025) | Upgrade vs. new-build classification |
| SRTM DEM (90 m) | NASA/USGS | Slope → construction cost |
| ESA WorldCover 2021 (10 m) | ESA | Land cover → base cost per km |
| Construction cost parameters | Literature (EWV model) | Cost lookup tables |
| Hansen Global Forest Change v1.12 | Hansen et al. (2013) | Treecover 2000 + loss 2001–2024 |
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

Each scenario produces 21 files in `model/results/impact-assessment/`.

### `assessment_{scenario}.csv`

One row per scenario.

**Road classification**

| Column | Unit | Description |
|---|---|---|
| `paved_km` | km | Along existing paved roads — no action |
| `unpaved_km` | km | Existing unpaved roads to be upgraded |
| `to_be_built_km` | km | New road construction required |
| `total_km` | km | Total path length across all mines |

**Construction cost**

| Column | Unit | Description |
|---|---|---|
| `upgrade_cost_usd` | USD | Cost to upgrade unpaved → paved |
| `new_build_cost_usd` | USD | Cost to construct new segments |
| `total_cost_usd` | USD | `upgrade + new_build` |

**Deforestation risk** — canopy-cover weighted (Hansen, ≥10% = forest); upgrade and build zones are mutually exclusive (build takes priority)

| Column | Unit | Description |
|---|---|---|
| `defor_upgrade_0km_km2` | km² | Forest on direct footprint of upgrade roads |
| `defor_build_0km_km2` | km² | Forest on direct footprint of new-build roads |
| `defor_action_0km_km2` | km² | Combined footprint |
| `defor_upgrade_{d}km_km2` | km² | Forest within *d* km of upgrade roads; *d* ∈ {0.1, 0.25, 0.5, 0.75, 1.0} |
| `defor_build_{d}km_km2` | km² | Forest within *d* km of new-build roads |
| `defor_action_{d}km_km2` | km² | Combined |

**Forest fragmentation** — index = (patches − 1) / (forest km² − 1); min patch 1 ha; *w* ∈ {50, 100} km

| Column | Unit | Description |
|---|---|---|
| `fragmentation_patches_before_{w}km` | count | Patch count, existing roads burned |
| `fragmentation_patches_after_{w}km` | count | Patch count, scenario roads also burned |
| `fragmentation_index_before_{w}km` | patches/km² | Index before |
| `fragmentation_index_after_{w}km` | patches/km² | Index after |
| `fragmentation_n_windows_with_action_roads_{w}km` | count | Grid cells containing action roads |

### `action_roads_{scenario}.gpkg`

Vector lines of upgrade and new-build segments.

| Attribute | Description |
|---|---|
| `mine_id`, `mine_name` | Mine identifier |
| `closest_port` | Destination port |
| `action_needed` | `upgrade` or `build` |
| `length_km` | Segment length |
| `construction_cost_usd` | Segment-level cost (spatial inspection only; use CSV for totals) |

### `classified_{scenario}.tif`

90 m raster: `0` = not a path pixel · `1` = paved · `2` = upgrade · `3` = new build

### `defor_buffer_{d}km_{scenario}.tif`

30 m raster (6 files, one per buffer distance): `0` = outside · `2` = upgrade buffer · `3` = build buffer

### `fragmentation_{type}_{w}km_{scenario}.tif`

12 rasters (2 window sizes × 6 types): `patches_before`, `patches_after`, `patches_delta`, `index_before`, `index_after`, `index_delta`. Nodata = −1 (int) or NaN (float) where no forest exists.

---

## References

- Hansen et al. (2013). *Science*, 342, 850–853.
- Siqueira-Gay et al. (2022). *Nature Sustainability*, 5, 853–860.
- Sonter et al. (2017). *Nature Communications*, 8, 1013.
