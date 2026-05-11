# Impact Assessment Model for Mine-Access Road Planning in Cameroon

This model evaluates the environmental and economic impact of road network scenarios that connect mining sites to export ports in Cameroon. For each scenario, it estimates construction cost, deforestation risk, and forest fragmentation caused by upgrading existing unpaved roads and constructing new roads through undeveloped land.

---

## Model Structure

![Model structure](figures/model-structure.png)

The model runs in two stages:

**Stage 1 — Least-cost path generation** (`model/path-generator.py`): For each combination of mining scenario, port, and friction layer, the model finds the optimal (least-cost) route from each mine to its nearest port using a cost-surface algorithm. Routes follow existing roads where possible and cut new paths only when necessary. Results are saved as GeoPackages containing the path geometries and mine attributes.

**Stage 2 — Impact assessment** (`model/impact-assessment.py`): For each generated path network, the model classifies every road segment by the action needed (paved — no action, upgrade, or new build), then computes three impact metrics: construction cost, deforestation risk within buffer zones, and forest fragmentation. Results are saved as a CSV (one row per scenario) alongside geospatial files for visualisation in QGIS.

---

## Data and Preprocessing

Run the following scripts once before the model, in this order:

| Order | Script | What it does |
|---|---|---|
| 1 | `notebook/00_get-DEM.ipynb` | Downloads SRTM digital elevation model tiles for Cameroon |
| 2 | `notebook/00_get-OSM.ipynb` | Downloads road network from OpenStreetMap / HeiGIT / Liu et al. |
| 3 | `notebook/01_clean-roads.ipynb` | Cleans and merges road datasets into a single GeoPackage |
| 4 | `notebook/02a_model-construction-cost-friction.py` | Builds the construction cost friction raster from slope using an earthwork volume model |
| 5 | `notebook/02b_merge-friction-tiles.py` | Merges and clips friction tiles to the Cameroon boundary |
| 6 | `notebook/01b_clean-hansen-forests.py` | Processes Hansen Global Forest Change tiles (2000–2024) into a single 30 m canopy cover raster for Cameroon |

---

## Running the Model

All commands are run from `strategic-road-planning/model/`:

```bash
# Run all scenario combinations
python path-generator-run.py

# Preview scenarios without running
python path-generator-run.py --dry-run

# Run impact assessment for one scenario
python impact-assessment.py results/least-cost-paths/<scenario>.gpkg
```

**Scenario naming convention:**

```
{mining_scope}__{port}__{friction}__{mask}__ds{downsample}
Example: late_stage__kribi__base__protected_areas__ds1
```

| Dimension | Options |
|---|---|
| Mining scope | `late_stage` (operational mines only), `late_and_early_stage` (all planned mines) |
| Port | `kribi`, `kribi_douala` |
| Friction | `base` (slope-based construction cost raster) |
| Mask | `no_mask`, `protected_areas` (routes avoid WDPA protected areas) |
| Downsample | `ds1` (native ~90 m resolution), `ds5`, `ds10` |

---

## Model Output

Each scenario produces **24 output files** saved in `model/results/impact-assessment/`.

---

### CSV — `assessment_{scenario}.csv`

One row per scenario containing all impact metrics.

#### Road Network Classification

| Column | Unit | Description |
|---|---|---|
| `scenario` | — | Scenario identifier |
| `paved_km` | km | Path length along existing paved roads — no construction needed |
| `unpaved_km` | km | Path length along existing unpaved roads to be upgraded to paved |
| `to_be_built_km` | km | Path length requiring entirely new road construction |
| `total_km` | km | Total mine-to-port path length across all mines in the scenario |

#### Construction Cost

Cost is estimated by summing friction raster values (USD/km) along each road segment. The friction raster encodes construction difficulty based on terrain slope using an earthwork volume (EWV) model.

| Column | Unit | Description |
|---|---|---|
| `upgrade_cost_usd` | USD | Estimated cost to upgrade unpaved roads to paved standard |
| `new_build_cost_usd` | USD | Estimated cost to construct entirely new road segments |
| `total_cost_usd` | USD | `upgrade_cost_usd + new_build_cost_usd` |

#### Deforestation Risk

Inspired by Sonter et al. (2017) and Siqueira-Gay et al. (2022), who quantify forest loss within road buffer zones. Forest area is **canopy-cover weighted** using Hansen et al. (2013) treecover data adjusted for loss through 2024 — a pixel with 60% canopy cover contributes 0.6 × pixel area rather than a full pixel. Forest is defined as ≥10% canopy cover (FAO definition). Buffer zones are computed via Euclidean distance transform at 90 m, upsampled to 30 m for counting. Upgrade and build zones are mutually exclusive — build takes priority where they overlap.

| Column | Unit | Description |
|---|---|---|
| `defor_upgrade_0m_ha` | ha | Canopy-weighted forest area on the direct footprint of upgrade roads |
| `defor_build_0m_ha` | ha | Canopy-weighted forest area on the direct footprint of new-build roads |
| `defor_action_0m_ha` | ha | Combined footprint: `defor_upgrade_0m_ha + defor_build_0m_ha` |
| `defor_upgrade_{d}m_ha` | ha | Forest area within distance *d* of upgrade roads (d = 100, 250, 500, 750, 1000 m) |
| `defor_build_{d}m_ha` | ha | Forest area within distance *d* of new-build roads |
| `defor_action_{d}m_ha` | ha | Combined: `defor_upgrade_{d}m_ha + defor_build_{d}m_ha` |

#### Forest Fragmentation

Following Siqueira-Gay et al. (2022): **Fragmentation index = (patches − 1) / (forest extent in km² − 1)**. Forest patches are identified using 8-connectivity with a minimum patch size of ≥1 ha to exclude sub-pixel noise. The **before** state burns the full existing road network (paved + unpaved) into the forest mask; the **after** state additionally burns the scenario's action roads. Computed across all non-overlapping grid windows covering Cameroon. Reported at four window sizes as a sensitivity test.

| Column | Unit | Description |
|---|---|---|
| `fragmentation_patches_before_{w}km` | count | Total forest patches (≥1 ha) across all windows with the existing road network burned |
| `fragmentation_patches_after_{w}km` | count | Same but with scenario action roads additionally burned |
| `fragmentation_index_before_{w}km` | patches / km² | Country-level index before: `(patches_before − 1) / (forest_km² − 1)` |
| `fragmentation_index_after_{w}km` | patches / km² | Country-level index after adding scenario roads |
| `fragmentation_n_windows_with_action_roads_{w}km` | count | Number of grid cells that contained at least one action road pixel |

*w* ∈ {10, 50, 100, 150} km

---

### Geospatial Outputs

#### Action Roads — `action_roads_{scenario}.gpkg`

A vector line file of all upgrade and new-build road segments, split by road class. Use this in QGIS to inspect individual segment costs or to style roads by action type.

| Attribute | Description |
|---|---|
| `mine_id` | Mine identifier |
| `mine_name` | Mine name |
| `closest_port` | Port this segment connects to |
| `action_needed` | `upgrade` or `build` |
| `length_km` | Segment length (km) |
| `construction_cost_usd` | Estimated construction cost for this segment |
| `scenario` | Scenario identifier |

Note: segment-level costs are for spatial inspection only. Use the CSV for scenario-level cost totals, which correctly deduplicate shared corridors used by multiple mines.

#### Classified Road Raster — `classified_{scenario}.tif`

A 90 m raster encoding road classification for every pixel along the mine-to-port paths.

| Value | Meaning |
|---|---|
| 0 | Not a path pixel |
| 1 | Existing paved road (no action needed) |
| 2 | Existing unpaved road to be upgraded |
| 3 | New road to be built |

#### Deforestation Buffer Rasters — `defor_buffer_{d}m_{scenario}.tif`

Six rasters (one per buffer distance: 0, 100, 250, 500, 750, 1000 m) at 30 m resolution. Each pixel inside the buffer zone is labelled by the road action type that created it. Useful for overlaying with other spatial data (biodiversity index, protected areas) in QGIS.

| Value | Meaning |
|---|---|
| 0 | Outside the buffer |
| 2 | Within buffer of an upgrade road |
| 3 | Within buffer of a new-build road |

#### Fragmentation Rasters — `fragmentation_{type}_{w}km_{scenario}.tif`

Sixteen rasters (4 window sizes × 4 types). Each pixel represents one grid window — the raster resolution is therefore coarser for larger window sizes. Subtract `index_before` from `index_after` in QGIS or a notebook to map where fragmentation increases the most.

| File | Format | Description |
|---|---|---|
| `fragmentation_patches_before_{w}km_{scenario}.tif` | int32 | Forest patch count per window cell, before; nodata = −1 (no forest) |
| `fragmentation_patches_after_{w}km_{scenario}.tif` | int32 | Forest patch count per window cell, after |
| `fragmentation_index_before_{w}km_{scenario}.tif` | float32 | Fragmentation index (patches/km²) per window cell, before; nodata = NaN |
| `fragmentation_index_after_{w}km_{scenario}.tif` | float32 | Fragmentation index (patches/km²) per window cell, after |

---

## References

- Hansen, M. C. et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342, 850–853.
- Siqueira-Gay, J. et al. (2022). Strategic planning to mitigate mining impacts on protected areas in the Brazilian Amazon. *Nature Sustainability*, 5, 853–860.
- Sonter, L. J. et al. (2017). Mining drives extensive deforestation in the Brazilian Amazon. *Nature Communications*, 8, 1013.
