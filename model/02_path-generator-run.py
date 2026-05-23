#!/usr/bin/env python3
"""
path-generator-run.py — Experiment configuration for the least-cost path model.

Runs all combinations of mining scenarios, port choices, friction layers,
and spatial masks. Each combination is one experiment; results are saved
to a GeoPackage named after the combination keys and downsample factor.

Usage:
    python path-generator-run.py              # run all combinations
    python path-generator-run.py --dry-run    # list experiments without running
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

import path_generator
from path_generator import ExperimentConfig, run_experiment

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data/output"
OUTPUT_DIR = BASE_DIR / "model/results/least-cost-paths"
MINES_FP   = DATA_DIR / "cmr-mine-locations/all_mines_with_id.csv"

# =============================================================================
# CHOICE LIBRARIES
# =============================================================================

_mines_df = pd.read_csv(MINES_FP)

mining_scenario_choices = {
    "late_stage":           _mines_df[_mines_df["DEV_STAGE_AGGREGATED_SNL"] == "Late-stage"]["ID"].tolist(),
    "late_and_early_stage": _mines_df[_mines_df["DEV_STAGE_AGGREGATED_SNL"].isin(["Late-stage", "Early-stage"])]["ID"].tolist(),
}

port_choices = {
    "kribi":         {"Kribi":  (2.940594, 9.910192)},
    # "kribi_douala":  {"Kribi":  (2.940594, 9.910192), "Douala": (4.0511,   9.7679)},
}

_FRICTION_DIR = DATA_DIR / "cmr-cost-friction-90m-for-experiments"
_MASK_DIR     = DATA_DIR / "processed-protected-areas"

friction_layer_choices = {
    "construction_only":           _FRICTION_DIR / "friction__construction_only.tif",
    "carbon_low":                  _FRICTION_DIR / "friction__carbon_low.tif",
    "carbon_central":              _FRICTION_DIR / "friction__carbon_central.tif",
    "carbon_high":                 _FRICTION_DIR / "friction__carbon_high.tif",
    "carbon_low__biodiversity":    _FRICTION_DIR / "friction__carbon_low__biodiversity.tif",
    "carbon_central__biodiversity":_FRICTION_DIR / "friction__carbon_central__biodiversity.tif",
    "carbon_high__biodiversity":   _FRICTION_DIR / "friction__carbon_high__biodiversity.tif",
}

spatial_mask_choices = {
    "no_mask":                  None,
    "protected_areas":          _MASK_DIR / "with-buffer/cmr-protected-areas__buffer0km.gpkg",
    "protected_areas_buf3km":   _MASK_DIR / "with-buffer/cmr-protected-areas__buffer3km.gpkg",
    "protected_areas_buf5km":   _MASK_DIR / "with-buffer/cmr-protected-areas__buffer5km.gpkg",
    "protected_areas_buf10km":  _MASK_DIR / "with-buffer/cmr-protected-areas__buffer10km.gpkg",
}

# Downsampling factor applied to all experiments
DOWNSAMPLE = 1   # 1=~90m, 5=~450m, 10=~900m

# =============================================================================
# RUN ALL COMBINATIONS
# =============================================================================

def build_experiments():
    experiments = []
    combos = itertools.product(
        mining_scenario_choices.items(),
        port_choices.items(),
        friction_layer_choices.items(),
        spatial_mask_choices.items(),
    )
    for (mining_key, mine_ids), (port_key, ports), (friction_key, friction_fp), (mask_key, mask_fp) in combos:
        name = f"{mining_key}__{port_key}__{friction_key}__{mask_key}__ds{DOWNSAMPLE}"
        experiments.append(ExperimentConfig(
            name        = name,
            mine_ids    = mine_ids,
            ports       = ports,
            friction_fp = friction_fp,
            mines_fp    = MINES_FP,
            mask_fp     = mask_fp,
            output_dir  = OUTPUT_DIR,
            downsample  = DOWNSAMPLE,
        ))
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Run all least-cost path experiment combinations")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all experiment names without running them.")
    args = parser.parse_args()

    experiments = build_experiments()

    print(f"{len(experiments)} experiment(s) to run:\n")
    for exp in experiments:
        print(f"  {exp.name}")

    if args.dry_run:
        return

    print()
    for cfg in experiments:
        run_experiment(cfg)

    print(f"\n{'=' * 62}")
    print(f"  All done. {len(experiments)} experiment(s) completed.")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
