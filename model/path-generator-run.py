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
import importlib.util
import itertools
from pathlib import Path

import pandas as pd

# Load path-generator.py by file path (hyphens are not valid in module names)
_spec = importlib.util.spec_from_file_location(
    "path_generator", Path(__file__).parent / "path-generator.py"
)
_pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pg)
ExperimentConfig = _pg.ExperimentConfig
run_experiment   = _pg.run_experiment

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
    "kribi_douala":  {"Kribi":  (2.940594, 9.910192),
                      "Douala": (4.0511,   9.7679)},
}

friction_layer_choices = {
    "base": DATA_DIR / "cmr-construction-cost-friction-90m/cmr_friction_90m_clipped.tif",
}

spatial_mask_choices = {
    "protected_areas": DATA_DIR / "processed-protected-areas/cmr-protected-areas.gpkg",
    "no_mask":         None,
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
