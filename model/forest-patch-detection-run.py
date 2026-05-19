"""
Forest patch detection — experiment runner.

Edit the EXPERIMENTS list below to configure which combinations to run.
Each entry is a PatchConfig. Results are saved to OUTPUT_DIR.

Usage
-----
python forest-patch-detection-run.py
python forest-patch-detection-run.py --dry-run
"""

import argparse
import importlib.util
from pathlib import Path

# Load forest-patch-detection.py by file path (hyphens are not valid in module names)
_spec = importlib.util.spec_from_file_location(
    "forest_patch_detection", Path(__file__).parent / "forest-patch-detection.py"
)
_fpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fpd)
PatchConfig         = _fpd.PatchConfig
run_patch_detection = _fpd.run_patch_detection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent

TREECOVER_PATH = _REPO_ROOT / "data/output/processed-hansen-treecover/cameroon_treecover2024_30m.tif"
OUTPUT_DIR     = _REPO_ROOT / "data/output/forest-patch-detection"

# ---------------------------------------------------------------------------
# Experiment matrix
# Edit this list to add, remove, or adjust runs.
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # Baseline — no road burning
    PatchConfig(
        treecover_path=TREECOVER_PATH,
        prefix="no_action",
        output_dir=OUTPUT_DIR,
        forest_threshold=10,
        min_patch_size_ha=1.0,
        admin_level=2,
        burn_paved_roads=False,
        burn_unpaved_roads=False,
    ),
    # Paved roads as barriers
    PatchConfig(
        treecover_path=TREECOVER_PATH,
        prefix="no_action",
        output_dir=OUTPUT_DIR,
        forest_threshold=10,
        min_patch_size_ha=1.0,
        admin_level=2,
        burn_paved_roads=True,
        burn_unpaved_roads=False,
    ),
    # All roads as barriers
    PatchConfig(
        treecover_path=TREECOVER_PATH,
        prefix="no_action",
        output_dir=OUTPUT_DIR,
        forest_threshold=10,
        min_patch_size_ha=1.0,
        admin_level=2,
        burn_paved_roads=True,
        burn_unpaved_roads=True,
    ),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run forest patch detection experiments")
    p.add_argument("--dry-run", action="store_true",
                   help="List experiments without running them")
    args = p.parse_args()

    print(f"Forest patch detection — {len(EXPERIMENTS)} experiment(s)")

    for i, config in enumerate(EXPERIMENTS, 1):
        road_types = []
        if config.burn_paved_roads:
            road_types.append("paved")
        if config.burn_unpaved_roads:
            road_types.append("unpaved")
        roads_label = ", ".join(road_types) if road_types else "none"
        print(f"  [{i}] thresh={config.forest_threshold}%  "
              f"min_patch={config.min_patch_size_ha}ha  "
              f"adm={config.admin_level}  "
              f"roads={roads_label}")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    print()
    for config in EXPERIMENTS:
        run_patch_detection(config)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
