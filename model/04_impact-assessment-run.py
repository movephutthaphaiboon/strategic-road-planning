"""
04-impact-assessment-run.py — Batch runner for impact assessment.

Defines which experiment folders to assess and calls run_assessment()
for every .gpkg file found in each folder.

Outputs go to results/impact-assessment/<folder_name>/ (created if absent).

Usage:
    python 04-impact-assessment-run.py
"""

from pathlib import Path
import sys

# Make sure impact_assessment.py is importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from impact_assessment import run_assessment

# =============================================================================
# CONFIGURATION — add / remove experiment folders here
# =============================================================================

# Each entry: (folder_name, downsample_factor)
#   folder_name   : subfolder under results/least-cost-paths/
#   downsample    : 1 = native ~90 m, 5 = ~450 m (faster, less memory)
EXPERIMENTS = [
    ("experiment_00", 1),
]

# =============================================================================

_model_dir  = Path(__file__).parent
_paths_root = _model_dir / "results/least-cost-paths"
_out_root   = _model_dir / "results/impact-assessment"

for folder_name, downsample in EXPERIMENTS:
    paths_dir  = _paths_root / folder_name
    gpkg_files = sorted(paths_dir.glob("*.gpkg"))

    if not paths_dir.exists():
        print(f"[SKIP] Folder not found: {paths_dir}")
        continue
    if not gpkg_files:
        print(f"[SKIP] No .gpkg files in: {paths_dir}")
        continue

    out_dir = _out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Experiment: {folder_name}  ({len(gpkg_files)} scenario(s))")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*62}")

    for i, gpkg_fp in enumerate(gpkg_files, 1):
        print(f"\n{'━'*62}")
        print(f"  [{i}/{len(gpkg_files)}] {gpkg_fp.name}")
        print(f"{'━'*62}")
        run_assessment(gpkg_fp, downsample=downsample, out_dir=out_dir)
