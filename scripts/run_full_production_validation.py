#!/usr/bin/env python3
"""Run the resumable production-validation workflow.

Examples from the repository root, after activating the project environment:
    python scripts/run_full_production_validation.py --stage preflight
    python scripts/run_full_production_validation.py --stage mode-library
    python scripts/run_full_production_validation.py --stage realization
    python scripts/run_full_production_validation.py --stage fields --grid 0
    python scripts/run_full_production_validation.py --stage analysis
    python scripts/run_full_production_validation.py --stage compare

The full publication-size run is deliberate and may be expensive.  Ordinary
unit validation is handled by scripts/check_full_production_validation.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required; activate the project environment first.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from azlens.production_validation import STAGES, make_run_context, run_stages, smoke_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/production_validation.yaml", help="Workflow YAML/JSON configuration")
    parser.add_argument("--run-id", default=None, help="Override run identifier; default comes from config or UTC timestamp")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow a dirty Git tree for development runs")
    parser.add_argument("--smoke", action="store_true", help="Use small reference-state smoke configuration instead of the production YAML")
    parser.add_argument("--stage", action="append", choices=[*STAGES, "all"], required=True,
                        help="Stage to run; repeatable. Use --stage all for all stages.")
    parser.add_argument("--grid", type=int, action="append", help="Restrict fields stage to selected grid(s)")
    args = parser.parse_args()

    if args.smoke:
        # Create a context from the built-in smoke configuration by writing it as an in-memory override.
        context = make_run_context(ROOT, config_path=None, run_id=args.run_id or "smoke", output_root=args.output_root, allow_dirty=True)
        context = type(context)(context.repository_root, context.run_dir, smoke_config(), None, True)
    else:
        context = make_run_context(ROOT, config_path=args.config, run_id=args.run_id, output_root=args.output_root, allow_dirty=args.allow_dirty)

    stages = []
    for stage in args.stage:
        if stage == "all":
            stages.extend(STAGES)
        else:
            stages.append(stage)
    records = run_stages(context, stages, grid_indices=args.grid)
    print(f"Run directory: {context.run_dir}")
    for record in records:
        print(f"{record['stage']}: {record['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
