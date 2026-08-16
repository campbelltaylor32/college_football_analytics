#!/usr/bin/env python
"""Chains the pipeline stages end-to-end. Each stage is a subprocess invocation of the
corresponding scripts/<stage>.py, run with this project's own venv Python so imports
resolve the same way as running each script directly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    "load_and_validate_dataset",
    "run_eda",
    "select_features",
    "train_models",
    "evaluate_models",
    "analyze_train_vs_holdout",
    "analyze_feature_stability",
    "explain_model",
]

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(name: str) -> None:
    script_path = SCRIPTS_DIR / f"{name}.py"
    print(f"\n{'=' * 80}\n[run_pipeline] Stage: {name}\n{'=' * 80}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        raise SystemExit(f"[run_pipeline] Stage {name!r} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default=None, help="Run only this single stage.")
    parser.add_argument("--from-stage", type=str, default=None, help="Start from this stage (inclusive).")
    parser.add_argument("--to-stage", type=str, default=None, help="Stop after this stage (inclusive).")
    args = parser.parse_args()

    if args.stage:
        if args.stage not in STAGES:
            raise SystemExit(f"Unknown stage {args.stage!r}. Valid stages: {STAGES}")
        run_stage(args.stage)
        return

    start = STAGES.index(args.from_stage) if args.from_stage else 0
    end = STAGES.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
    for stage in STAGES[start:end]:
        run_stage(stage)

    print(f"\n{'=' * 80}\n[run_pipeline] Done. See outputs/model_comparison/final_summary.json\n{'=' * 80}")


if __name__ == "__main__":
    main()
