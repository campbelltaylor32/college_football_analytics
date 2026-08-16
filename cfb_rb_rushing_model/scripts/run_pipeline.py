#!/usr/bin/env python
"""Thin end-to-end sequencer -- runs the full pipeline stage-by-stage by calling into the
already-built stage scripts' main() functions. No new logic lives here.

Stages: inspect -> build_dataset -> quality_gate -> eda -> train -> evaluate

`predict` (scripts/generate_week_predictions.py) is intentionally NOT part of the default
stage list -- unlike the sibling projects' single-target-season `predict` stage, this
project's weekly inference needs an explicit --season/--week and is meant to be re-run once
per week during the season, not as part of a full historical rebuild.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --from-stage eda --to-stage evaluate
    python scripts/run_pipeline.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import build_modeling_dataset as build_modeling_dataset_script
import evaluate_models as evaluate_models_script
import inspect_database as inspect_database_script
import run_eda as run_eda_script
import train_models as train_models_script

from cfb_rb_rushing_model.config import load_modeling_config
from cfb_rb_rushing_model.utils.logging import get_logger
from cfb_rb_rushing_model.utils.paths import DATA_PROCESSED_DIR, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

STAGE_ORDER = ["inspect", "build_dataset", "quality_gate", "eda", "train", "evaluate"]


def stage_inspect(args) -> None:
    rc = inspect_database_script.main()
    if rc != 0:
        raise RuntimeError("Database inspection failed (duplicate-key integrity violation)")


def stage_build_dataset(args) -> None:
    argv_backup = sys.argv
    sys.argv = ["build_modeling_dataset.py"] + (["--rebuild"] if args.rebuild else [])
    try:
        build_modeling_dataset_script.main()
    finally:
        sys.argv = argv_backup


def stage_quality_gate(args) -> None:
    """Re-runs integrity checks against the BUILT modeling table (not just the raw DB): no
    duplicate (athlete_id, game_id) keys, no unexpected nulls in required non-feature
    columns, and the target's realized-vs-zero-filled split is within a sane range."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- build_dataset stage did not run")
    df = pd.read_parquet(DATASET_PATH)

    if df.duplicated(subset=["athlete_id", "game_id"]).any():
        raise AssertionError("Quality gate FAILED: duplicate (athlete_id, game_id) rows in modeling dataset")

    required_non_null = ["athlete_id", "game_id", "team", "opponent", "season", "week", "rushing_yards"]
    for col in required_non_null:
        if df[col].isna().any():
            raise AssertionError(f"Quality gate FAILED: unexpected nulls in required column '{col}'")

    zero_carry_share = float((~df["played"]).mean()) if "played" in df.columns else None
    if zero_carry_share is not None and zero_carry_share > 0.5:
        logger.warning(f"Quality gate WARNING: {zero_carry_share:.1%} of eligible rows had zero realized carries -- unexpectedly high, check eligibility thresholds")
    else:
        logger.info(f"Quality gate PASSED: no dup keys, no unexpected nulls, zero-carry share={zero_carry_share:.1%}" if zero_carry_share is not None else "Quality gate PASSED: no dup keys, no unexpected nulls")


def stage_eda(args) -> None:
    run_eda_script.main()


def stage_train(args) -> None:
    train_models_script.main()


def stage_evaluate(args) -> None:
    evaluate_models_script.main()


STAGE_FUNCS = {
    "inspect": stage_inspect,
    "build_dataset": stage_build_dataset,
    "quality_gate": stage_quality_gate,
    "eda": stage_eda,
    "train": stage_train,
    "evaluate": stage_evaluate,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, help="Run only this single stage")
    parser.add_argument("--from-stage", choices=STAGE_ORDER, help="Start from this stage")
    parser.add_argument("--to-stage", choices=STAGE_ORDER, help="Stop after this stage")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the modeling dataset cache")
    args = parser.parse_args()

    ensure_dirs()
    load_modeling_config()  # fail fast if config/.env is misconfigured

    if args.stage:
        stages = [args.stage]
    else:
        start = STAGE_ORDER.index(args.from_stage) if args.from_stage else 0
        end = STAGE_ORDER.index(args.to_stage) if args.to_stage else len(STAGE_ORDER) - 1
        stages = STAGE_ORDER[start : end + 1]

    logger.info(f"Running pipeline stages: {stages}")
    for stage in stages:
        logger.info(f"=== Stage: {stage} ===")
        STAGE_FUNCS[stage](args)

    logger.info("Pipeline complete. Run scripts/generate_week_predictions.py --season <S> --week <N> for weekly inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
