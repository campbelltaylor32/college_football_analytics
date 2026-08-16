#!/usr/bin/env python
"""Thin end-to-end sequencer -- runs the full pipeline stage-by-stage by calling into the
already-built stage scripts' main() functions. No new logic lives here.

Stages: inspect -> build_dataset -> quality_gate -> eda -> train -> evaluate -> predict

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --from-stage eda --to-stage predict
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
import generate_predictions as generate_predictions_script
import inspect_database as inspect_database_script
import run_eda as run_eda_script
import train_models as train_models_script

from cfb_win_total_model import data_validation as dv
from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.database import get_engine
from cfb_win_total_model.targets import get_fbs_teams_by_season
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_DATA_INVENTORY, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"

STAGE_ORDER = ["inspect", "build_dataset", "quality_gate", "eda", "train", "evaluate", "predict"]


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
    """Re-runs the integrity checks against the BUILT modeling table (not just the raw DB):
    row counts per season should match the verified FBS team counts, no duplicate
    (school, season) keys, and no unexpected nulls in non-optional columns."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} missing -- build_dataset stage did not run")
    df = pd.read_parquet(DATASET_PATH)

    if df.duplicated(subset=["school", "season"]).any():
        raise AssertionError("Quality gate FAILED: duplicate (school, season) rows in modeling dataset")

    required_non_null = ["school", "season", "regular_season_wins", "regular_season_losses", "scheduled_games"]
    for col in required_non_null:
        if df[col].isna().any():
            raise AssertionError(f"Quality gate FAILED: unexpected nulls in required column '{col}'")

    engine = get_engine()
    mismatches = []
    for season in sorted(df["season"].unique()):
        expected = len(get_fbs_teams_by_season(engine, int(season)))
        actual = (df["season"] == season).sum()
        if actual != expected:
            mismatches.append((season, expected, actual))
    if mismatches:
        logger.warning(f"Quality gate: row-count mismatches vs FBS team counts: {mismatches}")
    else:
        logger.info("Quality gate PASSED: dataset row counts match FBS team counts per season, no dup keys, no unexpected nulls")


def stage_eda(args) -> None:
    run_eda_script.main()


def stage_train(args) -> None:
    train_models_script.main()


def stage_evaluate(args) -> None:
    evaluate_models_script.main()


def stage_predict(args) -> None:
    generate_predictions_script.main()


STAGE_FUNCS = {
    "inspect": stage_inspect,
    "build_dataset": stage_build_dataset,
    "quality_gate": stage_quality_gate,
    "eda": stage_eda,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "predict": stage_predict,
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

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
