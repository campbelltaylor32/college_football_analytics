#!/usr/bin/env python
"""Thin end-to-end sequencer -- runs the full pipeline stage-by-stage by calling into the
already-built stage scripts' main() functions. No new logic lives here.

Stages: load -> eda -> select_features -> train -> evaluate -> explain -> calibration -> generalization -> holdout_comparison -> predict

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

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_rank_calibration as analyze_rank_calibration_script
import analyze_train_vs_holdout as analyze_train_vs_holdout_script
import compare_models_on_holdout as compare_models_on_holdout_script
import evaluate_models as evaluate_models_script
import explain_model as explain_model_script
import load_and_validate_dataset as load_and_validate_dataset_script
import run_eda as run_eda_script
import select_features as select_features_script
import train_models as train_models_script

from cfb_spread_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import ensure_dirs

logger = get_logger(__name__)

STAGE_ORDER = [
    "load",
    "eda",
    "select_features",
    "train",
    "evaluate",
    "explain",
    "calibration",
    "generalization",
    "holdout_comparison",
    "predict",
]


def stage_load(args) -> None:
    argv_backup = sys.argv
    sys.argv = ["load_and_validate_dataset.py"] + (["--rebuild"] if args.rebuild else [])
    try:
        load_and_validate_dataset_script.main()
    finally:
        sys.argv = argv_backup


def stage_eda(args) -> None:
    run_eda_script.main()


def stage_select_features(args) -> None:
    select_features_script.main()


def stage_train(args) -> None:
    train_models_script.main()


def stage_evaluate(args) -> None:
    evaluate_models_script.main()


def stage_explain(args) -> None:
    explain_model_script.main()


def stage_calibration(args) -> None:
    analyze_rank_calibration_script.main()


def stage_generalization(args) -> None:
    analyze_train_vs_holdout_script.main()


def stage_holdout_comparison(args) -> None:
    compare_models_on_holdout_script.main()


def stage_predict(args) -> None:
    logger.info(
        "'predict' has no default target week -- once a production model exists (produced by "
        "the 'evaluate' stage), run `python scripts/generate_week_predictions.py --week <N>` directly."
    )


STAGE_FUNCS = {
    "load": stage_load,
    "eda": stage_eda,
    "select_features": stage_select_features,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "explain": stage_explain,
    "calibration": stage_calibration,
    "generalization": stage_generalization,
    "holdout_comparison": stage_holdout_comparison,
    "predict": stage_predict,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, help="Run only this single stage")
    parser.add_argument("--from-stage", choices=STAGE_ORDER, help="Start from this stage")
    parser.add_argument("--to-stage", choices=STAGE_ORDER, help="Stop after this stage")
    parser.add_argument("--rebuild", action="store_true", help="Force reload of the CSV cache (load stage)")
    args = parser.parse_args()

    ensure_dirs()
    load_data_config()
    load_features_config()
    load_modeling_config()  # fail fast if config is misconfigured

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
