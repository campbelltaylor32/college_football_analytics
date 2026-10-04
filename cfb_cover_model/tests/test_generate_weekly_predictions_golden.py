"""Characterization test for scripts/generate_weekly_predictions.py's real output, run
against real local data when available (data/processed/modeling_dataset.parquet and
outputs/threshold_selection/chosen_threshold_per_model.csv are both gitignored, so this
never runs in a fresh clone or CI - it's a local safety net for the day
generate_weekly_predictions.py gets refactored to call serving.artifact/serving.scoring
instead of duplicating their logic, per docs/serving_and_monitoring.md's "why the script
wasn't refactored yet" note).

If you're refactoring generate_weekly_predictions.py: run this test BEFORE your change to
capture today's output, then again AFTER to confirm serving.artifact.train_production_artifact
+ serving.scoring.score_week produce byte-for-byte-equivalent agreement_bet/flag columns for
a real week (probabilities may drift slightly since a freshly-trained artifact and a live
refit are two different model fits on the same data, not the same fit reused - only
agreement_bet and the two *_flag columns are asserted here for that reason).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet"
THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
WEEK_CSV = PROJECT_ROOT.parent / "Data" / "CFB_Pred_Week_12.csv"

requires_local_data = pytest.mark.skipif(
    not (DATASET_PATH.exists() and THRESHOLD_PATH.exists() and WEEK_CSV.exists()),
    reason="Local training data / week file not present (gitignored) - this test only runs "
    "against a real local checkout with the pipeline already run once.",
)


@requires_local_data
def test_generate_weekly_predictions_produces_sane_output(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_weekly_predictions.py"), "--week", "12"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    out_path = PROJECT_ROOT / "outputs" / "predictions" / "week_12_dual_model_predictions.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)

    expected_columns = {
        "game_id", "home_team", "away_team", "season", "week", "spread",
        "logistic_regression_probability", "logistic_regression_flag",
        "xgboost_regressor_probability", "xgboost_regressor_flag",
        "agreement_bet", "avg_probability",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) > 0
    assert (df["agreement_bet"] == (df["logistic_regression_flag"] & df["xgboost_regressor_flag"])).all()
    assert df["logistic_regression_probability"].between(0, 1).all()
    assert df["xgboost_regressor_probability"].between(0, 1).all()
