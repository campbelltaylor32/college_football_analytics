"""Prediction/calibration drift - no Evidently here, since this is really just "grade last
week's predictions now that outcomes are known" plus a rolling trend, not a distributional
comparison. Reuses the same precision/calibration functions the offline evaluation pipeline
already uses (modeling/evaluation.py), so "how we score a week in production" and "how we
scored candidate models during development" are the same code.

IMPORTANT: agreement-bet volume is small by design (~5-15 games/week on holdout evidence,
sometimes 0) - a single week's precision is not a reliable signal on its own. The
rolling-window mean (PREDICTION_DRIFT_WINDOW_WEEKS) is the number worth alerting on;
single-week precision is reported for visibility, not as a trigger.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from cfb_cover_model.modeling.evaluation import calibration_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTION_DRIFT_DIR = PROJECT_ROOT / "outputs" / "monitoring" / "prediction_drift"
ROLLING_LOG_PATH = PREDICTION_DRIFT_DIR / "rolling_precision_log.csv"

CLASSIFIER_MODEL = "logistic_regression"
REGRESSOR_MODEL = "xgboost_regressor"

DEFAULT_WINDOW_WEEKS = int(os.environ.get("PREDICTION_DRIFT_WINDOW_WEEKS", 4))
DEFAULT_ALERT_DELTA = float(os.environ.get("PREDICTION_DRIFT_ALERT_DELTA", 0.10))


def evaluate_completed_week(
    predictions_df: pd.DataFrame,
    realized_outcomes_df: pd.DataFrame,
    classifier_threshold: float,
    regressor_threshold: float,
    *,
    season: int,
    week: int,
    baseline_precision: float = 0.606,
    window_weeks: int = DEFAULT_WINDOW_WEEKS,
    alert_delta: float = DEFAULT_ALERT_DELTA,
    rolling_log_path: Path = ROLLING_LOG_PATH,
) -> dict:
    """`realized_outcomes_df` must have `game_id` and `home_covered` (0/1, pushes already
    excluded/labeled upstream the same way the training data is). Joins on game_id - any
    predicted game with no matching realized outcome is dropped from grading (and counted),
    rather than silently treated as a miss."""
    merged = predictions_df.merge(
        realized_outcomes_df[["game_id", "home_covered"]], on="game_id", how="inner"
    )
    n_ungraded = len(predictions_df) - len(merged)

    agreement = merged[merged["agreement_bet"]]
    n_agreement_bets = len(agreement)
    precision_this_week = float(agreement["home_covered"].mean()) if n_agreement_bets > 0 else None

    classifier_calibration = calibration_report(merged["home_covered"], merged[f"{CLASSIFIER_MODEL}_probability"])
    regressor_calibration = calibration_report(merged["home_covered"], merged[f"{REGRESSOR_MODEL}_probability"])

    rolling_log_path = Path(rolling_log_path)
    rolling_log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(rolling_log_path) if rolling_log_path.exists() else pd.DataFrame(
        columns=["season", "week", "n_agreement_bets", "precision_this_week"]
    )
    log = log[~((log["season"] == season) & (log["week"] == week))]  # idempotent re-runs
    new_row = pd.DataFrame([{"season": season, "week": week, "n_agreement_bets": n_agreement_bets, "precision_this_week": precision_this_week}])
    log = pd.concat([log, new_row], ignore_index=True).sort_values(["season", "week"])

    recent = log.tail(window_weeks)
    graded_recent = recent.dropna(subset=["precision_this_week"])
    rolling_mean_precision = float(graded_recent["precision_this_week"].mean()) if len(graded_recent) else None
    delta_vs_baseline = (rolling_mean_precision - baseline_precision) if rolling_mean_precision is not None else None
    alert = delta_vs_baseline is not None and delta_vs_baseline < -alert_delta

    log["rolling_mean_precision"] = None
    log.loc[log.index[-1], "rolling_mean_precision"] = rolling_mean_precision
    log["baseline_precision"] = baseline_precision
    log["delta_vs_baseline"] = None
    log.loc[log.index[-1], "delta_vs_baseline"] = delta_vs_baseline
    log.to_csv(rolling_log_path, index=False)

    summary = {
        "season": season,
        "week": week,
        "n_games_predicted": len(predictions_df),
        "n_games_graded": len(merged),
        "n_ungraded": n_ungraded,
        "n_agreement_bets": n_agreement_bets,
        "precision_this_week": precision_this_week,
        "rolling_mean_precision": rolling_mean_precision,
        "rolling_window_weeks": window_weeks,
        "baseline_precision": baseline_precision,
        "delta_vs_baseline": delta_vs_baseline,
        "alert": alert,
        "alert_threshold_delta": -alert_delta,
        "calibration": {
            CLASSIFIER_MODEL: {"brier_score": classifier_calibration["brier_score"], "roc_auc": classifier_calibration["roc_auc"]},
            REGRESSOR_MODEL: {"brier_score": regressor_calibration["brier_score"], "roc_auc": regressor_calibration["roc_auc"]},
        },
    }

    PREDICTION_DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    (PREDICTION_DRIFT_DIR / f"{season}_week_{week}_prediction_drift.json").write_text(json.dumps(summary, indent=2))

    return summary
