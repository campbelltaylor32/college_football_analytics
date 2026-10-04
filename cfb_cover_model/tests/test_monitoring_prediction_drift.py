import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.monitoring.prediction_drift import evaluate_completed_week


def _predictions_df(agreement_flags, probs):
    n = len(agreement_flags)
    return pd.DataFrame(
        {
            "game_id": range(1, n + 1),
            "agreement_bet": agreement_flags,
            "logistic_regression_probability": probs,
            "xgboost_regressor_probability": probs,
        }
    )


def _outcomes_df(home_covered):
    n = len(home_covered)
    return pd.DataFrame({"game_id": range(1, n + 1), "home_covered": home_covered})


def test_precision_on_agreement_subset_is_exact(tmp_path):
    # 5 games: agreement_bet True for games 1,2,4. Of those, games 1 and 4 covered (correct),
    # game 2 didn't (incorrect) -> precision = 2/3.
    predictions = _predictions_df(
        agreement_flags=[True, True, False, True, False],
        probs=[0.7, 0.6, 0.3, 0.65, 0.2],
    )
    outcomes = _outcomes_df([1, 0, 0, 1, 1])

    summary = evaluate_completed_week(
        predictions, outcomes, classifier_threshold=0.56, regressor_threshold=0.60,
        season=2025, week=1, rolling_log_path=tmp_path / "log.csv",
    )

    assert summary["n_agreement_bets"] == 3
    assert summary["precision_this_week"] == 2 / 3
    assert summary["n_games_graded"] == 5
    assert summary["n_ungraded"] == 0


def test_ungraded_games_are_dropped_and_counted(tmp_path):
    predictions = _predictions_df(agreement_flags=[True, True, False], probs=[0.7, 0.6, 0.3])
    outcomes = _outcomes_df([1])  # only game_id=1 has a known outcome
    # patch game_ids so only one overlaps
    predictions["game_id"] = [1, 2, 3]

    summary = evaluate_completed_week(
        predictions, outcomes, classifier_threshold=0.56, regressor_threshold=0.60,
        season=2025, week=1, rolling_log_path=tmp_path / "log.csv",
    )

    assert summary["n_games_graded"] == 1
    assert summary["n_ungraded"] == 2


def test_no_agreement_bets_gives_none_precision_not_error(tmp_path):
    predictions = _predictions_df(agreement_flags=[False, False], probs=[0.3, 0.2])
    outcomes = _outcomes_df([0, 1])

    summary = evaluate_completed_week(
        predictions, outcomes, classifier_threshold=0.56, regressor_threshold=0.60,
        season=2025, week=1, rolling_log_path=tmp_path / "log.csv",
    )

    assert summary["n_agreement_bets"] == 0
    assert summary["precision_this_week"] is None


def test_rolling_mean_averages_across_weeks_and_alerts_below_baseline(tmp_path):
    log_path = tmp_path / "log.csv"

    # Week 1: precision 1.0 (both agreement bets correct)
    evaluate_completed_week(
        _predictions_df([True, True], [0.7, 0.7]), _outcomes_df([1, 1]),
        0.56, 0.60, season=2025, week=1, rolling_log_path=log_path,
        window_weeks=4, alert_delta=0.10, baseline_precision=0.6,
    )
    # Week 2: precision 0.0 (both agreement bets wrong)
    summary = evaluate_completed_week(
        _predictions_df([True, True], [0.7, 0.7]), _outcomes_df([0, 0]),
        0.56, 0.60, season=2025, week=2, rolling_log_path=log_path,
        window_weeks=4, alert_delta=0.10, baseline_precision=0.6,
    )

    assert summary["rolling_mean_precision"] == 0.5  # mean of 1.0 and 0.0
    assert summary["delta_vs_baseline"] == pytest.approx(0.5 - 0.6)
    assert summary["alert"] is False  # -0.1 delta is not < -0.10 alert_delta

    log = pd.read_csv(log_path)
    assert len(log) == 2
    assert set(log["week"]) == {1, 2}


def test_rerunning_same_week_is_idempotent_not_duplicated(tmp_path):
    log_path = tmp_path / "log.csv"
    for _ in range(2):
        evaluate_completed_week(
            _predictions_df([True], [0.7]), _outcomes_df([1]),
            0.56, 0.60, season=2025, week=1, rolling_log_path=log_path,
        )
    log = pd.read_csv(log_path)
    assert len(log) == 1
