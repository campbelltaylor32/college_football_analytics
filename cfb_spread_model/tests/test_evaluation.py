from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb_spread_model.modeling.evaluation import (
    bucket_rank_monotonicity,
    generalization_gap,
    probabilistic_fit_metrics,
    top_vs_bottom_summary,
)


def _bucket_df(actual_rates: list[float], n_per_bucket: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bucket": range(len(actual_rates)),
            "n": [n_per_bucket] * len(actual_rates),
            "mean_predicted": np.linspace(0.1, 0.9, len(actual_rates)),
            "actual_cover_rate": actual_rates,
        }
    )


def test_bucket_rank_monotonicity_perfectly_increasing():
    df = _bucket_df([0.2, 0.35, 0.5, 0.65, 0.8])
    assert bucket_rank_monotonicity(df) == pytest.approx(1.0)


def test_bucket_rank_monotonicity_perfectly_decreasing():
    df = _bucket_df([0.8, 0.65, 0.5, 0.35, 0.2])
    assert bucket_rank_monotonicity(df) == pytest.approx(-1.0)


def test_bucket_rank_monotonicity_no_relationship():
    """A bucket order with no consistent trend should score well below the perfect cases."""
    df = _bucket_df([0.5, 0.2, 0.8, 0.3, 0.5])
    monotonic_score = bucket_rank_monotonicity(df)
    assert monotonic_score < 1.0
    assert abs(monotonic_score) < 1.0


def test_bucket_rank_monotonicity_single_bucket_is_nan():
    df = _bucket_df([0.5])
    assert np.isnan(bucket_rank_monotonicity(df))


def test_top_vs_bottom_summary_matches_hand_calculation():
    df = _bucket_df([0.20, 0.50, 0.80])
    y_true_overall = np.array([1] * 49 + [0] * 51)  # base rate = 0.49
    summary = top_vs_bottom_summary(df, y_true_overall)

    assert summary["overall_base_rate"] == pytest.approx(0.49)
    assert summary["top_bucket_actual_rate"] == pytest.approx(0.80)
    assert summary["top_bucket_lift_vs_base_rate"] == pytest.approx(0.80 - 0.49)
    assert summary["bottom_bucket_actual_cover_rate"] == pytest.approx(0.20)
    assert summary["bottom_bucket_other_side_rate"] == pytest.approx(0.80)
    assert summary["bottom_bucket_other_side_lift_vs_base_rate"] == pytest.approx(0.80 - (1 - 0.49))
    assert summary["monotonicity"] == pytest.approx(1.0)
    assert summary["n_buckets"] == 3
    assert summary["top_bucket_n"] == 20
    assert summary["bottom_bucket_n"] == 20


def test_top_vs_bottom_summary_flags_inverted_ranking_with_negative_monotonicity():
    """If the model's "top" bucket actually covers less than its "bottom" bucket, that should
    show up as negative monotonicity -- a red flag this function must surface, not hide."""
    df = _bucket_df([0.80, 0.50, 0.20])  # inverted: lowest predicted prob has the HIGHEST actual rate
    y_true_overall = np.array([1] * 50 + [0] * 50)
    summary = top_vs_bottom_summary(df, y_true_overall)
    assert summary["monotonicity"] == pytest.approx(-1.0)
    assert summary["top_bucket_actual_rate"] < summary["bottom_bucket_actual_cover_rate"]


def test_probabilistic_fit_metrics_lower_log_loss_for_confident_correct_predictions():
    y_true = np.array([1, 1, 0, 0])
    confident_correct = np.array([0.95, 0.95, 0.05, 0.05])
    unsure = np.array([0.55, 0.55, 0.45, 0.45])

    confident_metrics = probabilistic_fit_metrics(y_true, confident_correct)
    unsure_metrics = probabilistic_fit_metrics(y_true, unsure)

    assert confident_metrics["log_loss"] < unsure_metrics["log_loss"]
    assert confident_metrics["average_precision"] >= unsure_metrics["average_precision"]


def test_generalization_gap_positive_when_reference_looks_better():
    train_metrics = {"precision": 0.90, "roc_auc": 0.95, "log_loss": 0.10}
    holdout_metrics = {"precision": 0.50, "roc_auc": 0.55, "log_loss": 0.80}

    gaps = generalization_gap(train_metrics, holdout_metrics, keys=("precision", "roc_auc", "log_loss"))

    # Higher-is-better metrics: positive gap means train looks better, as here.
    assert gaps["precision_gap"] == pytest.approx(0.40)
    assert gaps["roc_auc_gap"] == pytest.approx(0.40)
    # log_loss is lower-is-better: train's lower log_loss (0.10 < 0.80) should STILL produce a
    # positive gap under this function's "positive = reference looks better" convention.
    assert gaps["log_loss_gap"] == pytest.approx(0.70)


def test_generalization_gap_skips_missing_keys():
    gaps = generalization_gap({"precision": 0.5}, {"recall": 0.5}, keys=("precision", "recall", "roc_auc"))
    assert gaps == {}
