from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb_spread_model.modeling.importance import (
    build_feature_importance_report,
    compute_gain_importance,
    summarize_by_temporal_transform,
)

_FEATURES = [
    "home_prev_week_total_yards",  # prev_week
    "home_total_yards_avg_all",  # avg_all
    "home_total_yards_avg3",  # avg3
    "home_talent",  # non_temporal
    "spread",  # context (not home_/away_ prefixed)
]


def _make_gain_importance() -> pd.Series:
    return pd.Series([0.4, 0.3, 0.1, 0.15, 0.05], index=_FEATURES, name="gain_importance")


def _make_permutation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": _FEATURES,
            "permutation_importance_mean": [0.02, 0.05, 0.01, 0.015, 0.005],
            "permutation_importance_std": [0.001] * 5,
        }
    )


def test_compute_gain_importance_sorts_descending():
    class FakeModel:
        feature_importances_ = np.array([0.4, 0.3, 0.1, 0.15, 0.05])

    result = compute_gain_importance(FakeModel(), _FEATURES)
    assert list(result.index)[:2] == ["home_prev_week_total_yards", "home_total_yards_avg_all"]
    assert result.iloc[0] == pytest.approx(0.4)


def test_compute_gain_importance_raises_without_feature_importances():
    class NoImportanceModel:
        pass

    with pytest.raises(AttributeError):
        compute_gain_importance(NoImportanceModel(), _FEATURES)


def test_build_feature_importance_report_categorizes_every_feature():
    report = build_feature_importance_report(_make_gain_importance(), _make_permutation_df())
    categories = dict(zip(report["feature"], report["temporal_transform"]))
    assert categories["home_prev_week_total_yards"] == "prev_week"
    assert categories["home_total_yards_avg_all"] == "avg_all"
    assert categories["home_total_yards_avg3"] == "avg3"
    assert categories["home_talent"] == "non_temporal"
    assert categories["spread"] == "context"

    sides = dict(zip(report["feature"], report["side"]))
    assert sides["home_prev_week_total_yards"] == "home"
    assert pd.isna(sides["spread"])  # pandas may store the None as NaN once column dtype is inferred


def test_build_feature_importance_report_sorted_by_permutation_importance():
    report = build_feature_importance_report(_make_gain_importance(), _make_permutation_df())
    assert report["permutation_importance_mean"].is_monotonic_decreasing
    assert report.iloc[0]["feature"] == "home_total_yards_avg_all"  # highest perm importance (0.05)


def test_summarize_by_temporal_transform_counts_and_percentages():
    report = build_feature_importance_report(_make_gain_importance(), _make_permutation_df())
    summary = summarize_by_temporal_transform(report)

    counts = dict(zip(summary["temporal_transform"], summary["n_features"]))
    assert counts == {"prev_week": 1, "avg_all": 1, "avg3": 1, "non_temporal": 1, "context": 1}

    # Percentages of features across all categories should sum to 1.
    assert summary["pct_of_features"].sum() == pytest.approx(1.0)
    # Percentages of (non-negative-clipped) permutation importance mass should sum to 1.
    assert summary["pct_of_permutation_importance"].sum() == pytest.approx(1.0)


def test_summarize_by_temporal_transform_handles_missing_gain_importance():
    """When gain importance isn't available (e.g. a non-tree model), build_feature_importance_report
    still runs (gain_importance passed as an empty Series) and the rollup shouldn't crash --
    it should report 0 gain-importance mass rather than raising."""
    empty_gain = pd.Series(dtype=float, name="gain_importance")
    report = build_feature_importance_report(empty_gain, _make_permutation_df())
    summary = summarize_by_temporal_transform(report)
    assert (summary["pct_of_gain_importance"] == 0).all()
    assert summary["pct_of_permutation_importance"].sum() == pytest.approx(1.0)
