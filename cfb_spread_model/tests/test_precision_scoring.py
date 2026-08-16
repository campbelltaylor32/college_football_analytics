from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score

from cfb_spread_model.feature_selection.precision_scoring import (
    coverage,
    coverage_threshold,
    precision_at_coverage_floor,
    precision_at_coverage_floor_scorer,
    precision_at_threshold,
)


def test_coverage_threshold_flags_at_least_the_floor_fraction():
    y_score = np.linspace(0, 1, 100)
    threshold = coverage_threshold(y_score, min_coverage=0.1)
    flagged = (y_score >= threshold).sum()
    assert flagged >= 10


def test_precision_at_coverage_floor_matches_manual_computation():
    y_true = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # min_coverage=0.3 on n=10 -> top 3 rows flagged -> y_true[:3] = [1,1,1] -> precision 1.0
    assert precision_at_coverage_floor(y_true, y_score, min_coverage=0.3) == 1.0


def test_precision_at_coverage_floor_zero_when_no_positives_at_threshold():
    y_true = np.zeros(10)
    y_score = np.linspace(0, 1, 10)
    assert precision_at_coverage_floor(y_true, y_score, min_coverage=0.2) == 0.0


def test_precision_at_threshold_matches_sklearn():
    y_true = np.array([1, 0, 1, 1, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.6, 0.1])
    threshold = 0.5
    expected = precision_score(y_true, (y_score >= threshold).astype(int))
    assert precision_at_threshold(y_true, y_score, threshold) == expected


def test_coverage_fraction():
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    assert coverage(y_score, threshold=0.5) == 0.6


def test_scorer_factory_matches_sklearn_estimator_protocol():
    class FakeEstimator:
        def predict_proba(self, X):
            return np.column_stack([1 - X, X])

    scorer = precision_at_coverage_floor_scorer(min_coverage=0.4)
    X = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    y = np.array([1, 0, 1, 0, 1])
    score = scorer(FakeEstimator(), X, y)
    assert 0.0 <= score <= 1.0
