"""Custom precision-focused scorers. This is the direct fix for the current notebook's core
gap: every scoring="roc_auc" call site in Python Scripts/CFB_Gambling_Model.ipynb is replaced
by one of these throughout feature_selection/selection.py, modeling/tuning.py, and
modeling/threshold_selection.py.

Every scorer here follows sklearn's `scoring`-callable protocol directly -- `scorer(estimator,
X, y) -> float` -- rather than sklearn.metrics.make_scorer, because precision_at_coverage_floor
needs predict_proba (to find its own threshold), which make_scorer's needs_proba plumbing
makes awkward. This same callable signature is accepted by GridSearchCV, RFECV, and
sklearn.inspection.permutation_importance's `scoring` parameter, so one scorer factory serves
all three call sites.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_score

Scorer = Callable[[object, np.ndarray, np.ndarray], float]


def coverage_threshold(y_score: np.ndarray, min_coverage: float) -> float:
    """The highest decision threshold that still flags at least `min_coverage` fraction of
    rows. Returns the k-th highest score, k = ceil(min_coverage * n), so `y_score >= threshold`
    keeps exactly k (or more, in the presence of ties) rows."""
    y_score = np.asarray(y_score)
    n = len(y_score)
    if n == 0:
        return 1.0
    k = max(1, int(np.ceil(min_coverage * n)))
    return np.sort(y_score)[::-1][k - 1]


def precision_at_coverage_floor(y_true: np.ndarray, y_score: np.ndarray, min_coverage: float) -> float:
    """Precision at the highest threshold that still meets the coverage floor. This is the
    scorer used to drive feature/model/hyperparameter selection throughout this project: it
    directly answers "how precise can we be while still flagging a usable number of games,"
    preventing selection from degenerating to a single ultra-confident pick (which would
    trivially maximize plain precision but be useless for actually placing bets)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_score) == 0:
        return 0.0
    threshold = coverage_threshold(y_score, min_coverage)
    preds = (y_score >= threshold).astype(int)
    if preds.sum() == 0:
        return 0.0
    return precision_score(y_true, preds, zero_division=0)


def precision_at_coverage_floor_scorer(min_coverage: float) -> Scorer:
    def scorer(estimator, X, y) -> float:
        proba = estimator.predict_proba(X)[:, 1]
        return precision_at_coverage_floor(np.asarray(y), proba, min_coverage)

    return scorer


def precision_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true)
    preds = (np.asarray(y_score) >= threshold).astype(int)
    if preds.sum() == 0:
        return 0.0
    return precision_score(y_true, preds, zero_division=0)


def precision_at_threshold_scorer(threshold: float) -> Scorer:
    def scorer(estimator, X, y) -> float:
        proba = estimator.predict_proba(X)[:, 1]
        return precision_at_threshold(np.asarray(y), proba, threshold)

    return scorer


def average_precision_scorer() -> Scorer:
    """Threshold-agnostic area-under-precision-recall-curve scorer -- used as a secondary,
    cross-check signal for model/feature-count comparison (config/features.yaml
    selection_methods), never as the sole selection objective."""

    def scorer(estimator, X, y) -> float:
        proba = estimator.predict_proba(X)[:, 1]
        return average_precision_score(np.asarray(y), proba)

    return scorer


def coverage(y_score: np.ndarray, threshold: float) -> float:
    """Fraction of rows flagged at a given threshold -- reported alongside precision/recall in
    every evaluation output so a reader can see the coverage/precision tradeoff directly."""
    y_score = np.asarray(y_score)
    if len(y_score) == 0:
        return 0.0
    return float(np.mean(y_score >= threshold))
