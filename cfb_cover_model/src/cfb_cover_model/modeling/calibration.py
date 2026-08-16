"""Post-hoc probability calibration, fit only on pooled walk-forward OOF predictions and
applied (never refit) to the true holdout - tests whether a model's raw scores can be
turned into a trustworthy confidence signal, the failure mode flagged in prior work on
this data (near-zero rank monotonicity in the raw scores)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_isotonic(y_true: np.ndarray, y_proba: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(y_proba, y_true)
    return model


def apply_isotonic(model: IsotonicRegression, y_proba: np.ndarray) -> np.ndarray:
    return model.predict(y_proba)


def fit_platt(y_true: np.ndarray, y_proba: np.ndarray) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(y_proba.reshape(-1, 1), y_true)
    return model


def apply_platt(model: LogisticRegression, y_proba: np.ndarray) -> np.ndarray:
    return model.predict_proba(y_proba.reshape(-1, 1))[:, 1]


CALIBRATORS = {
    "isotonic": (fit_isotonic, apply_isotonic),
    "platt": (fit_platt, apply_platt),
}
