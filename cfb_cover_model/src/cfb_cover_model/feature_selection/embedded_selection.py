"""Stage 4 (primary reducer): elastic-net logistic regression selection.

Chosen over permutation-importance threshold sweeps or RFECV because both were found
unstable across folds in prior work on this same data (RFECV's selected feature count
swung from 10 to 641 across six folds). An L1-leaning elastic net zeros out redundant/
noisy coefficients directly as part of a single convex fit, which is a much lower-variance
selection mechanism than iterative wrapper methods.

The l1_ratio/C grid is chosen via an inner TimeSeriesSplit (chronological, not random) on
the fold's own training rows only - this keeps hyperparameter selection consistent with
the project's no-lookahead stance rather than reintroducing a random-CV leak one level
down, which is what the production notebook's RandomizedSearchCV/GridSearchCV did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def select_features_embedded(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    season_train: pd.Series,
    l1_ratio_grid: list[float],
    C_grid: list[float],
    inner_cv_folds: int,
    max_features: int,
    random_state: int = 42,
) -> tuple[list[str], dict]:
    order = np.argsort(season_train.to_numpy(), kind="stable")
    X_ordered = X_train.iloc[order]
    y_ordered = y_train.iloc[order].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ordered)

    n_splits = min(inner_cv_folds, max(2, len(X_ordered) // 50 - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score, best_l1, best_C = -np.inf, l1_ratio_grid[0], C_grid[0]
    for l1_ratio in l1_ratio_grid:
        for C in C_grid:
            fold_scores = []
            for tr_idx, va_idx in tscv.split(X_scaled):
                if len(np.unique(y_ordered[tr_idx])) < 2 or len(np.unique(y_ordered[va_idx])) < 2:
                    continue
                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=3000,
                    random_state=random_state,
                )
                clf.fit(X_scaled[tr_idx], y_ordered[tr_idx])
                proba = clf.predict_proba(X_scaled[va_idx])[:, 1]
                fold_scores.append(roc_auc_score(y_ordered[va_idx], proba))
            if not fold_scores:
                continue
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score, best_l1, best_C = mean_score, l1_ratio, C

    final_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=best_l1,
        C=best_C,
        max_iter=5000,
        random_state=random_state,
    )
    final_clf.fit(X_scaled, y_ordered)

    coefs = final_clf.coef_[0]
    ranked = sorted(
        ((f, abs(c)) for f, c in zip(X_train.columns, coefs) if abs(c) > 1e-8),
        key=lambda pair: -pair[1],
    )
    selected = [f for f, _ in ranked[:max_features]]

    report = {
        "best_l1_ratio": best_l1,
        "best_C": best_C,
        "inner_cv_mean_auc": None if best_score == -np.inf else best_score,
        "n_nonzero_before_cap": len(ranked),
        "n_selected": len(selected),
        # Full ranked (feature, |coef|) list, not just the top-max_features names - lets
        # callers (e.g. scripts/analyze_feature_stability.py) inspect coefficient
        # magnitude for every nonzero feature, not only which ones made the cutoff.
        "ranked_coefficients": ranked,
    }
    return selected, report
