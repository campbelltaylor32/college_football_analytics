"""Stage 2 dimensionality reduction: model-based selection scored on precision, not ROC-AUC.
Two independent methods are run and reported side by side (never silently merged) so a reader
can sanity-check they roughly agree -- the current notebook only ever ran one method
(permutation importance) to completion.

Must be called with a single fold's training rows only (the fold's *inner* CV splits, built by
modeling/tuning.build_inner_season_cv, are used for both methods' internal scoring) -- never
fit once globally and reused across outer folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

from cfb_spread_model.feature_selection.precision_scoring import Scorer, average_precision_scorer
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


def _default_estimator(random_seed: int):
    """XGBoost, matching the current notebook's chosen model family for the importance signal
    -- the cheapest reasonable choice, not necessarily the final production model (model
    comparison across families happens separately in modeling/models.py)."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=random_seed, n_estimators=200, max_depth=4, eval_metric="logloss")
    except ImportError:
        logger.warning("xgboost not installed; falling back to GradientBoostingClassifier for Stage 2 selection")
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=random_seed)


@dataclass
class PermutationImportanceSweepResult:
    importances: pd.Series
    sweep_table: pd.DataFrame
    best_n_features: int
    selected_features: list[str]


def compute_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scorer: Scorer,
    n_repeats: int,
    random_seed: int,
) -> pd.Series:
    model = _default_estimator(random_seed)
    model.fit(X_train, y_train)
    result = permutation_importance(
        model, X_val, y_val, scoring=scorer, n_repeats=n_repeats, random_state=random_seed, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)


def sweep_feature_counts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    importances: pd.Series,
    candidate_counts: list[int],
    scorer: Scorer,
    random_seed: int,
) -> PermutationImportanceSweepResult:
    """Refits on each candidate top-N feature subset (ranked by permutation importance) and
    scores it on the validation split with the SAME precision-focused scorer used to compute
    the importances -- this is what the current notebook's ROC-AUC-driven sweep never did
    (it scored the sweep on ROC-AUC even after computing importances)."""
    ap_scorer = average_precision_scorer()
    rows = []
    for n in sorted(set(c for c in candidate_counts if c <= len(importances))):
        top_features = importances.index[:n].tolist()
        model = _default_estimator(random_seed)
        model.fit(X_train[top_features], y_train)
        rows.append(
            {
                "n_features": n,
                "precision_at_coverage_floor": scorer(model, X_val[top_features], y_val),
                "average_precision": ap_scorer(model, X_val[top_features], y_val),
            }
        )
    sweep_table = pd.DataFrame(rows)
    best_row = sweep_table.loc[sweep_table["precision_at_coverage_floor"].idxmax()]
    best_n = int(best_row["n_features"])
    return PermutationImportanceSweepResult(
        importances=importances,
        sweep_table=sweep_table,
        best_n_features=best_n,
        selected_features=importances.index[:best_n].tolist(),
    )


def run_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    scorer: Scorer,
    random_seed: int,
    min_features_to_select: int = 10,
    step: float = 0.1,
) -> list[str]:
    """Independent second selection method, scored with the same precision-at-coverage-floor
    scorer as the permutation-importance sweep. RFECV needs an estimator with
    feature_importances_/coef_; XGBoost exposes feature_importances_."""
    if len(cv_splits) < 2:
        logger.warning("Fewer than 2 inner CV splits; skipping RFECV for this fold")
        return list(X_train.columns)

    estimator = _default_estimator(random_seed)
    selector = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=cv_splits,
        scoring=scorer,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    logger.info(f"RFECV selected {len(selected)} of {len(X_train.columns)} features")
    return selected
