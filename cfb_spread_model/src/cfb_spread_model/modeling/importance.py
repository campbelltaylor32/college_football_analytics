"""Feature importance for an already-fitted production model. Two measures, reported side by
side (never merged into one score) so a reader can see where they agree/disagree:

1. Gain-based, model-native importance (`XGBClassifier.feature_importances_`) -- free, no
   refit, no extra dependency.
2. Permutation importance on real (not synthetic) holdout rows, scored with the SAME
   precision-at-coverage-floor scorer used throughout this project's feature selection and
   threshold selection (feature_selection/precision_scoring.py) -- consistent with the
   project's precision-first objective, rather than a generic accuracy-based importance.

Both are joined with each feature's (side, temporal_transform, base_metric), via the
already-built data.parse_side_and_metric, so category rollups (e.g. "how much does the
prev_week_* category matter") are a groupby away, not a re-parse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cfb_spread_model.data import parse_side_and_metric
from cfb_spread_model.feature_selection.precision_scoring import Scorer


def compute_gain_importance(fitted_model, feature_names: list[str]) -> pd.Series:
    """`fitted_model` is the final pipeline step (e.g. an already-fit XGBClassifier), not the
    whole sklearn Pipeline -- callers extract it via `pipeline.named_steps["model"]`. Assumes
    feature_names is in the same column order the model was fit on (true for this project's
    preprocessing.build_preprocessing_pipeline, which uses a single ColumnTransformer branch
    that preserves input column order)."""
    if not hasattr(fitted_model, "feature_importances_"):
        raise AttributeError(
            f"{type(fitted_model).__name__} has no feature_importances_ -- gain-based importance "
            f"is only available for tree-ensemble models"
        )
    return pd.Series(fitted_model.feature_importances_, index=feature_names, name="gain_importance").sort_values(
        ascending=False
    )


def compute_permutation_importance_for_model(
    fitted_pipeline, X: pd.DataFrame, y: pd.Series, scorer: Scorer, n_repeats: int, random_seed: int
) -> pd.DataFrame:
    """`fitted_pipeline` is the full (already-fit) sklearn Pipeline, since permutation
    importance needs predict_proba on raw (pre-preprocessing) feature columns, matching how
    every other scorer(estimator, X, y) call in this project works."""
    result = permutation_importance(
        fitted_pipeline, X, y, scoring=scorer, n_repeats=n_repeats, random_state=random_seed, n_jobs=-1
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)


def build_feature_importance_report(gain_importance: pd.Series, permutation_df: pd.DataFrame) -> pd.DataFrame:
    """Joins gain + permutation importance for the same feature set and adds category columns
    (side, temporal_transform, base_metric) via data.parse_side_and_metric. Context columns
    that aren't home_/away_-prefixed (spread, home_favored, neutral_site, conference_game) get
    temporal_transform="context" rather than a parse failure."""
    gain_df = gain_importance.rename("gain_importance").reset_index().rename(columns={"index": "feature"})
    report = permutation_df.merge(gain_df, on="feature", how="outer")

    sides, transforms, bases = [], [], []
    for feature in report["feature"]:
        parsed = parse_side_and_metric(feature)
        if parsed is None:
            sides.append(None)
            transforms.append("context")
            bases.append(feature)
        else:
            side, transform, base = parsed
            sides.append(side)
            transforms.append(transform)
            bases.append(base)
    report["side"] = sides
    report["temporal_transform"] = transforms
    report["base_metric"] = bases

    report["permutation_importance_rank"] = report["permutation_importance_mean"].rank(ascending=False, method="min")
    report["gain_importance_rank"] = report["gain_importance"].rank(ascending=False, method="min")
    return report.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def summarize_by_temporal_transform(report: pd.DataFrame) -> pd.DataFrame:
    """Count + importance-mass rollup by temporal_transform category (prev_week/avg_all/avg3/
    non_temporal/context) -- the direct, reusable answer to "are we using 1-game predictors,
    and do they matter" for whatever feature set the current production model actually has."""
    total_gain = report["gain_importance"].sum()
    total_perm = report["permutation_importance_mean"].clip(lower=0).sum()

    summary = report.groupby("temporal_transform", observed=True).agg(
        n_features=("feature", "count"),
        total_gain_importance=("gain_importance", "sum"),
        total_permutation_importance=("permutation_importance_mean", lambda s: s.clip(lower=0).sum()),
    )
    summary["pct_of_features"] = summary["n_features"] / len(report)
    summary["pct_of_gain_importance"] = np.where(total_gain > 0, summary["total_gain_importance"] / total_gain, 0.0)
    summary["pct_of_permutation_importance"] = np.where(
        total_perm > 0, summary["total_permutation_importance"] / total_perm, 0.0
    )
    return summary.reset_index().sort_values("total_permutation_importance", ascending=False)
