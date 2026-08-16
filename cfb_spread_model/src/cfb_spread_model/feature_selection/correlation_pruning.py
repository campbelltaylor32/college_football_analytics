"""Stage 1 dimensionality reduction: cheap, correlation-based pruning done BEFORE any model
fitting. Must always be called with a single fold's training rows only (see
tests/test_leakage.py::test_correlation_pruning_never_fit_on_validation_rows) -- never fit once
globally and reused across folds.

Two passes, both grounded in the source CSV's verified structure (see docs/data_dictionary.md):

1. Temporal-transform collapse: within each {prev_week_X, X_avg_all, X_avg3} triplet (336
   metric x side groups), drop members whose correlation with the strongest-univariate-
   association member of the triplet exceeds a (loose) threshold -- these are definitionally
   rolling transforms of the same underlying stat.
2. General redundancy pass: on the survivors, drop one of every pair whose |Pearson| exceeds a
   (stricter) global threshold, keeping the higher-univariate-association member. Catches
   offense/defense mirror pairs and algebraic-ratio columns against their raw components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from cfb_spread_model.config import CorrelationPruningConfig
from cfb_spread_model.data import build_temporal_triplet_groups
from cfb_spread_model.utils.logging import get_logger

logger = get_logger(__name__)


def point_biserial_association(feature: pd.Series, label: pd.Series) -> float:
    """abs(point-biserial correlation) between a continuous feature and the binary label.
    Point-biserial correlation is mathematically identical to Pearson correlation between a
    continuous variable and a binary variable -- pointbiserialr is used here (rather than a
    plain .corr() call) so the metric name in config/features.yaml has a literal implementation
    to point to."""
    if feature.std(ddof=0) == 0:
        return 0.0
    corr, _ = pointbiserialr(label.to_numpy(), feature.to_numpy())
    return abs(corr) if np.isfinite(corr) else 0.0


def collapse_temporal_triplets(
    X: pd.DataFrame, y: pd.Series, cfg: CorrelationPruningConfig
) -> tuple[list[str], list[dict]]:
    groups = build_temporal_triplet_groups(list(X.columns))
    dropped: set[str] = set()
    report: list[dict] = []

    for (side, base), transform_map in groups.items():
        cols = [c for c in transform_map.values() if c in X.columns]
        if len(cols) < 2:
            continue
        assoc = {c: point_biserial_association(X[c], y) for c in cols}
        ordered = sorted(cols, key=lambda c: assoc[c], reverse=True)
        kept_col = ordered[0]
        for c in ordered[1:]:
            corr = X[[kept_col, c]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) > cfg.temporal_collapse_corr_threshold:
                dropped.add(c)
                report.append(
                    {"column": c, "kept_alternative": kept_col, "correlation": float(corr), "reason": "temporal_collapse"}
                )

    kept = [c for c in X.columns if c not in dropped]
    logger.info(f"Temporal-transform collapse: dropped {len(dropped)} of {len(X.columns)} columns")
    return kept, report


def general_redundancy_prune(
    X: pd.DataFrame, y: pd.Series, columns: list[str], cfg: CorrelationPruningConfig
) -> tuple[list[str], list[dict]]:
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(X[c])]
    assoc = {c: point_biserial_association(X[c], y) for c in numeric_cols}

    corr_matrix = X[numeric_cols].corr().abs()
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            corr_val = corr_matrix.loc[c1, c2]
            if pd.notna(corr_val) and corr_val > cfg.general_corr_threshold:
                pairs.append((corr_val, c1, c2))
    pairs.sort(key=lambda t: t[0], reverse=True)

    dropped: set[str] = set()
    report: list[dict] = []
    for corr_val, c1, c2 in pairs:
        if c1 in dropped or c2 in dropped:
            continue
        keep, drop = (c1, c2) if assoc[c1] >= assoc[c2] else (c2, c1)
        dropped.add(drop)
        report.append({"column": drop, "kept_alternative": keep, "correlation": float(corr_val), "reason": "general_redundancy"})

    kept = [c for c in columns if c not in dropped]
    logger.info(f"General redundancy pass: dropped {len(dropped)} of {len(columns)} columns")
    return kept, report


def prune(X: pd.DataFrame, y: pd.Series, cfg: CorrelationPruningConfig) -> tuple[list[str], pd.DataFrame]:
    """Runs both passes. Must be called on a single fold's training rows only."""
    stage1_kept, stage1_report = collapse_temporal_triplets(X, y, cfg)
    stage2_kept, stage2_report = general_redundancy_prune(X, y, stage1_kept, cfg)
    report = pd.DataFrame(stage1_report + stage2_report)
    logger.info(f"Correlation pruning: {len(X.columns)} -> {len(stage2_kept)} columns")
    return stage2_kept, report
