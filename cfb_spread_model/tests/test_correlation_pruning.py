from __future__ import annotations

from cfb_spread_model.feature_selection import correlation_pruning

_DROP_COLS = ["game_id", "home_team", "away_team", "season", "week", "home_covered"]


def test_temporal_triplet_collapses_highly_correlated_columns(synthetic_df, features_cfg):
    X = synthetic_df.drop(columns=_DROP_COLS)
    y = synthetic_df["home_covered"]
    kept, report = correlation_pruning.collapse_temporal_triplets(X, y, features_cfg.correlation_pruning)

    total_yards_cols = [c for c in kept if "total_yards" in c and c.startswith("home_") and "allowed" not in c]
    assert len(total_yards_cols) == 1

    dropped_cols = {row["column"] for row in report}
    assert dropped_cols & {"home_total_yards_avg_all", "home_total_yards_avg3", "home_prev_week_total_yards"}


def test_temporal_triplet_keeps_uncorrelated_metrics(synthetic_df, features_cfg):
    X = synthetic_df.drop(columns=_DROP_COLS)
    y = synthetic_df["home_covered"]
    kept, _ = correlation_pruning.collapse_temporal_triplets(X, y, features_cfg.correlation_pruning)

    sacks_cols = [c for c in kept if "sacks" in c]
    assert len(sacks_cols) == 3, "independent (uncorrelated) triplet members should all survive Stage 1"


def test_general_redundancy_prune_keeps_independent_columns(synthetic_df, features_cfg):
    X = synthetic_df.drop(columns=_DROP_COLS)
    y = synthetic_df["home_covered"]
    kept, _ = correlation_pruning.general_redundancy_prune(X, y, list(X.columns), features_cfg.correlation_pruning)

    assert "home_talent" in kept
    assert "away_talent" in kept


def test_prune_never_mutates_input(synthetic_df, features_cfg):
    X = synthetic_df.drop(columns=_DROP_COLS)
    y = synthetic_df["home_covered"]
    original_cols = list(X.columns)
    correlation_pruning.prune(X, y, features_cfg.correlation_pruning)
    assert list(X.columns) == original_cols


def test_prune_reduces_column_count_and_reports_reasons(synthetic_df, features_cfg):
    X = synthetic_df.drop(columns=_DROP_COLS)
    y = synthetic_df["home_covered"]
    kept, report = correlation_pruning.prune(X, y, features_cfg.correlation_pruning)
    assert len(kept) < len(X.columns)
    assert not report.empty
    assert set(report["reason"].unique()) <= {"temporal_collapse", "general_redundancy"}
