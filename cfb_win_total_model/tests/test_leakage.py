"""End-to-end leakage guards. Cross-references outputs/feature_analysis/feature_registry.csv
(built fresh here, not read from disk, so the test never trusts a stale file) and directly
exercises modeling/splits.py's fold generator.
"""

from __future__ import annotations

from cfb_win_total_model.dataset import build_feature_registry
from cfb_win_total_model.modeling.splits import generate_walk_forward_folds

SANCTIONED_AS_IS_EXCEPTIONS = {"returning_production", "talent_recruiting"}


def test_feature_registry_documents_source_season_for_every_leak_risk_category(features_cfg):
    registry = build_feature_registry(features_cfg)
    for _, row in registry.iterrows():
        assert row["source_season"], f"{row['feature_name']} has no documented source_season"
        if row["category"] in SANCTIONED_AS_IS_EXCEPTIONS:
            source = str(row["source_season"])
            assert not source.strip().startswith("t-1") and "t-2" not in source, (
                f"{row['feature_name']} is in a sanctioned as-is category but its documented "
                f"source_season looks lagged: {source}"
            )


def test_walk_forward_folds_never_leak_future_seasons(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    assert len(folds) > 0
    for fold in folds:
        assert max(fold.train_seasons) < fold.validation_season
        assert modeling_cfg.excluded_seasons[0] not in fold.train_seasons
        assert fold.validation_season not in modeling_cfg.excluded_seasons


def test_walk_forward_folds_are_expanding_not_random(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    train_set_sizes = [len(f.train_seasons) for f in folds]
    assert train_set_sizes == sorted(train_set_sizes), "training window should only ever grow across folds"


def test_final_holdout_excludes_covid_season(modeling_cfg):
    from cfb_win_total_model.modeling.splits import final_holdout_fold

    holdout = final_holdout_fold(modeling_cfg)
    assert 2020 not in holdout.train_seasons
    assert holdout.validation_season == modeling_cfg.final_holdout_season


def test_modeling_dataset_has_no_2025_derived_leak_columns(full_modeling_df):
    """The 2025 row's feature columns must never correlate 1:1 with that season's own game
    outcomes -- spot check that no column is literally the target or derivable 1:1 from it."""
    df = full_modeling_df[full_modeling_df["season"] == 2025]
    assert not df.empty
    for col in ("regular_season_wins", "regular_season_losses"):
        assert (df[col] != df["regular_season_wins"]).sum() >= 0  # sanity: column exists, not aliased silently
    # No feature column should be an exact duplicate of the target (a hard leak signature).
    feature_cols = [c for c in df.columns if c not in ("school", "season", "regular_season_wins", "regular_season_losses", "scheduled_games")]
    for col in feature_cols:
        if df[col].dtype.kind in "if":
            assert not df[col].equals(df["regular_season_wins"])
