"""Cross-references the auto-generated feature registry against the actual modeling table and
NON_FEATURE_COLS -- catches a feature module that's accidentally tagged known_before_kickoff
incorrectly, or a raw realized-outcome column that's slipped into the feature set."""

from __future__ import annotations

from cfb_rb_rushing_model.dataset import NON_FEATURE_COLS, build_feature_registry
from cfb_rb_rushing_model.modeling.train import get_feature_columns
from cfb_rb_rushing_model.modeling.splits import generate_walk_forward_folds


def test_every_registered_feature_is_known_before_kickoff(features_cfg, data_cfg):
    registry = build_feature_registry(features_cfg, data_cfg)
    assert (registry["known_before_kickoff"] == True).all()  # noqa: E712


def test_feature_registry_columns_all_present_in_modeling_dataset(small_modeling_df, features_cfg, data_cfg):
    registry = build_feature_registry(features_cfg, data_cfg)
    missing = [f for f in registry["feature_name"] if f not in small_modeling_df.columns]
    assert missing == [], f"Registry lists features not present in the modeling table: {missing}"


def test_raw_realized_outcome_columns_are_excluded_from_model_features(small_modeling_df):
    """carries/rushing_yards/explosive_runs/etc. are this player's OWN actual stats from the
    target game -- using them as inputs would leak the target directly. Must never appear in
    the trainable feature-column set."""
    feature_cols = get_feature_columns(small_modeling_df, NON_FEATURE_COLS)
    leaky_cols = {"rushing_yards", "carries", "explosive_runs", "red_zone_carries", "success_rate", "yards_per_carry", "played"}
    assert not (leaky_cols & set(feature_cols)), f"Leaky same-game realized columns found in feature set: {leaky_cols & set(feature_cols)}"


def test_walk_forward_folds_never_include_validation_season_in_training(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    for fold in folds:
        assert fold.validation_season not in fold.train_seasons
        assert max(fold.train_seasons) < fold.validation_season
