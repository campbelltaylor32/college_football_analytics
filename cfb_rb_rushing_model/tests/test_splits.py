from __future__ import annotations

from cfb_rb_rushing_model.config import ModelingConfig
from cfb_rb_rushing_model.modeling.splits import final_holdout_fold, generate_walk_forward_folds


def _cfg(**overrides) -> ModelingConfig:
    base = dict(
        target_season=2025,
        full_feature_start_season=2014,
        excluded_seasons=[2020],
        min_train_seasons=4,
        first_validation_season=2019,
        walk_forward_validation_seasons=[2019, 2021, 2022, 2023],
        final_holdout_season=2025,
        final_holdout_max_week=8,
        random_seed=42,
        clip_min_yards=0,
        prediction_interval_method="oof_residual_quantiles",
        prediction_interval_levels=(0.10, 0.90),
        baseline_models=["player_rolling3_avg"],
        candidate_models=["ols"],
        hyperparam_grids={},
    )
    base.update(overrides)
    return ModelingConfig(**base)


def test_walk_forward_folds_never_leak_a_future_season():
    cfg = _cfg()
    folds = generate_walk_forward_folds(cfg)
    assert len(folds) > 0
    for fold in folds:
        assert max(fold.train_seasons) < fold.validation_season
        assert fold.validation_season not in fold.train_seasons


def test_excluded_seasons_never_appear_in_training():
    cfg = _cfg()
    folds = generate_walk_forward_folds(cfg)
    for fold in folds:
        assert 2020 not in fold.train_seasons
        assert 2025 not in fold.train_seasons


def test_folds_below_min_train_seasons_are_dropped():
    # full_feature_start_season=2020 gives val_season=2022 only 2 prior training seasons
    # (2020, 2021) -- below min_train_seasons=4, so that fold must be dropped, not crash.
    cfg = _cfg(full_feature_start_season=2020, excluded_seasons=[], walk_forward_validation_seasons=[2022, 2023], min_train_seasons=4)
    folds = generate_walk_forward_folds(cfg)
    assert folds == []


def test_final_holdout_fold_excludes_holdout_season_from_training():
    cfg = _cfg()
    holdout = final_holdout_fold(cfg)
    assert holdout.validation_season == 2025
    assert 2025 not in holdout.train_seasons
    assert max(holdout.train_seasons) < 2025
    assert 2024 in holdout.train_seasons  # no longer the holdout itself -- now training data
