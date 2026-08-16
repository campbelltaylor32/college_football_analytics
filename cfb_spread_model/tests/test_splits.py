from __future__ import annotations

from cfb_spread_model.modeling.splits import final_holdout_fold, generate_walk_forward_folds


def test_generate_walk_forward_folds_matches_config(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    assert len(folds) > 0
    val_seasons = [f.validation_season for f in folds]
    assert set(val_seasons).issubset(set(modeling_cfg.walk_forward_validation_seasons))


def test_fold_train_seasons_exclude_configured_exclusions(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    for fold in folds:
        for excluded in modeling_cfg.excluded_seasons:
            assert excluded not in fold.train_seasons


def test_fold_respects_min_train_seasons(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    for fold in folds:
        assert len(fold.train_seasons) >= modeling_cfg.min_train_seasons


def test_final_holdout_fold_trains_on_all_prior_non_excluded_seasons(modeling_cfg):
    holdout = final_holdout_fold(modeling_cfg)
    assert holdout.validation_season == modeling_cfg.final_holdout_season
    expected = [
        s
        for s in range(modeling_cfg.full_feature_start_season, modeling_cfg.final_holdout_season)
        if s not in modeling_cfg.excluded_seasons
    ]
    assert holdout.train_seasons == expected
