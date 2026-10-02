from cfb_playcall_predictability.modeling.splits import final_holdout_fold, generate_walk_forward_folds


def test_walk_forward_trains_only_on_earlier_seasons(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    assert folds
    for fold in folds:
        assert max(fold.train_seasons) < fold.validation_season
        assert min(fold.train_seasons) >= modeling_cfg.full_feature_start_season


def test_holdout_excluded_from_walk_forward(modeling_cfg):
    holdout = final_holdout_fold(modeling_cfg)
    assert all(f.validation_season < holdout.validation_season for f in generate_walk_forward_folds(modeling_cfg))
    assert max(holdout.train_seasons) == holdout.validation_season - 1
