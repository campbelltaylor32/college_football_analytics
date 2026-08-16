"""End-to-end leakage guards. Feature engineering (and its lookahead-bias prevention) already
happened upstream in R -- these tests are a regression safety net, not a rebuild of that logic.
See docs/data_leakage_rules.md for the per-category source-season table these tests check
against.
"""

from __future__ import annotations

import inspect

from cfb_spread_model.data import build_feature_matrix, get_feature_columns
from cfb_spread_model.data_validation import POSTGAME_EXACT_DENYLIST, POSTGAME_SUBSTRING_DENYLIST
from cfb_spread_model.feature_selection import correlation_pruning
from cfb_spread_model.modeling.splits import final_holdout_fold, generate_walk_forward_folds


def test_no_postgame_columns_present(real_dataset):
    exact_hits = [c for c in real_dataset.columns if c in POSTGAME_EXACT_DENYLIST]
    assert not exact_hits, f"Found post-game column(s): {exact_hits}"
    for pattern in POSTGAME_SUBSTRING_DENYLIST:
        hits = [c for c in real_dataset.columns if pattern in c]
        assert not hits, f"Found post-game-looking column(s) matching '{pattern}': {hits}"


def test_season_week_excluded_from_feature_matrix_by_default(real_dataset, data_cfg):
    """Directly targets the current notebook's confirmed bug: Python Scripts/CFB_Gambling_Model.ipynb's
    exclude_vars only drops game_id/home_team/away_team, leaving season/week inside X."""
    assert data_cfg.split_only_columns == ["season", "week"]
    assert data_cfg.include_split_columns_as_features is False
    X, _ = build_feature_matrix(real_dataset, data_cfg)
    assert "season" not in X.columns
    assert "week" not in X.columns


def test_walk_forward_folds_never_leak_future_seasons(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    assert len(folds) > 0
    for fold in folds:
        assert max(fold.train_seasons) < fold.validation_season
        for excluded in modeling_cfg.excluded_seasons:
            assert excluded not in fold.train_seasons
        assert fold.validation_season not in modeling_cfg.excluded_seasons


def test_walk_forward_folds_are_expanding_not_random(modeling_cfg):
    folds = generate_walk_forward_folds(modeling_cfg)
    train_set_sizes = [len(f.train_seasons) for f in folds]
    assert train_set_sizes == sorted(train_set_sizes), "training window should only ever grow across folds"


def test_final_holdout_excludes_covid_season(modeling_cfg):
    holdout = final_holdout_fold(modeling_cfg)
    assert 2020 not in holdout.train_seasons
    assert holdout.validation_season == modeling_cfg.final_holdout_season


def test_correlation_pruning_signature_only_accepts_a_single_slice(modeling_cfg):
    """correlation_pruning.prune() is a pure function over whatever X/y it's given -- this
    pins down its signature so a future refactor that threads in a full unsplit dataframe +
    mask (instead of an already fold-sliced X/y) fails a review, not silently leaks."""
    sig = inspect.signature(correlation_pruning.prune)
    assert list(sig.parameters) == ["X", "y", "cfg"]


def test_no_within_week_completeness_gap(real_dataset, data_cfg):
    assert real_dataset["week"].min() >= data_cfg.week_min
    assert real_dataset["week"].max() <= data_cfg.week_max

    feature_cols = get_feature_columns(list(real_dataset.columns), data_cfg)
    temporal_cols = [c for c in feature_cols if "prev_week_" in c or c.endswith("_avg3") or c.endswith("_avg_all")]
    assert temporal_cols, "expected at least some prev_week_/avg3/avg_all columns"
    na_counts = real_dataset[temporal_cols].isna().sum()
    assert (na_counts == 0).all(), (
        f"Unexpected NA in lag columns (upstream R na.omit() guarantee broken?): {na_counts[na_counts > 0].to_dict()}"
    )


def test_home_covered_matches_documented_derivation():
    """Synthetic fixture recomputing the label via the documented 3-condition rule
    (R Scripts/Full_CFB_Game_Outcome_Historical.R lines 112-127) -- pins the derivation down in
    executable form. The real CSV doesn't carry raw home_points/away_points, so this uses a
    small hand-built fixture, not real_dataset.

    `spread` here is the RAW signed value as it exists at that point in the R pipeline --
    BEFORE Merge_Predictors_CFB_Historical.R:91's `spread = abs(spread)` conversion -- using
    the standard betting convention: negative means the home team is favored (e.g. -7.0 means
    home favored by 7). This sign is what makes `home_minus_away > (-spread)` mean "home won
    by more than the number" for a favored home team.
    """

    def compute_home_covered(home_favored: int, spread: float, home_minus_away: float) -> int:
        away_favored = 0 if home_favored == 1 else 1
        cond_favorite_covers = (home_favored == 1) and (home_minus_away > -spread)
        cond_dog_wins_outright = (away_favored == 1) and (home_minus_away >= 0)
        cond_dog_covers = (
            (away_favored == 1) and (spread >= 0) and (home_minus_away < 0) and (home_minus_away > -spread)
        )
        return 1 if (cond_favorite_covers or cond_dog_wins_outright or cond_dog_covers) else 0

    # Home favored by 7 (spread=-7): wins by 10 -> covers; wins by 3 -> fails; wins by exactly
    # 7 (push) -> falls through to 0, the documented no-push-class simplification.
    assert compute_home_covered(home_favored=1, spread=-7, home_minus_away=10) == 1
    assert compute_home_covered(home_favored=1, spread=-7, home_minus_away=3) == 0
    assert compute_home_covered(home_favored=1, spread=-7, home_minus_away=7) == 0
    # Home underdog by 7 (spread=+7): wins outright -> covers; loses by 3 (within the number)
    # -> covers; loses by 10 (beyond the number) -> fails.
    assert compute_home_covered(home_favored=0, spread=7, home_minus_away=3) == 1
    assert compute_home_covered(home_favored=0, spread=7, home_minus_away=-3) == 1
    assert compute_home_covered(home_favored=0, spread=7, home_minus_away=-10) == 0
