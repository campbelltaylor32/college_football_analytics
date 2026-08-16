import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.modeling.splits import get_eligible_frame, get_holdout_split, walk_forward_folds


def make_data_cfg(exclude, holdout):
    return {"seasons": {"exclude": exclude, "final_holdout": holdout}}


def test_excluded_seasons_removed_entirely(synthetic_modeling_frame):
    data_cfg = make_data_cfg([2017], [2019])
    eligible = get_eligible_frame(synthetic_modeling_frame, data_cfg)
    assert 2017 not in eligible["season"].unique()
    assert set(eligible["season"].unique()) == {2015, 2016, 2018, 2019}


def test_holdout_never_appears_in_train_pool(synthetic_modeling_frame):
    data_cfg = make_data_cfg([], [2019])
    train_pool, holdout = get_holdout_split(synthetic_modeling_frame, data_cfg)
    assert set(holdout["season"].unique()) == {2019}
    assert 2019 not in train_pool["season"].unique()
    assert len(train_pool) + len(holdout) == len(synthetic_modeling_frame)


def test_train_pool_and_holdout_indices_are_disjoint_and_index_the_original_frame_correctly(
    synthetic_modeling_frame,
):
    """Regression test for a real bug: get_holdout_split used to reset_index(drop=True) on
    both train_pool and holdout, so both index spaces started at 0 again - overlapping
    labels that silently mapped back to the *wrong* (and overlapping) rows when a caller
    indexed into a frame built from the original, un-split frame (exactly what
    evaluate_models.py does). This must never regress: indices must stay disjoint AND must
    still correctly select each row's own season when applied to the original frame."""
    data_cfg = make_data_cfg([], [2019])
    train_pool, holdout = get_holdout_split(synthetic_modeling_frame, data_cfg)

    assert train_pool.index.intersection(holdout.index).empty
    assert set(train_pool.index) | set(holdout.index) == set(synthetic_modeling_frame.index)

    # The critical check: slicing the *original* frame with holdout's index must recover
    # exactly the holdout season, not a mix of training rows.
    recovered_holdout = synthetic_modeling_frame.loc[holdout.index]
    assert set(recovered_holdout["season"].unique()) == {2019}

    recovered_train = synthetic_modeling_frame.loc[train_pool.index]
    assert 2019 not in recovered_train["season"].unique()


def test_walk_forward_folds_never_train_on_future_seasons(synthetic_modeling_frame):
    data_cfg = make_data_cfg([], [2019])
    modeling_cfg = {"validation": {"min_train_seasons": 2}}
    train_pool, _holdout = get_holdout_split(synthetic_modeling_frame, data_cfg)
    folds = walk_forward_folds(train_pool, modeling_cfg)

    assert len(folds) > 0
    for fold in folds:
        assert max(fold.train_seasons) < fold.val_season
        train_seasons_seen = set(train_pool.loc[fold.train_idx, "season"].unique())
        val_seasons_seen = set(train_pool.loc[fold.val_idx, "season"].unique())
        assert train_seasons_seen == set(fold.train_seasons)
        assert val_seasons_seen == {fold.val_season}
        # no row index appears in both train and val
        assert set(fold.train_idx).isdisjoint(set(fold.val_idx))


def test_walk_forward_folds_are_expanding(synthetic_modeling_frame):
    data_cfg = make_data_cfg([], [])
    modeling_cfg = {"validation": {"min_train_seasons": 2}}
    train_pool, _holdout = get_holdout_split(synthetic_modeling_frame, data_cfg)
    folds = walk_forward_folds(train_pool, modeling_cfg)

    sizes = [len(f.train_seasons) for f in folds]
    assert sizes == sorted(sizes)  # strictly non-decreasing training window
    for earlier, later in zip(folds, folds[1:]):
        assert set(earlier.train_seasons) < set(later.train_seasons) | {earlier.val_season}
