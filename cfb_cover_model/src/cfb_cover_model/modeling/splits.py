"""Expanding-window walk-forward validation by season, plus the final train/holdout split.

Never a random split, never a fold where a validation-season game could influence a
training-season fit: within a fold, train rows are strictly every season earlier than the
validation season. This is the single mechanism every other stage (feature selection,
hyperparameter tuning, stacking) is built on top of - see docs/data_leakage_rules.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_seasons: tuple[int, ...]
    val_season: int
    train_idx: pd.Index
    val_idx: pd.Index


def get_eligible_frame(frame: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    """Drop config-excluded seasons (e.g. COVID-shortened 2020) entirely - not just from
    being used as a validation fold, but from training data too, since the anomaly (an
    8-game conference-only slate) makes those rows unrepresentative training signal as
    well as an unrepresentative validation target.

    Deliberately does NOT reset_index: every row keeps its original index label from the
    saved modeling frame (data/processed/modeling_dataset.parquet), so any variant built
    from the *original* frame elsewhere (e.g. evaluate_models.py's build_full_variant,
    which must include holdout rows for feature-engineering purposes) can still be sliced
    with train_pool.index / holdout.index and get the *same* rows back. Resetting to a
    fresh 0-based RangeIndex here would make train_pool.index and holdout.index both start
    at 0 again - two different rows sharing the same label - so `.loc[holdout.index]`
    against a frame indexed by the *original* labels would silently pull training rows
    instead. This was a real bug caught by comparing walk-forward vs. holdout precision:
    tree models scored ~0.95+ holdout precision, which turned out to be near-perfect
    recall of rows they'd been trained on, not real signal.
    """
    exclude = set(data_cfg["seasons"]["exclude"])
    if not exclude:
        return frame
    return frame.loc[~frame["season"].isin(exclude)]


def get_holdout_split(
    frame: pd.DataFrame, data_cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train_pool, holdout). train_pool is everything eligible for walk-forward
    folds/feature selection/model training; holdout is the final_holdout season(s), never
    touched until the single final evaluation pass. Index labels are preserved from the
    input frame (see get_eligible_frame's docstring) - callers that need a fresh RangeIndex
    for a specific downstream frame should reset it themselves, on that frame, consistently."""
    eligible = get_eligible_frame(frame, data_cfg)
    holdout_seasons = set(data_cfg["seasons"]["final_holdout"])
    train_pool = eligible.loc[~eligible["season"].isin(holdout_seasons)]
    holdout = eligible.loc[eligible["season"].isin(holdout_seasons)]
    assert train_pool.index.intersection(holdout.index).empty, (
        "train_pool and holdout must never share index labels - see get_eligible_frame docstring."
    )
    return train_pool, holdout


def walk_forward_folds(train_pool: pd.DataFrame, modeling_cfg: dict) -> list[Fold]:
    """Build expanding-window folds over train_pool (already holdout/exclusion-filtered).

    Fold i validates on the (min_train_seasons + i)-th earliest eligible season, training on
    every strictly-earlier eligible season. Uses label-based indexing throughout (train_pool
    keeps its original row labels from get_holdout_split - see that function's docstring),
    so the returned Fold.train_idx/val_idx are safe to use against any frame that shares
    train_pool's original index, not just train_pool itself.
    """
    min_train_seasons = modeling_cfg["validation"]["min_train_seasons"]
    seasons = sorted(train_pool["season"].unique())
    folds: list[Fold] = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = tuple(seasons[:i])
        train_idx = train_pool.index[train_pool["season"].isin(train_seasons)]
        val_idx = train_pool.index[train_pool["season"] == val_season]
        folds.append(Fold(train_seasons, val_season, train_idx, val_idx))
    return folds
