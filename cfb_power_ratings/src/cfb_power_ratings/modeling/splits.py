"""Expanding-window walk-forward validation by season -- never a random split, same discipline
every sibling project in this repo uses. Each fold trains on every eligible season strictly
before the validation season (COVID-excluded) and validates on that one season."""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward_folds(
    df: pd.DataFrame, validation_seasons: list[int], excluded_seasons: list[int], min_train_seasons: int
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    for val_season in validation_seasons:
        train_seasons = sorted(
            s for s in df["season"].unique() if s < val_season and s not in excluded_seasons
        )
        if len(train_seasons) < min_train_seasons:
            continue
        train = df[df["season"].isin(train_seasons)]
        val = df[df["season"] == val_season]
        if val.empty:
            continue
        yield val_season, train, val
