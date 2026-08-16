"""Baseline predictors, each implementing .fit(train_df)/.predict(eval_df). Every real
candidate model in models.py is compared against these -- especially player_rolling3_avg,
the headline "did the model beat just using this player's own recent average" comparison.

player_rolling3_avg and player_season_avg reuse the modeling dataset's own
`rushing_yards_avg3_asof`/`rushing_yards_avg_all_asof` columns (eligibility.py, via
merge_asof against the player's most recently played game) rather than issuing fresh
computation -- those columns already carry exactly the rolling-average information these
baselines need, and are guaranteed to reflect the same no-lookahead discipline as every other
feature.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

TARGET_COL = "rushing_yards"


class Baseline(Protocol):
    def fit(self, train_df: pd.DataFrame) -> "Baseline": ...
    def predict(self, eval_df: pd.DataFrame) -> np.ndarray: ...


class PlayerRolling3AvgBaseline:
    name = "player_rolling3_avg"

    def fit(self, train_df: pd.DataFrame) -> "PlayerRolling3AvgBaseline":
        self.fallback_mean = train_df[TARGET_COL].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        if "rushing_yards_avg3_asof" not in eval_df.columns:
            return np.full(len(eval_df), self.fallback_mean)
        return eval_df["rushing_yards_avg3_asof"].fillna(self.fallback_mean).to_numpy()


class PlayerSeasonAvgBaseline:
    name = "player_season_avg"

    def fit(self, train_df: pd.DataFrame) -> "PlayerSeasonAvgBaseline":
        self.fallback_mean = train_df[TARGET_COL].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        if "rushing_yards_avg_all_asof" not in eval_df.columns:
            return np.full(len(eval_df), self.fallback_mean)
        return eval_df["rushing_yards_avg_all_asof"].fillna(self.fallback_mean).to_numpy()


class PositionAverageBaseline:
    """The "know nothing about this specific player" floor: predicts the trailing mean
    rushing_yards across every eligible RB in the training data, for every row."""

    name = "position_avg"

    def fit(self, train_df: pd.DataFrame) -> "PositionAverageBaseline":
        self.overall_mean = train_df[TARGET_COL].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(eval_df), self.overall_mean)


def get_baselines(names: list[str]) -> dict[str, Baseline]:
    registry = {
        "player_rolling3_avg": lambda: PlayerRolling3AvgBaseline(),
        "player_season_avg": lambda: PlayerSeasonAvgBaseline(),
        "position_avg": lambda: PositionAverageBaseline(),
    }
    return {name: registry[name]() for name in names if name in registry}
