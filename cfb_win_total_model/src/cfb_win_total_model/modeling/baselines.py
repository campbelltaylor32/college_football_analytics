"""Baseline predictors, each implementing .fit(train_df)/.predict(eval_df). Every real
candidate model in models.py is compared against these, especially PrevSeasonWinsBaseline --
the headline "did the model beat just guessing last year's win total" comparison.

PrevSeasonWinsBaseline and Rolling3YrAvgBaseline reuse the modeling dataset's own
`prior_season_wins` (coaching.py, sourced from coaches.wins for season t-1) and
`rolling_win_total_3` (program_history.py) columns rather than issuing fresh DB queries --
those columns already carry exactly the win-history information these baselines need.

No market/poll baseline in v1: coaches.preseason_rank is 0% populated for the 2025 demo
target season (verified) and betting_lines is rejected as a preseason signal (spreads are set
weekly, not preseason) -- see docs/assumptions_and_limitations.md. The
features.yaml `use_coach_preseason_rank` toggle is the hook for adding one later.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query


class Baseline(Protocol):
    def fit(self, train_df: pd.DataFrame) -> "Baseline": ...
    def predict(self, eval_df: pd.DataFrame) -> np.ndarray: ...


class OverallMeanBaseline:
    name = "overall_mean"

    def fit(self, train_df: pd.DataFrame) -> "OverallMeanBaseline":
        self.mean_wins = train_df["regular_season_wins"].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(eval_df), self.mean_wins)


class PrevSeasonWinsBaseline:
    name = "prev_season_wins"

    def fit(self, train_df: pd.DataFrame) -> "PrevSeasonWinsBaseline":
        self.fallback_mean = train_df["regular_season_wins"].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        if "prior_season_wins" not in eval_df.columns:
            return np.full(len(eval_df), self.fallback_mean)
        return eval_df["prior_season_wins"].fillna(self.fallback_mean).to_numpy()


class Rolling3YrAvgBaseline:
    name = "rolling_3yr_avg"

    def fit(self, train_df: pd.DataFrame) -> "Rolling3YrAvgBaseline":
        self.fallback_mean = train_df["regular_season_wins"].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        if "rolling_win_total_3" not in eval_df.columns or "rolling_window_actual_seasons_3" not in eval_df.columns:
            return np.full(len(eval_df), self.fallback_mean)
        avg = eval_df["rolling_win_total_3"] / eval_df["rolling_window_actual_seasons_3"].replace(0, np.nan)
        return avg.fillna(self.fallback_mean).to_numpy()


def _team_conference_by_season(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["school", "season", "conference"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"""
        SELECT season, home_team AS school, home_conference AS conference FROM games WHERE season IN ({placeholders})
        UNION ALL
        SELECT season, away_team AS school, away_conference AS conference FROM games WHERE season IN ({placeholders})
    """
    df = run_query(sql, params=params, engine=engine)
    return df.groupby(["school", "season"])["conference"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None).reset_index()


class ConferenceAverageBaseline:
    name = "conference_avg"

    def __init__(self, engine: Engine):
        self.engine = engine

    def fit(self, train_df: pd.DataFrame) -> "ConferenceAverageBaseline":
        seasons = sorted(train_df["season"].unique().tolist())
        conf_map = _team_conference_by_season(self.engine, seasons)
        merged = train_df.merge(conf_map, on=["school", "season"], how="left")
        self.conf_means = merged.groupby("conference")["regular_season_wins"].mean()
        self.overall_mean = train_df["regular_season_wins"].mean()
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        seasons = sorted(eval_df["season"].unique().tolist())
        conf_map = _team_conference_by_season(self.engine, seasons)
        merged = eval_df.merge(conf_map, on=["school", "season"], how="left")
        return merged["conference"].map(self.conf_means).fillna(self.overall_mean).to_numpy()


class OLSPriorWinsTalentBaseline:
    name = "ols_prior_wins_talent"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, train_df: pd.DataFrame) -> "OLSPriorWinsTalentBaseline":
        X = self._features(train_df)
        self.fill_values = X.median()
        X = X.fillna(self.fill_values)
        self.model.fit(X, train_df["regular_season_wins"])
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        X = self._features(eval_df).fillna(self.fill_values)
        return self.model.predict(X)

    @staticmethod
    def _features(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["prior_season_wins", "talent"]
        for c in cols:
            if c not in df.columns:
                df = df.assign(**{c: np.nan})
        return df[cols]


def get_baselines(names: list[str], engine: Engine) -> dict[str, Baseline]:
    registry = {
        "overall_mean": lambda: OverallMeanBaseline(),
        "prev_season_wins": lambda: PrevSeasonWinsBaseline(),
        "rolling_3yr_avg": lambda: Rolling3YrAvgBaseline(),
        "conference_avg": lambda: ConferenceAverageBaseline(engine),
        "ols_prior_wins_talent": lambda: OLSPriorWinsTalentBaseline(),
    }
    return {name: registry[name]() for name in names if name in registry}
