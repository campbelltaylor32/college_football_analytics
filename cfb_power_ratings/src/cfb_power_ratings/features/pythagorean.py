"""Preseason feature: last season's gap between actual win% and Pythagorean-expected win%
(points-for/points-against, Bill James style) -- a regression-to-expectation signal distinct
from `program_history.py`'s `srs_lag1`/`srs_trailing_mean`. SRS is already opponent- and
site-adjusted margin; Pythagorean win% is raw scoring differential turned into an implied win
probability, with no opponent adjustment at all. Teams whose actual record diverged from what
their raw scoring margin implied (over- or under-performing their Pythagorean expectation) tend
to regress toward it the following season -- a well-documented sports-analytics phenomenon this
project's SRS-based features don't isolate on their own.

Ported from `cfb_pythagorean_model/pythagorean_analysis.py`, which validated the classic k=2
exponent at R²=0.797 against actual 2025 win% (a numerically-fit exponent, k=2.181, only reached
R²=0.801 -- negligible improvement, so k=2 is used here rather than adding exponent-fitting
complexity). That project's own methodology is preserved exactly: each team's FULL season slate
counts (including money games vs. non-FBS opponents, matching real season record), and no
home-field adjustment is applied to the raw PF/PA sums (unlike SRS, which explicitly does).
"""
from __future__ import annotations

import pandas as pd

from cfb_power_ratings.database import run_query
from cfb_power_ratings.srs import games_to_team_game_frame


def pythagorean_win_pct(points_for: pd.Series, points_against: pd.Series, exponent: float = 2.0) -> pd.Series:
    pf_k = points_for**exponent
    pa_k = points_against**exponent
    return pf_k / (pf_k + pa_k)


def build_pythagorean_features(engine, seasons: list[int]) -> pd.DataFrame:
    """One row per (team, season) with `pythagorean_win_pct_lag1` and
    `win_pct_over_pythagorean_lag1`, both built ONLY from season t-1's completed games -- never
    season t itself. A team with no games in t-1 (first FBS season, etc.) simply has no row
    here; callers left-merge and get NaN, handled the same NaN-tolerant way every other
    preseason feature is."""
    prior_seasons = sorted({s - 1 for s in seasons})
    games = run_query(
        "SELECT * FROM games WHERE completed = 1 AND season IN :seasons",
        params={"seasons": tuple(prior_seasons)}, engine=engine,
    )
    if games.empty:
        return pd.DataFrame(columns=["team", "season", "pythagorean_win_pct_lag1", "win_pct_over_pythagorean_lag1"])

    team_games = games_to_team_game_frame(games)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)

    team_season = team_games.groupby(["team", "season"]).agg(
        games_played=("win", "count"),
        wins=("win", "sum"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
    ).reset_index()
    team_season["actual_win_pct"] = team_season["wins"] / team_season["games_played"]
    team_season["pythagorean_win_pct_lag1"] = pythagorean_win_pct(
        team_season["points_for"], team_season["points_against"]
    )
    team_season["win_pct_over_pythagorean_lag1"] = (
        team_season["actual_win_pct"] - team_season["pythagorean_win_pct_lag1"]
    )

    # Shift the label forward one season: a row computed from season (t-1)'s games becomes
    # the feature for season t.
    out = team_season[["team", "season", "pythagorean_win_pct_lag1", "win_pct_over_pythagorean_lag1"]].copy()
    out["season"] = out["season"] + 1
    return out[out["season"].isin(seasons)]
