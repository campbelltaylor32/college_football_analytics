"""Opponent-adjusted, home-field-adjusted Simple Rating System (SRS) — the numerical method
this project's rating target AND its in-season update engine (rating_engine.py) both build on.

Generalizes cfb_pythagorean_model/opponent_adjusted_analysis.py::compute_srs (the same
fixed-point iteration: adj_margin = margin + opponent_srs, team_srs = mean(adj_margin),
recentered to 0 each pass, 200 iterations by default) in three ways that file didn't need for
its one-shot, single-completed-season use case:

1. Home-field advantage is estimated from data and backed out of each game's margin before
   iterating (not: raw, unadjusted point margins), so the resulting rating is a neutral-field
   power rating rather than one that silently rewards teams for a home-heavy schedule.
2. Non-FBS opponents are pooled into one fixed-rating pseudo-team rather than dropped, so an
   FBS team's money games against FCS/G5-adjacent opponents still count toward its profile
   (mirrors cfb_transfer_portal_flow's "Non-FBS/Other" bucketing decision) without having to
   accurately rate hundreds of lower-division programs.
3. Works for any subset of games (a full season, or a partial in-season slice through week
   N-1) — this file has no season-boundary assumption baked in, which is what lets
   rating_engine.py reuse it for weekly updates, not just end-of-season targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SRS_ITERATIONS = 200


def games_to_team_game_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """One row per game -> two rows per game (one per team's perspective). Expects the
    `games` table's own column names (game_id, season, home_team, away_team, home_points,
    away_points, home_division, away_division, neutral_site)."""
    common = {"game_id": games_df["game_id"], "season": games_df["season"]}
    home = pd.DataFrame({
        **common,
        "team": games_df["home_team"],
        "opponent": games_df["away_team"],
        "points_for": games_df["home_points"],
        "points_against": games_df["away_points"],
        "team_is_fbs": games_df["home_division"] == "fbs",
        "opponent_is_fbs": games_df["away_division"] == "fbs",
        "is_home": True,
        "neutral_site": games_df["neutral_site"].astype(bool),
    })
    away = pd.DataFrame({
        **common,
        "team": games_df["away_team"],
        "opponent": games_df["home_team"],
        "points_for": games_df["away_points"],
        "points_against": games_df["home_points"],
        "team_is_fbs": games_df["away_division"] == "fbs",
        "opponent_is_fbs": games_df["home_division"] == "fbs",
        "is_home": False,
        "neutral_site": games_df["neutral_site"].astype(bool),
    })
    return pd.concat([home, away], ignore_index=True)


def estimate_home_field_advantage(games_df: pd.DataFrame) -> float:
    """League-average home-field advantage, in points: mean(home_points - away_points) over
    FBS-vs-FBS, non-neutral-site, completed games. One constant for the whole league and
    history — a documented v1 simplification (see docs/assumptions_and_limitations.md), not a
    per-team or per-season estimate."""
    g = games_df[
        (games_df["home_division"] == "fbs")
        & (games_df["away_division"] == "fbs")
        & (~games_df["neutral_site"].astype(bool))
        & games_df["home_points"].notna()
        & games_df["away_points"].notna()
    ]
    if g.empty:
        return 0.0
    return float((g["home_points"] - g["away_points"]).mean())


def site_adjusted_margin(team_games: pd.DataFrame, hfa: float) -> pd.Series:
    """Backs the home-field effect out of each row's raw margin: a home team's raw margin is
    reduced by `hfa` (some of it was the crowd, not the team), an away team's raw margin is
    increased by `hfa` (it overcame a disadvantage to get that margin); neutral-site games are
    left alone."""
    raw_margin = team_games["points_for"] - team_games["points_against"]
    site_adj = np.where(
        team_games["neutral_site"], 0.0,
        np.where(team_games["is_home"], -hfa, hfa),
    )
    return raw_margin + site_adj


def compute_srs(
    team_games: pd.DataFrame,
    hfa: float,
    fbs_teams: set[str],
    non_fbs_pool_name: str = "generic_low_major",
    iterations: int = DEFAULT_SRS_ITERATIONS,
) -> pd.Series:
    """Returns one rating per FBS team (index = team name), mean 0, in points on a neutral
    field. `team_games` must be in the two-row-per-game format games_to_team_game_frame()
    produces, and may cover any span of games (a full season or a partial in-season slice)."""
    teams = sorted(fbs_teams)
    g = team_games[team_games["team"].isin(fbs_teams)].copy()
    if g.empty:
        return pd.Series(0.0, index=teams)

    g["opponent_rated"] = np.where(g["opponent"].isin(fbs_teams), g["opponent"], non_fbs_pool_name)
    g["site_adj_margin"] = site_adjusted_margin(g, hfa)

    non_fbs_mask = g["opponent_rated"] == non_fbs_pool_name
    # Fixed, not iterated: calibrated so that a perfectly-average FBS team's expected margin
    # against a non-FBS opponent, plus this rating, nets to 0 -- see module docstring point 2.
    non_fbs_pool_rating = (
        -float(g.loc[non_fbs_mask, "site_adj_margin"].mean()) if non_fbs_mask.any() else 0.0
    )

    return iterate_ratings(
        teams, g["team"].values, g["opponent_rated"].values, g["site_adj_margin"].values,
        fixed_opponent_ratings={non_fbs_pool_name: non_fbs_pool_rating},
        iterations=iterations,
    )


def iterate_ratings(
    teams: list[str],
    team_col: np.ndarray,
    opponent_col: np.ndarray,
    margin_col: np.ndarray,
    fixed_opponent_ratings: dict[str, float],
    iterations: int = DEFAULT_SRS_ITERATIONS,
) -> pd.Series:
    """The core opponent-adjusted fixed-point iteration, factored out so rating_engine.py's
    in-season blended update can reuse it with extra synthetic ("phantom game") rows mixed
    into team_col/opponent_col/margin_col, rather than duplicating this loop. `teams` is the
    set of entities actually being solved for (their ratings update every pass and recenter to
    mean 0); `fixed_opponent_ratings` holds any additional opponent identities (e.g. the
    non-FBS pool, or rating_engine.py's league-average anchor) whose rating never updates.
    """
    srs = pd.Series(0.0, index=teams)
    for _ in range(iterations):
        lookup = srs.to_dict()
        lookup.update(fixed_opponent_ratings)
        opp_srs = np.array([lookup[o] for o in opponent_col])
        adj_margin = margin_col + opp_srs
        new_srs = (
            pd.Series(adj_margin, index=team_col).groupby(level=0).mean().reindex(teams).fillna(0.0)
        )
        new_srs -= new_srs.mean()
        srs = new_srs

    return srs


def build_historical_srs_table(
    engine, seasons: list[int], hfa: float | None = None, iterations: int = DEFAULT_SRS_ITERATIONS
) -> pd.DataFrame:
    """One row per (team, season): that team's actual, opponent- and site-adjusted power
    rating for that season, computed from every completed game it played. This is the
    training TARGET for the preseason model (modeling/) -- "how good did this team actually
    turn out to be," in the same point units a final rating needs to be in."""
    from cfb_power_ratings.database import get_fbs_teams_by_season, run_query

    if hfa is None:
        all_games = run_query(
            "SELECT * FROM games WHERE completed = 1 AND season IN :seasons",
            params={"seasons": tuple(seasons)}, engine=engine,
        )
        hfa = estimate_home_field_advantage(all_games)

    rows = []
    for season in seasons:
        games = run_query(
            "SELECT * FROM games WHERE completed = 1 AND season = :season",
            params={"season": season}, engine=engine,
        )
        if games.empty:
            continue
        fbs_teams = get_fbs_teams_by_season(engine, season)
        team_games = games_to_team_game_frame(games)
        srs = compute_srs(team_games, hfa, fbs_teams, iterations=iterations)
        rows.append(pd.DataFrame({"team": srs.index, "season": season, "srs": srs.values}))

    if not rows:
        return pd.DataFrame(columns=["team", "season", "srs"])
    return pd.concat(rows, ignore_index=True)
