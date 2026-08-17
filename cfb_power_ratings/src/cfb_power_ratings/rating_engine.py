"""In-season blended rating update -- "SRS with a prior" -- the one genuinely novel piece of
this project (nothing like it exists anywhere else in the repo; see docs/methodology.md).

Each team's preseason model prediction is treated as `phantom_games` synthetic games against a
fixed, rating-0 "league average" opponent, mixed into the same opponent-adjusted fixed-point
iteration srs.py already uses for the historical target. A team with zero real games played
this season gets back its preseason prior, shifted by one constant applied equally to every
team: every week's ratings are recentered to mean 0 across the full FBS field (the same
convention build_historical_srs_table's target already uses, so a model whose raw predictions
average close to 0 -- as a model trained against a mean-0 target should -- only shifts by a
small amount). A team with many real games played is dominated by its actual opponent-adjusted
results, with a smooth transition in between -- no manual if/else blending or discontinuity,
the fade-out falls out of the weighted-average math for free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb_power_ratings.srs import DEFAULT_SRS_ITERATIONS, site_adjusted_margin, games_to_team_game_frame, iterate_ratings

LEAGUE_AVERAGE = "__league_average__"


def update_ratings(
    preseason_priors: pd.Series,
    completed_games_so_far: pd.DataFrame,
    hfa: float,
    fbs_teams: set[str],
    phantom_games: int = 5,
    non_fbs_pool_name: str = "generic_low_major",
    iterations: int = DEFAULT_SRS_ITERATIONS,
) -> pd.DataFrame:
    """`preseason_priors` is a Series indexed by team name (that team's preseason prior
    rating -- points on a neutral field). `completed_games_so_far` is the `games` table's own
    row shape (one row per game), already filtered to whatever the caller considers "so far"
    (e.g. week < N) -- this function has no season/week awareness itself, matching srs.py's
    same any-span-of-games design.

    Returns one row per FBS team: team, rating, games_played, effective_prior_weight (the
    fraction of that team's blended rating that's still attributable to its preseason prior,
    i.e. phantom_games / (phantom_games + games_played) -- purely descriptive, not itself used
    in the rating math, which handles the fade-out implicitly)."""
    teams = sorted(fbs_teams)
    priors = preseason_priors.reindex(teams).fillna(preseason_priors.mean() if len(preseason_priors) else 0.0)

    real_games = pd.DataFrame(columns=["team", "opponent_rated", "site_adj_margin"])
    non_fbs_pool_rating = 0.0
    games_played = pd.Series(0, index=teams)

    if not completed_games_so_far.empty:
        tg = games_to_team_game_frame(completed_games_so_far)
        tg = tg[tg["team"].isin(fbs_teams)].copy()
        if not tg.empty:
            tg["opponent_rated"] = np.where(tg["opponent"].isin(fbs_teams), tg["opponent"], non_fbs_pool_name)
            tg["site_adj_margin"] = site_adjusted_margin(tg, hfa)
            non_fbs_mask = tg["opponent_rated"] == non_fbs_pool_name
            if non_fbs_mask.any():
                non_fbs_pool_rating = -float(tg.loc[non_fbs_mask, "site_adj_margin"].mean())
            real_games = tg[["team", "opponent_rated", "site_adj_margin"]]
            games_played = tg.groupby("team").size().reindex(teams).fillna(0).astype(int)

    phantom = pd.DataFrame({
        "team": np.repeat(teams, phantom_games),
        "opponent_rated": LEAGUE_AVERAGE,
        "site_adj_margin": np.repeat(priors.values, phantom_games),
    })

    combined = pd.concat([real_games, phantom], ignore_index=True)
    # An empty real_games frame (week 1, before any games are played) has no dtype info of
    # its own -- without this cast, concatenating it with phantom's float column upcasts
    # site_adj_margin to object dtype, which silently breaks the arithmetic below.
    combined["site_adj_margin"] = combined["site_adj_margin"].astype(float)
    fixed_ratings = {non_fbs_pool_name: non_fbs_pool_rating, LEAGUE_AVERAGE: 0.0}
    rating = iterate_ratings(
        teams, combined["team"].values, combined["opponent_rated"].values, combined["site_adj_margin"].values,
        fixed_opponent_ratings=fixed_ratings, iterations=iterations,
    )

    effective_prior_weight = phantom_games / (phantom_games + games_played)

    return pd.DataFrame({
        "team": teams,
        "rating": rating.reindex(teams).values,
        "games_played": games_played.reindex(teams).values,
        "effective_prior_weight": effective_prior_weight.reindex(teams).values,
    }).sort_values("rating", ascending=False).reset_index(drop=True)


def implied_matchup(rating_home: float, rating_away: float, hfa: float) -> dict:
    predicted_margin = rating_home - rating_away + hfa
    return {
        "predicted_margin": predicted_margin,
        "favored_team": "home" if predicted_margin > 0 else ("away" if predicted_margin < 0 else "even"),
    }


def win_probability(rating_diff_with_hfa: float | np.ndarray, residual_std: float) -> float | np.ndarray:
    """Same conversion methodology as cfb_cover_model's ResidualProbabilityRegressor
    (modeling/regressor.py): P(home win) = Phi(predicted_margin / residual_std), with
    residual_std fit once from historical (actual_margin - predicted_margin) -- see
    fit_residual_std below. Deliberately reused rather than inventing a new approach."""
    return norm.cdf(np.asarray(rating_diff_with_hfa) / residual_std)


def fit_residual_std(actual_margins: np.ndarray, predicted_margins: np.ndarray) -> float:
    residuals = np.asarray(actual_margins) - np.asarray(predicted_margins)
    std = float(np.std(residuals))
    return std or 1.0


def historical_site_adjusted_residuals(engine, seasons: list[int], hfa: float) -> tuple[np.ndarray, np.ndarray]:
    """(actual, predicted) site-adjusted-margin arrays across every FBS-vs-FBS game in
    `seasons`, `predicted` = that season's own end-of-season SRS differential
    (team_srs - opponent_srs). Used to calibrate win_probability's residual_std once from
    history -- the same "fit once from training residuals" approach
    ResidualProbabilityRegressor uses in cfb_cover_model, applied here to this project's own
    rating rather than a spread-relative one."""
    from cfb_power_ratings.database import get_fbs_teams_by_season, run_query
    from cfb_power_ratings.srs import build_historical_srs_table, games_to_team_game_frame

    srs_table = build_historical_srs_table(engine, seasons, hfa=hfa)
    actual_all, predicted_all = [], []
    for season in seasons:
        games = run_query("SELECT * FROM games WHERE completed = 1 AND season = :season", params={"season": season}, engine=engine)
        if games.empty:
            continue
        fbs_teams = get_fbs_teams_by_season(engine, season)
        tg = games_to_team_game_frame(games)
        tg = tg[tg["team"].isin(fbs_teams) & tg["opponent"].isin(fbs_teams)].copy()
        if tg.empty:
            continue
        tg["site_adj_margin"] = site_adjusted_margin(tg, hfa)
        srs_season = srs_table.loc[srs_table["season"] == season].set_index("team")["srs"]
        predicted = tg["team"].map(srs_season) - tg["opponent"].map(srs_season)
        actual_all.append(tg["site_adj_margin"].to_numpy())
        predicted_all.append(predicted.to_numpy())

    if not actual_all:
        return np.array([]), np.array([])
    return np.concatenate(actual_all), np.concatenate(predicted_all)
