"""Walk-forward evaluation of every baseline + candidate model against the actual SRS target,
plus a market-based external sanity check (does the winning model's implied spread track the
real consensus opening line?) independent of the model's own target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from cfb_power_ratings.dataset import FEATURE_COLUMNS
from cfb_power_ratings.modeling.baselines import BASELINES
from cfb_power_ratings.modeling.models import get_candidate_models
from cfb_power_ratings.modeling.splits import walk_forward_folds


def walk_forward_evaluate(df: pd.DataFrame, modeling_cfg) -> pd.DataFrame:
    """Returns one row per (model_name, season) with mae + n; callers pool across seasons
    themselves (see main()'s pooled-MAE summary) since fold sizes vary."""
    df = df.dropna(subset=["target_srs"])
    candidate_models = get_candidate_models(modeling_cfg.candidate_models, modeling_cfg.random_seed)

    rows = []
    for val_season, train, val in walk_forward_folds(
        df, modeling_cfg.walk_forward_validation_seasons, modeling_cfg.excluded_seasons, modeling_cfg.min_train_seasons
    ):
        y_val = val["target_srs"].to_numpy()

        for name, fn in BASELINES.items():
            pred = fn(train, val)
            rows.append({"model": name, "season": val_season, "mae": mean_absolute_error(y_val, pred), "n": len(val)})

        X_train, y_train = train[FEATURE_COLUMNS], train["target_srs"]
        X_val = val[FEATURE_COLUMNS]
        for name, model in candidate_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rows.append({"model": name, "season": val_season, "mae": mean_absolute_error(y_val, pred), "n": len(val)})

    return pd.DataFrame(rows)


def walk_forward_predictions(df: pd.DataFrame, modeling_cfg, model_name: str) -> pd.DataFrame:
    """Out-of-fold predictions for one specific model across every walk-forward validation
    season -- used both for the consensus-spread sanity check and for eyeballing individual
    team predictions. `model_name` must be a key in BASELINES or modeling_cfg.candidate_models."""
    df = df.dropna(subset=["target_srs"])
    is_baseline = model_name in BASELINES
    rows = []
    for val_season, train, val in walk_forward_folds(
        df, modeling_cfg.walk_forward_validation_seasons, modeling_cfg.excluded_seasons, modeling_cfg.min_train_seasons
    ):
        if is_baseline:
            pred = BASELINES[model_name](train, val)
        else:
            model = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
            model.fit(train[FEATURE_COLUMNS], train["target_srs"])
            pred = model.predict(val[FEATURE_COLUMNS])
        rows.append(pd.DataFrame({
            "team": val["team"].values, "season": val_season,
            "predicted_srs": pred, "actual_srs": val["target_srs"].values,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["team", "season", "predicted_srs", "actual_srs"])


def pooled_mae(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Weights each fold by its row count rather than averaging fold MAEs unweighted, so a
    56-team fold and a 136-team fold don't count equally."""
    return (
        fold_results.assign(weighted_error=fold_results["mae"] * fold_results["n"])
        .groupby("model")
        .agg(total_weighted_error=("weighted_error", "sum"), n=("n", "sum"))
        .assign(pooled_mae=lambda d: d["total_weighted_error"] / d["n"])
        ["pooled_mae"]
        .sort_values()
    )


def evaluate_against_consensus_spread(engine, predictions: pd.DataFrame, season: int, hfa: float) -> dict:
    """External, target-independent sanity check: for every real game that season, does
    `predicted_srs_home - predicted_srs_away + hfa` track the actual market spread?
    `predictions` must have columns team, season, predicted_srs (one row per team-season, e.g.
    a walk-forward validation fold's predictions or a trained model's fitted values).
    betting_lines.spread is negative when the home team is favored (verified against real
    data), so the implied home margin is `-spread`.

    Averages `spread` across every provider for a game rather than filtering to
    `provider = 'consensus'` -- verified live that CFBD's own "consensus" field is only
    populated through the 2022 season (832-1201 games/season 2013-2022, 29 in 2023, 0 in
    2024-2025), while per-game rows from *some* provider exist for every season through 2025.
    A same-game average across whichever books reported that week is a reasonable synthetic
    consensus and keeps this check usable for recent seasons."""
    from cfb_power_ratings.database import run_query

    games = run_query(
        """
        SELECT g.game_id, g.home_team, g.away_team, AVG(bl.spread) AS spread
        FROM games g JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.season = :season AND g.completed = 1 AND bl.spread IS NOT NULL
        GROUP BY g.game_id, g.home_team, g.away_team
        """,
        params={"season": season}, engine=engine,
    )
    if games.empty:
        return {"season": season, "n_games": 0, "mae": None, "correlation": None}

    ratings = predictions[predictions["season"] == season].set_index("team")["predicted_srs"]
    games["home_rating"] = games["home_team"].map(ratings)
    games["away_rating"] = games["away_team"].map(ratings)
    games = games.dropna(subset=["home_rating", "away_rating", "spread"])
    if games.empty:
        return {"season": season, "n_games": 0, "mae": None, "correlation": None}

    games["predicted_margin"] = games["home_rating"] - games["away_rating"] + hfa
    games["implied_market_margin"] = -games["spread"]

    mae = float(mean_absolute_error(games["implied_market_margin"], games["predicted_margin"]))
    corr = float(np.corrcoef(games["predicted_margin"], games["implied_market_margin"])[0, 1])
    return {"season": season, "n_games": len(games), "mae": mae, "correlation": corr}
