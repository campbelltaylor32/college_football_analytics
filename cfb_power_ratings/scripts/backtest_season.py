#!/usr/bin/env python
"""Reconstructs the full week-by-week rating trajectory for a past, fully-completed season and
reports how well it tracked reality as the season progressed -- the main evidence for whether
the phantom-game prior-blending design (rating_engine.py) actually behaves as intended (fading
out smoothly, converging toward market-quality accuracy) rather than just architecturally
existing.

The preseason prior used here is a genuinely out-of-sample prediction: a fresh model trained
only on seasons strictly before the target season (never the target season's own outcome),
independent of whatever scripts/train_preseason_model.py's persisted production model saw
during its own training (which does include recent seasons, since it's meant to make the best
real 2026 prediction, not stay backtest-clean).

Usage: python scripts/backtest_season.py --season 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine, get_fbs_teams_by_season, run_query
from cfb_power_ratings.preseason import load_preseason_model_metadata, predict_out_of_sample_preseason_ratings
from cfb_power_ratings.rating_engine import fit_residual_std, historical_site_adjusted_residuals, update_ratings, win_probability
from cfb_power_ratings.srs import estimate_home_field_advantage
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)


def _market_spreads(engine, season: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT g.game_id, g.week, g.home_team, g.away_team, g.home_points, g.away_points,
               AVG(bl.spread) AS spread
        FROM games g JOIN betting_lines bl ON g.game_id = bl.game_id
        WHERE g.season = :season AND g.completed = 1 AND bl.spread IS NOT NULL
        GROUP BY g.game_id, g.week, g.home_team, g.away_team, g.home_points, g.away_points
        """,
        params={"season": season}, engine=engine,
    )


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    season = args.season

    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    model_name = load_preseason_model_metadata()["model_name"]

    if season not in modeling_cfg.walk_forward_validation_seasons:
        logger.warning(
            f"season={season} isn't in modeling.yaml's walk_forward_validation_seasons "
            f"({modeling_cfg.walk_forward_validation_seasons}) -- still runs an honest "
            "out-of-sample preseason prior (trained on seasons < season), but wasn't part of "
            "the model-selection walk-forward evaluation itself."
        )

    logger.info(f"Training an out-of-sample preseason model for season={season}...")
    preseason_priors = predict_out_of_sample_preseason_ratings(engine, season, features_cfg, modeling_cfg, model_name)

    fbs_teams = get_fbs_teams_by_season(engine, season)
    all_games = run_query("SELECT * FROM games WHERE completed = 1", engine=engine)
    hfa = estimate_home_field_advantage(all_games)

    season_games = run_query("SELECT * FROM games WHERE completed = 1 AND season = :season", params={"season": season}, engine=engine)
    max_week = int(season_games["week"].max())
    market = _market_spreads(engine, season)

    calibration_seasons = [s for s in (modeling_cfg.walk_forward_validation_seasons + [modeling_cfg.final_holdout_season]) if s != season]
    cal_actual, cal_predicted = historical_site_adjusted_residuals(engine, calibration_seasons, hfa)
    residual_std = fit_residual_std(cal_actual, cal_predicted) if len(cal_actual) else 15.0

    sweep_results = []
    weekly_rows = []
    for phantom_games in modeling_cfg.rating_engine.phantom_games_sweep:
        for week in range(1, max_week + 1):
            games_so_far = season_games[season_games["week"] < week]
            ratings = update_ratings(preseason_priors, games_so_far, hfa, fbs_teams, phantom_games=phantom_games).set_index("team")["rating"]

            week_games = market[market["week"] == week].copy()
            if week_games.empty:
                continue
            week_games["home_rating"] = week_games["home_team"].map(ratings)
            week_games["away_rating"] = week_games["away_team"].map(ratings)
            week_games = week_games.dropna(subset=["home_rating", "away_rating"])
            if week_games.empty:
                continue

            week_games["predicted_margin"] = week_games["home_rating"] - week_games["away_rating"] + hfa
            week_games["implied_market_margin"] = -week_games["spread"]
            week_games["actual_margin"] = week_games["home_points"] - week_games["away_points"]
            week_games["home_win_prob"] = win_probability(week_games["predicted_margin"].to_numpy(), residual_std)
            week_games["home_won"] = (week_games["actual_margin"] > 0).astype(int)

            spread_mae = mean_absolute_error(week_games["implied_market_margin"], week_games["predicted_margin"])
            actual_mae = mean_absolute_error(week_games["actual_margin"], week_games["predicted_margin"])
            brier = brier_score_loss(week_games["home_won"], week_games["home_win_prob"])

            weekly_rows.append({
                "phantom_games": phantom_games, "week": week, "n_games": len(week_games),
                "mae_vs_market_spread": spread_mae, "mae_vs_actual_margin": actual_mae, "brier_score": brier,
            })

    weekly_df = pd.DataFrame(weekly_rows)
    for phantom_games, group in weekly_df.groupby("phantom_games"):
        early = group[group["week"] <= 4]
        late = group[group["week"] > 4]
        sweep_results.append({
            "phantom_games": phantom_games,
            "mean_mae_vs_market_all_weeks": float(group["mae_vs_market_spread"].mean()),
            "mean_mae_vs_actual_all_weeks": float(group["mae_vs_actual_margin"].mean()),
            "mean_brier_all_weeks": float(group["brier_score"].mean()),
            "mean_mae_vs_market_weeks_1_4": float(early["mae_vs_market_spread"].mean()) if len(early) else None,
            "mean_mae_vs_market_weeks_5_plus": float(late["mae_vs_market_spread"].mean()) if len(late) else None,
        })

    print(f"\nBacktest: season={season}, model={model_name}, hfa={hfa:.2f}, residual_std={residual_std:.2f}")
    print("\nPhantom-games sensitivity sweep:")
    print(pd.DataFrame(sweep_results).to_string(index=False))

    out_dir = OUTPUTS_MODEL_COMPARISON
    weekly_df.to_csv(out_dir / f"backtest_{season}_weekly.csv", index=False)
    (out_dir / f"backtest_{season}_summary.json").write_text(json.dumps(sweep_results, indent=2))
    print(f"\nWrote {out_dir / f'backtest_{season}_weekly.csv'}")
    print(f"Wrote {out_dir / f'backtest_{season}_summary.json'}")


if __name__ == "__main__":
    main()
