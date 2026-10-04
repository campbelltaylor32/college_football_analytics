#!/usr/bin/env python
"""Weekly in-season update: blends the preseason prior with every completed game through week
N-1, writes the resulting power ratings, and scores every real matchup on week N's schedule
(predicted margin, favored team, home win probability).

Completed games/FBS-team-list are read from the MySQL DB first (fast, no rate limit); if the DB
has no rows yet for this season (e.g. a new season SQL Scripts/ingest_to_mysql.R hasn't been
re-run for), falls back to a live CFBD pull. The upcoming week's schedule is ALWAYS pulled live
-- games/betting_lines in the DB are completed-only by design (see SQL Scripts/README.md), so
there is no DB path for "who plays whom next week."

Each game's margin is a blend of its actual score and its play-by-play efficiency-implied
margin (efficiency.py; weight from modeling.yaml's efficiency.scoring_weight or
--scoring-weight). The ratings CSV also carries reporting-only offense/defense EPA/play and
success-rate columns, raw and opponent-adjusted.

Usage: python scripts/update_ratings.py --season 2026 --week 4 [--phantom-games 3] [--scoring-weight 0.5]
    (blends games through week 3, scores week 4's real matchups)

Outputs are versioned by the weighting used, so runs with different weightings sit side by
side instead of overwriting each other: week_<NN>_ratings_pg<P>_sw<W>.csv and
week_<NN>_matchups_pg<P>_sw<W>.csv (e.g. week_05_ratings_pg5_sw0.5.csv). Pass the same
`pg<P>_sw<W>` tag to the plot scripts as a third arg to chart a given version.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cfb_power_ratings.config import load_features_config, load_modeling_config
from cfb_power_ratings.database import get_engine, get_fbs_teams_by_season, run_query
from cfb_power_ratings.efficiency import (
    efficiency_margins,
    fit_efficiency_to_points,
    load_team_game_efficiency,
    opponent_adjusted_off_def,
    team_game_frame,
)
from cfb_power_ratings.preseason import load_preseason_model_metadata, predict_preseason_ratings
from cfb_power_ratings.rating_engine import (
    fit_residual_std,
    historical_site_adjusted_residuals,
    implied_matchup,
    update_ratings,
    win_probability,
)
from cfb_power_ratings.utils.logging import get_logger
from cfb_power_ratings.utils.paths import OUTPUTS_RATINGS

logger = get_logger(__name__)


def _completed_games_through(engine, season: int, week: int) -> pd.DataFrame:
    db_games = run_query(
        "SELECT * FROM games WHERE completed = 1 AND season = :season AND week < :week",
        params={"season": season, "week": week}, engine=engine,
    )
    if not db_games.empty:
        return db_games

    logger.warning(f"No completed games in the DB for season={season} week<{week} -- falling back to a live CFBD pull.")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_completed_games

    return fetch_completed_games(get_client(), season, list(range(1, week)))


def _fbs_teams(engine, season: int) -> set[str]:
    teams = get_fbs_teams_by_season(engine, season)
    if teams:
        return teams
    logger.warning(f"No FBS team list in the DB for season={season} -- falling back to a live CFBD pull.")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_fbs_teams

    return fetch_fbs_teams(get_client(), season)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True, help="Week to score; games through week-1 are blended in")
    parser.add_argument("--phantom-games", type=float, default=None, help="Override modeling.yaml's default_phantom_games")
    parser.add_argument(
        "--scoring-weight", type=float, default=None,
        help="Share of each game's margin from the actual score (rest from efficiency); overrides modeling.yaml. 1.0 = scoring only",
    )
    args = parser.parse_args()

    engine = get_engine()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    # `is not None`, not `or` -- --phantom-games 0 (pure results, no prior) is a valid override.
    phantom_games = (
        args.phantom_games if args.phantom_games is not None else modeling_cfg.rating_engine.default_phantom_games
    )
    eff_cfg = modeling_cfg.efficiency
    if args.scoring_weight is not None:
        scoring_weight = args.scoring_weight
    else:
        scoring_weight = eff_cfg.scoring_weight if eff_cfg.enabled else 1.0

    metadata = load_preseason_model_metadata()
    hfa = float(metadata["hfa"])

    logger.info(f"Predicting preseason priors for season={args.season}")
    preseason_priors = predict_preseason_ratings(engine, args.season, features_cfg, modeling_cfg.srs_history_start_season)

    logger.info(f"Pulling completed games for season={args.season} through week {args.week - 1}")
    games_so_far = _completed_games_through(engine, args.season, args.week)
    fbs_teams = _fbs_teams(engine, args.season)

    logger.info(f"Pulling play-by-play efficiency for season={args.season} through week {args.week - 1}")
    season_eff = team_game_frame(
        load_team_game_efficiency(engine, [args.season], max_week=args.week, garbage=eff_cfg.garbage_time)
    )
    eff_margins = None
    if scoring_weight != 1.0:
        if season_eff.empty:
            logger.warning("No play-by-play rows for this season in the DB -- falling back to scoring-only ratings.")
            scoring_weight = 1.0
        else:
            calib_seasons = [
                s for s in range(eff_cfg.calibration_start_season, args.season) if s not in modeling_cfg.excluded_seasons
            ]
            calibration = fit_efficiency_to_points(
                engine, calib_seasons, {s: get_fbs_teams_by_season(engine, s) for s in calib_seasons},
                garbage=eff_cfg.garbage_time,
            )
            logger.info(
                f"Efficiency->points calibration ({calib_seasons[0]}-{calib_seasons[-1]}): "
                f"{calibration.coef_epa:.2f} * net EPA/play + {calibration.coef_sr:.2f} * net success rate "
                f"(R^2 {calibration.r2:.3f}, MAE {calibration.mae:.2f}, n={calibration.n_team_games})"
            )
            eff_margins = efficiency_margins(season_eff, calibration)

    blended = update_ratings(
        preseason_priors, games_so_far, hfa, fbs_teams, phantom_games=phantom_games,
        efficiency_margins=eff_margins, scoring_weight=scoring_weight,
    )
    blended.insert(0, "rank", range(1, len(blended) + 1))
    # Tagged after the efficiency fallback above, so the filename reflects the weight actually used.
    version = f"pg{phantom_games:g}_sw{scoring_weight:g}"

    # Reporting-only offense/defense columns: season-to-date raw (play-weighted) and
    # opponent/site-adjusted. None of these feed the rating itself.
    if not season_eff.empty:
        w_off = season_eff["off_plays"]
        w_def = season_eff["def_plays"]
        raw_eff = season_eff.assign(
            _oe=season_eff["off_epa"] * w_off, _os=season_eff["off_sr"] * w_off,
            _de=season_eff["def_epa_allowed"] * w_def, _ds=season_eff["def_sr_allowed"] * w_def,
        ).groupby("team")[["_oe", "_os", "_de", "_ds", "off_plays", "def_plays"]].sum()
        raw_eff = pd.DataFrame({
            "off_epa": raw_eff["_oe"] / raw_eff["off_plays"], "off_sr": raw_eff["_os"] / raw_eff["off_plays"],
            "def_epa_allowed": raw_eff["_de"] / raw_eff["def_plays"], "def_sr_allowed": raw_eff["_ds"] / raw_eff["def_plays"],
        }).reset_index()
        adj_eff = opponent_adjusted_off_def(season_eff, games_so_far, fbs_teams, modeling_cfg.srs.non_fbs_pool_name)
        blended = blended.merge(raw_eff, on="team", how="left").merge(adj_eff, on="team", how="left")

    out_dir = OUTPUTS_RATINGS / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / f"week_{args.week:02d}_ratings_{version}.csv"
    blended.to_csv(ratings_path, index=False)
    print(
        f"Ratings through week {args.week - 1}, season {args.season}, phantom_games={phantom_games:g}, "
        f"scoring_weight={scoring_weight:g} (top 25):"
    )
    print(blended.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {ratings_path}")

    logger.info(f"Pulling week {args.week}'s real schedule live and scoring matchups")
    from cfb_power_ratings.cfbd_client import get_client
    from cfb_power_ratings.live_data import fetch_games

    schedule = fetch_games(get_client(), args.season, args.week)
    schedule = schedule[schedule["home_team"].isin(fbs_teams) | schedule["away_team"].isin(fbs_teams)]

    calibration_seasons = modeling_cfg.walk_forward_validation_seasons + [modeling_cfg.final_holdout_season]
    actual, predicted = historical_site_adjusted_residuals(engine, calibration_seasons, hfa)
    residual_std = fit_residual_std(actual, predicted) if len(actual) else 15.0

    ratings_lookup = blended.set_index("team")["rating"]
    matchup_rows = []
    for _, g in schedule.iterrows():
        home_rating = ratings_lookup.get(g["home_team"])
        away_rating = ratings_lookup.get(g["away_team"])
        if home_rating is None or away_rating is None:
            continue
        im = implied_matchup(home_rating, away_rating, hfa)
        prob = float(win_probability(im["predicted_margin"], residual_std))
        matchup_rows.append({
            "home_team": g["home_team"], "away_team": g["away_team"],
            "predicted_margin": im["predicted_margin"], "favored_team": im["favored_team"],
            "home_win_probability": prob,
        })

    matchups = pd.DataFrame(matchup_rows).sort_values("predicted_margin", ascending=False)
    matchups_path = out_dir / f"week_{args.week:02d}_matchups_{version}.csv"
    matchups.to_csv(matchups_path, index=False)
    print(f"\nWeek {args.week} matchups ({len(matchups)} games):")
    print(matchups.to_string(index=False))
    print(f"\nWrote {matchups_path}")


if __name__ == "__main__":
    main()
