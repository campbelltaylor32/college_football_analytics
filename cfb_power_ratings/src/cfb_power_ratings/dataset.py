"""Assembles the full modeling dataset: one row per (team, season) for every FBS team-season
in `seasons`, joining every preseason feature category plus the target (that season's actual
opponent-adjusted SRS, from srs.build_historical_srs_table).
"""
from __future__ import annotations

import pandas as pd

from cfb_power_ratings.config import FeaturesConfig
from cfb_power_ratings.database import get_fbs_teams_by_season
from cfb_power_ratings.features.coaching import build_coaching_features
from cfb_power_ratings.features.program_history import build_program_history_features
from cfb_power_ratings.features.pythagorean import build_pythagorean_features
from cfb_power_ratings.features.returning_production import build_returning_production_features
from cfb_power_ratings.features.roster_experience import build_roster_experience_features
from cfb_power_ratings.features.roster_turnover import build_roster_turnover_features
from cfb_power_ratings.features.talent_recruiting import build_talent_recruiting_features
from cfb_power_ratings.srs import build_historical_srs_table

FEATURE_COLUMNS = [
    "talent_composite", "blue_chip_ratio", "n_matched_recruits", "avg_recruit_rating",
    "total_ppa", "percent_ppa", "percent_passing_ppa", "percent_receiving_ppa",
    "percent_rushing_ppa", "usage_pct",
    "transfers_in", "transfers_out", "net_transfer_rating",
    "coach_tenure_years", "coach_career_win_pct_prior", "coaching_change",
    "srs_lag1", "srs_lag2", "srs_lag3", "srs_trailing_mean",
    "pythagorean_win_pct_lag1", "win_pct_over_pythagorean_lag1",
]
# Tested and reverted, not included above: features/roster_experience.py's class_avg,
# class_valid_row_share, avg_roster_experience, veteran_roster_share. Walk-forward MAE got
# WORSE with either included (class-based: 6.412 -> 6.618; tenure-based: 6.412 -> 6.450;
# combined: 6.709) -- a real regression, not just a null result like the Pythagorean features.
# The module/its merge into build_modeling_dataset below are kept as tested, working
# infrastructure (the columns are still assembled into the dataframe, just not fed to the
# model) in case a future revision of either signal is worth re-testing. See
# docs/methodology.md for the full diagnostic writeup.


def build_modeling_dataset(
    engine, seasons: list[int], features_cfg: FeaturesConfig, srs_history_start_season: int,
    historical_srs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Returns one row per (team, season) with every column in FEATURE_COLUMNS plus
    `target_srs` (NaN only if that season's SRS couldn't be computed -- shouldn't happen for
    any season with completed games in the DB). `historical_srs` can be injected (tests, or to
    avoid recomputing it once per caller in scripts that need it for other purposes too);
    otherwise it's built fresh, covering `srs_history_start_season` through the latest season
    requested (needed both as the target itself and as program_history's lag features for
    later seasons)."""
    if historical_srs is None:
        history_seasons = list(range(srs_history_start_season, max(seasons) + 1))
        historical_srs = build_historical_srs_table(engine, history_seasons)

    fbs_frames = []
    for s in seasons:
        teams = get_fbs_teams_by_season(engine, s)
        if not teams:
            # get_fbs_teams_by_season reads division info off the `games` table -- for a
            # not-yet-started season (no completed games ingested), that's structurally empty,
            # not just missing. Fall back to a live CFBD pull (same as live_data.py's other
            # DB-completeness fallbacks) rather than silently building an empty frame, which
            # would otherwise surface downstream as a confusing pandas dtype error rather than
            # a clear "no FBS team list available" message.
            from cfb_power_ratings.cfbd_client import get_client
            from cfb_power_ratings.live_data import fetch_fbs_teams

            teams = fetch_fbs_teams(get_client(), s)
            if not teams:
                raise ValueError(f"No FBS team list available for season={s} (DB and live CFBD both empty).")
        fbs_frames.append(pd.DataFrame({"team": sorted(teams), "season": s}))
    base = pd.concat(fbs_frames, ignore_index=True)

    talent = build_talent_recruiting_features(engine, seasons, features_cfg.min_matched_recruits_for_blue_chip_ratio)
    returning = build_returning_production_features(engine, seasons)
    turnover = build_roster_turnover_features(engine, seasons, features_cfg.transfer_portal_start_season)
    coaching = build_coaching_features(engine, seasons)
    history = build_program_history_features(historical_srs, seasons, features_cfg.program_history_trailing_seasons)
    pythagorean = build_pythagorean_features(engine, seasons)
    roster_experience = build_roster_experience_features(
        engine, seasons, features_cfg.min_valid_class_rows, features_cfg.tenure_lookback_seasons
    )

    df = base
    for feature_frame in (talent, returning, turnover, coaching, history, pythagorean, roster_experience):
        df = df.merge(feature_frame, on=["team", "season"], how="left")

    target = historical_srs.rename(columns={"srs": "target_srs"})
    df = df.merge(target, on=["team", "season"], how="left")

    return df
