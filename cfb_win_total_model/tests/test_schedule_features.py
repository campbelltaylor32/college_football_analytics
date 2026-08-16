from __future__ import annotations

import pandas as pd

from cfb_win_total_model.database import run_query
from cfb_win_total_model.features.schedule import build_schedule_features

TARGET_SEASON = 2022


def test_n_games_matches_direct_db_count(engine, features_cfg):
    df = build_schedule_features(engine, TARGET_SEASON, features_cfg)
    row = df[df["school"] == "Georgia"].iloc[0]

    direct = run_query(
        """
        SELECT COUNT(*) AS n FROM games
        WHERE season = :season
          AND ((home_team = :school AND home_division = 'fbs') OR (away_team = :school AND away_division = 'fbs'))
        """,
        params={"season": TARGET_SEASON, "school": "Georgia"},
        engine=engine,
    )
    assert row["n_games"] == direct["n"].iloc[0]


def test_home_road_neutral_sum_to_total(engine, features_cfg):
    df = build_schedule_features(engine, TARGET_SEASON, features_cfg)
    assert (df["n_games"] == df["n_home"] + df["n_road"] + df["n_neutral"]).all()


def test_power_conference_classification_is_season_aware(features_cfg):
    # Pre-realignment: Pac-12 was a Power conference.
    assert features_cfg.is_power_conference_opponent("Pac-12", 2022) is True
    # Post-realignment (2024+): Pac-12 label is no longer treated as Power.
    assert features_cfg.is_power_conference_opponent("Pac-12", 2024) is False
    assert features_cfg.is_power_conference_opponent("Pac-12", 2025) is False


def test_bye_and_short_rest_never_cross_season_boundary(engine, features_cfg):
    """The rest/travel gap calculation sorts by start_date within a single season's schedule
    pull (_pull_schedule filters WHERE season = :season) -- verify no gap computation could
    span into a different season by checking the source data itself only contains one
    season's start_dates."""
    from cfb_win_total_model.features.schedule import _pull_schedule

    games = _pull_schedule(engine, TARGET_SEASON)
    dates = pd.to_datetime(games["start_date"])
    assert dates.dt.year.isin([TARGET_SEASON, TARGET_SEASON + 1]).all()  # bowl-season games can spill into Jan
    assert dates.min().year >= TARGET_SEASON
