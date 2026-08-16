"""The most important test file in this project: verifies every feature module's source
season is correctly shifted relative to a target season. Two categories of assertion:

1. t-1 modules (prior_performance, roster_turnover, coaching, schedule's opponent-strength
   columns, program_history): source season must be STRICTLY LESS than target_season.
2. Sanctioned as-is exceptions (returning_production, talent_recruiting): source season must
   be EXACTLY EQUAL to target_season -- a positive assertion guarding against a future
   "fix" that wrongly lags these into an off-by-one bug (see docs/data_leakage_rules.md).
"""

from __future__ import annotations

from cfb_win_total_model.database import run_query
from cfb_win_total_model.features import coaching, prior_performance, program_history, returning_production, roster_turnover, schedule, talent_recruiting

TARGET_SEASON = 2022


def test_prior_performance_source_season_strictly_before_target():
    assert prior_performance._source_season(TARGET_SEASON) < TARGET_SEASON
    assert prior_performance._source_season(TARGET_SEASON) == TARGET_SEASON - 1


def test_roster_turnover_source_seasons_include_target_and_prior():
    t1, t = roster_turnover._source_seasons(TARGET_SEASON)
    assert t1 == TARGET_SEASON - 1
    assert t == TARGET_SEASON


def test_coaching_source_season_strictly_before_target():
    assert coaching._source_season(TARGET_SEASON) < TARGET_SEASON
    assert coaching._source_season(TARGET_SEASON) == TARGET_SEASON - 1


def test_schedule_opponent_strength_source_season_strictly_before_target():
    assert schedule._opponent_strength_source_season(TARGET_SEASON) < TARGET_SEASON


def test_program_history_windows_end_before_target(features_cfg):
    max_window = max(features_cfg.rolling_windows)
    seasons = program_history._source_seasons(TARGET_SEASON, max_window)
    assert all(s < TARGET_SEASON for s in seasons)
    assert max(seasons) == TARGET_SEASON - 1


def test_returning_production_source_season_equals_target():
    """SANCTIONED EXCEPTION: this must equal target_season, not target_season-1."""
    assert returning_production._source_season(TARGET_SEASON) == TARGET_SEASON


def test_talent_recruiting_source_season_equals_target():
    """SANCTIONED EXCEPTION: this must equal target_season, not target_season-1."""
    assert talent_recruiting._source_season(TARGET_SEASON) == TARGET_SEASON


def test_coach_of_record_career_win_pct_excludes_target_season_row(engine, features_cfg):
    """End-to-end: for a school with a known coaching change, career_win_pct_entering_t for
    the incoming coach must not include any wins/games from the coaches row where
    season==target_season, verified by independently recomputing it via raw SQL."""
    df = coaching.build_coaching_features(engine, TARGET_SEASON, features_cfg)
    changed = df[df["coaching_change_indicator"] == True]  # noqa: E712
    assert not changed.empty, "expected at least one coaching change in the test season"

    history = coaching._pull_coaches_history(engine, TARGET_SEASON)
    cor = coaching._coach_of_record(history)
    incoming = cor[cor["season"] == TARGET_SEASON]

    sample_school = changed.iloc[0]["school"]
    incoming_coach = incoming[incoming["school"] == sample_school].iloc[0]

    independent = run_query(
        """
        SELECT SUM(wins) AS w, SUM(games) AS g FROM coaches
        WHERE first_name = :fn AND last_name = :ln AND season < :t AND games > 0
        """,
        params={"fn": incoming_coach["first_name"], "ln": incoming_coach["last_name"], "t": TARGET_SEASON},
        engine=engine,
    )
    if independent["g"].iloc[0] and independent["g"].iloc[0] > 0:
        expected_pct = independent["w"].iloc[0] / independent["g"].iloc[0]
        actual_pct = changed[changed["school"] == sample_school]["career_win_pct_entering_t"].iloc[0]
        assert abs(expected_pct - actual_pct) < 1e-9


def test_returning_production_pre_2014_seasons_flagged_missing(engine, features_cfg):
    df = returning_production.build_returning_production_features(engine, 2013, features_cfg.winsorize_percent_ppa_limits)
    assert df.empty or df["returning_production_missing"].all()
