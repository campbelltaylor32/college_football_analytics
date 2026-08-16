"""Pure-function tests against small synthetic DataFrames -- no DB needed. Exercises
eligibility.compute_eligibility_from_asof_source directly, the DB-independent core extracted
from build_eligibility_spine specifically for this kind of test."""

from __future__ import annotations

import pandas as pd

from cfb_rb_rushing_model.config import EligibilityConfig
from cfb_rb_rushing_model.eligibility import compute_eligibility_from_asof_source

CFG = EligibilityConfig(min_trailing3_avg_carries=8, min_season_to_date_carries=15, min_games_played_for_avg3=3)


def _asof_source_row(athlete_id, start_date, carries_avg3, carries_avg_all, games_played_lag1, **extra):
    row = {
        "athlete_id": athlete_id,
        "start_date": pd.Timestamp(start_date),
        "carries_avg3": carries_avg3,
        "carries_avg_all": carries_avg_all,
        "career_games_played_lag1": games_played_lag1,
    }
    other_cols = ["rushing_yards", "yards_per_carry", "success_rate", "explosive_runs", "explosive_run_rate", "stuffed_run_rate", "red_zone_carries", "avg_epa_per_rush", "first_down_rate"]
    for col in other_cols:
        row[f"{col}_avg3"] = extra.get(col, 0.0)
        row[f"{col}_avg_all"] = extra.get(col, 0.0)
    return row


def test_player_with_no_prior_played_game_is_ineligible():
    """A true debut/transfer with zero prior recorded games is eligible=False by construction.
    asof_source carries a real, properly-typed row for a DIFFERENT athlete (999) -- realistic
    shape of the production table, where the candidate athlete (1) simply has no rows of
    their own yet, rather than the whole table being empty."""
    candidates = pd.DataFrame({"athlete_id": [1], "team": ["Ohio"], "game_id": [100], "start_date": [pd.Timestamp("2023-09-02")]})
    asof_source = pd.DataFrame([_asof_source_row(999, "2023-08-26", carries_avg3=20.0, carries_avg_all=20.0, games_played_lag1=0)])
    result = compute_eligibility_from_asof_source(candidates, asof_source, CFG)
    assert result.loc[0, "eligible"] == False  # noqa: E712
    assert result.loc[0, "prior_games_played"] == 0


def test_merge_asof_carries_workload_forward_across_a_bye_week():
    """A player who played weeks 1-3 (well above threshold) then had a bye in week 4 should
    still be eligible for their week-5 game, using week 3's carried-forward inclusive average
    -- not treated as having no history just because week 4 has no played-game row."""
    athlete_id = 42
    asof_source = pd.DataFrame(
        [
            _asof_source_row(athlete_id, "2023-09-02", carries_avg3=15.0, carries_avg_all=15.0, games_played_lag1=0),
            _asof_source_row(athlete_id, "2023-09-09", carries_avg3=14.0, carries_avg_all=14.5, games_played_lag1=1),
            _asof_source_row(athlete_id, "2023-09-16", carries_avg3=16.0, carries_avg_all=15.0, games_played_lag1=2),
        ]
    )
    # Week 4 (bye, no played-game row) then week 5 target game.
    candidates = pd.DataFrame(
        {"athlete_id": [athlete_id], "team": ["Ohio"], "game_id": [999], "start_date": [pd.Timestamp("2023-09-30")]}
    )
    result = compute_eligibility_from_asof_source(candidates, asof_source, CFG)
    assert result.loc[0, "prior_games_played"] == 3
    assert result.loc[0, "carries_avg3_asof"] == 16.0  # carried forward from the 2023-09-16 game, not reset by the bye
    assert result.loc[0, "eligible"] == True  # noqa: E712


def test_allow_exact_matches_false_excludes_same_game_self_reference():
    """A candidate row whose start_date exactly matches a played-game row (i.e. this IS that
    played game) must NOT pull in that same game's own inclusive stats -- allow_exact_matches
    must be False, or this would be a direct lookahead leak."""
    athlete_id = 7
    same_date = pd.Timestamp("2023-09-16")
    asof_source = pd.DataFrame(
        [
            _asof_source_row(athlete_id, "2023-09-02", carries_avg3=20.0, carries_avg_all=20.0, games_played_lag1=0),
            _asof_source_row(athlete_id, same_date, carries_avg3=99.0, carries_avg_all=99.0, games_played_lag1=1),
        ]
    )
    candidates = pd.DataFrame({"athlete_id": [athlete_id], "team": ["Ohio"], "game_id": [999], "start_date": [same_date]})
    result = compute_eligibility_from_asof_source(candidates, asof_source, CFG)
    # Must resolve to the PRIOR game (carries_avg3=20), never the same-date row (99).
    assert result.loc[0, "carries_avg3_asof"] == 20.0


def test_early_season_fallback_gate_uses_total_season_to_date_carries():
    """A player with only 1 prior game (below min_games_played_for_avg3=3) is judged on total
    season-to-date carries, not the (not-yet-meaningful) trailing-3 average."""
    athlete_id = 3
    asof_source = pd.DataFrame(
        [_asof_source_row(athlete_id, "2023-09-02", carries_avg3=20.0, carries_avg_all=20.0, games_played_lag1=0)]
    )
    candidates = pd.DataFrame({"athlete_id": [athlete_id], "team": ["Ohio"], "game_id": [999], "start_date": [pd.Timestamp("2023-09-09")]})
    result = compute_eligibility_from_asof_source(candidates, asof_source, CFG)
    assert result.loc[0, "prior_games_played"] == 1
    # total season-to-date carries = carries_avg_all(20) * prior_games_played(1) = 20 >= 15 -> eligible
    assert result.loc[0, "eligible"] == True  # noqa: E712


def test_below_both_thresholds_is_ineligible():
    athlete_id = 5
    asof_source = pd.DataFrame(
        [_asof_source_row(athlete_id, "2023-09-02", carries_avg3=2.0, carries_avg_all=2.0, games_played_lag1=0)]
    )
    candidates = pd.DataFrame({"athlete_id": [athlete_id], "team": ["Ohio"], "game_id": [999], "start_date": [pd.Timestamp("2023-09-09")]})
    result = compute_eligibility_from_asof_source(candidates, asof_source, CFG)
    assert result.loc[0, "eligible"] == False  # noqa: E712
