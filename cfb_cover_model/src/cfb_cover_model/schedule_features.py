"""Bye-week detection from a team's own played-game schedule.

Deliberately NOT part of engineered_features.py's apply_engineered_features hook: that hook
only ever sees one game-row-per-team at a time (a single week's slate for the live path), which
isn't enough to tell whether a team's *previous* week was idle - that requires the team's whole
season-to-date schedule. Callers (load_and_validate_dataset.py for historical,
ingest/pipeline.py for live) must attach home_off_bye/away_off_bye as raw columns before the
row ever reaches build_clean_modeling_frame / prepare_week_frame, at which point they flow
through candidate_feature_columns like any other context column (neutral_site,
conference_game) - no config/data.yaml registration needed.

Also deliberately sourced from the *unfiltered* game schedule (every game a team played,
including season-opening weeks that a stats-completeness join later drops), not from the
already-filtered modeling frame - see docs/data_leakage_rules.md's bye-week rule: deriving this
from a frame that's already missing early-season rows (predictors_csv's week>=3 na.omit()) would
misread "row absent because stats were incomplete" as "row absent because of a bye."
"""
from __future__ import annotations

import pandas as pd


def compute_bye_flags(schedule: pd.DataFrame) -> pd.DataFrame:
    """schedule: game rows with season, week, home_team, away_team - the complete,
    unfiltered set of games each team played (or is scheduled to play), not just the rows
    that survived candidate-feature filtering.

    Returns one row per (team, season, week) the team appears in (as home or away),
    with off_bye: True if the team has an earlier played week that same season but did not
    play the week immediately before this one. A team's first appearance of a season is
    always off_bye=False - there's no prior week for it to have skipped."""
    home = schedule[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away = schedule[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    team_week = pd.concat([home, away], ignore_index=True).drop_duplicates()
    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)

    prev_played_week = team_week.groupby(["team", "season"])["week"].shift(1)
    team_week["off_bye"] = prev_played_week.notna() & (team_week["week"] - prev_played_week > 1)

    return team_week[["team", "season", "week", "off_bye"]]


def attach_bye_flags(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Left-join home_off_bye/away_off_bye (0/1 ints) onto `games` (season, week, home_team,
    away_team, ... one row per game to score/train on) using flags computed from `schedule`
    (see compute_bye_flags - pass the complete, unfiltered schedule here even when `games`
    itself is a filtered subset, e.g. the predictors frame with week>=3 already applied)."""
    flags = compute_bye_flags(schedule)

    home_flags = flags.rename(columns={"team": "home_team", "off_bye": "home_off_bye"})
    away_flags = flags.rename(columns={"team": "away_team", "off_bye": "away_off_bye"})

    result = games.merge(home_flags, on=["season", "week", "home_team"], how="left")
    result = result.merge(away_flags, on=["season", "week", "away_team"], how="left")
    result["home_off_bye"] = result["home_off_bye"].fillna(False).astype(int)
    result["away_off_bye"] = result["away_off_bye"].fillna(False).astype(int)
    return result
