"""Live CFBD API fallbacks for anything the MySQL DB can't answer: a genuinely future/upcoming
schedule (games/betting_lines are completed-only, see SQL Scripts/README.md), or completed
games for a season/week range the DB hasn't been re-ingested for yet. Column names are kept
identical to the `games` table's own shape so srs.py's functions work on either source
unchanged.
"""
from __future__ import annotations

import pandas as pd


def _game_to_row(g) -> dict:
    return {
        "game_id": g.id, "season": g.season, "week": g.week,
        "home_team": g.home_team, "away_team": g.away_team,
        "home_points": g.home_points, "away_points": g.away_points,
        "home_division": str(getattr(g.home_classification, "value", g.home_classification) or "").lower(),
        "away_division": str(getattr(g.away_classification, "value", g.away_classification) or "").lower(),
        "neutral_site": bool(g.neutral_site), "completed": bool(g.completed),
    }


def fetch_games(client, season: int, week: int) -> pd.DataFrame:
    """Every game scheduled for one season/week, completed or not (completed=False rows have
    NaN home_points/away_points, matching the DB's own shape once a game finishes)."""
    import cfbd

    api = cfbd.GamesApi(client)
    games = api.get_games(year=season, week=week, classification="fbs")
    if not games:
        return pd.DataFrame(columns=["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points", "home_division", "away_division", "neutral_site", "completed"])
    return pd.DataFrame([_game_to_row(g) for g in games])


def fetch_season_schedule(client, season: int) -> pd.DataFrame:
    """The full season's schedule in one call (CFBD's `week` filter is optional) -- every FBS
    game, completed or not. Used for schedule-strength analysis, where the whole season's slate
    is needed up front rather than one week at a time."""
    import cfbd

    api = cfbd.GamesApi(client)
    games = api.get_games(year=season, classification="fbs")
    if not games:
        return pd.DataFrame(columns=["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points", "home_division", "away_division", "neutral_site", "completed"])
    return pd.DataFrame([_game_to_row(g) for g in games])


def fetch_completed_games(client, season: int, weeks: list[int]) -> pd.DataFrame:
    frames = [fetch_games(client, season, w) for w in weeks]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df[df["completed"]] if not df.empty else df


def fetch_fbs_teams(client, season: int) -> set[str]:
    import cfbd

    api = cfbd.TeamsApi(client)
    teams = api.get_teams(year=season)
    return {
        t.school for t in teams
        if str(getattr(t.classification, "value", t.classification)).lower() == "fbs"
    }


def fetch_team_talent(client, season: int) -> pd.DataFrame:
    """Team talent composite for a season straight from CFBD -- used when the local DB's
    team_talent table has no rows for that season yet (verified live: true for 2026 as of this
    writing, even though CFBD's own talent endpoint already has the real composite for every
    team). Same shape as talent_recruiting.py's DB query: columns season, team, talent_composite."""
    import cfbd

    api = cfbd.TeamsApi(client)
    talent = api.get_talent(year=season)
    if not talent:
        return pd.DataFrame(columns=["season", "team", "talent_composite"])
    return pd.DataFrame([{"season": t.year, "team": t.team, "talent_composite": t.talent} for t in talent])
