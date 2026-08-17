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
