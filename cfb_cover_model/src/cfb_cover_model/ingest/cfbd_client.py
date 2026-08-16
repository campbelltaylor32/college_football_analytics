"""Thin wrapper around the official `cfbd` Python client, returning flat pandas
DataFrames with snake_case columns matching what the R layer's `cfbfastR`-based pull
produces (verified against real API responses during development - see
docs/api_ingestion.md). Every function here corresponds to one of the CFBD calls made in
`../R Scripts/Full_CFB_Game_Outcome_Historical.R` / `2025_Game_Update.R`.

Authentication: reads CFBD_API_KEY from the repo root's .env (../. env relative to this
project), via python-dotenv - the same key the R scripts already use.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import cfbd
import pandas as pd
from dotenv import load_dotenv

_ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"  # repo root, sibling of cfb_cover_model/
load_dotenv(_ROOT_ENV)

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    """Only splits after a lowercase/digit and before an uppercase letter, so runs of
    capitals stay together (e.g. "defensiveTDs" -> "defensive_tds", not "defensive_t_ds") -
    matches the base-stat names already documented in docs/data_dictionary.md."""
    return _CAMEL_RE.sub("_", name).lower()


def get_client() -> cfbd.ApiClient:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError(f"CFBD_API_KEY not found - expected it in {_ROOT_ENV}")
    config = cfbd.Configuration(access_token=api_key)
    return cfbd.ApiClient(config)


def _with_backoff(fn, *args, max_retries: int = 4, **kwargs):
    """CFBD free-tier rate limits are modest and undocumented precisely - simple
    exponential backoff on any ApiException, matching the retry behavior
    ../SQL Scripts/ingest_to_mysql.R already relies on for the same API."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except cfbd.ApiException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


# --- games / betting / box score ------------------------------------------------------

def fetch_games(client: cfbd.ApiClient, year: int, week: int | None = None) -> pd.DataFrame:
    """FBS-only, matching the historical R-generated training data's game population (an
    ATS spread-cover model for FBS games only - confirmed empirically: an unfiltered pull
    returns FCS/D2/D3 matchups with no betting line, talent, coaching, or returning-
    production data at all, which would otherwise surface as NaN feature columns in the
    live path - caught during --live smoke testing, see docs/api_ingestion.md)."""
    api = cfbd.GamesApi(client)
    games = _with_backoff(api.get_games, year=year, week=week, season_type="regular", classification="fbs")
    rows = [
        {
            "game_id": g.id, "season": g.season, "week": g.week, "season_type": g.season_type,
            "start_date": g.start_date, "completed": g.completed,
            "neutral_site": g.neutral_site, "conference_game": g.conference_game,
            "venue_id": g.venue_id,
            "home_id": g.home_id, "home_team": g.home_team, "home_conference": g.home_conference,
            "home_points": g.home_points,
            "away_id": g.away_id, "away_team": g.away_team, "away_conference": g.away_conference,
            "away_points": g.away_points,
        }
        for g in games
    ]
    return pd.DataFrame(rows)


def fetch_betting_lines(client: cfbd.ApiClient, year: int, week: int | None = None) -> pd.DataFrame:
    """One row per (game, provider) - mirrors R's tot_betting before the
    consensus/fallback collapse (done separately, see box_score_features.py's
    consensus_or_average_spread)."""
    api = cfbd.BettingApi(client)
    games = _with_backoff(api.get_lines, year=year, week=week, season_type="regular")
    rows = []
    for g in games:
        for line in g.lines or []:
            rows.append(
                {
                    "game_id": g.id, "home_team": g.home_team, "away_team": g.away_team,
                    "provider": line.provider, "spread": line.spread,
                    "formatted_spread": line.formatted_spread, "over_under": line.over_under,
                }
            )
    return pd.DataFrame(rows)


# cfbfastR (R) column names that don't match a plain camelCase->snake_case conversion of
# CFBD's raw field name - verified by diffing against the real historical CSV during
# pipeline validation (see docs/api_ingestion.md).
_STAT_RENAME = {
    "rushing_tds": "rush_tds",
}


def fetch_game_team_stats(client: cfbd.ApiClient, year: int, week: int | None = None) -> pd.DataFrame:
    """Flattens the nested {team: {stats: [{category, stat}]}} response into one row per
    (game, team) with snake_case stat columns - the shape cfbfastR's own R-side
    flattening already produces."""
    api = cfbd.GamesApi(client)
    results = _with_backoff(api.get_game_team_stats, year=year, week=week, season_type="regular")
    rows = []
    for game in results:
        for team in game.teams:
            row = {
                "game_id": game.id, "team": team.team, "conference": team.conference,
                "home_away": team.home_away, "points": team.points,
            }
            for stat in team.stats or []:
                col = _camel_to_snake(stat.category)
                row[_STAT_RENAME.get(col, col)] = stat.stat
            rows.append(row)
    df = pd.DataFrame(rows)
    df["week"] = week
    df["year"] = year
    return df


def fetch_plays(client: cfbd.ApiClient, year: int, week: int) -> pd.DataFrame:
    """KNOWN LIMITATION, not a bug: CFBD's raw `ppa` (predicted points added) field, used
    here as the EPA column, is a *different* model than cfbfastR's own `EPA` column - the R
    package computes EPA with its own internally-trained model, not a passthrough of the
    raw API's `ppa`. Validated during pipeline development (see
    docs/api_ingestion.md): all non-EPA box-score-derived features matched the real
    historical CSV exactly after fixing genuine bugs (column naming, an off-by-one-game
    staleness issue), while EPA-derived features (Total_Offense_EPA and everything built on
    top of it - EPA_per_Play, avg_all/avg3/prev_week variants) show a real, systematic but
    modest per-play gap that accumulates over a game. Reproducing cfbfastR's exact EPA
    model would mean re-deriving a separate, undocumented statistical model - out of scope
    here. `ppa` is a legitimate, correlated proxy, not nonsense, but EPA-derived features
    should be read as approximate when computed via this live pipeline versus the
    R-trained historical values the model was actually fit on."""
    api = cfbd.PlaysApi(client)
    plays = _with_backoff(api.get_plays, year=year, week=week, season_type="regular")
    rows = [
        {
            "id_play": p.id, "drive_id": p.drive_id, "game_id": p.game_id,
            "pos_team": p.offense, "def_pos_team": p.defense,
            "period": p.period, "down": p.down, "distance": p.distance,
            "yards_gained": p.yards_gained, "play_type": p.play_type,
            "EPA": p.ppa,  # CFBD's `ppa` (predicted points added) is the direct raw-API
                            # equivalent of cfbfastR's `EPA` column
        }
        for p in plays
    ]
    df = pd.DataFrame(rows)
    df["week"] = week
    df["year"] = year
    return df


def fetch_drives(client: cfbd.ApiClient, year: int, week: int) -> pd.DataFrame:
    api = cfbd.DrivesApi(client)
    drives = _with_backoff(api.get_drives, year=year, week=week, season_type="regular")
    rows = [
        {
            "drive_id": d.id, "offense": d.offense, "defense": d.defense,
            "scoring": d.scoring, "drive_result": d.drive_result,
        }
        for d in drives
    ]
    df = pd.DataFrame(rows)
    df["week"] = week
    df["year"] = year
    return df


# --- talent / coaching / roster / recruiting / returning production -------------------

def fetch_team_talent(client: cfbd.ApiClient, year: int) -> pd.DataFrame:
    api = cfbd.TeamsApi(client)
    talent = _with_backoff(api.get_talent, year=year)
    df = pd.DataFrame([{"year": t.year, "team": t.team, "talent": t.talent} for t in talent])
    if not df.empty:
        df["Scaled_Talent"] = (df["talent"] - df["talent"].mean()) / df["talent"].std(ddof=0)
    return df


def fetch_coaches(client: cfbd.ApiClient, year: int) -> pd.DataFrame:
    api = cfbd.CoachesApi(client)
    coaches = _with_backoff(api.get_coaches, year=year)
    rows = []
    for c in coaches:
        name = f"{c.first_name} {c.last_name}"
        for s in c.seasons or []:
            if s.year != year:
                continue
            rows.append(
                {"Name": name, "team": s.school, "year": s.year, "games": s.games, "wins": s.wins}
            )
    return pd.DataFrame(rows)


def fetch_roster(client: cfbd.ApiClient, year: int) -> pd.DataFrame:
    api = cfbd.TeamsApi(client)
    roster = _with_backoff(api.get_roster, year=year)
    rows = [
        {"athlete_id": p.id, "team": p.team, "year": year, "position": p.position}
        for p in roster
    ]
    return pd.DataFrame(rows)


def fetch_recruits(client: cfbd.ApiClient, year: int) -> pd.DataFrame:
    api = cfbd.RecruitingApi(client)
    recruits = _with_backoff(api.get_recruits, year=year)
    rows = [
        {"athlete_id": r.athlete_id, "recruit_year": r.year, "stars": r.stars, "rating": r.rating}
        for r in recruits
    ]
    return pd.DataFrame(rows)


def fetch_returning_production(client: cfbd.ApiClient, year: int) -> pd.DataFrame:
    api = cfbd.PlayersApi(client)
    rp = _with_backoff(api.get_returning_production, year=year)
    rows = [
        {
            "season": r.season, "team": r.team,
            "total_ppa": r.total_ppa, "total_passing_ppa": r.total_passing_ppa,
            "total_receiving_ppa": r.total_receiving_ppa, "total_rushing_ppa": r.total_rushing_ppa,
            "percent_ppa": r.percent_ppa, "percent_passing_ppa": r.percent_passing_ppa,
            "percent_receiving_ppa": r.percent_receiving_ppa, "percent_rushing_ppa": r.percent_rushing_ppa,
            "usage": r.usage, "passing_usage": r.passing_usage,
            "receiving_usage": r.receiving_usage, "rushing_usage": r.rushing_usage,
        }
        for r in rp
    ]
    return pd.DataFrame(rows)
