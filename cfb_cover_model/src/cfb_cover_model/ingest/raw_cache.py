"""Fetch-and-cache raw CFBD pulls as local parquet files under data/raw/<endpoint>/ -
idempotent (skips re-fetching a season/week already cached, mirroring the resumable
per-season/week logic in ../SQL Scripts/ingest_to_mysql.R, just file-based instead of a
DB upsert).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfb_cover_model.ingest import cfbd_client

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# (endpoint_name, fetch_fn, is_weekly) - is_weekly=True means cached per (year, week),
# False means cached per year only (talent/coaches/roster/recruits/returning production
# are season-level pulls in the R source, not looped over weeks).
_WEEKLY_ENDPOINTS = {
    "games": cfbd_client.fetch_games,
    "betting_lines": cfbd_client.fetch_betting_lines,
    "game_team_stats": cfbd_client.fetch_game_team_stats,
    "plays": cfbd_client.fetch_plays,
    "drives": cfbd_client.fetch_drives,
}
_SEASON_ENDPOINTS = {
    "team_talent": cfbd_client.fetch_team_talent,
    "coaches": cfbd_client.fetch_coaches,
    "roster": cfbd_client.fetch_roster,
    "recruits": cfbd_client.fetch_recruits,
    "returning_production": cfbd_client.fetch_returning_production,
}


def _path_weekly(endpoint: str, year: int, week: int) -> Path:
    return RAW_DIR / endpoint / f"{year}_{week:02d}.parquet"


def _path_season(endpoint: str, year: int) -> Path:
    return RAW_DIR / endpoint / f"{year}.parquet"


def get_weekly(endpoint: str, year: int, week: int, client, force_refresh: bool = False) -> pd.DataFrame:
    if endpoint not in _WEEKLY_ENDPOINTS:
        raise ValueError(f"Unknown weekly endpoint: {endpoint!r}")
    path = _path_weekly(endpoint, year, week)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch_fn = _WEEKLY_ENDPOINTS[endpoint]
    df = fetch_fn(client, year, week) if endpoint != "games" else fetch_fn(client, year, week)
    df.to_parquet(path, index=False)
    return df


def get_season(endpoint: str, year: int, client, force_refresh: bool = False) -> pd.DataFrame:
    if endpoint not in _SEASON_ENDPOINTS:
        raise ValueError(f"Unknown season endpoint: {endpoint!r}")
    path = _path_season(endpoint, year)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch_fn = _SEASON_ENDPOINTS[endpoint]
    df = fetch_fn(client, year)
    df.to_parquet(path, index=False)
    return df


def get_weekly_range(endpoint: str, year: int, weeks: list[int], client, force_refresh: bool = False) -> pd.DataFrame:
    frames = [get_weekly(endpoint, year, w, client, force_refresh) for w in weeks]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
