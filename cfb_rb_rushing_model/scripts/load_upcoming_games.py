#!/usr/bin/env python
"""Insert one upcoming week's not-yet-played games into `games`, so schedule_spine (and
therefore generate_week_predictions.py) has rows to score. SQL Scripts/ingest_to_mysql.R only
writes completed games, so without this an upcoming week is invisible to the model.

ONLY run this against a throwaway DB (the GitHub Actions weekly-rb job restores the
pipeline-state dump into a service container and never dumps it back). In a DB that persists,
ingest_to_mysql.R's needed_weeks() would see this season/week as already ingested and never
pull its final scores -- hence the required --ephemeral flag.

Usage:
    python scripts/load_upcoming_games.py --season 2026 --week 9 --ephemeral
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests
from sqlalchemy import text

from cfb_rb_rushing_model.database import get_engine
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

CFBD_GAMES_URL = "https://api.collegefootballdata.com/games"


def fetch_games(season: int, week: int, season_type: str) -> pd.DataFrame:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set")
    resp = requests.get(
        CFBD_GAMES_URL,
        params={"year": season, "week": week, "seasonType": season_type},
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    resp.raise_for_status()
    games = pd.DataFrame(resp.json())
    if games.empty:
        return games
    return pd.DataFrame({
        "game_id": games["id"],
        "season": games["season"],
        "week": games["week"],
        "season_type": games["seasonType"],
        "start_date": pd.to_datetime(games["startDate"], utc=True).dt.tz_localize(None),
        "start_time_tbd": games["startTimeTBD"].astype(int),
        "completed": games["completed"].astype(int),
        "neutral_site": games["neutralSite"].astype(int),
        "conference_game": games["conferenceGame"].astype(int),
        "home_id": games["homeId"],
        "home_team": games["homeTeam"],
        "home_division": games["homeClassification"],
        "home_conference": games["homeConference"],
        "away_id": games["awayId"],
        "away_team": games["awayTeam"],
        "away_division": games["awayClassification"],
        "away_conference": games["awayConference"],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season-type", default="regular")
    parser.add_argument("--ephemeral", action="store_true", help="Confirm the target DB is a throwaway copy (see module docstring)")
    args = parser.parse_args()
    if not args.ephemeral:
        parser.error("refusing to write without --ephemeral: this DB must not be dumped back (see module docstring)")

    games = fetch_games(args.season, args.week, args.season_type)
    if games.empty:
        logger.warning(f"CFBD returned no games for season={args.season} week={args.week}")
        return 0
    games = games[(games["home_division"] == "fbs") | (games["away_division"] == "fbs")]
    games = games.astype(object).where(games.notna(), None)

    engine = get_engine()
    with engine.begin() as con:
        existing = {r[0] for r in con.execute(text("SELECT game_id FROM games WHERE season = :s AND week = :w"), {"s": args.season, "w": args.week})}
        new = games[~games["game_id"].isin(existing)]
        schools = sorted(set(new["home_team"]) | set(new["away_team"]))
        if schools:
            con.execute(text("INSERT IGNORE INTO teams (school) VALUES (:school)"), [{"school": s} for s in schools])
        if not new.empty:
            cols = list(new.columns)
            con.execute(
                text(f"INSERT IGNORE INTO games ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"),
                new.to_dict("records"),
            )
    logger.info(f"season={args.season} week={args.week}: {len(new)} upcoming games inserted, {len(existing)} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
