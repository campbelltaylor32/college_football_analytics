#!/usr/bin/env python
"""Long-running weekly-cadence scheduler (its own docker-compose service). Runs, once per
week on a configurable day/hour:

  1. scripts/ingest_and_update_history.py - pull the just-completed week directly from the
     CFBD API (no R dependency), appending to data/processed/extended_history.parquet.
  2. POST {API_BASE_URL}/admin/retrain - the API (not this process) retrains and persists a
     new artifact and hot-swaps it in, so it's the only writer of
     outputs/models/production/. See docs/serving_and_monitoring.md for why.
  3. scripts/run_weekly_monitoring.py - grade the completed week's already-generated
     predictions against these newly-known outcomes, and check data drift on the upcoming
     week's incoming feature frame.

PRODUCTION_SEASON/PRODUCTION_WEEK are read fresh from the environment on every job fire (not
cached at process start) - "the week that just completed" - matching the same manual-bump
convention the R pipeline's week_update/week variables already use elsewhere in this repo.
Bumping them for a new week requires restarting this container
(`docker compose restart scheduler` or `up -d`).

Usage:
    python scripts/run_scheduler.py            # long-running, fires weekly per SCHEDULE_*
    python scripts/run_scheduler.py --run-once  # run the weekly job immediately, once, then exit
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

SCRIPTS_DIR = Path(__file__).resolve().parent


def _env_int(name: str, required: bool = True, default: int | None = None) -> int | None:
    val = os.environ.get(name)
    if val is None:
        if required and default is None:
            raise RuntimeError(f"{name} is not set - it must be bumped weekly (see module docstring).")
        return default
    return int(val)


def _run_subprocess(args: list[str], label: str) -> bool:
    print(f"[{label}] running: {' '.join(args)}")
    result = subprocess.run([sys.executable] + args, cwd=SCRIPTS_DIR.parent)
    ok = result.returncode == 0
    print(f"[{label}] {'OK' if ok else f'FAILED (exit {result.returncode})'}")
    return ok


def weekly_job() -> None:
    season = _env_int("PRODUCTION_SEASON")
    week = _env_int("PRODUCTION_WEEK")
    api_base_url = os.environ.get("API_BASE_URL", "http://api:8000")
    admin_token = os.environ.get("ADMIN_API_TOKEN", "")
    monitoring_live = os.environ.get("MONITORING_LIVE", "true").lower() in ("1", "true", "yes")

    print(f"=== weekly_job start: season={season} week={week} ===")

    ingest_ok = _run_subprocess(
        [str(SCRIPTS_DIR / "ingest_and_update_history.py"), "--season", str(season), "--weeks", str(week)],
        "ingest",
    )
    if not ingest_ok:
        print("[scheduler] ingest failed - continuing to retrain/monitor anyway, on whatever history already exists.")

    try:
        resp = requests.post(
            f"{api_base_url}/admin/retrain",
            json={"season": season, "week": week},
            headers={"X-Admin-Token": admin_token},
            timeout=300,
        )
        resp.raise_for_status()
        print(f"[retrain] OK: {resp.json()}")
    except requests.RequestException as e:
        print(f"[retrain] FAILED: {e}", file=sys.stderr)

    monitoring_args = [
        str(SCRIPTS_DIR / "run_weekly_monitoring.py"),
        "--completed-season", str(season), "--completed-week", str(week),
        "--upcoming-season", str(season), "--upcoming-week", str(week + 1),
    ]
    if monitoring_live:
        monitoring_args.append("--live")
    _run_subprocess(monitoring_args, "monitoring")

    print(f"=== weekly_job end: season={season} week={week} ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true", help="Run the weekly job immediately, once, then exit")
    args = parser.parse_args()

    if args.run_once:
        weekly_job()
        return

    from apscheduler.schedulers.blocking import BlockingScheduler

    day_of_week = os.environ.get("SCHEDULE_DAY_OF_WEEK", "tue")
    hour = int(os.environ.get("SCHEDULE_HOUR", 6))

    scheduler = BlockingScheduler()
    scheduler.add_job(weekly_job, "cron", day_of_week=day_of_week, hour=hour)
    print(f"Scheduler started - weekly_job will fire every {day_of_week} at {hour:02d}:00 UTC.")
    scheduler.start()


if __name__ == "__main__":
    main()
