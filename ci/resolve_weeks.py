#!/usr/bin/env python
"""Decide what this scheduled GitHub Actions run should do, and write it to $GITHUB_OUTPUT.

    python ci/resolve_weeks.py results   # Sunday job: publish week C's results
    python ci/resolve_weeks.py picks     # Tuesday job: publish week C+1's picks

Weeks come from the CFBD calendar (cfb_cover_model.schedule.resolve_weeks) unless the
workflow_dispatch inputs set SEASON / COMPLETED_WEEK. Outputs: season, completed, upcoming,
run ("true"/"false") and reason.

The results job also runs for the final regular-season week (resolve_weeks reports that as
in_season=False), and is skipped when that week's ratings are already in the repo, so the
weekly cron goes quiet through the offseason instead of republishing the same week.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cfb_cover_model" / "src"))

RATINGS_TAG = os.environ.get("RATINGS_TAG", "pg1_sw0.75")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("results", "picks"):
        sys.exit("usage: resolve_weeks.py results|picks")

    season = os.environ.get("SEASON") or None
    completed = os.environ.get("COMPLETED_WEEK") or None
    forced = completed is not None

    if forced:
        if season is None:
            sys.exit("SEASON is required when COMPLETED_WEEK is given")
        season, completed = int(season), int(completed)
        upcoming = completed + 1
        in_season = True
        note = f"manual override: season {season} week {completed}"
    else:
        from cfb_cover_model.ingest import cfbd_client
        from cfb_cover_model.schedule import resolve_weeks

        r = resolve_weeks(cfbd_client.get_client(), season=int(season) if season else None)
        season, completed, upcoming, in_season, note = (
            r.season, r.completed_week, r.upcoming_week, r.in_season, r.note
        )

    run, reason = True, note
    if completed is None:
        run, reason = False, f"no completed week ({note})"
    elif mode == "results" and not forced:
        published = (ROOT / "cfb_power_ratings" / "outputs" / "ratings" / str(season)
                     / f"week_{completed + 1:02d}_ratings_{RATINGS_TAG}.csv")
        if published.exists():
            run, reason = False, f"week {completed} results already published ({published.name})"
    elif mode == "picks" and (upcoming is None or not (in_season or forced)):
        run, reason = False, f"no upcoming week to pick ({note})"

    # Ratings are always "entering week C+1", even after the final week (no matchups to score).
    upcoming = upcoming if upcoming is not None else completed + 1 if completed is not None else ""
    out = {"season": season, "completed": completed if completed is not None else "",
           "upcoming": upcoming, "run": str(run).lower(), "reason": reason}
    for k, v in out.items():
        print(f"{k}={v}")
    if gh := os.environ.get("GITHUB_OUTPUT"):
        with open(gh, "a") as f:
            for k, v in out.items():
                f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()
