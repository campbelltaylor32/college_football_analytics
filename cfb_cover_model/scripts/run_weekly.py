#!/usr/bin/env python
"""The weekly Sunday job (run by deploy/com.cfb.cover-model-weekly.plist via launchd).

One hands-off pass, in order:

  1. resolve which regular-season week just finished + which is next, off the CFBD calendar
     (cfb_cover_model.schedule.resolve_weeks) - no hand-bumped week number
  2. ingest the season so far (weeks 1..completed) -> data/processed/extended_history.parquet
  3. retrain the dual-model production artifact on all history through that week
     -> outputs/models/production/<version>/ + latest.json
  4. score the upcoming week against that just-retrained artifact
     -> outputs/predictions/live_<season>_week_<next>_dual_model_predictions.csv
  5. monitoring: grade last week's picks vs. now-known outcomes, and data-drift the
     upcoming week's feature frame -> outputs/monitoring/

Steps are independent: a failure in one is logged and the rest still run, matching
scripts/run_scheduler.py's existing posture. Every run also appends a summary to
logs/weekly_<YYYYMMDD>.log.

Usage:
    python scripts/run_weekly.py
        (auto-resolve season/weeks from the calendar - the launchd invocation)
    python scripts/run_weekly.py --season 2025 --completed-week 8 --upcoming-week 9
        (explicit override, for manual re-runs and backfills; --upcoming-week optional)
    python scripts/run_weekly.py --only ingest,monitor
        (run a subset of steps: the Sunday results job; the Tuesday picks job uses
         --only ingest,retrain,score. retrain/score read the persisted extended_history, so they
         don't need ingest in the same run)
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts imported below

from cfb_cover_model.ingest import cfbd_client, pipeline
from cfb_cover_model.schedule import resolve_weeks
from cfb_cover_model.serving.artifact import load_latest_artifact
from cfb_cover_model.serving.weekly import score_and_save_upcoming_week

import ingest_and_update_history
import train_production_artifact

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
STEPS = ("ingest", "retrain", "score", "monitor")


def _monitoring():
    """Imported lazily: run_weekly_monitoring pulls in evidently (the `serving` extra), and a
    missing/broken monitoring dependency must not stop ingest/retrain/predict."""
    import run_weekly_monitoring

    return run_weekly_monitoring


class _Tee:
    """Mirror stdout/stderr to a per-day logfile without swallowing the console output
    launchd already captures to logs/launchd.out."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def __getattr__(self, name):
        # fileno/isatty/encoding/etc. - defer to the real stream for anything a library pokes at
        return getattr(self._stream, name)


def _step(label: str, fn) -> tuple[bool, object]:
    print(f"\n=== {label} ===", flush=True)
    try:
        result = fn()
        print(f"[{label}] OK")
        return True, result
    except Exception as exc:  # noqa: BLE001 - deliberately broad; one step must not kill the rest
        print(f"[{label}] FAILED: {exc}")
        traceback.print_exc()
        return False, None


def run_weekly(
    *,
    season: int | None,
    completed_week: int | None,
    upcoming_week: int | None,
    only: set[str] = frozenset(STEPS),
) -> int:
    client = cfbd_client.get_client()

    if completed_week is None:
        resolved = resolve_weeks(client, season=season)
        print(f"[calendar] {resolved.note}")
        if not resolved.in_season:
            print("[calendar] not in the regular season - nothing to do.")
            return 0
        season = resolved.season
        completed_week = resolved.completed_week
        if upcoming_week is None:
            upcoming_week = resolved.upcoming_week
    elif season is None:
        raise SystemExit("--season is required when --completed-week is given")

    print(
        f"[plan] season={season} completed_week={completed_week} "
        f"upcoming_week={upcoming_week} steps={','.join(s for s in STEPS if s in only)}"
    )

    results: dict[str, str] = {}
    hard_failure = False

    # 1. ingest -------------------------------------------------------------------------
    # Pass the whole season-to-date, not just the last week: build_historical_rows needs a
    # team's prior weeks in the same call to compute its rolling/lag features (a lone week
    # gets dropped by drop_incomplete=True). Re-fetches are cheap (raw_cache parquet) and
    # extended_history dedupes on game_id, so this also back-fills any week a prior run
    # missed.
    ingest_weeks = list(range(1, completed_week + 1))
    if "ingest" not in only:
        ok, frame = None, None
        results["ingest"] = "skipped (--only)"
    else:
        ok, frame = _step(
            f"ingest season={season} weeks=1..{completed_week}",
            lambda: ingest_and_update_history.run(season, ingest_weeks),
        )
    if ok is None:
        pass
    elif not ok:
        results["ingest"] = "FAILED"
        hard_failure = True
    elif frame is None or len(frame) == 0:
        results["ingest"] = "ok (no completed games yet this season)"
    else:
        results["ingest"] = f"ok ({len(frame)} rows in extended_history)"

    # 2. retrain -------------------------------------------------------------------------
    if "retrain" not in only:
        results["retrain"] = "skipped (--only)"
    elif completed_week is None or completed_week < pipeline.MIN_WEEK_HISTORICAL:
        results["retrain"] = f"skipped (week {completed_week} < {pipeline.MIN_WEEK_HISTORICAL})"
        print(f"\n[retrain] {results['retrain']}")
    else:
        ok, artifact = _step(
            f"retrain artifact through season={season} week={completed_week}",
            lambda: train_production_artifact.run(season, completed_week),
        )
        if ok:
            results["retrain"] = f"{artifact.version} ({artifact.training_row_count} rows)"
        else:
            results["retrain"] = "FAILED"
            hard_failure = True

    # 3. score the upcoming week -------------------------------------------------------
    if "score" not in only:
        results["predict"] = "skipped (--only)"
    elif upcoming_week is None:
        results["predict"] = "skipped (no upcoming week - regular season ending)"
        print(f"\n[predict] {results['predict']}")
    elif upcoming_week < pipeline.MIN_WEEK_LIVE:
        results["predict"] = f"skipped (week {upcoming_week} < {pipeline.MIN_WEEK_LIVE}, too early for live features)"
        print(f"\n[predict] {results['predict']}")
    elif load_latest_artifact() is None:
        results["predict"] = "skipped (no production artifact to score with)"
        print(f"\n[predict] {results['predict']}")
    else:
        ok, path = _step(
            f"score upcoming season={season} week={upcoming_week}",
            lambda: score_and_save_upcoming_week(season, upcoming_week),
        )
        results["predict"] = str(path) if ok else "FAILED"
        hard_failure |= not ok

    # 4. monitoring (soft - never a hard failure) -------------------------------------
    if "monitor" not in only:
        results["prediction_drift"] = "skipped (--only)"
    else:
        ok, summary = _step(
            f"prediction-drift check season={season} week={completed_week}",
            lambda: _monitoring().run_prediction_drift_check(season, completed_week, None, None),
        )
        results["prediction_drift"] = (
            f"week_precision={summary.get('precision_this_week')} "
            f"rolling={summary.get('rolling_mean_precision')} alert={summary.get('alert')}"
            if ok and isinstance(summary, dict) else ("ok" if ok else "skipped/failed")
        )

    can_data_drift = (
        "monitor" in only
        and upcoming_week is not None
        and upcoming_week >= pipeline.MIN_WEEK_LIVE
        and load_latest_artifact() is not None
    )
    if can_data_drift:
        ok, summary = _step(
            f"data-drift check season={season} week={upcoming_week}",
            lambda: _monitoring().run_data_drift_check(season, upcoming_week, live=True),
        )
        results["data_drift"] = (
            f"dataset_drift={summary.get('dataset_drift')}"
            if ok and isinstance(summary, dict) else ("ok" if ok else "skipped/failed")
        )
    else:
        results["data_drift"] = "skipped"

    # summary --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"weekly run summary  (season={season}, completed=w{completed_week}, upcoming=w{upcoming_week})")
    for k, v in results.items():
        print(f"  {k:18s} {v}")
    print("=" * 60)

    return 1 if hard_failure else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--completed-week", type=int, default=None,
                        help="Skip calendar auto-resolution; treat this week as just-completed")
    parser.add_argument("--upcoming-week", type=int, default=None,
                        help="Week to score (default: completed-week + 1 / calendar's next week)")
    parser.add_argument("--only", default=",".join(STEPS),
                        help=f"Comma-separated subset of steps to run: {','.join(STEPS)} (default: all)")
    args = parser.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if bad := only - set(STEPS):
        parser.error(f"--only: unknown step(s) {sorted(bad)}; choose from {','.join(STEPS)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    logfile = open(LOG_DIR / f"weekly_{stamp}.log", "a")
    logfile.write(f"\n\n########## run_weekly {datetime.now(timezone.utc).isoformat()} ##########\n")
    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)

    try:
        code = run_weekly(
            season=args.season,
            completed_week=args.completed_week,
            upcoming_week=args.upcoming_week,
            only=only,
        )
    finally:
        logfile.flush()
        logfile.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    sys.exit(code)


if __name__ == "__main__":
    main()
