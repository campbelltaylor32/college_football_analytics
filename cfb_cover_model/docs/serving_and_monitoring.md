# Serving and monitoring

A FastAPI inference service for the dual-model agreement signal (`logistic_regression` +
`xgboost_regressor`, see `docs/final_writeup_2026.md`), plus weekly data-drift and
prediction/calibration-drift monitoring — both run locally via Docker Compose, no cloud
dependency. This closes the gap `docs/final_writeup_2026.md`'s "what actually shipped" section
and (the former) `outputs/models/README.md` both flagged: until this existed, nothing was ever
persisted to disk, and `scripts/generate_weekly_predictions.py` refit both models from scratch on
every single run.

## 1. Overview

- `src/cfb_cover_model/serving/artifact.py` — trains the dual-model signal on all eligible
  history and persists it as a versioned artifact (`outputs/models/production/<version>/`).
- `src/cfb_cover_model/serving/scoring.py` — scores a week's raw feature rows against an
  already-loaded artifact (CSV or live-CFBD-API input, same two modes
  `generate_weekly_predictions.py` has always had).
- `src/cfb_cover_model/api/` — the FastAPI app. Loads the latest artifact at startup; serves
  predictions, monitoring status, health/readiness, and an admin-triggered retrain.
- `src/cfb_cover_model/monitoring/` — Evidently-based data-drift checks + a custom
  prediction/calibration-drift module graded against realized outcomes.
- `scripts/train_production_artifact.py`, `scripts/run_weekly_monitoring.py`,
  `scripts/run_scheduler.py` — the CLI/scheduling layer around the above.

## 2. Artifact lifecycle

```
outputs/models/production/
├── latest.json                              # {"version": "v2026w03_20260816T193000Z"}
└── v2026w03_20260816T193000Z/
    ├── classifier.joblib                    # fitted logistic_regression
    ├── regressor.joblib                     # fitted ResidualProbabilityRegressor(XGBRegressor)
    ├── selected_columns.json                # the "reduced" feature-set column list
    ├── reference_training_features.parquet  # X_train snapshot - the data-drift reference
    └── metadata.json                        # thresholds, feature_columns, transforms,
                                              #   representation, training_row_count,
                                              #   git_commit, library_versions
```

`latest.json` is written via a temp-file + atomic `os.replace()`, never a symlink (Docker-volume
portable). Version strings are `v{season}w{week}_{UTC timestamp}` — sortable and self-describing.
Every retrain prunes old versions beyond `PRODUCTION_ARTIFACT_RETENTION` (default 8, roughly two
months of weekly runs) unless `--no-prune` is passed.

**The API is the only writer of this directory.** `scripts/run_scheduler.py` calls the API's own
`POST /admin/retrain` over HTTP rather than running `train_production_artifact.py` directly in a
second process — this makes the in-process hot-swap (`ArtifactStore.replace(...)`) happen in the
exact request that persisted the new artifact, and avoids two processes ever writing the same
directory concurrently.

## 3. Running locally without Docker

```bash
cd cfb_cover_model
pip install -e ".[dev,boosting,serving]"

# Bootstrap: nothing is trained yet on a fresh clone.
python scripts/train_production_artifact.py --season <Y> --week <N>

uvicorn cfb_cover_model.api.app:create_app --factory --reload
```

Before the first `train_production_artifact.py` run, `/health` still returns 200 (pure
liveness), but `/ready` returns `{"ready": false}` and `/predictions/*` return 503 with a
message pointing at the command above.

## 4. Running with Docker Compose

```bash
cd cfb_cover_model
cp .env.example .env   # then set ADMIN_API_TOKEN to something real
docker compose up --build
```

Two services, one shared image:
- `api` — port 8000, mounts `outputs/models/`, `outputs/monitoring/`, `outputs/predictions/`,
  `data/processed/` so training/monitoring output survives container restarts.
- `scheduler` — same mounts, no exposed port, runs `scripts/run_scheduler.py` (long-running,
  fires the weekly job per `SCHEDULE_DAY_OF_WEEK`/`SCHEDULE_HOUR`).

Both read `env_file: [../.env, .env]` — the repo-root `.env` for `CFBD_API_KEY`, this project's
own `.env` for everything else. See `.env.example` for the full variable list.

Bootstrap the first artifact inside the running `api` container:
```bash
docker compose exec api python scripts/train_production_artifact.py --season <Y> --week <N>
```

## 5. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness only, always 200 once the process is up |
| GET | `/ready` | 200 with artifact version/trained_at, or `ready: false` if none loaded |
| GET | `/predictions/week/{week}?season=&file=&live=` | Score a week — CSV (default, `Data/CFB_Pred_Week_<N>.csv` or `?file=` override) or live CFBD pull (`?live=true`) |
| GET | `/monitoring/status?season=&week=` | Latest data-drift + prediction-drift reports for that week (404 if neither has run yet) |
| POST | `/admin/retrain` | `{"season": Y, "week": N}` body, `X-Admin-Token` header — retrains and hot-swaps in-process |

`season` is always a **required** query parameter on `/predictions/week/{week}`, even in CSV
mode — `CFB_Pred_Week_<N>.csv`'s own `season` column is unreliable (observed all-`NaN` in real
weekly files during development), so the caller's value is authoritative and is what's returned,
never inferred from the scored frame.

Response probabilities are documented as poorly calibrated (Brier 0.257 on the 2025 holdout, see
`docs/final_writeup_2026.md`) — treat `*_flag`/`agreement_bet` as the primary signal.

`/admin/retrain`'s `X-Admin-Token` check is shared-secret auth only — proportionate for a
local-only deployment, **not** internet-safe. Don't expose port 8000 beyond localhost/your own
network without adding real auth in front of it.

## 6. Weekly operational runbook

1. After a week's games finish, bump `PRODUCTION_SEASON`/`PRODUCTION_WEEK` in `.env` to that
   just-completed week (same manual-bump convention as the R pipeline's `week_update`/`week`
   variables).
2. `docker compose restart scheduler` (env vars are read fresh from the environment on each job
   fire, but a restart guarantees a clean pickup rather than waiting for the next cron fire).
3. The scheduler's weekly job (or `docker compose exec scheduler python scripts/run_scheduler.py
   --run-once` to trigger it immediately) runs: ingest the completed week → retrain via the API →
   grade last week's predictions + check data drift on the upcoming week.
4. Check `GET /monitoring/status?season=<Y>&week=<completed week>` and container logs.

**Triage a bad week:**
- `prediction_drift.alert == true` (rolling precision dropped >10 points below the 0.606
  baseline) → check `calibration` in the same response; a real accuracy regression usually shows
  up in both `brier_score` getting worse and precision dropping together, not just precision
  alone (which is noisy at ~5-15 games/week).
- `data_drift.dataset_drift == true` on the upcoming week → check `drifted_columns`; a **single**
  week's feature frame compared against a **multi-season** reference will show meaningfully high
  nominal drift by construction (observed ~90%+ share_drifted_columns even in an unremarkable
  week during development) — this is expected background noise, not automatically a problem.
  Watch the *trend* week over week, not the absolute number.
- Scheduler log shows `[ingest] FAILED` → the CFBD API pull failed; retrain/monitoring still ran
  on whatever history already existed, so predictions kept flowing, but the new week's outcomes
  weren't incorporated into training. Re-run `scripts/ingest_and_update_history.py` manually.

### 6.1 Automated weekly job via launchd (no Docker)

`scripts/run_weekly.py` is a single wrapper that does the whole cycle in one process — no
API, no container, no `PRODUCTION_WEEK` to bump. It runs, in order:

1. **Resolve weeks** off the CFBD calendar (`cfb_cover_model.schedule.resolve_weeks`):
   which regular-season week just finished, and which is next. Exits 0 (nothing to do) in
   the pre-/post-season.
2. **Ingest** every completed week of the season so far →
   `data/processed/extended_history.parquet` (the whole range, not just the last week —
   `build_historical_rows` needs a team's prior weeks in the same call to build its rolling
   features; re-fetches are cheap and `game_id`-deduped).
3. **Retrain** the dual-model artifact through the completed week →
   `outputs/models/production/<version>/` + `latest.json` (skipped before week 3).
4. **Score the upcoming week** against that just-retrained artifact →
   `outputs/predictions/live_<season>_week_<next>_dual_model_predictions.csv` (skipped
   before week 4, `MIN_WEEK_LIVE`, when the live feature build has too little history).
5. **Monitor**: grade the completed week's picks vs. now-known outcomes, and data-drift the
   upcoming week's frame → `outputs/monitoring/`.

Each step is independent — a failure is logged and the rest still run. Every run appends to
`logs/weekly_<YYYYMMDD>.log`; launchd's own capture is `logs/launchd.out` / `logs/launchd.err`.

**One-time setup:**

```bash
cd cfb_cover_model
python -m venv .venv && .venv/bin/pip install -e ".[dev,boosting,serving]"
# CFBD_API_KEY must be in the repo-root .env

cp deploy/com.cfb.cover-model-weekly.plist ~/Library/LaunchAgents/
# edit the 4 absolute paths in the plist if the repo isn't at
# ~/Projects/CFB_Projects/college_football_analytics
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cfb.cover-model-weekly.plist
```

It then fires **every Sunday at 09:17 local** (launchd runs one missed Sunday when the Mac
next wakes). Sunday is early enough that a still-open betting line on the *upcoming* week can
shift before kickoff — the retrain inputs (completed games, final scores) are settled by
then, the picks are a first read to refresh mid-week if lines move.

```bash
launchctl kickstart -k gui/$(id -u)/com.cfb.cover-model-weekly   # force a run now
launchctl print gui/$(id -u)/com.cfb.cover-model-weekly          # inspect / next fire time
launchctl bootout gui/$(id -u)/com.cfb.cover-model-weekly        # uninstall
```

**Manual re-run / backfill** (bypasses calendar resolution):

```bash
.venv/bin/python scripts/run_weekly.py --season 2025 --completed-week 8 --upcoming-week 9
```

This is the launchd counterpart to the Docker `scheduler` service in §4 — same
ingest→retrain→monitor, plus the upcoming-week scoring the Docker job never did, and it
trains the artifact directly instead of over `POST /admin/retrain` (safe here because
nothing else writes `outputs/models/production/` in this deployment). Run one or the other,
not both.

## 7. Drift monitoring methodology

**Data drift** (`monitoring/data_drift.py`): Evidently's `DataDriftPreset` (per-column K-S test,
`p < 0.05` threshold) comparing the artifact's training-time reference snapshot
(post-feature-engineering, pre-selection frame — i.e. before `fit_feature_set("reduced")`
narrows to the ~60 columns the models actually consume, so drift in a feature that's currently
selected out but could get selected back in on a future retrain is still visible) against the
upcoming week's same-stage frame. Reports saved as HTML (full Evidently visualization) + a
trimmed JSON summary (`dataset_drift`, `share_drifted_columns`, `drifted_columns`) under
`outputs/monitoring/data_drift/`.

**Prediction/calibration drift** (`monitoring/prediction_drift.py`): no Evidently — grades a
completed week's already-generated predictions against realized outcomes once known. Computes
precision on the `agreement_bet` subset, Brier score + ROC-AUC per model
(`modeling/evaluation.calibration_report`, the same function the offline evaluation pipeline
uses), and appends to a rolling log (`outputs/monitoring/prediction_drift/rolling_precision_log.csv`).
Alerts (`alert: true`) when the rolling mean over `PREDICTION_DRIFT_WINDOW_WEEKS` (default 4)
falls more than `PREDICTION_DRIFT_ALERT_DELTA` (default 0.10) below the known 0.606 holdout
baseline.

**Small-sample caveat**: agreement-bet volume is small by design — the 2025 holdout evidence is
~33 agreement bets across an entire season, so single-week counts are often single digits or
zero. A single week's precision is not a reliable signal on its own; the rolling multi-week trend
is what's worth alerting on, and that's reflected in the default 4-week window above.

## 8. Known limitations

- Evidently is pinned at `0.4.33` (the "classic" `Report`/`metric_preset` API) — all calls are
  isolated in `monitoring/data_drift.py` so a future forced version bump is a one-file change.
- A Docker image rebuild with a newer xgboost/scikit-learn could fail to unpickle an
  older artifact; `metadata.json` records `library_versions` and `load_artifact()` warns on
  mismatch, but doesn't auto-migrate.
- `POST /admin/retrain` runs synchronously in-process, blocking that worker for the full
  training duration (tens of seconds against real data) — fine for local, single-operator,
  weekly-cadence use; not concurrency-safe under real traffic.
- The **Docker `scheduler` service** (§4) has no NCAA-calendar detection —
  `PRODUCTION_SEASON`/`PRODUCTION_WEEK` must be bumped manually, matching the rest of this
  repo's convention. The **launchd job** (§6.1) resolves both off the CFBD calendar
  (`cfb_cover_model.schedule.resolve_weeks`) and needs no bump; its one assumption is that
  it runs on a Sunday (a Thu/Fri run could call a week complete mid-slate). Manual
  `run_weekly.py --completed-week N` runs are unaffected.
- `run_weekly.py`'s upcoming-week picks are generated Sunday; betting lines for that week
  can still move before kickoff. Re-run `scripts/generate_weekly_predictions.py --live` (or
  `GET /predictions/week`) mid-week for a refreshed read.
