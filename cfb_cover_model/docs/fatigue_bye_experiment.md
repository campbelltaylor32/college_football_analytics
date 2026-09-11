# Experiment: prev-week workload ("snap count") + bye-week flag

**Status: null result — not promoted.** Stays as tested, disabled-by-default experimental
feature code behind `CFB_EXPERIMENT` (mirrors `cfb_win_total_model`'s
`docs/power_rating_experiment.md` diagnostic-feature convention).

## What was tested

Two candidate signals, both gated behind `CFB_EXPERIMENT` /
`src/cfb_cover_model/experiment_paths.py::include_experimental_features()`:

1. **Workload ("snap count") composite** — `engineered_features.py::add_fatigue_and_rest_features`.
   `cfbd` exposes no direct snap-count endpoint, so total offensive + defensive play count
   (already-ingested `Total_Offense_Plays`/`Total_Defense_Plays`) is used as the proxy, summed
   per side into `{home,away}_total_snaps_{prev_week,avg_all,avg3}`.
2. **Bye-week flag** — `schedule_features.py::compute_bye_flags`/`attach_bye_flags` — derives
   `home_off_bye`/`away_off_bye` from gaps in each team's own played-week schedule.

Both were already fully implemented and unit-tested (`tests/test_engineered_features.py`,
`tests/test_schedule_features.py`) but had never been run through the actual training pipeline.
This experiment wires them in with `CFB_EXPERIMENT=fatigue_bye_v1` and trains + persists a real
dual-model artifact (`scripts/train_production_artifact.py`, now experiment-tag aware) to
`outputs/experiments/fatigue_bye_v1/models/production/`, entirely isolated from the confirmed
baseline in `outputs/models/production/`.

## Results

Ran the full chain (`load_and_validate_dataset.py` → `select_features.py` → `train_models.py` →
`evaluate_models.py` → `train_production_artifact.py`) both untagged (baseline, regenerated
fresh for a clean comparison) and tagged `fatigue_bye_v1`, on the same underlying history
(2,260 training rows through 2025 week 12).

| | Baseline (untagged) | `fatigue_bye_v1` |
|---|---|---|
| `select_features.py` winning transform/representation | `all_three` / `differential` (491 candidates) | `avg3_only` / `raw_dual` (346 candidates, incl. the 8 new columns) |
| Best reduction strategy in that search | `reduced` (best of 3) | `pca_reduced` (best of 3) — **`reduced` ranked *worst*** |
| `logistic_regression` walk-forward precision | 0.5396 | 0.5202 |
| `xgboost_regressor` walk-forward precision | 0.5390 | 0.5386 (flat) |
| Overall single-best-model search: holdout precision | 0.5357 (meets 0.53 target) | 0.3478 (**misses** target) |
| Persisted dual-model artifact (`reduced` mode, always forced regardless of the above) | 60 selected columns | 7 selected columns |

The persisted production-style artifact always uses `feature_set_mode="reduced"` for both the
classifier and regressor (hardcoded in `serving/artifact.py::train_production_artifact`,
independent of whichever strategy the ablation in `select_features.py` ranks best). **None of
the 8 new columns (`home`/`away_off_bye`, `home`/`away_total_snaps_{prev_week,avg_all,avg3}`)
were selected into that final 7-column set** — every selected column is a pre-existing
rushing/receiving-usage or down-success-rate feature.

## Interpretation

- The new columns flowed correctly through the full pipeline (candidate set → transform/lag →
  selection), but were not judged useful enough by the deterministic `reduced` selection
  strategy production actually uses — in either the baseline or tagged run.
- Turning the experimental flag on also changed which transform/representation combination won
  the broader ablation (`avg3_only`/`raw_dual` instead of `all_three`/`differential`), and in
  that combination `reduced` — the one strategy production is hardcoded to — ranked *worst*
  of three, a reversal from baseline. The single-best-model search's holdout precision collapse
  (0.536 → 0.348) tracks with that instability, not obviously with the two new features
  themselves (which weren't in the winning strategy's selected set either).
- The apples-to-apples dual-model comparison (`logistic_regression`/`xgboost_regressor`
  walk-forward precision, same models production always fits) is flat-to-slightly-worse, not
  improved.

## Recommendation

**Do not promote.** Leave both features exactly as they are: implemented, unit-tested, and
gated behind `CFB_EXPERIMENT`. Possible follow-ups if revisited:
- Test the two features independently (rather than bundled under one flag) to isolate whether
  either individually changes the `reduced`-strategy selection outcome.
- Force the baseline's `all_three`/`differential` transform/representation even when the flag
  is on, to separate "do these two columns help" from "does adding 346-vs-491 raw candidates
  change which ablation config wins."

## Reproducing

```bash
cd cfb_cover_model
CFB_EXPERIMENT=fatigue_bye_v1 python scripts/load_and_validate_dataset.py
CFB_EXPERIMENT=fatigue_bye_v1 python scripts/select_features.py
CFB_EXPERIMENT=fatigue_bye_v1 python scripts/train_models.py
CFB_EXPERIMENT=fatigue_bye_v1 python scripts/evaluate_models.py
CFB_EXPERIMENT=fatigue_bye_v1 python scripts/train_production_artifact.py --season 2025 --week 12
```

Outputs land entirely under `outputs/experiments/fatigue_bye_v1/` and
`data/processed/experiments/fatigue_bye_v1/` — the confirmed baseline in `outputs/` and
`data/processed/` is never touched by a tagged run.

## Known follow-up (out of scope here)

Both features are wired into the historical R-CSV path
(`scripts/load_and_validate_dataset.py`) only. The live CFBD ingestion path
(`src/cfb_cover_model/ingest/pipeline.py`, `scripts/ingest_and_update_history.py`) does not
compute either column. This is a no-op today because `data/processed/extended_history.parquet`
doesn't exist yet in this checkout — but if that file is populated before this experiment is
revisited, `train_production_artifact.py`'s history-merge missing-column check will start
raising for tagged runs unless the live path is updated to match.
