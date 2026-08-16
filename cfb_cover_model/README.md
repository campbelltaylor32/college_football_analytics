# cfb_cover_model

An independent, walk-forward-validated rebuild of the spread-cover classifier currently living in
`../Python Scripts/CFB_Gambling_Model.ipynb`. Same target (`home_covered`: does the home team
cover the point spread) and same source data as that notebook, but built and evaluated from
scratch: proper push handling, a controlled ablation of the source data's triple temporal
redundancy, dimensionality reduction fit only inside each walk-forward training fold, and a
three-track model comparison — direct classification, a regression-on-margin track converted to a
cover probability, and a stacked ensemble of the two — rather than a single classifier family.

## Why this exists

The production notebook selects its 52 features by which subset maximizes **test-set** ROC-AUC,
then picks a betting threshold (0.60) by eyeballing a manual sweep table, evaluated once on a
single fixed split (2015-2022 train / 2023-2024 test) — a textbook test-set-reuse pattern. There
is no time-aware cross-validation anywhere in the notebook's hyperparameter search (plain 3-fold
`KFold`), no push handling (an exact push on the spread is silently labeled "did not cover"), no
probability calibration, and a confirmed upstream R bug where the `Offense_EPA_per_Run` /
`Defense_EPA_per_Run` columns actually contain pass-play EPA (a duplicate `summarise()` name
overwrote the true per-run values before they were ever written to the CSV).

A separate, already-mature sibling project (`../cfb_spread_model/`) independently walk-forward
validated this same feature set and found a low honest ceiling (~0.51-0.54 pooled precision) with
plain logistic regression on every raw column beating every feature-selection/model-complexity
attempt it tried. This project does not start from that project's conclusions — it re-derives its
own answers with its own pipeline, and tests two angles the sibling project didn't cleanly run: a
controlled single-temporal-transform ablation, and a regression-to-probability track stacked with
direct classification. See `docs/assumptions_and_limitations.md` for what's carried over
unchanged vs. fixed, and `docs/project_story.md` for the full, honestly-reported results (winning
or not).

## Data source

Two parallel paths, kept deliberately separate (see "Direct API ingestion" below):

- **Historical training data**: `../Data/CFB_Gambling_Predictors_Final_PBP.csv` (features) and
  `../Data/CFB_Gambling_Results.csv` (joined on `game_id` to recover signed spread / final score,
  needed for push detection and the continuous margin target), produced by
  `../R Scripts/Full_CFB_Game_Outcome_Historical.R` and
  `../R Scripts/Merge_Predictors_CFB_Historical.R`. This is still how `modeling_dataset.parquet`
  (what every model is trained on) gets built — the R layer has not been retired.
- **Live/current-season data**: pulled directly from the CollegeFootballData (CFBD) API via
  `src/cfb_cover_model/ingest/`, with no R dependency at all. This is what
  `scripts/generate_weekly_predictions.py --live` and `scripts/ingest_and_update_history.py`
  use. Needs `CFBD_API_KEY` in the repo root's `.env` (the same key the R scripts already use).

## Repository structure

```
cfb_cover_model/
├── config/                  data.yaml, features.yaml, modeling.yaml
├── docs/                    data dictionary, leakage rules, methodology, limitations, project story
├── notebooks/                01-04, exploratory companions to the scripts below
├── scripts/                  one script per pipeline stage (see "Running the pipeline")
├── src/cfb_cover_model/
│   ├── config.py, data.py, targets.py, cleaning.py, data_validation.py
│   ├── feature_selection/    correlation_pruning.py, embedded_selection.py,
│   │                          transform_ablation.py, pca_reduction.py
│   ├── modeling/              splits.py, classifiers.py, regressor.py, stacking.py,
│   │                           calibration.py, evaluation.py
│   └── ingest/                 direct-CFBD-API port of the R feature pipeline — cfbd_client.py,
│                                 raw_cache.py, box_score_features.py, pbp_features.py,
│                                 rolling_features.py, talent_coach_returning.py, pipeline.py
├── tests/                     pytest suite, including leakage + push-handling + OOF-integrity
│                               tests and per-module ingest unit tests
├── data/{raw,interim,processed}   cached intermediates (gitignored) — data/raw/ is the ingest
│                                    package's per-endpoint API cache
└── outputs/                   eda, feature_analysis, model_comparison, threshold_selection,
                                calibration, predictions, models, validation
```

## Installation

```bash
cd cfb_cover_model
pyenv install -s 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,boosting]"
```

`CFBD_API_KEY` in the repo root's `.env` is only required for the direct-API path
(`--live`, `ingest_and_update_history.py`) — everything else only reads the two CSVs above and
needs no credentials.

## Running the pipeline

One command end-to-end:

```bash
python scripts/run_pipeline.py
```

Or stage by stage:

```bash
python scripts/load_and_validate_dataset.py   # -> data/processed/modeling_dataset.parquet
python scripts/run_eda.py                     # -> outputs/eda/
python scripts/select_features.py             # transform ablation + correlation pruning +
                                                #    embedded/PCA selection -> outputs/feature_analysis/
python scripts/train_models.py                # walk-forward OOF, tracks A/B/C -> outputs/model_comparison/
python scripts/evaluate_models.py             # threshold selection, calibration, full holdout
                                                #    comparison of every candidate -> outputs/threshold_selection/,
                                                #    outputs/calibration/
python scripts/analyze_train_vs_holdout.py    # in-sample vs. walk-forward vs. holdout accuracy,
                                                #    overfitting diagnostic -> outputs/model_comparison/
python scripts/analyze_feature_stability.py   # fold-stability of selected features across two
                                                #    transform configs -> outputs/feature_analysis/
python scripts/explain_model.py               # holdout permutation importance for the production
                                                #    model -> outputs/model_comparison/
python scripts/generate_week_predictions.py --week <N>   # replacement for Week_Predictions.ipynb
```

`run_pipeline.py` also accepts `--stage <name>` and `--from-stage`/`--to-stage`.

## 2026 deployment

`docs/final_writeup_2026.md` is the up-to-date synthesis: final results, the three real
bugs caught along the way, why `logistic_regression` + `xgboost_regressor` agreement is the
recommended signal, and honest limitations to read before trading on it.
`scripts/generate_weekly_predictions.py` is the weekly production script — scores both
models and flags a bet only when they agree — in two modes:

```bash
# Direct from the CFBD API, no R dependency at all:
python scripts/generate_weekly_predictions.py --live --season 2026 --week <N>

# Original path, reading a week file the R scripts already produced:
python scripts/generate_weekly_predictions.py --week <N>
```

`scripts/ingest_and_update_history.py --season 2026 --weeks <N> [<N> ...]` pulls completed
weeks directly from the API and appends them to `data/processed/extended_history.parquet`,
which `--live` automatically unions onto `modeling_dataset.parquet` before refitting — so a
finished 2026 week becomes usable training history for the next week's run without waiting
on a new R pull.

### Direct API ingestion — what it is and isn't

`src/cfb_cover_model/ingest/` is a from-scratch Python port of the R feature-engineering
pipeline (~168 per-team-week stats, including play-by-play EPA/success-rate/explosiveness
aggregation), built so the whole weekly cycle can run standalone from `cfb_cover_model` in
2026 without manually running the R scripts first. It was validated column-by-column against
the real, R-generated historical CSV (`scripts/validate_against_r_pipeline.py`) — 752/1048
columns match exactly, with the rest traced to two understood, documented causes rather than
silent bugs:
- **EPA discrepancy**: CFBD's raw `ppa` field is a genuinely different statistical model than
  `cfbfastR`'s own internally-trained `EPA` column, not a bug in this port's aggregation
  logic. Live EPA-derived features are a correlated but not identical proxy for what the
  models were trained on — the single biggest open accuracy risk in the live path.
- **Coach career-length lookback**: fetches 20 years of coaching history vs. the R source's
  exact 2004 start year, undercounting very long-tenured coaches' career totals slightly.

**Current status**: pipeline-validated, not yet production-validated. A full `--live` run
has been exercised end-to-end against one real, already-completed week (2025 week 8 — 60
games scored, sensible output) and caught two real bugs in the process (a missing FBS-only
filter on the games endpoint, and a missing blanket 0-fill matching the R live script's
`tot_pred[is.na(tot_pred)] <- 0`, both now fixed). Still open before trusting this for a real
upcoming week:
- Never run against a genuinely *upcoming* (not-yet-played) week — betting lines can be thin
  or unposted days out, which is exactly the kind of gap that broke the run twice already.
- `ingest_and_update_history.py` has never been executed against real data.
- Early-season weeks (1-3, before `MIN_WEEK_LIVE`'s 3-game history requirement is met) are
  untested.
- The hardcoded CFBD API key in `R Scripts/2025_Game_Update.R` and the missing `.env` entry
  in the root `.gitignore` are still unresolved, and now more load-bearing since the live
  path depends on that same key — rotate the key and fix `.gitignore` before relying on this.

## Feature deep-dive

`scripts/analyze_feature_stability.py` and `scripts/explain_model.py` identify which
individual features are actually driving the model, using two complementary lenses: fold-
to-fold stability (does the same feature get selected across all 6 walk-forward folds, and
under two different temporal-transform configs, or only in one snapshot?) and single-shot
holdout permutation importance. Features are rolled up by a domain taxonomy
(`src/cfb_cover_model/feature_categories.py`: EPA/success-rate, turnover/penalty, down-
conversion, talent, coaching, etc.) rather than by temporal transform, since the winning
config is ~100% `prev_week` already. See `docs/feature_importance_findings.md` for the
synthesized write-up and feature-engineering recommendations.

## Dimensionality reduction, in brief

Every one of the ~160 base game stats in the source CSV is tripled into `prev_week_*` /
`*_avg_all` / `*_avg3` versions. Stage 2 (`feature_selection/transform_ablation.py`) runs a
controlled grid over which single transform (or `avg_all` + trend-delta pair) carries the signal,
rather than feeding a model all three at once. Stage 3 (`correlation_pruning.py`) collapses
remaining collinearity via fold-local hierarchical clustering. Stage 4
(`embedded_selection.py`) uses elastic-net logistic regression as the primary reducer (more
stable across folds than permutation-importance sweeps or RFECV — see
`docs/feature_selection_methodology.md`). Stage 5 (`pca_reduction.py`) is an alternative,
head-to-head-compared reducer, not a step chained after 3-4.

## Time-based validation

Expanding-window walk-forward by season, never a random split. Season 2020 (COVID-shortened) is
excluded, 2025 is held out as the final, never-touched-during-selection test season. See
`docs/modeling_methodology.md`.

## Modeling: three tracks

- **Track A — direct classification**: logistic regression (with and without feature selection),
  random forest, gradient boosting, XGBoost, LightGBM, CatBoost.
- **Track B — regression-to-probability**: predicts the continuous `cover_margin` (points beyond
  the spread), converted to a cover probability via the fold-specific residual distribution.
- **Track C — stacked ensemble**: a meta-learner trained on out-of-fold predictions from Track A
  and Track B base models (proper nested OOF stacking, no base model sees its own stacked row).

Every candidate — not just the walk-forward-selected winner — is scored on the true 2025 holdout.

## Leakage prevention

Feature engineering (and its own lookahead-bias handling — lagging by week/year) already happened
upstream in R; this project's leakage tests are a regression safety net over that plus this
project's own walk-forward/selection code. See `docs/data_leakage_rules.md` and
`tests/test_leakage.py`.

## Known limitations

See `docs/assumptions_and_limitations.md`. In brief: the source CSV only covers weeks 4-12 of each
season (bowls/championship games excluded upstream); `Offense_EPA_per_Run`/`Defense_EPA_per_Run`
are excluded (mislabeled at source, see above); no new data sources (line movement, injuries,
weather) are in scope this phase; `../Python Scripts/CFB_Gambling_Model.ipynb` and
`Week_Predictions.ipynb` remain the production pipeline unless this project's honestly-validated
precision beats the documented, re-verified baseline. For the direct-API `--live` path
specifically, see "Direct API ingestion — what it is and isn't" above.
