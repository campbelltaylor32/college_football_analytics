# cfb_spread_model

A leakage-tested, precision-optimized rebuild of the spread-cover classifier currently living in
`../Python Scripts/CFB_Gambling_Model.ipynb`. Same target (`home_covered`: does the home team
cover the point spread), same source data, but a config-driven pipeline with walk-forward
validation and dimensionality reduction that is explicitly scored on **precision**, not ROC-AUC —
the metric that matters when a flagged game becomes a real bet.

## Why this exists

The current notebook selects its 52 production features by which subset maximizes test-set
ROC-AUC, then picks a betting threshold (0.60) by eyeballing a manual sweep table, evaluated on a
single fixed train/test split (2015-2022 vs. 2023-2024). See `docs/project_story.md` for the full
before/after comparison and `docs/assumptions_and_limitations.md` for known simplifications
carried over unchanged (e.g. spread pushes are labeled `home_covered=0`, not a separate class).

## Data source

This project reads `../Data/CFB_Gambling_Predictors_Final_PBP.csv` directly — the same file the
current notebook uses, produced by `../R Scripts/Full_CFB_Game_Outcome_Historical.R` and
`../R Scripts/Merge_Predictors_CFB_Historical.R`. It does **not** query the local MySQL
`cfb_football` database populated by `../SQL Scripts/`; that database has no engineered features
yet (see `../SQL Scripts/README.md`). `docs/data_dictionary.md` cross-references this CSV's
column-naming conventions back to the raw tables in `../SQL Scripts/schema.sql` purely as
documentation.

## Repository structure

```
cfb_spread_model/
├── config/                  data.yaml, features.yaml, modeling.yaml
├── docs/                    data dictionary, leakage rules, methodology, limitations, project story
├── notebooks/               01-04, exploratory companions to the scripts below
├── scripts/                 one script per pipeline stage (see "Running the pipeline")
├── src/cfb_spread_model/
│   ├── config.py, data.py, data_validation.py, cleaning.py
│   ├── feature_selection/   correlation_pruning.py, precision_scoring.py, selection.py
│   └── modeling/            splits, preprocessing, models, tuning, threshold_selection, evaluation
├── tests/                   pytest suite, including leakage tests
├── data/{interim,processed} cached intermediates (gitignored)
└── outputs/                 data_inventory, eda, feature_analysis, model_comparison,
                              threshold_selection, predictions, models
```

## Installation

```bash
cd cfb_spread_model
pyenv install -s 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,boosting]"
```

No `.env`/database credentials needed — this project only reads a CSV.

## Running the pipeline

One command end-to-end:

```bash
python scripts/run_pipeline.py
```

Or stage by stage:

```bash
python scripts/load_and_validate_dataset.py   # -> data/processed/modeling_dataset.parquet
python scripts/run_eda.py                     # -> outputs/eda/
python scripts/select_features.py             # -> outputs/feature_analysis/
python scripts/train_models.py                # walk-forward OOF predictions -> outputs/model_comparison/
python scripts/evaluate_models.py             # threshold selection, final holdout -> outputs/threshold_selection/
python scripts/explain_model.py               # feature importance for the production model -> outputs/model_comparison/
python scripts/analyze_rank_calibration.py    # top/bottom predicted-probability calibration -> outputs/calibration/
python scripts/analyze_train_vs_holdout.py    # overfitting vs. insufficient-signal diagnostic -> outputs/model_comparison/
python scripts/compare_models_on_holdout.py   # is the walk-forward winner also the true-holdout winner? -> outputs/model_comparison/
python scripts/replicate_notebook_features.py # why does ../Python Scripts/CFB_Gambling_Model.ipynb report higher precision? -> outputs/model_comparison/
python scripts/generate_week_predictions.py --week <N>   # replacement for Week_Predictions.ipynb
```

`run_pipeline.py` also accepts `--stage <name>`, `--from-stage`/`--to-stage`, and `--rebuild`.

## Dimensionality reduction, in brief

Stage 1 (`feature_selection/correlation_pruning.py`) collapses the CSV's built-in 3x temporal
redundancy (`prev_week_*`/`*_avg_all`/`*_avg3` triplets of the same underlying stat) and
offense/defense-mirror pairs via correlation pruning, computed fresh inside each fold's training
data only. Stage 2 (`feature_selection/selection.py`) runs permutation importance and RFECV
scored by a custom precision-at-coverage-floor metric (`feature_selection/precision_scoring.py`),
not ROC-AUC. Full writeup: `docs/feature_selection_methodology.md`.

## Time-based validation

Expanding-window walk-forward validation by season, never a random split — see
`docs/modeling_methodology.md`. Season 2020 (COVID-shortened) is excluded, same justification as
the sibling `../cfb_win_total_model/` project.

## Leakage prevention

Feature engineering (and its own lookahead-bias handling — lagging by week/year) already happened
upstream in R; this project's leakage tests are a regression safety net, not a rebuild of that
logic. See `docs/data_leakage_rules.md` and `tests/test_leakage.py`.

## Known limitations

See `docs/assumptions_and_limitations.md`. In brief: spread pushes are folded into
`home_covered=0` (not a separate class, carried over unchanged from the current labels); the
source CSV only covers weeks 4-12 of each season (bowls/championship games excluded upstream);
`../Python Scripts/CFB_Gambling_Model.ipynb` and `Week_Predictions.ipynb` remain the production
pipeline until this project's pipeline is validated to match or beat their documented baseline.
