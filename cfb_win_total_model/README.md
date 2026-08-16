# cfb_win_total_model

A leakage-safe regression pipeline that predicts each FBS team's **regular-season win total**
for an upcoming season, using only information available before that season's first game.

## Goal and target

`regular_season_wins`: the number of games an FBS team wins in a season, one row per
(team, season). Every feature used to predict season `t` is either from season `t-1` or
earlier, or is a genuinely preseason-known measure for season `t` itself (team talent,
returning production — see `docs/data_leakage_rules.md` for exactly which features fall into
each category and why).

## Database requirement

This project reads from a local MySQL database (`cfb_football`) populated by the sibling
`SQL Scripts/` directory in this repo — a raw cache of the CollegeFootballData API, pulled via
`cfbfastR`. It must already exist and be populated before running anything here:

```bash
mysql -u root -e "CREATE DATABASE IF NOT EXISTS cfb_football;"
mysql -u root cfb_football < "../SQL Scripts/schema.sql"
cd .. && nohup Rscript "SQL Scripts/ingest_to_mysql.R" > ingest.log 2>&1 &
```

See `../SQL Scripts/README.md` for details. This project never modifies that database or the
R ingestion pipeline — it only reads.

## Repository structure

```
cfb_win_total_model/
├── config/                  database/modeling/features YAML configs
├── docs/                    leakage rules, methodology, limitations, data dictionary,
│                            compression/drift investigation (see below)
├── notebooks/               01-04, exploratory companions to the scripts below
├── scripts/                 one script per pipeline stage (see "Running the pipeline")
│   └── diagnostics/         compression/drift investigation tooling — see its own README
├── src/cfb_win_total_model/
│   ├── config.py, database.py, data_validation.py, cleaning.py, targets.py, dataset.py
│   ├── features/            one module per feature category
│   └── modeling/            splits, preprocessing, baselines, models, tuning, evaluation, diagnostics
├── tests/                   pytest suite, integration tests against the live local DB
├── data/{interim,processed} cached intermediates (gitignored)
└── outputs/                 data_inventory, eda, feature_analysis, model_comparison, diagnostics,
                             predictions, models, diagnostics_compression
```

## Installation

The ambient system Python (3.14) is newer than XGBoost/LightGBM/SHAP wheel support typically
allows, so this project pins its own interpreter via `pyenv`:

```bash
cd cfb_win_total_model
pyenv install -s 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,boosting,shap]"
```

## Environment setup

```bash
cp .env.example .env
# .env defaults already match a local `mysql -u root` install with no password, db=cfb_football
```

## Running the pipeline

One command end-to-end:

```bash
python scripts/run_pipeline.py
```

Or stage by stage:

```bash
python scripts/inspect_database.py          # data inventory -> outputs/data_inventory/
python scripts/build_modeling_dataset.py     # -> data/processed/modeling_dataset.parquet + feature_registry.csv
python scripts/run_eda.py                    # -> outputs/eda/
python scripts/train_models.py               # walk-forward OOF predictions -> outputs/model_comparison/oof_predictions.csv
python scripts/evaluate_models.py            # model selection + final holdout + diagnostics
python scripts/generate_predictions.py       # -> outputs/predictions/predicted_win_totals_<season>.csv
```

`run_pipeline.py` also accepts `--stage <name>`, `--from-stage`/`--to-stage`, and `--rebuild`
(forces the modeling-dataset cache to rebuild).

## Leakage-prevention methodology

Every feature's source season is documented in `docs/data_leakage_rules.md` and enforced by
`tests/test_feature_shifting.py` and `tests/test_leakage.py`. Two feature categories
(`returning_production`, `talent_recruiting`) are sanctioned exceptions that legitimately use
season `t` itself rather than `t-1` — read that doc before touching either module.

## Time-based validation

Expanding-window walk-forward validation by season, never a random split — see
`docs/modeling_methodology.md` for the exact fold table and season boundaries. Season 2020
(COVID-shortened) is excluded from training and validation by default.

## Feature groups

`prior_performance`, `returning_production`, `talent_recruiting`, `roster_turnover`,
`coaching`, `schedule`, `program_history` — toggleable in `config/features.yaml`. Full list in
`outputs/feature_analysis/feature_registry.csv` (see `docs/data_dictionary.md`).

## Evaluation metrics

MAE is the primary metric (the target is directly in wins). Also reported: RMSE, median AE,
R², mean bias, % within 1/2 wins, calibration by predicted-win bucket, and breakdowns by
season/conference/talent-tier/new-coach/QB-turnover/high-transfer-activity. See
`docs/modeling_methodology.md` for this build's actual results (selected model:
`gradient_boosting`, walk-forward MAE 1.54, final-holdout MAE 2.01 on season 2025 — beating
every baseline including previous-season-wins by a wide margin).

## Prediction compression and the 2024-2025 accuracy decline

Predictions were found to be significantly more conservative than reality (a team's predicted
win total moved roughly half as much as its actual outcome did), and out-of-sample accuracy has
declined for three straight seasons — even as the training set has grown each year, which is
the opposite of what should happen if this were ordinary underfitting. The root cause of the
compression was diagnosed and confirmed (an MAE-only tuning objective consistently selects the
most-shrunk hyperparameters available), and a working fix exists — but it does not close the
gap on the true 2025 holdout, pointing instead to real non-stationarity in the sport itself
(accelerating transfer-portal churn, conference realignment). Full story:
`docs/project_story.md`; full technical detail and every reproducible number:
`docs/diagnostics_compression_report.md`; the tooling behind both: `scripts/diagnostics/`.

## Known limitations

See `docs/assumptions_and_limitations.md` for the full list, in brief:
- No true 2026 schedule yet — this build backtests on season 2025 rather than forecasting a
  genuine future season.
- Conference championship games are indistinguishable from regular-season games in this
  database's `season_type` field.
- Power/Group-of-5 conference mapping is a hardcoded, season-aware assumption — and its 2024+
  override for the Pac-12 collapse was never independently verified, exactly the seasons where
  out-of-sample accuracy declined most (see the section above).
- Transfer activity is inferred from roster set differences, not ground truth, and has grown
  roughly 7x since 2019 — the historical relationships the model learned describe an earlier,
  much lower-churn era of the sport.
- OC/DC coaching changes are not derivable from this schema.
