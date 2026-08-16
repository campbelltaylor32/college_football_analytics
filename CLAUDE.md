# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A machine learning pipeline that predicts whether the home team covers the point spread in college football games. Data pulls and feature engineering are done in R; model training and weekly inference are done in Python (Jupyter notebooks). There is no package manager manifest (no `requirements.txt`/`renv.lock`) — dependencies are installed ad hoc (`cfbfastR`, `tidyverse`, `data.table` in R; `pandas`, `xgboost`, `joblib` in Python).

## Running the pipeline

There is no test suite, build step, or linter. This is a data-science pipeline run manually and sequentially, script by script, typically re-run once per week during the season. Order matters — each stage reads CSVs written by the previous one from `Data/`:

1. **`R Scripts/Full_CFB_Game_Outcome_Historical.R`** — pulls historical games/betting lines/PBP stats via `cfbfastR` (2013–2025+), engineers rolling/moving-average features, derives the cover/push outcome label, writes historical CSVs to `Data/`. Only re-run to rebuild the full historical base (rarely).
2. **`R Scripts/Merge_Predictors_CFB_Historical.R`** — joins talent, coaching, returning-production, and game-average data into `Data/CFB_Gambling_Predictors_Final_PBP.csv`, the historical training set. Sets `setwd('Data/')` internally, so paths inside it are relative to `Data/`, not the repo root.
3. **`R Scripts/2025_Game_Update.R`** — the weekly refresh: pulls the current 2025-season games/betting lines/stats, writes `Data/*_2025_<week>.csv` files. Has a `week_update` variable at the top that must be bumped each week.
4. **`R Scripts/2025_Pred_Update.R`** — rebuilds current-season predictors the same way as step 2 (talent + coaching + returning production + lagged game stats), producing `Data/CFB_Pred_Week_<N>.csv` — the feature row per upcoming game, with `home_covered`/labels blank (`NA`) since these are future games. Also has a `week` variable to bump.
5. **`Python Scripts/CFB_Gambling_Model.ipynb`** — trains/tunes the classifier (XGBoost) on the historical predictors file, evaluates precision, and saves a model + its feature list to `Model Information/` (`best_xgb_model_<date>.pkl`, `selected_features_best_model_<date>.json`).
6. **`Python Scripts/Week_Predictions.ipynb`** — loads the current week's `CFB_Pred_Week_<N>.csv` plus the saved model/feature list, scores cover probability, and outputs the week's picks (probability threshold currently 0.6).

When resuming work mid-season, check which `week_update`/`week` values and which `*_2025_<N>.csv` / model artifact filenames are most recent before assuming a script's hardcoded values are current.

## Architecture notes

- **No lookahead bias is a first-class design constraint.** Coaching stats are shifted with `lag()` by year before joining, and in-season game stats (`prev_week_*`, `_avg_all`, `_avg3` columns) are lagged by one week per team before being used as that week's predictors. Any change to feature engineering must preserve this — a predictor row for week N must only use information available before week N kicked off.
- **Home/away symmetry via renaming, not row-per-team.** Team-level predictors are computed once per team-week, then duplicated into `home_*`-prefixed and `away_*`-prefixed column sets and joined back onto the game-outcome table by `game_id`. This pattern repeats in both the historical merge and the weekly update scripts.
- **Historical vs. current-season are parallel, separately-maintained pipelines.** `Merge_Predictors_CFB_Historical.R`/`Full_CFB_Game_Outcome_Historical.R` (training data) and `2025_Pred_Update.R`/`2025_Game_Update.R` (weekly inference data) duplicate the same feature-engineering logic against different source files (`*_CFB.csv` vs `*_2025_WeekN.csv`). A feature engineering fix generally needs to be made in both places to keep train/inference features consistent.
- **`Data/` is the single interchange point** between R and Python — nothing is passed in memory or via a database. All scripts read/write CSVs there and expect prior stages' outputs to already exist. Actual CSVs are gitignored (`Data/*.csv`); only the code that generates them is tracked.
- **`Model Information/`** holds trained model binaries (`.pkl`) and their matching feature-name JSON, dated by training run. Prediction code must load the feature list alongside the model and subset/reorder columns to match (`model.get_booster().feature_names`), since the predictor CSV has far more columns than the model was trained on.
