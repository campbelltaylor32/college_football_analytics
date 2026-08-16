# College Football Analytics

A collection of college football data pipelines and machine learning models built on
[`cfbfastR`](https://github.com/sportsdataverse/cfbfastR)/[collegefootballdata.com](https://collegefootballdata.com/)
data: a production spread-cover prediction pipeline (R feature engineering + Python/XGBoost
inference), several newer standalone Python rebuilds and extensions of that model, a win-total
regression model, a running-back rushing-yards model, and a handful of one-off analytical
deep-dives (Pythagorean win expectation, recruiting-talent distribution, transfer-portal flow).

Weekly predictions for the 2025 season are posted on Twitter/X: [@camtaylor_4](https://twitter.com/camtaylor_4).

## Repository map

```
college_football_analytics/
├── Data/                    shared CSV interchange point (gitignored, see below)
├── R Scripts/                production feature-engineering pipeline (historical + weekly)
├── Python Scripts/           production model training + weekly inference notebooks
├── Model Information/        trained model binaries + feature lists from the notebooks above
├── SQL Scripts/               MySQL raw-data cache of the CFBD API (parallel, newer infra)
├── cfb_cover_model/           standalone rebuild of the spread-cover classifier (walk-forward CV)
├── cfb_spread_model/           sibling rebuild of the spread-cover classifier (precision-optimized)
├── cfb_win_total_model/        season win-total regression model
├── cfb_rb_rushing_model/       weekly RB rushing-yards regression model
├── cfb_pythagorean_model/      retrospective Pythagorean win% analysis (2025 season)
├── cfb_talent_distribution/    recruiting-talent distribution analysis (2015-2025)
└── cfb_transfer_portal_flow/   conference-level transfer-portal flow analysis (2021-2025)
```

Each `cfb_*` directory is a self-contained Python (or R) project with its own `README.md`,
dependencies, and (where applicable) `pyproject.toml` — see the linked sections below for details
rather than duplicating them here.

---

## Environment setup

A `.env` file in the repo root holds the shared CollegeFootballData API key, read by both the R
scripts and (for the direct-API paths) some of the `cfb_*` Python projects:

```bash
CFBD_API_KEY=your_key_here
```

`.env` is gitignored — never commit it. Get a free key at
[collegefootballdata.com/key](https://collegefootballdata.com/key).

> **Security note:** an earlier version of this repo had a CFBD key hardcoded directly in several
> R scripts and committed to git history (this repo is public). All scripts have since been fixed
> to read `Sys.getenv("CFBD_API_KEY")` instead, but the old key is still recoverable from git
> history — it should be treated as compromised and rotated at the provider, not just removed from
> the current files.

---

## Production pipeline

The original, currently-deployed pipeline: R does data pulls and feature engineering, Python does
model training and weekly inference. There is no test suite, build step, or linter — it's a
data-science pipeline run manually, script by script, roughly once per week during the season.
Order matters; each stage reads CSVs written by the previous one from `Data/`.

**Task:** binary classification — does the home team cover the spread? Trained on 2015-2022,
tested on 2023-2024, achieving **57% precision** on predicted home covers (probability threshold
0.6). See `CLAUDE.md` for the full architectural constraints (no-lookahead-bias handling,
home/away symmetry pattern, train/inference parity between historical and weekly scripts).

### `Data/`
Single canonical folder for all CSV inputs/outputs — historical training data
(`CFB_Gambling_Results.csv`, `CFB_Team_Talent_Data.csv`, `Coaches_Winning_CFB.csv`,
`Game_Stats_Averages_CFB_PBP_Added.csv`, `Returning_Production_CFB.csv`,
`CFB_Gambling_Predictors_Final(_PBP).csv`), the 2025-season weekly series
(`*_2025_N.csv` / `*_2025_WeekN.csv`), and weekly prediction outputs (`CFB_Pred_Week_N.csv`).
Gitignored (`Data/*.csv`) — only the code that generates these files is tracked. All scripts
read/write here; there should be no loose data CSVs at the repo root.

### `R Scripts/`
1. `Full_CFB_Game_Outcome_Historical.R` — pulls historical games/betting lines/PBP stats via
   `cfbfastR` (2013-2025+), engineers rolling/moving-average features, derives the cover/push
   outcome label, writes historical CSVs to `Data/`. Only re-run to rebuild the full historical
   base (rarely).
2. `Merge_Predictors_CFB_Historical.R` — joins talent, coaching, returning-production, and
   game-average data into `Data/CFB_Gambling_Predictors_Final_PBP.csv`, the historical training
   set. Sets `setwd('Data/')` internally.
3. `2025_Game_Update.R` — the weekly refresh: pulls the current 2025-season games/betting
   lines/stats, writes `Data/*_2025_<week>.csv`. Has a `week_update` variable that must be bumped
   each week.
4. `2025_Pred_Update.R` — rebuilds current-season predictors the same way as step 2, producing
   `Data/CFB_Pred_Week_<N>.csv` — the feature row per upcoming game (labels blank since these are
   future games). Also has a `week` variable to bump.

### `Python Scripts/`
5. `CFB_Gambling_Model.ipynb` — trains/tunes an XGBoost classifier on the historical predictors
   file, evaluates precision, saves a model + its feature list to `Model Information/`
   (`best_xgb_model_<date>.pkl`, `selected_features_best_model_<date>.json`).
6. `Week_Predictions.ipynb` — loads the current week's `CFB_Pred_Week_<N>.csv` plus the saved
   model/feature list, scores cover probability, and outputs the week's picks (threshold 0.6).

### `Model Information/`
Trained model binaries (`.pkl`) and their matching feature-name JSON, dated by training run.
Prediction code loads the feature list alongside the model and subsets/reorders columns to match
(`model.get_booster().feature_names`), since the predictor CSV has far more columns than the
model was trained on.

### Architecture notes (full detail in `CLAUDE.md`)
- **No lookahead bias is a first-class design constraint** — coaching stats are `lag()`-shifted
  by year, in-season game stats are lagged by one week per team, before being used as predictors.
- **Home/away symmetry via renaming, not row-per-team** — team-level predictors computed once per
  team-week, duplicated into `home_*`/`away_*` column sets, joined back by `game_id`.
- **Historical vs. current-season are parallel, separately-maintained pipelines** — a feature
  engineering fix generally needs to be made in both places to keep train/inference features
  consistent.

---

## `SQL Scripts/` — MySQL raw-data cache

A newer, parallel data store: a raw cache of the CFBD API in a local MySQL database
(`cfb_football`), populated via `cfbfastR`, mirroring each endpoint's raw response column-for-
column rather than the aggregated/lagged features the R pipeline derives. Nothing in
`R Scripts/` reads from it yet — it exists so the `cfb_win_total_model` and `cfb_rb_rushing_model`
projects (and any future ones) can query structured raw data directly instead of re-deriving
everything from the flat `Data/*.csv` files.

```bash
mysql -u root -e "CREATE DATABASE IF NOT EXISTS cfb_football;"
mysql -u root cfb_football < "SQL Scripts/schema.sql"
nohup Rscript "SQL Scripts/ingest_to_mysql.R" > ingest.log 2>&1 &
```

Idempotent and resumable — safe to `kill` and restart. Pulls `games`, `betting_lines`,
`team_talent`, `coaches`, `team_rosters`, `recruiting_players`, `returning_production`,
`game_team_stats`, and `plays` (2013-2025+ depending on endpoint). See
[`SQL Scripts/README.md`](SQL%20Scripts/README.md) for the full schema and per-endpoint column
list.

---

## Standalone modeling projects

Each of these lives in its own directory with a full `pyproject.toml`/`config/`/`src/`/`tests/`
layout, independent of the production notebook pipeline above. Follow each project's own README
for installation and usage — only a summary is given here.

### [`cfb_cover_model/`](cfb_cover_model/README.md)
An independent, walk-forward-validated rebuild of the spread-cover classifier, built to fix
several issues found in the production notebook (test-set reuse during feature selection, no
push handling, no probability calibration, a confirmed upstream R bug in the EPA-per-run columns).
Three modeling tracks (direct classification, regression-to-probability, stacked ensemble) plus a
from-scratch direct-CFBD-API ingestion path so the weekly cycle can run without the R scripts.
See `docs/final_writeup_2026.md` for the up-to-date results and recommended production signal.

### [`cfb_spread_model/`](cfb_spread_model/README.md)
A sibling rebuild of the same classifier, explicitly optimized for **precision** (not ROC-AUC)
with a config-driven, walk-forward-validated pipeline. Found a low honest ceiling (~0.51-0.54
pooled precision) — plain logistic regression on every raw column beat every feature-selection/
model-complexity variant it tried. See `docs/project_story.md` for the full before/after.

### [`cfb_win_total_model/`](cfb_win_total_model/README.md)
Predicts each FBS team's regular-season win total ahead of the season, using only
preseason-available information (talent, returning production, prior performance, coaching,
schedule). Reads from the `SQL Scripts/`-populated MySQL database. Selected model
(`gradient_boosting`) achieves walk-forward MAE 1.54, final-holdout MAE 2.01 (season 2025). Also
investigates and documents a real prediction-compression / declining-accuracy issue tied to
accelerating transfer-portal churn — see `docs/project_story.md`.

### [`cfb_rb_rushing_model/`](cfb_rb_rushing_model/README.md)
Predicts a running back's rushing yards for their upcoming game, refreshed weekly, restricted to
workload-eligible RB-games via a rolling-carries gate. Reads from the same MySQL database. v1
scope is RB rushing only (QB rushing deferred). MAE is the primary metric, with 80% prediction
intervals via out-of-fold residual quantiles.

### [`cfb_pythagorean_model/`](cfb_pythagorean_model/README.md)
Standalone, retrospective (not predictive) analysis of how well the Pythagorean expectation
(points-for/points-against) explains a team's actual 2025 win percentage. R² ≈ 0.80 for both the
classic (k=2) and numerically-fitted exponent; an opponent-adjustment extension found strength of
schedule adds only a marginal improvement on top of raw scoring margin.

### [`cfb_talent_distribution/`](cfb_talent_distribution/README.md)
How recruiting talent has been distributed across FBS football since the transfer portal (2018)
and NIL (2021). Corrects a diluted `blue_chip_ratio` bug in the shared pipeline (see its README
for the fix) and finds the median team's blue-chip roster share more than doubled (4.0% → 10.5%,
2015-2025) while talent concentration (Gini coefficient) has recently *decreased*.

### [`cfb_transfer_portal_flow/`](cfb_transfer_portal_flow/README.md)
Conference-level analysis of who's winning and losing on transfer-portal movement (2021-2025).
Finds the Big 12 leads on both raw net talent gained and per-transfer quality delta, while the
SEC and Big Ten post negative raw net talent (roster-crunch churn) but still clearly trade up in
quality per transfer.

---

## Contributing / working in this repo

See `CLAUDE.md` for guidance aimed at AI coding assistants (and useful as onboarding notes for a
human, too) — pipeline run order, the no-lookahead-bias constraint, and where feature-engineering
changes need to be duplicated across the historical/weekly scripts.
