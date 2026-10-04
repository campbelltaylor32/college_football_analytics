# cfb_power_ratings

A weekly-updating FBS team power rating: one number per team, in points on a neutral field,
that answers "who would be favored" between any two teams and by how much. Seeded by a
**preseason model prior** (so a full ranking can be published before a single game is played)
that fades into **actual opponent-adjusted in-season results** as the season progresses.

Unlike the other `cfb_*` projects in this repo, this isn't a spread-cover or win-total
classifier — it's a full-league rating system, closer to what Sagarin/Massey/SRS-style ratings
do, but grounded in this repo's own preseason feature engineering and validated against real
historical betting markets.

## Why this exists / how it works

1. **Historical target** (`srs.py`): every team-season's actual, opponent- and home-field-
   adjusted power rating, computed via an iterative Simple Rating System (SRS) — a
   generalization of `cfb_pythagorean_model/opponent_adjusted_analysis.py::compute_srs` that
   adds home-field-advantage correction and pools non-FBS opponents rather than dropping them.
2. **Preseason model** (`dataset.py`, `modeling/`): a gradient-boosted/ridge regression trained
   on preseason-only features (talent, returning production, transfer portal activity,
   coaching, a team's own recent-season SRS trend, last season's Pythagorean win%-expectation
   gap) to predict that season's actual SRS — walk-forward validated by season, and
   sanity-checked against real historical betting-market spreads (see "Results" below).
3. **In-season blending** (`rating_engine.py`): the preseason prediction is treated as a fixed
   number of "phantom games" against a league-average opponent, mixed into the same SRS
   iteration used for the historical target. A team with 0 games played gets back its preseason
   prior; a team with many games played is dominated by real results — no manual blending
   logic, the fade-out falls out of the weighted-average math for free.

See `docs/methodology.md` for the full technical writeup.

## Results (validated against real data)

- **Preseason model**: pooled walk-forward MAE of **6.4 points** against actual end-of-season
  SRS (vs. 7.9 for "assume last season's rating unchanged" and 10.2 for "assume league
  average") — see `outputs/model_comparison/` after running `scripts/train_preseason_model.py`.
- **Market validation**: the preseason model's implied point spread correlates **0.74–0.84**
  with real historical consensus betting spreads across all 4 walk-forward validation seasons
  (2021–2024).
- **In-season backtest** (`scripts/backtest_season.py --season 2024`): the blended rating's
  implied spread tracks real market spreads within **~3.8–4.0 points MAE** across the season,
  visibly improving from early season (weeks 1–4: ~4.4 MAE) to mid/late season (weeks 5+:
  ~3.6 MAE) as real results accumulate — the intended fade-out behavior, not just an
  architectural claim.
- **Play-by-play efficiency blend** (`efficiency.py`): each game's margin is 75% actual score,
  25% an efficiency-implied margin (net EPA/play + net success rate on both sides of the ball,
  garbage time removed, converted to points by a regression on prior seasons -- R² 0.77).
  Backtested over 2021-2025: a small real gain when the preseason prior is weak (1-3 phantom
  games, up to ~0.1 pts MAE vs actual margin, most in weeks 1-4), roughly neutral at 5+.
  Efficiency-only is worse than scores-only. Ratings CSVs also carry opponent-adjusted
  offense/defense EPA and success-rate columns. Details: `docs/methodology.md` §4b.
- **Pythagorean win%-expectation feature, tested honestly**: adding last season's Pythagorean
  win% and its gap vs. actual win% (`features/pythagorean.py`, see `docs/methodology.md`)
  changed pooled walk-forward MAE by under 0.01 points — a real null result, not a win. Kept in
  the model since it doesn't measurably hurt, but it isn't earning meaningful weight either;
  `srs_lag1` already captures most of what it would offer.
- **Roster age/experience feature, tested and reverted**: a class-based signal
  (`team_rosters.year`, filtered for a confirmed data-corruption pattern in older seasons) and
  an independent athlete-tenure signal (`features/roster_experience.py`) both made pooled MAE
  *worse* (class: +0.21, tenure: +0.04, combined: +0.30) — a real regression, not a null result,
  so both were reverted rather than kept. Full diagnostic in `docs/methodology.md`.
- **Transition-team-aware imputation**: brand-new FBS transition teams (no prior-season FBS SRS
  at all, e.g. Sacramento State/North Dakota State entering 2026) now get every missing feature
  filled with the training data's minimum observed value instead of the median
  (`modeling/models.py::TransitionTeamAwareImputer`) — median imputation was scoring these teams
  as average FBS rosters (verified: Sacramento State ranked #46 nationally before the fix; #120
  after). Applying this to the 12 historical transition-team rows in training too, not just
  2026 scoring, genuinely *improved* pooled walk-forward MAE (6.412 → 6.390), not just a neutral
  correctness fix. See `docs/assumptions_and_limitations.md` for the full writeup.
- **Cross-project test: does this rating help `cfb_win_total_model`?** `scripts/export_win_total_
  feature.py` exports an honest out-of-sample preseason-rating series (`outputs/ratings/history/
  preseason_ratings_by_season.csv`, seasons 2020+) for that sibling project to test as a
  candidate feature. Result: a real null — no measurable improvement over its existing
  `sp_overall_entering_t` feature. Full writeup: `../cfb_win_total_model/docs/power_rating_experiment.md`.
- **2026 schedule difficulty**: `scripts/export_schedule_strength.py` +
  `scripts/plot_schedule_strength.R` rank every FBS team's 2026 schedule by mean site-adjusted
  opponent preseason rating (a true road opponent counts as `+hfa` harder, a home opponent
  `-hfa` easier, mirroring this project's own site-adjustment principle). Non-FBS opponents get
  a proxy rating calibrated from season 2025's actual results (`srs.estimate_non_fbs_pool_
  rating`), not an invented constant. Both a **mean** and a **median** version are produced --
  the mean answers "how hard is the schedule overall," the median "how hard is a typical game,"
  and they can disagree sharply (e.g. Michigan State: #3 nationally by mean thanks to two elite
  games against Oregon and Notre Dame, #22 by median once those two extreme games stop
  dominating the average). Output: `outputs/schedule_strength/2026/hardest_schedules_overall.png`
  / `..._median.png` (Top 25 league-wide) and one table per conference in
  `outputs/schedule_strength/2026/conferences/` / `conferences_median/`.

## Data source

The same local MySQL `cfb_football` database `cfb_win_total_model`/`cfb_rb_rushing_model` read
from, populated by the repo-root `SQL Scripts/` directory — see `../SQL Scripts/README.md`.
This project never modifies that database, only reads. One exception: the transfer-portal
endpoint (`cfbd_recruiting_transfer_portal`) isn't in the DB schema at all, so
`features/roster_turnover.py` pulls it live from the CFBD API for seasons 2021+ (falling back
to roster-diff inference for earlier seasons, which the portal endpoint has no data for).
`scripts/update_ratings.py` also pulls an upcoming week's real schedule live — `games`/
`betting_lines` in the DB are completed-games-only by design.

## Repository structure

```
cfb_power_ratings/
├── config/                  database.yaml, features.yaml, modeling.yaml
├── docs/                    methodology, data leakage rules, assumptions/limitations
├── scripts/                 train_preseason_model.py, generate_preseason_ratings.py,
│                             update_ratings.py, backtest_season.py,
│                             export_win_total_feature.py (cross-project rating export),
│                             export_schedule_strength.py + plot_schedule_strength.R
│                             (2026 schedule-difficulty tables)
├── src/cfb_power_ratings/
│   ├── config.py, database.py, cfbd_client.py, live_data.py
│   ├── srs.py                opponent- and site-adjusted SRS (historical target + shared
│   │                          fixed-point iteration core)
│   ├── rating_engine.py       in-season phantom-game prior blending, implied matchups,
│   │                          win-probability conversion
│   ├── dataset.py, preseason.py   modeling-dataset assembly, trained-model load/predict
│   ├── features/               talent_recruiting.py, returning_production.py,
│   │                            roster_turnover.py, coaching.py, program_history.py,
│   │                            pythagorean.py, roster_experience.py (built, tested, not
│   │                            currently used — see README "Results" / docs/methodology.md)
│   └── modeling/                baselines.py, models.py, splits.py, evaluate.py
├── tests/                    srs.py / rating_engine.py numerical correctness + regression tests
├── data/{interim,processed}  cached intermediates (gitignored)
└── outputs/                  models/ (trained preseason model), ratings/<season>/ (weekly
                               snapshots), model_comparison/ (walk-forward + backtest results)
```

## Installation

```bash
cd cfb_power_ratings
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,boosting]"
cp .env.example .env
```

`CFBD_API_KEY` is read from the **repo root's** `.env` (see `../.env.example`) — only needed
for `features/roster_turnover.py`'s transfer-portal pull and `scripts/update_ratings.py`'s live
schedule pull.

## Running the pipeline

```bash
# 1. Train the preseason model (walk-forward evaluates candidates, fits the winner on all
#    eligible history, writes outputs/models/preseason_model.joblib):
python scripts/train_preseason_model.py

# 2. Preseason-only rankings for a new season, before any games are played -- requires
#    season-<season> rows already ingested into team_talent/coaches/returning_production/
#    team_rosters/recruiting_players (run SQL Scripts/ingest_to_mysql.R for the new season
#    first; this script only reads):
python scripts/generate_preseason_ratings.py --season 2026

# 3. Weekly in-season update, once games start. --week N means "score week N; blend in every
#    completed game where week < N". So to fold in Week 1's results, run --week 2 (which also
#    scores the Week 2 slate). First re-run SQL Scripts/ingest_to_mysql.R with CURRENT_WEEK
#    bumped to the latest completed week so the DB has that season's games; otherwise this
#    falls back to a slow, rate-limited live CFBD pull. Writes outputs/ratings/<season>/
#    week_<NN>_ratings.csv and week_<NN>_matchups.csv:
python scripts/update_ratings.py --season 2026 --week 2 [--phantom-games 5] [--scoring-weight 0.75]

# 3b. In-season charts from a week_<NN>_ratings.csv (Top 25 + one table per conference). Same
#     scripts as the preseason charts -- pass the week as a second arg (default 0 = preseason).
#     Writes week_<NN>_top25.png and conferences/week_<NN>/*.png:
#     Outputs are tagged by weighting (week_<NN>_ratings_pg<P>_sw<W>.csv); pass that tag as a
#     third arg to chart a specific version:
Rscript scripts/plot_preseason_top25.R 2026 2 pg5_sw0.75
Rscript scripts/plot_conference_rankings.R 2026 2 pg5_sw0.75

# 4. Backtest the whole system against a past, fully-completed season (implied-spread MAE vs.
#    real market lines, win-probability calibration, phantom_games sensitivity sweep):
python scripts/backtest_season.py --season 2024

# 5. Rank 2026 schedule difficulty (overall Top 25 + one table per conference) -- requires
#    step 2 to have already written outputs/ratings/2026/week_00_ratings.csv:
python scripts/export_schedule_strength.py
Rscript scripts/plot_schedule_strength.R
```

## Weekly operational note

`PRODUCTION_SEASON`/`PRODUCTION_WEEK` aren't config-driven here (unlike `cfb_cover_model`'s
Docker service) — this project is standalone scripts only, matching every other sibling
project's convention. Bump `--season`/`--week` by hand each week; there's no automatic
NCAA-calendar detection anywhere in this repo.

## Known limitations

See `docs/assumptions_and_limitations.md` for the full list. In brief: one league-average
home-field-advantage constant (no per-team/per-season estimate); non-FBS opponents are pooled
into a single fixed-rating pseudo-team rather than individually rated; `phantom_games=5` is
config-tunable but not rigorously optimized beyond the 3/5/8 sensitivity sweep in
`scripts/backtest_season.py`; the preseason model's own accuracy ceiling is bounded by how
predictable a season's outcome genuinely is from preseason information alone (~6.4 points MAE
against actual SRS — real, but not small).
