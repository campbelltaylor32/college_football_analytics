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
│                             update_ratings.py, backtest_season.py
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

# 3. Weekly in-season update, once games start -- blends games through week N-1, scores every
#    real matchup on week N's schedule:
python scripts/update_ratings.py --season 2026 --week 4

# 4. Backtest the whole system against a past, fully-completed season (implied-spread MAE vs.
#    real market lines, win-probability calibration, phantom_games sensitivity sweep):
python scripts/backtest_season.py --season 2024
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
