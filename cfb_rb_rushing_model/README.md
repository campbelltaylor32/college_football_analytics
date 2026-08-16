# cfb_rb_rushing_model

A leakage-safe regression pipeline that predicts a running back's **rushing yards** for their
upcoming game, refreshed weekly during the season, using only information available before
that game's kickoff.

## Goal and target

`rushing_yards`: one running back's rushing yards in one game, one row per
`(athlete_id, game_id)` -- restricted to **workload-eligible** RB-games (a rolling-carries
threshold, see "Eligibility" below), not every roster RB every week. **v1 scope is RB only**
(QB rushing explicitly deferred to a later phase, per the approved plan).

## Database requirement

This project reads from the same local MySQL database (`cfb_football`) the sibling
`cfb_win_total_model`/`cfb_spread_model` projects use, populated by the repo-root `SQL Scripts/`
directory. It must already exist and be populated before running anything here -- see
`../SQL Scripts/README.md`. This project never modifies that database -- it only reads.

## Repository structure

```
cfb_rb_rushing_model/
├── config/                  database/data/features/modeling YAML configs
├── docs/                    leakage rules, data dictionary, methodology, assumptions/limitations, project story
├── scripts/                 one script per pipeline stage
├── src/cfb_rb_rushing_model/
│   ├── config.py, database.py, dataset.py, cleaning.py, data_validation.py
│   ├── schedule_spine.py     team-week opponent/schedule spine (works for future weeks too)
│   ├── player_game_rushing.py  raw player-game rushing aggregation from `plays`
│   ├── player_resolution.py   name -> athlete_id resolution
│   ├── eligibility.py          workload-relevance gate + authoritative player-grain rolling features
│   ├── targets.py               rushing_yards target construction
│   ├── features/                team_offense_context, opponent_defense_context, game_context, rushing_workload
│   └── modeling/                 splits, preprocessing, baselines, models, tuning, evaluation, diagnostics, artifacts
├── tests/                   pytest suite, integration tests against the live local DB + pure-function unit tests
├── data/{interim,processed} cached intermediates (gitignored)
└── outputs/                 data_inventory, eda, feature_analysis, model_comparison, diagnostics, predictions, models
```

## Installation

```bash
cd cfb_rb_rushing_model
pyenv install -s 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,boosting]"
```

## Environment setup

```bash
cp .env.example .env
# .env defaults already match a local `mysql -u root` install with no password, db=cfb_football
```

## Running the pipeline

```bash
python scripts/run_pipeline.py                 # inspect -> build_dataset -> quality_gate -> eda -> train -> evaluate
```

Or stage by stage:

```bash
python scripts/inspect_database.py              # data inventory -> outputs/data_inventory/
python scripts/build_modeling_dataset.py         # -> data/processed/modeling_dataset.parquet + feature_registry.csv
python scripts/run_eda.py                        # -> outputs/eda/, including the eligibility-threshold sensitivity sweep
python scripts/train_models.py                   # walk-forward OOF predictions -> outputs/model_comparison/oof_predictions.csv
python scripts/evaluate_models.py                 # model selection + final holdout + diagnostics + production artifact save
```

**Weekly inference** (run once per week during the season, not part of the full pipeline):

```bash
python scripts/generate_week_predictions.py --season 2024 --week 6
```

This resolves the latest production model artifact by glob (no hardcoded dates), reuses the
exact same feature-building code path as training, and runs a hard data-quality gate
(`check_rusher_name_completeness`) before scoring anything -- it aborts loudly rather than
silently producing predictions from a broken data week (see `docs/assumptions_and_limitations.md`
for why this check exists).

## Eligibility (workload-relevance gate)

Only RB-games where the player clears a rolling-carries threshold (evaluated as of their most
recently PLAYED game, not calendar week -- correctly carries workload history across a bye
week) are included, rather than predicting for every roster RB every week or building a
two-stage hurdle model. Config-driven, `config/features.yaml: eligibility` -- defaults are a
starting point grounded in the live carries distribution, not validated-optimal; see
`docs/modeling_methodology.md` and the sensitivity sweep `scripts/run_eda.py` produces.

## Leakage-prevention methodology

Every feature's timing rule is documented in `docs/data_leakage_rules.md` and enforced by
`tests/test_feature_shifting.py` and `tests/test_leakage.py`. Two-step compute-then-lag for
every rolling feature; `eligibility.py`'s `merge_asof(..., allow_exact_matches=False)` for the
player-grain features and the eligibility gate itself, so a target game reached after a bye
week or injury-missed game still only ever sees the player's last ACTUALLY PLAYED game's stats.

## Time-based validation

Expanding-window walk-forward validation by season (`modeling/splits.py`, ported verbatim from
the sibling projects), never a random split. Season 2020 (COVID) is excluded entirely. 2025 is
the final holdout season, but capped at week 8 -- weeks 9-14 have a confirmed, persistent
`rusher_player_name` gap at the source (re-verified live against the CollegeFootballData API
itself, not just observed once -- see `docs/assumptions_and_limitations.md`) and are dropped
post-build rather than silently zero-filled.

## Explosive-run predictors

Explosive-run count AND rate, on both the player side (`explosive_runs_avg3_asof`/
`explosive_run_rate_avg3_asof`, etc.) and the opposing-defense side
(`def_explosive_runs_allowed_avg3_lag1`/`def_explosive_rate_allowed_avg3_lag1`, etc.), sharing
one config threshold (`config/features.yaml: explosive_run_yard_threshold`) so both sides are
directly comparable -- called out explicitly per the approved plan.

## Feature categories

`rushing_workload` (player-grain, exposed via `eligibility.py`'s `_avg3_asof`/`_avg_all_asof`
columns), `team_offense_context` (rush/pass mix, tempo, time of possession),
`opponent_defense_context` (run defense allowed, including opposing time of possession),
`game_context` (home/away, neutral site, conference game, rest days, optional betting
context). Full list in `outputs/feature_analysis/feature_registry.csv` -- see
`docs/data_dictionary.md`.

## Evaluation metrics

MAE is the primary metric (rushing yards is the target's own natural unit); `median_ae` is
reported alongside it because it's more robust to the zero-carry-game noise a workload-eligible
RB's in-game injury or blowout benching produces (no injury-report table exists in this
database -- see `docs/assumptions_and_limitations.md`). Also reported: RMSE, R², mean bias,
`pct_within_10`/`pct_within_20` yards, calibration by predicted-value bucket, breakdowns by
season and by `played`. Prediction intervals via out-of-fold residual quantiles (80% interval
by default).

## Known limitations

See `docs/assumptions_and_limitations.md` for the full list, in brief:
- QB rushing is out of scope for v1 (deferred, not forgotten).
- Name-to-roster resolution is ~95-97% reliable against the full roster (~2-3% genuinely
  unmatched); a small, known miss (first-name-abbreviated box scores, e.g. "A. Jeanty") is not
  fixed by the current normalized-matching pass.
- A true debut game (first career carry, or a transfer's first game) is structurally
  unpredictable by this design -- no prior game exists to gate eligibility on or build rolling
  features from.
- No injury-report data exists in this database -- in-game injuries/benchings are unpredictable
  noise, not a model failure.
- The eligibility threshold is an explicit, tunable business rule, not a validated-optimal
  value.
