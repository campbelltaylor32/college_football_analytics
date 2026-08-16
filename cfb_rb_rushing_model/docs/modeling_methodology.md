# Modeling Methodology

## Target

`rushing_yards`: a single running back's rushing yards in a single game, for every
workload-eligible RB-game (see "Eligibility" below). One row per `(athlete_id, game_id)`.

## Row universe: workload-eligible RB-games, not every roster RB every week

Most RB roster spots in a given week have zero or trivial rushing involvement (backups,
committee-backfield complements, injuries, blowouts). Rather than predicting for every RB on
every roster (which would be dominated by structural zeros) or building a two-stage hurdle
model, this project restricts the modeling population to RB-games that clear a workload-
relevance gate, evaluated using only pre-game-known rolling history (`eligibility.py`):

```
eligible = (
    prior_games_played >= min_games_played_for_avg3 AND carries_avg3_asof >= min_trailing3_avg_carries
) OR (
    prior_games_played < min_games_played_for_avg3 AND (carries_avg_all_asof * prior_games_played) >= min_season_to_date_carries
)
```

Defaults (`config/features.yaml`): `min_trailing3_avg_carries=8`, `min_season_to_date_carries=15`,
`min_games_played_for_avg3=3`. These are starting points grounded in the live per-game carries
distribution (median ~5-6 carries/game across all rushers), not validated-optimal --
`scripts/run_eda.py` writes an `eligibility_threshold_sensitivity.csv` sweep (row count and
unique-player count across candidate thresholds) to inform tuning this before treating the
default as final.

A player with zero prior recorded games (true debut / transfer) is `eligible=False` by
construction -- a real, stated limitation (see `docs/assumptions_and_limitations.md`), not a bug.

## Train/validation: expanding-window walk-forward by season

Reused verbatim from the sibling `cfb_win_total_model`/`cfb_spread_model` projects'
`modeling/splits.py` -- appropriate unmodified at player-week grain because leakage
protection lives entirely in the per-row lag/merge_asof step (`data_leakage_rules.md`), not
the fold structure; the fold only needs to guarantee no season crosses the train/validation
boundary.

Season boundaries (`config/modeling.yaml`), independently justified for THIS project (not
copied from the sibling projects, which are driven by `team_talent`/`returning_production`
availability that this project doesn't use -- see `docs/assumptions_and_limitations.md` for
the live verification behind these numbers):

```
full_feature_start_season: 2014
excluded_seasons: [2020]                              # COVID-shortened
walk_forward_validation_seasons: [2019, 2021, 2022, 2023]
final_holdout_season: 2025
final_holdout_max_week: 8   # weeks 9-14 dropped post-build -- confirmed persistent rusher-name
                             # gap at the source (re-verified live, not just observed -- see
                             # docs/assumptions_and_limitations.md), not something a re-pull fixes
```

2025 is the final holdout (the most realistic recent-season test available), but only its
clean weeks 1-8 -- `scripts/build_modeling_dataset.py` drops holdout-season rows past
`final_holdout_max_week` after `dataset.build_modeling_dataset` runs, specifically because
those rows' targets would otherwise be silently zero-filled by the missing-data LEFT JOIN in
`targets.py`, indistinguishable from a real zero-carry game. 2024 (previously the holdout)
slides into the training pool for the 2025 holdout fold in its place.

## Baselines

Every candidate model is compared against three baselines (`modeling/baselines.py`), each
using ONLY the modeling table's own pre-computed rolling columns (never a fresh DB query):

- `player_rolling3_avg`: predicts `rushing_yards_avg3_asof` directly.
- `player_season_avg`: predicts `rushing_yards_avg_all_asof` directly.
- `position_avg`: predicts the training set's overall mean `rushing_yards` for every row (the
  "know nothing about this specific player" floor).

## Candidate models

OLS, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, HistGradientBoosting, XGBoost,
LightGBM (`modeling/models.py`) -- XGBoost/LightGBM load via try/except so the pipeline runs
without the `boosting` extra installed. Preprocessing: median-impute + StandardScaler in a
shared `ColumnTransformer` (`modeling/preprocessing.py`). Hyperparameters: small explicit
grids in `config/modeling.yaml`, tuned via `GridSearchCV` with an explicit season-ordered
inner CV (`modeling/tuning.py`) -- never sklearn's default random/K-fold, which would leak
future seasons into an "earlier" inner-validation fold.

## Evaluation

MAE is the primary metric (rushing yards is the target's own natural unit). `median_ae` is
reported alongside it specifically because it is more robust to the zero-carry-game noise
documented in `targets.py` -- a workload-eligible RB who left a game early (blowout, in-game
injury) produces a real `rushing_yards=0` row that MAE-alone would treat as a large,
"correctable" miss, when it is actually unpredictable noise (no injury-report table exists in
this DB). Also reported: RMSE, R², mean bias, `pct_within_10`/`pct_within_20` yards,
calibration by predicted-value bucket, and breakdowns by season and by `played`
(realized-carries vs. zero-carry eligible games).

## Prediction intervals

Out-of-fold residual quantile method (`modeling/evaluation.py::prediction_interval_from_residuals`):
for a new point prediction, the interval is `[y_hat + quantile(oof_residuals, lo),
y_hat + quantile(oof_residuals, hi)]` using the selected production model's pooled walk-forward
OOF residuals. Levels: `[0.10, 0.90]` (an 80% interval), `config/modeling.yaml:
prediction_interval_levels`.

## Weekly inference

`scripts/generate_week_predictions.py --season <S> --week <N>` reuses `dataset.build_modeling_dataset`
for the target season and filters to the target week -- the exact same feature-building code
path as training, so there is no separate inference-only feature logic that could silently
drift out of column-parity with the trained model. Before scoring anything, it runs a hard
data-quality gate (`data_validation.check_rusher_name_completeness`) against the target
week's `plays` data and aborts loudly if the NULL rate is too high -- the direct response to
the 2025-week-9+ ingestion gap discovered during planning (see
`docs/assumptions_and_limitations.md`).
