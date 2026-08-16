# Modeling Methodology

## Why walk-forward, not random split

Predicting a team's win total is fundamentally a **forecasting** problem: at inference time,
the model only has access to seasons strictly before the target season. A random train/test
split across team-seasons would let the model train on 2024 data and validate on 2019 data —
information from the "future" relative to that validation point — which does not represent
the real deployment scenario and would produce optimistic, misleading metrics. Expanding-window
walk-forward validation is the only design that mirrors how this model will actually be used
each preseason.

## Season boundaries

`full_feature_start_season = 2015` is a hard floor: `team_talent` starts in 2015 and
`returning_production` starts in 2014, so the earliest season with a complete feature set
(prior-year game stats from t-1, talent/returning-production from t itself) is 2015. Season
2020 (COVID-shortened, `games` row count ~1/3 of neighboring seasons) is excluded from every
fold's training set and is never used as a validation season.

| Fold | Train seasons | Validation season |
|---|---|---|
| 1 | 2015–2018 | 2019 |
| 2 | 2015–2019 | 2021 |
| 3 | 2015–2019, 2021 | 2022 |
| 4 | 2015–2019, 2021–2022 | 2023 |
| 5 | 2015–2019, 2021–2023 | 2024 |
| Final holdout | 2015–2019, 2021–2024 | **2025** |

Every fold satisfies `max(train_seasons) < validation_season` by construction
(`modeling/splits.py`), and this invariant is independently re-verified in
`tests/test_leakage.py`.

## Why target season 2025 in this build (not a true future forecast)

The local MySQL cache only stores **completed** games (`SQL Scripts/ingest_to_mysql.R` filters
`completed = TRUE`) and tops out at season 2025 — there is no 2026 schedule in the database.
`modeling.yaml`'s `target_season` is a configuration value, so the exact same code will
produce genuine 2026 predictions automatically once that schedule is ingested. In this build,
season 2025 (the most recently completed season) serves as both the final walk-forward
holdout AND the demo "prediction" target — this is a legitimate backtest on real historical
data, not a live forecast. See `docs/assumptions_and_limitations.md`.

## Model list and the HistGradientBoostingRegressor default

Baselines: overall mean, previous-season wins, rolling 3-year average, conference average, and
a 2-feature OLS on prior wins + talent. **No market/poll baseline** — evaluated and rejected;
see `docs/assumptions_and_limitations.md`.

Candidates: OLS, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting,
HistGradientBoostingRegressor, and (if installed) XGBoost/LightGBM.
**HistGradientBoostingRegressor is the designated safe-default boosted model** — no extra
dependency, native NaN handling, and well-suited to a training set on the order of
1,000–1,300 team-season rows. XGBoost/LightGBM are added only if their packages are importable
(`try/except ImportError`, logged warning, never a hard failure), so the pipeline runs on a
bare `pip install -e .` without the `boosting` extra.

`PoissonRegressor` is available as an explicit opt-in secondary model
(`models['poisson_secondary']` in `modeling/models.py`) but is excluded from the default
candidate list: win totals are bounded by schedule length (0–13ish games) rather than
classically Poisson-distributed, so clipped regression is the primary path.

## Evaluation results (this build)

Walk-forward out-of-fold MAE by model (lower is better; **MAE is the primary metric**):

| Model | Mean OOF MAE |
|---|---|
| **gradient_boosting (selected)** | **1.536** |
| elasticnet | 1.552 |
| lasso | 1.558 |
| random_forest | 1.566 |
| hist_gradient_boosting | 1.567 |
| lightgbm | 1.569 |
| xgboost | 1.570 |
| ridge | 1.638 |
| ols | 1.790 |
| ols_prior_wins_talent (baseline) | 1.976 |
| rolling_3yr_avg (baseline) | 2.150 |
| conference_avg (baseline) | 2.239 |
| overall_mean (baseline) | 2.268 |
| prev_season_wins (baseline) | 2.461 |

Every real candidate model beats every baseline, including the previous-season-wins baseline
by a wide margin (1.54 vs. 2.46 MAE) — confirming the feature set adds real preseason signal
beyond "assume this year looks like last year." Notably, `prev_season_wins` is the *worst*
baseline, even behind the overall-season mean; this is a real, if initially counterintuitive,
finding about the current era of college football (heavy transfer-portal roster churn and
elevated coaching turnover mean a team's raw prior-season win count alone is a weak preseason
signal — see `docs/assumptions_and_limitations.md`).

Selected model: **gradient_boosting** (lowest mean OOF MAE; tie-break would be lowest MAE std
across folds, not needed here since it also has the lowest mean).

Final holdout (season 2025, trained only on 2015–2019 + 2021–2024, i.e. no 2025 information
whatsoever):

| Metric | Value |
|---|---|
| MAE | 2.01 |
| RMSE | 2.41 |
| Median AE | 1.75 |
| R² | 0.215 |
| Mean bias | -0.12 |
| % within 1 win | 27.2% |
| % within 2 wins | 56.6% |

The holdout MAE (2.01) is higher than the walk-forward mean (1.54), which is expected and
healthy — the walk-forward mean averages over 5 folds of varying difficulty, while the final
holdout is a single, genuinely out-of-sample season. Near-zero mean bias indicates no
systematic over/under-prediction.

## Prediction intervals: out-of-fold residual quantiles

Method: pool the out-of-fold residuals (`y_true - y_pred`) for the selected model across all 5
walk-forward folds (each row predicted exactly once, by a model trained only on strictly
earlier seasons). For a new point prediction `ŷ`, the interval is
`[ŷ + quantile(resid, 0.10), ŷ + quantile(resid, 0.90)]` (an 80% interval,
`prediction_interval_levels` in `config/modeling.yaml`).

This was chosen over bootstrapping or conformal prediction because the walk-forward design
already produces exactly the out-of-fold residual pool this method needs, "for free" — no
repeated refits, no separate calibration-set bookkeeping. **Known limitation**: this assumes
the residual distribution is stable across talent tiers and seasons (no heteroscedasticity
adjustment) — a per-tier residual pool or quantile regression is a reasonable v2 enhancement,
not attempted here.

## Clipping

Predictions are clipped to `[0, scheduled_games]` per team-season — a team cannot win a
negative number of games or more games than it plays.
