# Modeling methodology

## Validation scheme

Expanding-window walk-forward by season (`modeling/splits.py`), never a random split.
Season 2020 (COVID-shortened, 88 games vs. ~200-270 in a normal season) is excluded
entirely - from training as well as validation - not just skipped as a fold, since its
anomalous 8-game conference-only slate is unrepresentative training signal too. The final
season (2025) is held out and never touched until `evaluate_models.py`'s single final pass
(see `data_leakage_rules.md`).

With `min_train_seasons: 3` and 2020 excluded, the eligible training seasons are
2015-2019, 2021-2024 (9 seasons), producing 6 walk-forward folds validating on
2018, 2019, 2021, 2022, 2023, 2024 in turn, each trained on every strictly-earlier eligible
season.

## Primary metric: precision at a coverage floor

`modeling/evaluation.py::best_precision_at_coverage_floor` scans `modeling.yaml`'s
`threshold_grid` and picks the threshold with the highest precision among thresholds whose
coverage (fraction of games flagged) clears `min_coverage_floor` (default 0.20 - higher
than a prior project's 0.10, since the goal is a usefully-sized slate of bets, not just
clearing a technicality). If no threshold clears the floor, the highest-coverage threshold
is reported anyway, flagged as `met_floor: false` - never hidden.

Every model comparison in this project pools predictions across all 6 walk-forward folds
before computing one precision number (`modeling/evaluation.py::pooled_precision_at_threshold`),
so a 260-game fold and a 190-game fold don't get equal weight if simply averaged.

## Three model tracks

- **Track A - direct classification** (`modeling/classifiers.py`): logistic regression
  (both with and without stage 3-5 feature reduction, plus a PCA-reduced variant), random
  forest, gradient boosting, XGBoost, LightGBM, CatBoost.
- **Track B - regression-to-probability** (`modeling/regressor.py`): predicts the
  continuous `cover_margin` (elastic net, XGBoost regressor, or a 3-quantile ensemble),
  then converts to `P(home_covered)` via `Phi(predicted_margin / residual_std)` - the
  training-fold's own fitted residual spread (or IQR-derived spread for the quantile
  variant), never a global constant estimated elsewhere.
- **Track C - stacked ensemble** (`modeling/stacking.py`): a logistic-regression meta-
  learner trained on out-of-fold predictions from `logistic_regression`, `xgboost`, and
  `elastic_net_regression`. See `data_leakage_rules.md` rule 6 for the nesting guarantee.

Two zero-parameter baselines (`modeling/classifiers.py`) anchor every comparison:
`majority_class` (predicts the training fold's own base rate for every row) and
`always_favorite` (predicts the training fold's conditional cover rate given
`home_favored`, using `home_favored` only as a diagnostic input, never a model feature -
see `data_dictionary.md`).

## Calibration is evaluated, not assumed

`modeling/evaluation.py::calibration_report` computes Brier score, ROC-AUC, and rank
monotonicity (Spearman correlation between predicted-probability decile and actual cover
rate) for every candidate, on both pooled walk-forward OOF and the true holdout.
`evaluate_models.py` additionally tests isotonic and Platt calibration on top of the
production model's raw scores, fit on pooled walk-forward OOF and applied (never refit) to
holdout - directly testing whether raw probabilities can be turned into a trustworthy
confidence signal, the failure mode flagged in prior work on this same data (near-zero
rank monotonicity in raw scores).

## Model selection is walk-forward, but every candidate is scored on holdout

`evaluate_models.py` selects thresholds from pooled walk-forward OOF only, then refits
**every** candidate (not just the walk-forward-highest-precision model) on the entire
`train_pool` and scores it on the true 2025 holdout at its walk-forward-chosen threshold.
`outputs/model_comparison/holdout_model_comparison.csv` reports both the walk-forward rank
and the holdout rank for every candidate side by side, so a mismatch between the two (the
walk-forward pick turning out not to be the holdout's best performer) is visible rather
than hidden by only ever checking the one model the walk-forward metric picked.
