# Modeling methodology

## Time-based validation

Expanding-window walk-forward validation by season, never a random split — this is the direct
fix for the current notebook's single fixed train/test split (train=2015-2022, test=2023-2024,
one shot, no sense of how stable precision is across seasons).

`config/modeling.yaml`:

```yaml
full_feature_start_season: 2015
excluded_seasons: [2020]              # COVID-shortened: 91 games vs 183-268 in neighboring seasons
min_train_seasons: 4
walk_forward_validation_seasons: [2019, 2021, 2022, 2023, 2024]
final_holdout_season: 2025
```

Verified per-season row counts in the source CSV: 2015:232, 2016:183, 2017:211, 2018:247,
2019:195, **2020:91** (COVID, ~39% of the neighboring-season median of 233.5 — excluded, same
justification the sibling `../cfb_win_total_model/` project uses), 2021:268, 2022:252, 2023:257,
2024:235, 2025:215. 2025 is a fully completed season in this dataset.

Each walk-forward fold's training set is every non-excluded season strictly before its
validation season (`modeling/splits.generate_walk_forward_folds`, ported near-verbatim from the
sibling project's `modeling/splits.py` — the season-boundary logic is target-agnostic). The
fold unit is always the **season**, never a mid-season split — a team's week-8 row shares
almost all its rolling-window inputs with its week-7 row, so splitting within a season would be
a much subtler leak than it first appears.

Hyperparameter tuning uses an **inner** expanding-window CV within each outer fold's training
seasons (`modeling/tuning.build_inner_season_cv`) — the same discipline one level down, so
`GridSearchCV` never sees a shuffled/random split either.

## Precision as the objective, throughout

Every `scoring="roc_auc"` call site in the current notebook is replaced by a precision-focused
scorer from `feature_selection/precision_scoring.py`:

- Feature selection (Stage 2, `feature_selection/selection.py`): permutation importance and
  RFECV both score candidates with `precision_at_coverage_floor_scorer`.
- Hyperparameter tuning (`modeling/tuning.tune_model`): `GridSearchCV`'s `scoring` parameter.
- Threshold selection (`modeling/threshold_selection.py`): grids over candidate thresholds and
  picks per-model whichever maximizes **mean precision across walk-forward folds**, subject to
  `coverage >= min_coverage_floor` in **every individual fold** — not just on average, since a
  combo that meets the floor on average but drops to near-zero coverage in one bad fold is a
  real "some seasons this model refuses to bet" failure a bettor would notice.

`min_coverage_floor` (default 0.10 in `config/modeling.yaml`) is set deliberately *below* the
current model's realized ~25% coverage at threshold 0.6, so the search isn't pre-constrained to
match the status quo — but high enough to rule out a degenerate "flag one ultra-confident game"
solution.

## Model comparison

`config/modeling.yaml`'s `models.candidates`: `logistic_regression`, `random_forest`,
`gradient_boosting`, `xgboost`, `lightgbm`, `catboost` (the last three all optional,
try/except-imported — the pipeline still runs on a bare `pip install -e .` without the
`boosting` extra, though `feature_selection/selection.py`'s Stage 2 importance signal does
default to XGBoost when available). Plus three baselines (`always_favorite`, `majority_class`,
`logistic_no_selection`) the current notebook has none of — see
`docs/feature_selection_methodology.md` for why each exists.

Hyperparameter grids are kept small and exhaustive-search-cheap (`config/modeling.yaml`
`hyperparam_grids`), same rationale as the sibling `../cfb_win_total_model/` project: the
training set is on the order of 1,000-2,000 rows, so a full grid search is affordable and a
randomized search is unnecessary.

## Production model selection

`scripts/evaluate_models.py` picks the winning `(model, threshold)` by highest mean walk-forward
precision among candidates (not baselines) that meet the coverage floor in every fold, falling
back to baselines only if no candidate qualifies. The winner is then refit on the final holdout
fold's training seasons (everything before 2025, excluding 2020), using whichever feature set
Stage 2 selected for that fold, and evaluated once on the true 2025 holdout season. See
`docs/project_story.md` for this build's actual numbers.
