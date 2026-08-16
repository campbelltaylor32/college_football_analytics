# Data leakage rules

This document is the checklist `tests/test_leakage.py`, `tests/test_splits.py`, and
`tests/test_stacking_oof.py` exist to enforce. If a change to this project violates one of
these rules, it should fail a test, not just look suspicious in review.

## 1. No row's features may reflect information unavailable before that game

Already handled upstream, in R: `Merge_Predictors_CFB_Historical.R` lags every
`prev_week_*`/`*_avg_all`/`*_avg3` column by one game within `group_by(team, year)`, and
lags coaching stats by one year. This project does not re-derive that logic - its own
leakage tests are a regression safety net over it (`tests/test_leakage.py`'s column-
exclusion tests), not a reimplementation.

## 2. Walk-forward folds train only on strictly earlier seasons

`modeling/splits.py`'s `walk_forward_folds` builds expanding-window folds where a fold
validating on season `S` trains only on seasons in `train_pool` strictly before `S`. Never
a random split. Verified by `tests/test_splits.py::test_walk_forward_folds_never_train_on_future_seasons`.

> **This rule caught a real bug, not a hypothetical one.** An earlier version of
> `modeling/splits.py` reset both `train_pool` and `holdout`'s index to a fresh 0-based
> range, which silently made `.loc[holdout.index]` against a frame built from the original
> source return training rows instead. It was caught because the resulting holdout
> precision numbers (up to 98.7%) were implausible, not because the bug was visually
> obvious in the code. See `docs/project_story.md`'s "A real bug, caught by the numbers
> looking too good" section and the regression test
> `tests/test_splits.py::test_train_pool_and_holdout_indices_are_disjoint_and_index_the_original_frame_correctly`.

## 3. The final holdout season is isolated until a single, final evaluation pass

`modeling/splits.py`'s `get_holdout_split` removes `data.yaml`'s `final_holdout` season(s)
(2025) from `train_pool` before any fold is built. Every stage - transform ablation
(`select_features.py`), correlation pruning, embedded selection, PCA reduction, Track A/B/C
training (`train_models.py`) - operates on `train_pool` only. The holdout is touched
exactly once, in `evaluate_models.py`, when every candidate model is refit on the entire
`train_pool` and scored on holdout at a threshold chosen before that scoring ever happens
(see rule 5). Verified by `tests/test_splits.py::test_holdout_never_appears_in_train_pool`.

## 4. Feature reduction is fit per-fold, on that fold's training rows only

`feature_selection/correlation_pruning.py`, `embedded_selection.py`, and
`pca_reduction.py` all take `X_train`/`y_train` as their only data arguments - there is no
code path in any of them that can see a fold's validation rows or the holdout. Every call
site (`scripts/select_features.py`'s reduction-strategy comparison,
`scripts/train_models.py`'s per-fold candidate fitting, `scripts/evaluate_models.py`'s
final refit) passes only that call's own training slice. Verified by
`tests/test_leakage.py::test_embedded_selection_never_receives_validation_rows`.

The one exception, documented rather than hidden: `feature_engineering.py`'s
`build_transform_variant`/`apply_home_away_representation` are pure column arithmetic (no
fitting), so they're applied once to the entire frame (`evaluate_models.py`'s
`build_full_variant`) for convenience - this is safe because no parameter is estimated
from the data in that step, unlike stages 3-5.

## 5. Threshold selection and calibration are fit on pooled walk-forward OOF, never on holdout

`evaluate_models.py`'s `select_thresholds` picks each model's betting threshold from
`modeling.yaml`'s `threshold_grid` by scoring against **pooled walk-forward OOF
predictions only** (`outputs/model_comparison/oof_predictions.csv`, produced entirely
within `train_pool`). That threshold is then applied - not re-tuned - when scoring the
holdout. Calibration (`modeling/calibration.py`'s isotonic/Platt fits) follows the same
rule: fit on pooled walk-forward OOF, applied (never refit) to holdout predictions
(`evaluate_models.py`'s calibration section).

This directly targets the failure mode found in the production notebook
(`../Python Scripts/CFB_Gambling_Model.ipynb`): its feature-count and threshold search is
scored by metrics computed **on the same 492-game test set** it then reports precision on
- a test-set-reuse pattern that inflates the reported number. Nothing in this project's
threshold or calibration selection ever looks at holdout rows before they're scored.

## 6. Stacking's meta-learner never trains on a base model's in-sample prediction

`modeling/stacking.py`'s `generate_oof_base_predictions` runs its own inner expanding-
window walk-forward *within* a given outer fold's training rows, so the meta-learner's
training data is out-of-fold base-model predictions, never a base model's prediction for a
row it was itself trained on. `safe_inner_min_seasons` clamps the inner split so this
still holds even for the outer fold with the least available training history. Verified by
`tests/test_stacking_oof.py` using a synthetic "fingerprint" model that reports exactly
which seasons it was trained on, then asserting every OOF prediction reflects a strictly
earlier season than the row it's predicting.

**Documented simplification**: all base models feeding the stack share one feature set -
the outer fold's own "reduced" (stage 3+4) selection - rather than each inner-fold step
re-running feature selection itself. Re-running stages 3-5 at both the outer-fold and
inner-OOF level was judged not worth the added compute for how much of a difference it
would make to the leakage picture (the outer-fold-level fold-locality is what matters for
avoiding a genuine leak; the inner loop only decides who the meta-learner's *training
labels* came from, not whether future information entered the model).
