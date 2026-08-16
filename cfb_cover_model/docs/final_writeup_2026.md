# cfb_cover_model: final write-up and 2026 deployment plan

## Executive summary

This project rebuilt the spread-cover classifier from scratch with leakage-safe walk-forward
validation, then ran two follow-on passes: a feature-importance deep dive, and a targeted
feature-engineering pass based on it. Along the way, **three real bugs were caught and fixed**
- two of them produced results that looked good enough to be suspicious, which is exactly
how they were caught. After the fixes, **the project's precision target was met on true
2025 holdout data for the first time**, and a cross-model-family agreement filter
(`logistic_regression` + `xgboost_regressor`, both flagging the same game) is the most
defensible actionable signal this project produced.

**Recommendation for 2026**: use `scripts/generate_weekly_predictions.py` to flag a bet only
when both models agree. Treat the first several weeks of live 2026 data as the real test -
everything below is validated against one already-completed holdout season that has, by
now, been examined from many angles in the course of this work. See "Honest limitations"
before acting on this at meaningful stakes.

## What was built

1. **A from-scratch pipeline** (`cfb_cover_model/`, replacing ad-hoc analysis in
   `../Python Scripts/CFB_Gambling_Model.ipynb`): push-aware labels, expanding-window
   walk-forward validation, three parallel modeling tracks (direct classification,
   regression-to-probability, stacked ensemble), and dimensionality reduction via
   correlation-cluster pruning + embedded elastic-net selection, compared against PCA and a
   no-reduction anchor. Full detail: `docs/project_story.md`, `docs/modeling_methodology.md`,
   `docs/feature_selection_methodology.md`.
2. **A feature-importance deep dive** (`scripts/analyze_feature_stability.py`,
   `scripts/explain_model.py`): fold-stability of selected features across two temporal-
   transform configs, cross-referenced with single-shot holdout permutation importance.
   Full detail: `docs/feature_importance_findings.md`.
3. **Domain-informed feature engineering** (`src/cfb_cover_model/engineered_features.py`),
   grounded in that analysis: opponent-adjusted matchup differentials for down-conversion
   and EPA stats, a trimmed 4-column returning-production family, and a special-teams family
   collapsed into one composite score per side.
4. **A model-agreement analysis** (`scripts/analyze_model_agreement.py`): every Track A
   classifier paired with every Track B regressor, scored on whether requiring both to agree
   improves on either alone.
5. **A production weekly-scoring script** (`scripts/generate_weekly_predictions.py`):
   the actual deliverable for 2026 - see "How to run this in 2026" below.

## Three real bugs, caught and fixed

Rigor here wasn't just "write tests up front" - two of these were caught specifically
because a result looked too good, and were run down before being trusted. That pattern is
worth continuing, not something to consider finished:

1. **Index-reset bug** (`modeling/splits.py`): an early evaluation run reported up to 98.7%
   holdout precision for tree models. Root cause: `get_holdout_split` reset both
   `train_pool` and `holdout` to a fresh 0-based index, so both index spaces started at 0
   again - `.loc[holdout.index]` against a frame built from the original data silently
   returned training rows. Fixed by preserving original index labels throughout; a
   regression test (`tests/test_splits.py::test_train_pool_and_holdout_indices_are_disjoint_and_index_the_original_frame_correctly`)
   now asserts disjointness directly.
2. **Transform-ablation suffix bug** (`feature_engineering.py`): the new engineered
   columns name their temporal transform with a trailing suffix
   (`matchup_adj_<stat>_prev_week`), but the ablation's column parser only recognized raw
   columns' leading `prev_week_` prefix. The `_prev_week`-suffixed engineered columns fell
   through to "always included" and leaked into every ablation candidate regardless of
   which transform was being tested - inflating their apparent importance in an
   intermediate run. Fixed by recognizing the trailing suffix too
   (`tests/test_feature_engineering.py::test_engineered_prev_week_suffix_recognized_as_temporal`).
3. **Weekly-scoring engineered-features gap** (`cleaning.py`): `CFB_Pred_Week_<N>.csv`
   files never went through `apply_engineered_features` - only historical training data
   did, inside `load_and_validate_dataset.py`. Any model using an engineered feature would
   `KeyError` when scoring a live week. This affected the original
   `generate_week_predictions.py` too, silently, for as long as it existed alongside the
   engineered features. Fixed with a shared `cleaning.prepare_week_frame` helper, used by
   both weekly scripts now
   (`tests/test_prepare_week_frame.py::test_prepare_week_frame_applies_engineered_features`).

## Final results (true 2025 holdout, 215 games, never touched during selection)

**Target met for the first time**: `logistic_regression` (the walk-forward-selected
production model) scored **53.6% precision at 26.0% coverage**, clearing the 53% precision
target and 20% coverage floor. Full 14-candidate comparison:
`outputs/model_comparison/holdout_model_comparison.csv` and
`outputs/model_comparison/holdout_comparison_table.png`.

| model | walk-forward rank | holdout precision | holdout coverage |
|---|---|---|---|
| lightgbm | 10th | 60.6% | 15.3% |
| logistic_no_selection | 4th | 58.5% | 19.1% |
| xgboost_regressor | 2nd | 58.1% | 28.8% |
| xgboost | 6th | 57.5% | 34.0% |
| elastic_net_regression | 3rd | 57.5% | 18.6% |
| **logistic_regression (production)** | **1st** | **53.6%** | **26.0%** |

Honest attribution (`docs/project_story.md`'s "Update" section): the improvement traces to
a combination of the second bug fix enabling a fair ablation, the ablation newly selecting
an `all_three transforms`/`differential` representation, and a real but partial
contribution from the engineered features - the returning-production trim is strongly
validated (2 of 4 kept sub-metrics selected in 6 of 6 walk-forward folds), the
opponent-adjustment work is only weakly present in the final model, and the special-teams
composite is neutral rather than a standout.

## Which model to actually trust: overfitting changes the ranking

Precision-at-threshold alone is misleading here. Checking train-vs-holdout accuracy and
ROC-AUC (`scripts/analyze_train_vs_holdout.py`,
`outputs/model_comparison/train_vs_holdout_accuracy.csv`) tells a different story:

| model | train accuracy | train ROC-AUC | holdout ROC-AUC | verdict |
|---|---|---|---|---|
| xgboost | 97.1% | 0.997 | 0.523 | severely overfit - holdout precision likely luck |
| lightgbm | 94.0% | 0.989 | 0.547 | severely overfit - same caution |
| xgboost_regressor | 87.7% | 0.952 | 0.538 | moderately overfit, still usable |
| logistic_regression | 63.0% | 0.681 | 0.539 | smallest gap of any classifier |
| **elastic_net_regression** | **59.8%** | **0.629** | **0.557** | **smallest gap overall; highest holdout AUC of anything tested** |

A model that memorized 94-97% of its training data and shows holdout ROC-AUC at or barely
above chance (xgboost, lightgbm) getting a good-looking precision number at one threshold
reads as a favorable roll on one season, not found signal - the same pattern this whole
project has been built to catch, just at the model-selection level instead of the
feature-selection level.

## Calibration: usable at a threshold, not as a graduated confidence score

`logistic_regression`'s raw holdout probabilities are not well calibrated - Brier score
(0.257) is worse than just guessing the base rate for every game (~0.250), and one
high-confidence decile (mean predicted 0.63) actually covers at only 38.1%, below a coin
flip. Isotonic/Platt calibration (fit on walk-forward OOF only, applied to holdout) gives a
modest Brier improvement but can't fix the ranking itself. **Use the model to flag/don't-flag
at its tuned threshold only - don't size bets by "how far the probability is from 0.5."**
Full detail in the conversation history around this analysis; raw numbers in
`outputs/calibration/holdout_summary.json`.

## The agreement approach: why `logistic_regression` + `xgboost_regressor`

Every Track A classifier was paired with every Track B regressor
(`outputs/model_comparison/model_agreement_combinations.csv`) - restricting to games where
*both* flag a bet, at each model's own tuned threshold. Nearly every combination showed a
positive lift over either model alone, but the combinations with the largest lift leaned on
the severely overfit tree classifiers (xgboost, lightgbm, catboost) - agreement between two
models isn't independent confirmation if one of them is mostly fitting noise.

Restricting to models that aren't severely overfit, `logistic_regression` +
`xgboost_regressor` is the best-supported pair: 60.6% precision on 33 games (a real sample,
not a 15-18 game fluke), with positive lift over both components alone (+7.0 vs.
logistic_regression solo, +2.5 vs. xgboost_regressor solo). They also differ on two axes at
once - linear vs. tree-ensemble, and classify-the-label-directly vs.
regress-the-margin-then-convert - which is exactly the kind of structural difference that
makes agreement between two models informative rather than redundant.

## Honest limitations - read before trading on this

- **One holdout season, examined from many angles.** No automated pipeline step ever
  re-touched holdout data during feature selection, model selection, or threshold tuning -
  that discipline is real and tested. But over the course of building and discussing this
  project, the same 215 games were checked from many different angles (per-model
  overfitting, per-pair agreement, calibration) before arriving at this recommendation. That
  iterative process is a milder version of the exact test-set-reuse pattern this project
  was built to catch in the original notebook. Getting a second, truly untouched
  holdout season is the only real fix.
- **Inherited, not independently re-derived**: the R-layer's lag logic (prev_week/avg_all/
  avg3 lagged by one game; coaching lagged by one year) was reviewed, not reimplemented or
  independently re-verified from raw play-by-play. Talent and returning-production data is
  *not* lagged across seasons (legitimate if frozen at preseason, but never confirmed
  against the underlying data source).
- **A known, uninvestigated selection-bias risk**: the R layer's `na.omit()` after
  requiring 3 games played may disproportionately drop games involving FCS/Group-of-5
  opponents with incomplete stat coverage.
- **Scope**: the source data only covers weeks 4-12 of each season (bowls/championship
  games excluded upstream) - this pipeline has never scored or validated on the parts of
  the season it doesn't cover.
- **No new data sources**: line movement, injury reports, weather, and personnel/roster
  changes remain out of scope, per the original project's own diagnosis that this is the
  most likely lever if the current feature family's ceiling proves too low
  (`docs/assumptions_and_limitations.md`).

## How to run this in 2026

```bash
cd cfb_cover_model
source .venv/bin/activate

# Direct from the CFBD API - no R scripts to run first (added after this write-up; see
# README.md's "Direct API ingestion" section for what's validated about this path and
# what isn't yet):
python scripts/generate_weekly_predictions.py --live --season 2026 --week <N>

# Original path, reading a week file the R scripts already produced:
python scripts/generate_weekly_predictions.py --week <N>
# or, if the 2026 weekly-update naming convention differs from CFB_Pred_Week_<N>.csv:
python scripts/generate_weekly_predictions.py --file ../Data/<whatever-the-file-is-named>.csv
```

This refits both `logistic_regression` and `xgboost_regressor` on all eligible history
(everything except the excluded 2020 season - 2025 becomes ordinary training data once
2026 games are being scored) and writes
`outputs/predictions/week_<N>_dual_model_predictions.csv` with both models' probabilities,
their individual bet flags, and the `agreement_bet` column - **treat `agreement_bet == True`
as the recommended slate, not either model's flag alone.**

Recommended operating procedure:
1. Run this weekly once the week's `CFB_Pred_Week_<N>.csv` (or 2026 equivalent) exists.
2. Bet only the `agreement_bet` rows, at reduced stakes through at least the first month of
   the season - this is the first genuinely unexamined data this whole project has produced.
3. Track actual outcomes against these predictions explicitly (a simple running log of
   `game_id, agreement_bet, predicted probability, actual home_covered` is enough) - that
   log is the real holdout test this project has been missing, and should inform whether to
   scale up, adjust the threshold, or reconsider the whole approach.
4. If a full season of 2026 agreement-bet precision lands meaningfully below ~53%, treat
   that as a real signal the ceiling documented here doesn't hold going forward - not
   as bad luck to wait out.
