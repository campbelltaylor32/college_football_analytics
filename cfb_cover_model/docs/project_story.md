# Project story: what this build actually found

## Update: target met on true holdout, after a feature-engineering pass and a second bug fix

Everything below this section describes the original build. A follow-up pass added three
domain-informed engineered features grounded in `docs/feature_importance_findings.md`'s
recommendations - opponent-adjusted matchup features for down-conversion/EPA
(`src/cfb_cover_model/engineered_features.py::add_opponent_adjusted_features`), a trimmed
4-column returning-production family (`consolidate_returning_production`), and a
special-teams family collapsed into one net composite score
(`consolidate_special_teams`) - then re-ran the full pipeline.

**A second real bug was caught before trusting any new number**: the new engineered
columns name their temporal transform with a *trailing* suffix
(`matchup_adj_<stat>_prev_week`), but `feature_engineering.py::categorize_features` (which
drives transform-ablation filtering) only recognized raw columns' *leading* `prev_week_`
prefix. The `_prev_week`-suffixed engineered columns fell through to "non_temporal" and
were silently included in *every* transform-ablation candidate regardless of which
transform was being tested - inflating their apparent importance in a run before this was
caught. Fixed in `feature_engineering.py` (recognizing the trailing `_prev_week` suffix
too), covered by a regression test
(`tests/test_feature_engineering.py::test_engineered_prev_week_suffix_recognized_as_temporal`),
and the full pipeline was re-run again after the fix - the numbers below are from that
corrected run.

**Headline result, first time in this project: the target was met on true holdout.** The
walk-forward-selected production model (`logistic_regression`) scored **53.6% precision at
26.0% coverage** on the never-touched 2025 holdout - clearing both the 53% precision target
and the 20% coverage floor. It's not a single lucky model either:
`outputs/model_comparison/holdout_model_comparison.csv` shows 5 of 14 candidates clearing
or nearly clearing both bars simultaneously (`xgboost`: 57.5% precision at 34% coverage;
`elastic_net_regression`: 57.5% at 18.6%; `logistic_no_selection`: 58.5% at 19.1%;
`lightgbm`: 60.6% at 15.3%; `catboost`: 57.5% at 18.6%) - a much more internally-consistent
picture than any earlier run, where walk-forward and holdout rank routinely inverted.
Overfitting also looks less severe: `scripts/analyze_train_vs_holdout.py` now shows holdout
ROC-AUC above chance for `logistic_regression` (0.539) and `gradient_boosting` (0.542),
versus below-chance in the original run.

**Honest attribution - the engineered features get partial, not full, credit.** The
winning feature-selection configuration also changed independently, from
`prev_week_only`/`raw_dual` to `all_three` (every temporal transform) /`differential`
(`home_X - away_X`) - a combination that was always in the pipeline's search space, not
something new added this pass. Checking exactly which features survived into the final
production model (`outputs/feature_analysis/feature_stability_winning_all_three.csv`,
`outputs/model_comparison/feature_importance_holdout.csv`):

- **Recommendation 2 (trim returning_production) is clearly validated.** Of the 4 kept
  sub-metrics, `rushing_usage` and `receiving_usage` are selected in **6 of 6** walk-forward
  folds and `percent_rushing_ppa` in 5 of 6 - the highest, densest consistency of any
  category in the whole analysis (mean selection frequency 4.25 of 6, on just 4 candidate
  columns).
- **Recommendation 1 (opponent-adjusted matchup features) is only weakly supported in this
  corrected run.** Only 1 of the 60 features in the final production model is a
  `matchup_adj_*` column, and only 5 of 54 candidate matchup-adjusted columns were ever
  selected across the 6 folds. (A run made *before* the trailing-suffix bug was fixed had
  shown these features far more prominently - that was substantially the bug artifact
  described above, not real signal, and should be disregarded.)
- **Recommendation 3 (special-teams composite) is neutral, not clearly activated.** 0 of 60
  final features are `special_teams_net_score`; the whole `special_teams` category dropped
  to the *least*-consistent of all 14 categories (0.4% of consistency-score mass). This is a
  reasonable outcome given the original recommendation was to de-prioritize this family, not
  necessarily to make it a star performer - consolidating 14 raw columns into 1 composite
  per side removed the earlier fold-stability/holdout-importance disagreement (a likely
  overfitting signature) without needing to drop the category outright.

**Caveat, unchanged from the rest of this document**: this is still one 215-game holdout
season. An encouraging, reproducible result - not proof the model is production-ready.
Treat it as a positive signal worth validating against a genuinely new holdout season
before relying on it, per the same statistical-power caution raised throughout this
document.

---

This document reports one full run of this pipeline against the real
`../Data/CFB_Gambling_Predictors_Final_PBP.csv`, including a real bug caught and fixed
along the way, compared honestly against the production notebook's documented baseline.
**The target (precision >= ~53% at >= 20% coverage, on a true holdout the selection
process never touched) was not met by the model the pipeline's own walk-forward metric
selected.** As with the sibling `../cfb_spread_model/` project, the most useful output
here is not a better number - it's a more trustworthy one, a caught-and-fixed leakage bug
worth remembering, and a second independent confirmation that this feature family's honest
ceiling is close to the betting break-even line, not comfortably above it.

## A real bug, caught by the numbers looking too good

The first full run of `evaluate_models.py` reported XGBoost at **98.7% holdout precision**
(and LightGBM 98.5%, Random Forest 95.7%). That is not a plausible number for spread-cover
prediction and was treated as a bug report, not a result. Root cause:
`modeling/splits.py`'s `get_holdout_split` called `.reset_index(drop=True)` on both
`train_pool` and `holdout`, so both index spaces started at 0 again. `train_models.py` and
`select_features.py` happened to be unaffected (they build every working frame *from*
`train_pool` and re-stamp its index onto the result, so everything stays internally
consistent). `evaluate_models.py` builds its variant from the *original*, unsplit frame
(`build_full_variant`) - and indexing that frame with `holdout.index` (labels `[0..214]`)
silently returned the **first 215 rows of the dataset by position** (mostly 2015 games),
which were *also* included in `train_idx` (labels `[0..2044]`, a strict superset). Tree
models trained on those rows and then "evaluated" on them again produced near-perfect
scores because they were partly scoring memorized rows, not held-out ones - and linear
models showed a smaller version of the same inflation for the same reason.

Fixed by removing the `reset_index(drop=True)` calls so `train_pool`/`holdout` keep their
original row labels from the saved modeling frame - disjoint by construction, and safe to
index into any frame built from the same original source. A regression test
(`tests/test_splits.py::test_train_pool_and_holdout_indices_are_disjoint_and_index_the_original_frame_correctly`)
now asserts this directly, and `get_holdout_split` itself asserts index disjointness at
runtime. Every number below is from the corrected pipeline, re-run in full after the fix.

## The baseline being compared against

`../Python Scripts/CFB_Gambling_Model.ipynb`, current live artifact: precision 0.569,
recall 0.298, 25% coverage, evaluated on a single fixed split (train 2015-2022, test
2023-2024), with its 52 features chosen by test-set ROC-AUC and its 0.60 threshold chosen
by eyeballing a manual sweep table on that same test set. The sibling `cfb_spread_model`
project already showed this number is inflated by test-set reuse during that search - an
honest refit of the same 52 features under walk-forward validation lands closer to 0.53-0.54.

## Stage 2: which temporal transform carries signal?

Contrary to the intuition that single-game (`prev_week_*`) stats are the noisiest of the
three temporal transforms (the working hypothesis in the sibling project, which excluded
them outright), the walk-forward-only probe-model ablation in this project found
`prev_week_only` / `raw_dual` scored highest (pooled precision 0.550 at 29% coverage),
narrowly ahead of `prev_week_only` / `differential` (0.550 at 45% coverage) and clearly
ahead of `avg3_only` (0.539) and `all_three` / `differential` (0.542, the closest
approximation to today's notebook's "use every transform" approach). See
`outputs/feature_analysis/transform_ablation_results.json` for the full 10-combination
grid. **This should be read as "which transform survived a probe-model comparison on this
specific walk-forward split," not a settled fact about the underlying signal** - the
margins between the top several candidates are small (0.539-0.550), well within the range
that could reorder under a different probe model or fold structure.

## Stage 3-5: does feature reduction help?

On the winning `prev_week_only`/`raw_dual` combination (350 candidate features),
correlation-pruning + embedded selection ("reduced", 52 features) narrowly outperformed
PCA collapse (247 components, 0.536) and no reduction at all (350 features, 0.529):

| feature_set_mode | mean n_features | pooled walk-forward precision | coverage |
|---|---|---|---|
| reduced | 52 | **0.546** | 32% |
| pca_reduced | 248 | 0.536 | 34% |
| deterministic_pruned_only (no reduction) | 350 | 0.529 | 31% |

Unlike the sibling project (where "use everything, no selection" beat every selection
attempt, twice), stage 3+4 selection here does modestly outperform the no-reduction
anchor on the probe-model walk-forward comparison - a genuinely different finding, though
a 1.7-point gap on a single probe model is not strong enough evidence to call this
selection process clearly better; see the full bakeoff below for what happens once real
model diversity (not just one probe) is in play.

## The full Track A/B/C bakeoff: walk-forward vs. true 2025 holdout

`logistic_regression` (Track A, elastic-net on the 52 "reduced" features) had the highest
pooled walk-forward precision (0.552 at 25% coverage) and was the model the pipeline's own
selection process would deploy. Scored on the true 2025 holdout:

| | walk-forward (pooled, 6 folds) | 2025 holdout |
|---|---|---|
| Precision | **0.552** | **0.467** |
| Coverage | 25% | 21% |
| Rank among 14 candidates | **1st** | **11th** |

**The walk-forward-selected model's holdout precision (0.467) is below the 49.3% base
rate and below the 53% target.** It is not the best-performing model on true holdout data
either - full comparison, every candidate refit once on the entire train_pool and scored
on the same never-touched holdout (`outputs/model_comparison/holdout_model_comparison.csv`):

| model | walk-forward rank | walk-forward precision | **holdout rank** | **holdout precision** | holdout n flagged |
|---|---|---|---|---|---|
| gradient_boosting | 11 | 0.518 | **1** | **0.548** | 31 (14%) |
| xgboost | 4 | 0.539 | 2 | 0.543 | 35 (16%) |
| logistic_no_selection | 8 | 0.529 | 3 | 0.533 | 30 (14%) |
| random_forest | 10 | 0.520 | 3 (tie) | 0.533 | 45 (21%) |
| xgboost_regressor | 5 | 0.536 | 5 | 0.529 | 34 (16%) |
| lightgbm | 7 | 0.535 | 6 | 0.515 | 33 (15%) |
| **logistic_regression (walk-forward pick)** | **1** | **0.552** | **11** | **0.467** | 45 (21%) |
| catboost | 7 (tie) | 0.509 | 7 (tie) | 0.500 | 92 (43%) |
| quantile_regression | 2 | 0.551 | 7 (tie) | 0.500 | 30 (14%) |
| logistic_regression_pca | 6 | 0.536 | 7 (tie) | 0.500 | 36 (17%) |
| elastic_net_regression | 9 | 0.525 | 7 (tie) | 0.500 | 40 (19%) |
| always_favorite (baseline) | 12 | 0.509 | 12 | 0.456 | 79 (37%) |
| stacking_ensemble | 3 | 0.539 | 13 | 0.450 | 60 (28%) |

**No model cleared both the 53% precision target and the 20% coverage floor on true
holdout.** `gradient_boosting` came closest on precision (0.548, just above target) but at
14% coverage - below the floor, and on only 31 flagged games, a small enough sample that
this should be read as noisy, not a reliable production candidate. This is the same
pattern the sibling project found with `logistic_no_selection`'s 24-flagged-game "win":
a good-looking number on a small slate is weak evidence.

**Walk-forward rank did not predict holdout rank.** `gradient_boosting` went from 11th to
1st; `logistic_regression` (the actual walk-forward winner) went from 1st to 11th;
`stacking_ensemble` went from 3rd to last. This is now two independent projects
(this one and `cfb_spread_model`) finding the same thing on the same underlying data:
walk-forward validation, done honestly, is still not a strong guarantee of true
out-of-sample rank on a single future season - the noise floor in a ~200-250-game season
is large relative to the size of the effects being measured here.

## Track B (regression-to-probability) and Track C (stacking): did the new angles help?

**Track B did not clearly beat Track A.** `xgboost_regressor` (0.529 holdout precision,
rank 5) and `quantile_regression` (0.500, tied rank 7) landed in the middle of the pack -
neither better nor worse in any way that looks like signal rather than noise given the
sample sizes involved. `elastic_net_regression` (0.500, tied rank 7) was similarly
unremarkable.

**Track C (stacking) was the worst-performing non-baseline model on holdout** (0.450,
rank 13 of 13 non-baseline candidates), despite ranking 3rd on walk-forward. The
regression-to-probability + classification ensemble idea - the one genuinely new angle
this project set out to test that the sibling project hadn't - did not pay off here. A
plausible explanation, not confirmed: the meta-learner's inner-OOF training set
(`modeling/stacking.py`) is necessarily smaller than any single base model's full training
set (it only uses rows where every base model had a genuine out-of-fold prediction
available), which combined with an already-weak underlying signal may leave too little
data for the meta-learner to find a real combination rule rather than overfitting to
walk-forward noise.

## Calibration: is the model's probability a trustworthy confidence signal?

No. Rank monotonicity (Spearman correlation between predicted-probability decile and
actual cover rate) on the walk-forward-selected model's raw scores:

| | walk-forward OOF (pooled, 6 folds, 1,437 games) | 2025 holdout (215 games) |
|---|---|---|
| Rank monotonicity | 0.685 | **-0.080** |
| Brier score | 0.263 | 0.266 |
| ROC-AUC | 0.536 | 0.480 |

The walk-forward view looks reasonably monotonic (0.685) - but that's exactly the sample
the threshold was chosen on, so some optimism there is expected. On the true holdout, rank
monotonicity is **slightly negative**: higher predicted probability does not reliably mean
a higher actual cover rate. Isotonic calibration (fit on pooled walk-forward OOF, applied
without refitting to holdout) improved holdout monotonicity to 0.257 - a real but modest
improvement, nowhere near making the raw score a reliable confidence gradient. Platt
scaling made no difference to monotonicity (it's a monotonic transform of the same raw
score by construction) and only marginally improved Brier score. **This matches the
sibling project's independent finding on different models and a different feature
representation**: whatever signal exists in this feature family does not currently produce
probabilities that rank games reliably beyond the single threshold they were tuned at.

## Train vs. walk-forward vs. holdout: overfitting, not just weak signal

`scripts/analyze_train_vs_holdout.py` refits three models on the entire train_pool and
scores them on their own training rows (in-sample), pooled walk-forward OOF, and the true
holdout, to distinguish two failure modes that look similar from OOF precision alone:
overfitting (fits training data well, that fit doesn't transfer) vs. insufficient signal
(doesn't even fit its own training data well).

| model | split | accuracy@0.5 | ROC-AUC | log loss |
|---|---|---|---|---|
| logistic_regression (walk-forward pick) | train (in-sample) | **0.629** | 0.675 | 0.646 |
| | walk-forward OOF | 0.532 | 0.536 | 0.726 |
| | holdout | 0.488 | 0.480 | 0.726 |
| gradient_boosting (best holdout precision) | train (in-sample) | **0.853** | 0.936 | 0.532 |
| | walk-forward OOF | 0.505 | 0.509 | 0.731 |
| | holdout | 0.512 | 0.502 | 0.712 |
| logistic_no_selection (no reduction, 350 features) | train (in-sample) | **0.676** | 0.739 | 0.598 |
| | walk-forward OOF | 0.522 | 0.523 | 1.020 |
| | holdout | 0.502 | 0.492 | 0.793 |

**This project's models show real overfitting, not just a low signal ceiling** - a
different diagnosis than the sibling `cfb_spread_model` project reached on its own models
(which found training log-loss barely beat the trivial base-rate baseline, i.e. not much
to overfit to in the first place). Here, `gradient_boosting` fits its own training data to
85.3% accuracy and 0.936 ROC-AUC - essentially memorizing individual games - then
collapses to a coin flip (50.5%/50.9 AUC walk-forward, 51.2%/0.502 AUC holdout).
`logistic_regression`, the most regularized of the three, has the smallest gap but still
loses 14 accuracy points and 0.20 ROC-AUC points from train to holdout. Every model's
out-of-sample accuracy - walk-forward or holdout, regardless of model family - lands
within a couple points of the 48-49% base rate. Two things are simultaneously true: there
is enough exploitable pattern in this feature set for a flexible model to fit training data
far above chance, and essentially none of that pattern generalizes to a new season. That
combination points toward regularization/model-capacity control and larger effective
training data (not just more features or fancier architectures) as the more relevant levers
if this feature family is revisited.

## Bottom line

> **Superseded**: the bullets below describe the *original* build, before the feature-
> engineering pass documented in the "Update" section at the top of this document. The
> target has since been met on true holdout - see that section for the current numbers.
> Kept here because everything below remains true of that earlier run and the reasoning
> (walk-forward/holdout instability, calibration, overfitting diagnosis) still informs how
> to read the newer results with appropriate caution.

- **A real leakage bug was caught and fixed before any number here was trusted** - see
  above. Worth remembering as a concrete example of why "check holdout numbers that look
  too good against a sanity floor" is a real, not theoretical, part of building this kind
  of pipeline.
- **The 53% precision / 20% coverage target was not met** on true holdout by the model
  this project's own walk-forward selection process would deploy (0.467, below the 49.3%
  base rate). No candidate among 14 (7 Track A, 3 Track B, 1 Track C, 3 baselines) cleared
  both the precision target and the coverage floor simultaneously on holdout.
- **Neither new angle this project set out to test - a controlled temporal-transform
  ablation, or a regression-to-probability/classification stacked ensemble - produced a
  result that beats what plain, honestly-validated classification already achieves.**
  Track B was unremarkable; Track C (stacking) was the worst non-baseline performer.
- **This is now two independently-built projects, on the same underlying ~1,000-column
  feature family, both finding a walk-forward-honest ceiling in the low-to-mid 50s on
  precision, with unstable walk-forward-to-holdout rank and unreliable probability
  calibration.** That convergence is itself information: it raises confidence that the
  ceiling is a property of the feature set (or of how much true signal exists in
  team-week aggregate box-score/EPA stats for ATS prediction at this sample size), not an
  artifact of either project's specific modeling choices.
- **Consistent with both projects' own diagnosis: the next lever most likely to move this
  needle is new information the current feature family doesn't carry at all** (line
  movement, injury reports, weather, personnel/roster changes) rather than further
  modeling sophistication on the existing ~350-970 candidate columns - this was
  deliberately out of scope for this phase (see `assumptions_and_limitations.md`) and is
  the natural next phase if that lever is worth pursuing.
- **Does not replace `../Python Scripts/CFB_Gambling_Model.ipynb` / `Week_Predictions.ipynb`**
  - it did not beat the honestly-reproduced baseline (~0.53-0.54), let alone the notebook's
  own optimistic self-reported number (0.569). *(Update: the post-engineering result above,
  53.6% on true holdout, roughly ties the honest ~0.536 baseline rather than clearly beating
  it - still not a strong enough, or long-enough-validated, case to replace the production
  pipeline on the strength of one holdout season.)*

## What actually shipped

`scripts/generate_week_predictions.py` is functional and was verified end-to-end against
`../Data/CFB_Pred_Week_14.csv` (66 games, 7 flagged at the `gradient_boosting` model's
walk-forward-chosen 0.58 threshold) - but given the holdout results above, **treat any
week's predictions from this pipeline as unproven, not production-grade**, until a future
phase either finds a configuration that clears the target on a fresh holdout season or
brings in new data sources per the "next steps" above.
