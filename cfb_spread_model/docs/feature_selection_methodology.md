# Feature selection methodology

The current notebook (`../Python Scripts/CFB_Gambling_Model.ipynb`) selects its 52 production
features by sweeping permutation-importance thresholds and picking whichever count maximizes
**test-set ROC-AUC** (0.575) — it never scores that sweep on precision, the metric that
actually matters for a model used to place bets. This project's two-stage selection process
fixes that: precision (specifically, precision at a coverage floor) is the objective at every
selection step, not a metric computed after the fact.

## Upstream exclusion: `prev_week_*` (single-game predictors)

Before Stage 1 even runs, `config/data.yaml`'s `excluded_column_patterns: [prev_week_]` drops
all 336 single-most-recent-game columns from `get_feature_columns()`'s output — a hard
exclusion, not a de-prioritization within selection. This was added after this project's first
production model (Run 1 in `docs/project_story.md`) turned out to be 45% `prev_week_*`
features and underperformed the 2025 season's base rate. See `docs/project_story.md` for the
full before/after result, which was mixed (walk-forward precision got slightly worse, 2025
holdout precision improved) rather than a clean validation of the hypothesis.

## Alternative representation: engineered differentials + trend (`feature_engineering.py`)

`config/data.yaml`'s `feature_representation` toggle (`raw_dual` default, or `differential`)
switches what `data.build_feature_matrix` hands to the rest of the pipeline, upstream of Stage
1/2 selection entirely:

- `raw_dual` (default): unchanged — separate `home_X`/`away_X` columns, as the CSV provides them.
- `differential`: replaces the `avg_all`/`avg3`/non-temporal `home_X`/`away_X` pairs with
  `diff_<transform>_<base> = home_X - away_X` (the matchup differential — arguably the more
  directly predictive quantity for a point-spread problem than two separate levels) and
  `trend_<side>_<base> = <side>_avg3 - <side>_avg_all` (recent form vs. season baseline). Both
  are pure arithmetic on already-lagged, already-leakage-verified columns
  (`docs/data_leakage_rules.md`) — no new data pull, no new leakage risk. Both deliberately
  exclude `prev_week_*`, for the same noise rationale as the exclusion above.

**Result (Run 3, `docs/project_story.md`): not a validated improvement.** The differential
representation didn't just fail to beat the benchmarks — it made walk-forward-to-holdout rank
correlation go strongly *negative* (-0.667, vs. +0.405 under `raw_dual`), meaning the standard
model-selection process became actively misleading under this representation, picking the model
that performed worst on the true holdout. One individual model's holdout number looked good in
isolation, but promoting it on that basis alone would repeat the exact test-set-peeking mistake
this project has otherwise been careful to avoid. See `docs/project_story.md`'s "Run 3" section
for the full result and why it isn't (yet) a recommendation to switch representations.

## Stage 1 — structural/correlation pruning (`feature_selection/correlation_pruning.py`)

Cheap, done before any model fitting, and re-run independently inside every walk-forward fold's
training data (never fit once globally — see `docs/data_leakage_rules.md`).

1. **Temporal-transform collapse.** The source CSV's 168 base metrics each appear three times
   per side (`prev_week_X`, `X_avg_all`, `X_avg3`) — 336 metric×side groups. Within each
   triplet, columns whose pairwise correlation with the strongest-univariate-association member
   exceeds `config/features.yaml`'s `temporal_collapse_corr_threshold` (0.85) are dropped.
   "Strongest univariate association" is measured as `|point-biserial correlation|` with
   `home_covered`, computed on the fold's training rows only.
2. **General redundancy pass.** On the survivors, a full pairwise correlation matrix is
   computed and one member of every pair exceeding `general_corr_threshold` (0.90) is dropped
   (again, keeping the higher-association member). This catches offense/defense-mirror pairs
   (41 verified `_allowed` pairs) and the 16 unpaired algebraic-ratio columns
   (`point_differential`, `turnover_margin`, `yards_per_pass`, etc.) against their raw
   components.

**Verified result on the first walk-forward fold** (validation_season=2019, training seasons
2015-2018): 1,048 candidate feature columns → 999 after temporal collapse (49 dropped) → 774
after the general redundancy pass (225 dropped). The temporal-transform collapse was
empirically less aggressive than the CSV's construction might suggest — `prev_week_*` (single
most-recent game, noisy) turns out not to be as tightly correlated with the smoothed
`*_avg_all`/`*_avg3` versions of the same stat as a naive "they're the same underlying
quantity" assumption would predict, at the chosen 0.85 threshold. This is documented here
rather than assumed, since it directly affects how much load Stage 2 has to carry.

Every drop is logged to `outputs/feature_analysis/correlation_pruning_report_fold_<season>.csv`
with a `reason` column (`temporal_collapse` or `general_redundancy`) and the `kept_alternative`
it was compared against.

## Stage 2 — model-based selection, scored on precision (`feature_selection/selection.py`)

Two independent methods are run and reported **side by side**, never silently merged, so a
reader can sanity-check they roughly agree — the current notebook only ever ran one method
(permutation importance) to completion, and its abandoned `VarianceThreshold`/`SelectKBest`
path was never compared against it.

Both methods use the same custom scorer,
`feature_selection/precision_scoring.precision_at_coverage_floor_scorer(min_coverage)`
(`config/modeling.yaml`'s `precision_objective.min_coverage_floor`, default 0.10) — this finds
the highest decision threshold that still flags at least `min_coverage` fraction of games, and
returns precision at that threshold. This is what prevents selection from degenerating to "1
ultra-confident pick," which would trivially maximize plain precision but be useless for
actually placing bets.

1. **Permutation-importance sweep.** An XGBoost model (matching the current notebook's chosen
   family) is fit on the Stage-1-pruned columns using the fold's *last inner season-ordered CV
   split* (see `modeling/tuning.build_inner_season_cv`), then `sklearn.inspection.permutation_importance`
   ranks features by mean importance under the coverage-floor scorer (`n_repeats=50`, matching
   the current notebook's setting for continuity). Candidate feature counts
   `[25, 40, 52, 75, 100, 150, 200]` (52 included specifically for an apples-to-apples
   comparison against the live `../Model Information/selected_features_best_model_20250915.json`)
   are each refit and scored on the same held-out inner-CV validation rows; the count
   maximizing precision-at-coverage-floor wins.
2. **RFECV**, independently, with the same scorer and the fold's full set of inner CV splits —
   a second selection method entirely, not a variant of the first.

Both methods' outputs are saved per fold
(`outputs/feature_analysis/selected_features_fold_<season>.json`,
`outputs/feature_analysis/permutation_sweep_fold_<season>.csv`) so their agreement (or
disagreement) is visible, not assumed.

**Observed result on this build (Run 1, before the `prev_week_*` exclusion above):** the two
methods did *not* agree well. RFECV's selected feature count varied wildly across the six folds
(10, 470, 641, 326, 82, 330), while the permutation-importance sweep stayed in a much tighter
range (25, 25, 75, 200, 75, 40). This suggests RFECV's greedy elimination is sensitive to
fold-specific noise in this feature space — the permutation-importance sweep, with its coarser,
explicitly precision-scored candidate-count grid, is the more trustworthy of the two here, and
is what the production feature set uses. Run 2 (after the exclusion) showed the same pattern.
See `docs/project_story.md` for the full result writeup, including both runs.

## What "production feature set" means

The feature set actually shipped in the production model artifact
(`outputs/models/selected_features_<date>.json`) is whatever Stage 2 selects on the **final
holdout fold's** training seasons (everything before 2025, excluding 2020) — not a fixed set
carried over from the walk-forward folds. Fold-to-fold stability of the selected features across
the five walk-forward folds is reported alongside the final numbers in `docs/project_story.md`,
since some fold-to-fold variation is expected (different training windows genuinely can surface
different top predictors) and is worth being able to see rather than hide.

## Baselines new to this project

`config/modeling.yaml`'s `models.baselines` (`always_favorite`, `majority_class`,
`logistic_no_selection`) don't exist in the current notebook at all:

- `always_favorite` predicts `home_covered=1` whenever the home team is favored — the "just bet
  the favorite" sanity floor.
- `majority_class` predicts the training-set class prior for every row — the trivial floor.
- `logistic_no_selection` fits plain L2 logistic regression on **all** raw feature columns
  (1,044 in Run 1, 712 in Run 2 after the `prev_week_*` exclusion), bypassing Stage 1/2
  entirely — this specifically stress-tests whether any precision gain this project reports
  actually comes from feature selection, or just from model family. **Observed result: it beat
  every selected-feature candidate model's walk-forward mean precision in both runs** — see
  `docs/project_story.md`'s "Next steps" for what that implies about the selection objective.

## Feature importance (`scripts/explain_model.py`)

Separate from selection, `scripts/explain_model.py` computes importance for the **already-fit
production model** (no retraining) on the true 2025 holdout: gain-based importance where
available (`feature_importances_`, tree ensembles only — falls back gracefully with a logged
warning for model families like logistic regression that only expose `coef_`), plus permutation
importance scored with the same `precision_at_coverage_floor_scorer` used everywhere else in
this project. Both are joined with each feature's `(side, temporal_transform, base_metric)` via
`data.parse_side_and_metric`, producing both a per-feature ranking
(`outputs/model_comparison/feature_importance_<date>.csv`) and a category rollup
(`outputs/model_comparison/feature_importance_by_category_<date>.csv`) — the latter is what
answers "how much do `prev_week_*`/`avg3`/`avg_all` predictors actually matter" for whatever
model is currently in production, not just "how many of each are present." See
`docs/project_story.md`'s "Feature importance" section for this build's actual numbers.
