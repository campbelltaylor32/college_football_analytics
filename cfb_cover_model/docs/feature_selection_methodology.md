# Feature selection methodology

Five stages, run in this order for the *ablation and reduction-strategy comparison* in
`scripts/select_features.py`, then stages 3-5 (the reduction step) are refit per fold
inside `scripts/train_models.py` and again once on the full `train_pool` in
`scripts/evaluate_models.py`. Stages 1-2 are decided once, from the walk-forward-only
comparison in `select_features.py`, and then baked into every downstream fold as a fixed
choice - re-running a full transform ablation inside every fold was judged not worth the
compute for a decision that shouldn't be fold-sensitive (which stat *transform* carries
signal is a property of the data-generating process, not of any particular training
window).

## Stage 1: deterministic redundancy removal

Not really "selection" - config-declared exact linear/ratio duplicates
(`data.yaml`'s `deterministic_redundant_base_stats`) are dropped before anything else runs.
See `data_dictionary.md`.

## Stage 2: temporal-transform ablation (`feature_selection/transform_ablation.py`)

Every base game stat exists as `prev_week_*` (last game), `*_avg_all` (season-to-date),
and `*_avg3` (trailing 3-game). Rather than assuming all three should be fed to a model
(today's notebook) or that a differential/trend representation is automatically better (an
untested assumption in prior work on this data), this stage runs a cheap probe model
(single elastic-net logistic regression, `TRANSFORM_ABLATION.PROBE_PARAMS`) through the
full walk-forward harness for every combination of:

- **temporal transform**: `avg_all_only`, `avg3_only`, `prev_week_only`, `avg_all_plus_trend`
  (`trend = avg3 - avg_all`, engineered fresh), `all_three` (today's notebook's approach,
  kept only as a comparison point)
- **home/away representation**: `raw_dual` (keep `home_X`/`away_X` separate) vs.
  `differential` (`diff_X = home_X - away_X`)

scored by pooled walk-forward precision-at-coverage-floor
(`modeling/evaluation.py::best_precision_at_coverage_floor`). The winning combination is
recorded in `outputs/feature_analysis/winning_feature_config.json` and used for every
subsequent stage. **The true holdout is never touched during this comparison** - see
`data_leakage_rules.md` rule 3.

## Stage 3: correlation-cluster pruning (`feature_selection/correlation_pruning.py`)

Fit fresh on a single fold's (or the full train_pool's, for the final production fit)
training rows only. Hierarchically clusters features by `1 - |Spearman correlation|`
(average linkage), cuts the dendrogram at `features.yaml`'s `correlation_threshold`
(default 0.90), and keeps one representative per cluster - the member most correlated with
that fold's own training target, not an arbitrary or alphabetical pick.

## Stage 4: embedded selection via elastic-net logistic regression (primary reducer)

Chosen over permutation-importance threshold sweeps or RFECV specifically because both
were found unstable across folds in prior work on this same data (RFECV's selected
feature count swung from 10 to 641 across six folds on a related project). An L1-leaning
elastic net zeros out redundant/noisy coefficients as part of one convex fit - a lower-
variance selection mechanism than iterative wrapper methods.

The `l1_ratio`/`C` grid is chosen via an **inner `TimeSeriesSplit`** (chronological, not
random) on the fold's own training rows, scored by ROC-AUC rather than precision-at-
coverage-floor - precision at a coverage floor is too noisy a target on the small inner-CV
slices this produces (few hundred rows split further), so ROC-AUC is used as a ranking-
ability proxy for *this inner step only*; the outer walk-forward folds (which have enough
rows) still use precision-at-coverage-floor as the project's real metric everywhere else.

**Runtime note**: the `l1_ratio_grid`/`C_grid`/`inner_cv_folds` in `features.yaml` are
deliberately small (9 combos x 2 inner splits = 18 fits per outer fold) to keep a full
pipeline run tractable on a laptop. Widening them is a reasonable next experiment if more
compute is available.

## Stage 5: PCA/factor collapse (alternative reducer, compared head-to-head)

`feature_selection/pca_reduction.py` collapses each semantically-related stat family
(columns matching `features.yaml`'s `stat_family_prefixes`, e.g. every `Offense_*`
EPA/success-rate column) into a handful of principal components retaining
`variance_retained` (default 90%) of variance, fit on training rows only. This is compared
against stages 3+4 combined - not chained after them - via the same probe-model walk-
forward comparison used for stage 2, in `scripts/select_features.py`'s
`compare_reduction_strategies`. The `logistic_regression_pca` candidate in
`modeling.yaml` also carries this reducer through the full Track A/B/C bakeoff, not just
the probe-model comparison.

## What the last run found

See `outputs/feature_analysis/transform_ablation_results.json`,
`reduction_strategy_comparison.json`, and `winning_feature_config.json` for the exact
numbers from the most recent run, and `docs/project_story.md` for the honest narrative
write-up (including whether the final chosen model actually beat the target).

## Which individual features drive the result

Stages 2-4 above answer "which transform / how much reduction" at an aggregate level -
they don't say which specific features matter, or whether the same ones show up
consistently across folds and across the (narrowly-decided) transform choice.
`scripts/analyze_feature_stability.py` and `scripts/explain_model.py` answer that,
rolled up by a domain taxonomy (`src/cfb_cover_model/feature_categories.py`) instead of by
temporal transform. See `docs/feature_importance_findings.md` for the synthesized findings
and feature-engineering recommendations they motivate.
