# Assumptions and limitations

## Fixed vs. carried over from the production notebook

**Fixed in this project:**
- Push handling: pushes are now excluded from training and evaluation (the production
  notebook labels an exact push `home_covered=0`, silently treating a no-money-changes-
  hands outcome as a loss). See `docs/data_dictionary.md`.
- `Offense_EPA_per_Run`/`Defense_EPA_per_Run`: excluded by name (confirmed mislabeled at
  the R layer - they hold pass-play EPA, not run-play EPA).
- Feature/threshold selection no longer looks at the same rows it's evaluated on (see
  `docs/data_leakage_rules.md`).
- Hyperparameter tuning uses time-aware inner splits (`TimeSeriesSplit`), not plain random
  K-fold.

**Carried over unchanged, out of scope this phase:**
- The source CSV only covers weeks 4-12 of each season (bowls/championship games are
  excluded upstream, in the R layer) - this project inherits that scope, not a full-season
  one.
- No new data sources. The sibling `../cfb_spread_model/` project's own diagnosis is that
  the likely lever for real improvement is information the current ~970-column feature
  family doesn't carry at all (line movement, injury reports, weather, personnel changes) -
  this phase deliberately stays a modeling/feature-engineering exercise on the existing R-
  pipeline output, per an explicit scoping decision, and defers new data sources to a later
  phase if this phase's ceiling turns out too low.
- No changes to the R feature-engineering scripts. The `Offense_EPA_per_Run` bug is worked
  around by exclusion here, not fixed at the source - fixing it there would affect the
  production notebook and every other project reading the same CSV, which is a
  cross-project change outside this project's scope.

## Known simplifications in this project's own pipeline

- **Deterministic-redundant column list is not claimed exhaustive.** `data.yaml`'s
  `deterministic_redundant_base_stats` was built from a manual R-script review, not a
  programmatic search for exact linear dependence. Stage 3 (correlation-cluster pruning)
  is the safety net for anything missed.
- **Embedded-selection hyperparameter grid is deliberately small** (`features.yaml`)
  for runtime feasibility on a laptop - see `docs/feature_selection_methodology.md`.
- **Stacking's base models share one feature set** rather than each inner-fold step
  re-running its own feature selection - see `docs/data_leakage_rules.md` rule 6.
- **Temporal-transform and home/away-representation choice is made once**, from a walk-
  forward-only probe-model comparison (`select_features.py`), then held fixed across every
  downstream fold and model - not re-derived per fold. This assumes which transform
  carries signal is a property of the underlying data-generating process rather than a
  fold-specific artifact; reasonable, but untested against the alternative.
- **`home_favored` is used only to build the `always_favorite` baseline**, not validated
  itself for correctness beyond the informal cross-check already done against
  `CFB_Gambling_Results.csv`'s independently-parsed `signed_spread` during development
  (see git history / development notes for the manual verification that produced the
  push-detection formula).

## What "success" means here, and what it doesn't

The target agreed for this phase is precision on flagged bets >= ~52-53% (clears the -110
vig) at a coverage floor of >= 20%, honestly measured: walk-forward-selected, then scored
on a true holdout the selection process never touched. Meeting that bar on this feature set
would be a genuine result. **Not meeting it is also a genuine, useful result** - it would
corroborate the sibling project's finding that this specific feature family has a low
signal ceiling, and would point toward new data sources rather than more modeling
sophistication as the next lever. See `docs/project_story.md` for which of these two
outcomes this run actually produced, reported honestly either way.

This project does not replace `../Python Scripts/CFB_Gambling_Model.ipynb` /
`Week_Predictions.ipynb` as the production pipeline unless its own holdout-validated
precision beats the documented, re-verified baseline (not the notebook's own optimistic
self-reported number - see `docs/project_story.md` for what that baseline actually is once
test-set reuse is corrected for).
