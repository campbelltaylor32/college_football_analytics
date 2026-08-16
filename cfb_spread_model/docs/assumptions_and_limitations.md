# Assumptions and limitations

## Inherited from the upstream R pipeline (not re-decided here)

- **Spread pushes are labeled `home_covered=0`, not a separate class.** The three-condition
  `home_covered` rule (`../R Scripts/Full_CFB_Game_Outcome_Historical.R:112-127`) has no branch
  that fires on an exact push, so it falls through to 0 by construction. This project carries
  that simplification forward unchanged for label consistency with the historical CSV, rather
  than silently reinterpreting existing labels. A follow-up experiment worth trying later:
  filtering push rows out of *training* only (not evaluation) — flagged as a "next steps" item,
  out of scope for this build.
- **The source CSV only covers weeks 4-12 of each season.** The R pull caps at week 12
  (`Full_CFB_Game_Outcome_Historical.R`'s `weeks <- seq(1,12)`), and the post-lag `na.omit()`
  (`Merge_Predictors_CFB_Historical.R:70`) pushes the practical minimum to week 4. Bowls and
  conference championship games are never in this dataset. Any production use of this model
  should not be expected to generalize to those excluded game types.
- **2020 is excluded from training and validation** (COVID-shortened: 91 games vs. a
  183-268-game neighboring-season range) — same call the sibling `../cfb_win_total_model/`
  project makes, for the same reason.
- **Lookahead-bias prevention is inherited, not re-verified from scratch.** Coaching stats are
  lagged by year, in-season game stats by week — both implemented upstream in R and spot-checked
  by direct read during this project's build (see `docs/data_leakage_rules.md`). This project's
  leakage tests are a regression safety net against future upstream changes, not an independent
  audit of the original logic.

## Decisions made in this project

- **`min_coverage_floor` (0.10) is a judgment call**, not a value derived from any external
  requirement — it exists to keep feature/threshold selection from degenerating to a
  single-pick solution, set below the current model's realized ~25% coverage so the search
  isn't artificially constrained to match it. A different bettor's risk tolerance might justify
  a different floor; this is a `config/modeling.yaml` value, not a hardcoded constant, so it's
  meant to be revisited.
- **Correlation-pruning thresholds (0.85 temporal-collapse, 0.90 general) are judgment calls**,
  chosen to be defensible but not derived from a formal optimization — `docs/feature_selection_methodology.md`
  documents the actual observed pruning yield at these thresholds on the real data so a reader
  can judge whether they seem too aggressive or too loose.
- **Stage 2's default importance-signal model is XGBoost**, matching the current notebook's
  chosen family — this is a "cheapest reasonable choice" decision, not a claim that XGBoost is
  the best possible importance signal. Model comparison across families happens separately
  (`modeling/models.py`) and is not constrained by this choice.

## Known gaps

- This project does not implement betting-specific evaluation (expected value under actual
  sportsbook vig, bankroll/Kelly-criterion sizing, closing-line-value tracking) — it optimizes
  and reports precision/recall/coverage only. Translating a precision number into "is this
  profitable against real -110 lines" is a deliberately separate, un-scoped follow-up.
- `../Python Scripts/CFB_Gambling_Model.ipynb` and `Week_Predictions.ipynb` remain the
  production pipeline until this project's pipeline is validated to match or beat their
  documented baseline (see `docs/project_story.md`) — retiring them is an explicit "next steps"
  item, out of scope for this build.
- No hyperparameter search beyond the small grids in `config/modeling.yaml`'s
  `hyperparam_grids` — kept intentionally small given the training set size (on the order of
  1,000-2,000 rows), matching the sibling `../cfb_win_total_model/` project's rationale.
