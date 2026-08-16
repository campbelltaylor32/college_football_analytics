# Feature importance deep-dive: what's actually consistent, and what to try next

## Update: the three recommendations below were implemented and tested

`src/cfb_cover_model/engineered_features.py` implements opponent-adjusted matchup features,
the returning-production trim, and the special-teams composite score proposed at the end of
this document. The full pipeline was re-run (twice - a second bug was caught along the way,
see `docs/project_story.md`'s "Update" section) and **the project's precision target was met
on true holdout for the first time**. Results, per recommendation, honestly reported:

- **Returning-production trim: clearly validated.** The 4 kept sub-metrics are the most
  fold-consistent feature family found anywhere in this project (2 of 4 selected 6/6 walk-
  forward folds).
- **Opponent-adjusted matchup features: only weakly supported**, once a second bug (these
  columns' transform-suffix naming wasn't recognized by transform-ablation filtering,
  inflating their apparent importance in an intermediate run) was fixed. Present in the
  final model, but not a dominant driver.
- **Special-teams composite: neutral**, not a strong positive signal but no longer showing
  the fold-stability/holdout-importance disagreement (a likely overfitting signature) that
  motivated de-prioritizing this family in the first place.

Full details, exact numbers, and the honest attribution of what actually drove the
improved result (the differential/all-transforms configuration the ablation newly
selected, not solely the three recommendations below) are in `docs/project_story.md`'s
"Update" section. The analysis below is preserved as-is - it's what motivated the changes,
and remains accurate for the *pre-engineering* feature set it was computed on.

---

Two complementary analyses, run against the walk-forward/holdout structure already built
for this project:

1. **Fold-stability** (`scripts/analyze_feature_stability.py`): refits correlation
   pruning + embedded elastic-net selection on each of the 6 walk-forward folds'
   training rows, for two configs - the winning `prev_week_only`/`raw_dual` (350
   candidate columns) and the best `avg_all_only`/`differential` (177 columns, the
   closest runner-up in the transform ablation, 0.548 vs. 0.550 pooled precision). A
   feature selected in most/all folds, under both transform choices, is a stronger
   "real predictor" candidate than one that shows up once.
2. **Single-shot holdout permutation importance** (`scripts/explain_model.py`): refits
   `logistic_regression` once on the entire train_pool, then measures how much shuffling
   each feature degrades precision-at-coverage-floor on the true, never-touched 2025
   holdout - mirrors `../cfb_spread_model/scripts/explain_model.py`'s approach.

Both are rolled up by a domain taxonomy (`src/cfb_cover_model/feature_categories.py`) -
EPA/success-rate, box-score scoring/yardage, turnover/penalty, possession, down-
conversion, explosiveness, pass-rush/pressure, talent, coaching, returning production,
special teams, play-volume-mix, drive-efficiency, context/market - rather than by
temporal transform, since the winning config is ~100% `prev_week` already.

## Part 1: most consistent predictors (fold-stability lens)

### Top individual features, winning `prev_week_only`/`raw_dual` config

| feature | domain | selected in (of 6 folds) |
|---|---|---|
| `home_prev_week_third_down_conversion` | down_conversion | **6/6** |
| `home_rushing_usage` | returning_production | **6/6** |
| `away_rushing_usage` | returning_production | **6/6** |
| `home_prev_week_qb_hurries` | pass_rush_pressure | **6/6** |
| `home_prev_week_punt_return_yards` | special_teams | **6/6** |

Full ranking: `outputs/feature_analysis/feature_stability_winning_prev_week_only.csv`.

### Category rollup (selection-frequency-weighted "consistency score"), both configs

| domain | winning config share | avg_all_only config share |
|---|---|---|
| returning_production | **16.3%** | 8.5% |
| down_conversion | 14.4% | **17.8%** |
| turnover_penalty | 13.1% | 12.0% |
| special_teams | 12.1% | 7.1% |
| box_score_scoring_yardage | 9.6% | 14.5% |
| pass_rush_pressure | 9.4% | 9.8% |
| explosiveness | 6.0% | 6.2% |
| **epa_success_rate** | **4.2%** | **4.9%** |
| talent | 3.5% | 1.1% |
| context_market (spread, week, ...) | 3.1% | 1.9% |

**Notable: EPA/success-rate metrics rank near the bottom in both configs**, despite being
the conventional go-to team-efficiency signal in football analytics. `returning_production`
and `down_conversion` are consistently the top two categories regardless of which temporal
transform won.

### Cross-transform robustness

66 of 177 base stats (37%) are selected in at least one fold under *both* the
`prev_week_only` and `avg_all_only` configs (`outputs/feature_analysis/feature_stability_cross_transform_comparison.csv`).
`rushing_usage` is selected 6/6 under both - the single most robust predictor found.
`third_down_conversion`, `percent_rushing_ppa`, `receiving_usage`, and
`Offense_Avg_3rd_Down_Distance` are also strong under both. Notably, `qb_hurries` (6/6
under `prev_week_only`) drops to 1/6 under `avg_all_only` - it looks like a genuinely
transform-specific signal (a team's single-game pass-rush pressure is informative; its
season-to-date average of the same stat is not), not a robust one.

**Caveat on the selection process itself**: the number of features actually selected per
fold swings from 9-12 up to the 60-feature cap depending on which training window the
inner elastic-net grid search landed on (e.g. the winning config's 2021 fold selected only
12 features vs. 60 in five other folds) - the *count* of selected features is itself
unstable, not just which ones are chosen. Worth keeping in mind before reading too much
into any single fold's list.

## Part 2: does fold-stability predict what mattered on the true holdout?

**Only 9 of 60 production-model features had positive holdout permutation importance at
all** (`outputs/model_comparison/feature_importance_holdout.csv`) - the other 51 made
holdout precision *no worse or slightly better* when shuffled into noise. This is
consistent with everything else already found about this model (holdout ROC-AUC 0.480,
holdout accuracy 48.8%, negative rank monotonicity - see `docs/project_story.md`): on this
specific 215-game season, essentially no individual feature is doing identifiable work.

**More strikingly: the fold-stability champions from Part 1 barely overlap with the
holdout-permutation leaders.** `rushing_usage` (6/6 in both configs) and
`third_down_conversion` (6/6 winning) don't appear in the holdout top 20 at all. The
closest matches are `away_percent_rushing_ppa` (5/6 fold-selected, holdout rank 10) and
`home_Winning_Percentage` (5/6 fold-selected, holdout rank 20) - both barely positive or
already negative. **This extends the project's existing finding that walk-forward rank
doesn't predict holdout rank for whole models (`docs/project_story.md`) down to the
individual-feature level.**

One genuine three-way agreement stands out: `home_prev_week_fourth_down_percentage_offense`
is the **#1 feature by holdout permutation importance**, and it belongs to
`down_conversion` - the category that was also #1 or #2 in *both* fold-stability configs.
That's the strongest evidence-backed signal in this entire analysis: consistent across 6
walk-forward folds, consistent across two transform choices, *and* the single most
important individual feature on true holdout data.

The opposite pattern is a warning sign: `special_teams` and
`box_score_scoring_yardage` rank in the top 5 categories by fold-stability in *both*
configs, but `special_teams` is the single **most negative** category in holdout
permutation importance (-0.236 total, the worst of all 13 categories) and
`box_score_scoring_yardage` is the second-most negative (-0.189). A category that looks
reliably selected across training windows but actively hurts (or does nothing for) true
out-of-sample precision is a plausible sign of fold-specific noise-fitting, not real
signal - the same generalization gap already diagnosed at the whole-model level in
`scripts/analyze_train_vs_holdout.py`, showing up here at the category level too.

## Feature-engineering opportunities

Ranked by how much empirical support each has, most to least:

1. **Opponent-adjust the down-conversion signal.** `down_conversion` is the single
   best-supported category in this analysis - top-2 in both fold-stability configs, and
   home to the #1 holdout-permutation feature. It's currently expressed as raw, un-adjusted
   single-game/season rates (`third_down_conversion`, `fourth_down_percentage_offense`,
   `Offense_Avg_3rd_Down_Distance`). None of it accounts for opponent quality - a 50%
   third-down rate against a good defense and against a bad one are treated identically.
   An opponent-adjusted version (e.g. team's rate minus the league-average rate allowed by
   that specific opponent, at the time of the game) is the most promising concrete next
   feature to build, precisely because the un-adjusted version already carries real signal
   - adjustment should sharpen it, not manufacture signal from nothing.

2. **Consolidate `returning_production`'s 12-column family around what's actually
   working.** This category is consistently top-2 by fold-stability, but only a few of its
   12 per-side sub-metrics (`rushing_usage`, `receiving_usage`, `percent_rushing_ppa`,
   `total_rushing_ppa`) do the driving - the rushing-specific ones, not the passing/
   receiving/overall ones. A narrower, rushing-usage-focused index (rather than feeding the
   model all 12 highly-correlated variants and relying on correlation-pruning to sort it
   out) could reduce redundancy at the source rather than after the fact.

3. **Investigate why raw EPA/success-rate metrics underperform expectations.** Despite
   being the standard team-efficiency signal in football analytics, `epa_success_rate`
   ranks near the bottom of both fold-stability configs and is unremarkable on holdout
   permutation. Rather than concluding EPA doesn't matter here, a more likely explanation
   is that single-game or simple season-average EPA is too noisy at this sample size to
   isolate - the same opponent-adjustment idea from (1), applied to EPA/success-rate
   columns specifically, is a natural next experiment given the raw form isn't working.

4. **Treat `special_teams` and `box_score_scoring_yardage` as a de-prioritization
   candidate, not a build target.** These are the two categories where fold-stability and
   holdout-permutation importance disagree most sharply (strong in-sample selection,
   among the most negative true-holdout permutation importance). Before building anything
   new in these families, a cheap, concrete follow-up experiment: re-run
   `scripts/select_features.py`'s reduction-strategy comparison with these two categories
   excluded entirely, and check whether walk-forward-to-holdout rank correlation improves -
   directly testing whether they're a source of the overfitting already diagnosed in
   `docs/project_story.md`, rather than assuming it from this analysis alone.

5. **Bigger-picture caveat.** The disagreement between fold-stability and single-holdout
   importance is itself the most important finding here, more than any specific feature.
   With only one true holdout season (~215 games), any feature-level effect this small is
   operating at the edge of what a single season can detect - the same underlying
   statistical-power problem noted for whole-model comparisons in `docs/project_story.md`.
   Feature engineering aimed at the categories above is worth trying, but the *evaluation*
   of whether it helped will face the same low-power holdout problem discussed there;
   getting a second true holdout season (waiting for 2026 data, or reconsidering data
   sourced from other levels/eras of the sport) may matter as much as which features go in.
