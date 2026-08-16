# Project story: what this build actually found

This document reports two end-to-end runs of this pipeline against the real
`../Data/CFB_Gambling_Predictors_Final_PBP.csv`, compared honestly against the current
notebook's documented baseline, and against each other. **Neither run is a win on paper
precision.** The most useful output of this build is not a better number — it's a more
trustworthy one, an honest test of a specific hypothesis (single-game predictors are hurting
more than they help), and a feature-importance readout that answers what the model is actually
using.

## The baseline being compared against

`../Python Scripts/CFB_Gambling_Model.ipynb`, current live artifact
(`../Model Information/selected_features_best_model_20250915.json`, 52 features):

| | |
|---|---|
| Precision | **0.569** |
| Recall | 0.298 |
| Coverage | 25% (123/492 games flagged) |
| Evaluation | **Single fixed split** — train 2015-2022, test 2023-2024 |
| Feature selection objective | ROC-AUC (0.575), not precision |
| Threshold | 0.60, chosen by eyeballing a manual sweep table |

## Why does the notebook look better? (verified empirically, `scripts/replicate_notebook_features.py`)

Every run below shows this project's precision below the notebook's 0.569. The original
hypothesis was "the notebook's number is an artifact of one lucky 2023-2024 split, not walk-forward
validated." That's directionally right but incomplete — the actual mechanism turned out to be
more specific, and this section verifies it rather than just asserting it.

**Test**: take the notebook's own exact 52 saved features
(`../Model Information/selected_features_best_model_20250915.json`), fit XGBoost on them with
this project's own leakage-safe fitting code (`modeling/fitting.py`, same
precision-at-coverage-floor-scored hyperparameter tuning used everywhere else in this project),
and score at the notebook's own threshold (0.60), two ways:

| evaluation | n | precision | recall | ROC-AUC |
|---|---|---|---|---|
| Notebook's own reported number | 492 | **0.569** | 0.298 | — |
| **This project's fit, reproducing the notebook's exact split** (train 2015-2022 incl. 2020, test 2023-2024) | 492 | **0.529** | 0.353 | 0.564 |
| This project's standard walk-forward (5 seasons, COVID excluded) | 1,207 | 0.536 | 0.366 | 0.556 |

**The single 2023-2024 split, by itself, is not the main story.** Refitting the *exact same
feature set* on the *exact same split* with this project's own (unbiased) hyperparameter tuning
gives 0.529 — nearly identical to the walk-forward estimate (0.536, gap of only -0.008), and
well below the notebook's reported 0.569 (gap of +0.040). If "one lucky split" were the whole
explanation, reproducing that split honestly should have landed close to 0.569, not close to the
walk-forward number.

**The more precise explanation: the notebook's own feature-count and threshold selection
process directly optimizes against the same 492-game test set it then reports precision on.**
The notebook sweeps candidate feature counts (scored by test-set ROC-AUC) and candidate
probability thresholds (scored by test-set precision in a manual table) and reports the
best-looking single result from that search — a textbook multiple-comparisons /
test-set-reuse pattern. It's a different, more specific mechanism than plain train/test split
luck: the *search itself* is what's biased, not just the one split it happened to run on. This
project's ~0.04-point remaining gap between the honest reproduction (0.529) and the notebook's
number (0.569) is a reasonable estimate of how much that search process alone is worth.

**Bottom line: no, the notebook's model is not actually better — 0.569 is an optimistic number
produced by evaluating many feature-count/threshold combinations against the same held-out set
the final metric is reported on.** This project's ~0.53 (both the honest single-split
reproduction and the walk-forward estimate agree closely) is the more trustworthy figure for
this exact feature set. See `docs/feature_selection_methodology.md` for how this project's own
Stage 2 selection avoids the same trap (inner-CV-scored, never touching the outer validation/
holdout rows it's later evaluated on).

### But the honest 0.536 still beats everything this project has built

Every configuration below is measured the identical way — walk-forward, pooled across the same
1,207 games (all predictions from the 5 walk-forward folds combined, then one precision
computed over the pool, so fold size doesn't distort the number):

| configuration | features | pooled walk-forward precision |
|---|---|---|
| **Notebook's 52 features, refit with this project's code** | 52 (incl. `prev_week_*`) | **0.536** |
| `logistic_no_selection` (Run 2 baseline, no selection at all) | 712 | 0.535 |
| Run 1 production model (XGBoost, incl. `prev_week_*`) | 40 | ~0.519 |
| `logistic_regression` — **Run 2's production model** | 75 | 0.491 |
| catboost / lightgbm / xgboost (Run 2 candidates) | 75 | 0.476–0.479 |
| random_forest | 75 | 0.463 |
| gradient_boosting | 75 | 0.457 |

**The notebook's already-selected 52 features, refit honestly, outperform every model this
project has actually shipped** (0.536 vs. 0.491 for the current production model — a real
~4.5-point gap) and are statistically indistinguishable from the best "use everything, no
selection" baseline (0.535). This is a stronger and more specific version of the
"`logistic_no_selection` keeps beating selected-feature candidates" finding above: **it isn't
just that this project's Stage 1/2 selection underperforms "use everything" — it also hasn't
beaten a completely different (and, per the analysis above, test-set-biased) selection process
someone else ran on this same data months earlier.** That raises the bar for what "the
selection pipeline is working" should mean going forward: matching or beating 0.536, not just
beating the in-project baselines.

## Run 1: full feature set (1,048 candidate columns, including `prev_week_*`)

Production model: **XGBoost, threshold=0.55, 40 features**, 18 of them (45%) `prev_week_*`
single-game predictors.

| | Walk-forward (5 folds: 2019, 2021, 2022, 2023, 2024) | 2025 final holdout |
|---|---|---|
| Precision | 0.522 (mean across folds) | **0.429** |
| Recall | 0.389 (mean) | 0.226 |
| Coverage | 36.2% (mean) | 26.0% (56/215 games) |
| ROC-AUC | — | 0.478 |

**2025 home_covered base rate is 49.3%.** This model's precision on its 56 flagged 2025 games
(0.429) was *below* that base rate — flagging games at random would have done better. Combined
with the walk-forward mean (0.522) sitting well under the notebook's single-split number
(0.569), the initial reading was that the notebook's 0.569 figure was inflated relative to a
reliable estimate — later verified and made precise in "Why does the notebook look better?"
above: it's specifically the notebook's feature-count/threshold selection process re-using the
test set across many candidates, not just ordinary single-split variance.

Reviewing this model's feature list surfaced that single-game (`prev_week_*`) predictors made
up the largest category by count (18 of 40) — the noisiest of the three temporal transforms
available for the same underlying stats (`prev_week_X` vs. the smoothed `X_avg_all`/`X_avg3`
alternatives). That observation drove Run 2.

## Run 2: `prev_week_*` excluded (712 candidate columns)

`config/data.yaml`'s `excluded_column_patterns: [prev_week_]` drops all 336 single-game columns
before Stage 1 correlation pruning ever sees them — a hard exclusion applied once, at the data
layer, so every downstream stage (selection, training, threshold selection) is affected
automatically.

Production model: **Logistic Regression, threshold=0.50, 75 features**, 0 of them `prev_week_*`
(confirmed by `scripts/explain_model.py`'s category rollup).

| | Walk-forward (5 folds) | 2025 final holdout |
|---|---|---|
| Precision | **0.492** (mean across folds) | **0.473** |
| Recall | 0.464 (mean) | 0.491 |
| Coverage | 45.8% (mean) | 51.2% (110/215 games) |
| ROC-AUC | — | 0.506 |

### The result is mixed, not a clean win

- **Walk-forward mean precision got slightly worse** (0.522 → 0.492) — removing `prev_week_*`
  did not improve the harder, 5-season honest estimate.
- **2025 holdout precision improved** (0.429 → 0.473) and moved to just under the 49.3% base
  rate, instead of clearly below it, as in Run 1. Coverage and recall both roughly doubled
  (26%→51%, 0.226→0.491) — Run 2's production model (plain logistic regression) is simply less
  confident/more moderate than Run 1's XGBoost, flagging about half of all games rather than a
  quarter.
- Neither run beats the notebook's 0.569 baseline. `beats_baseline_precision: false` in both
  runs' `outputs/model_comparison/final_summary.json`.

**Per-model walk-forward means, Run 2** (full comparison in `outputs/model_comparison/walk_forward_results.csv`):

| model | mean precision | mean recall | mean coverage |
|---|---|---|---|
| logistic_no_selection (baseline, all 712 raw columns) | **0.540** | 0.290 | 0.266 |
| logistic_regression (selected, production) | 0.492 | 0.464 | 0.458 |
| catboost | 0.480 | 0.329 | 0.334 |
| lightgbm | 0.479 | 0.469 | 0.472 |
| xgboost | 0.476 | 0.480 | 0.488 |
| always_favorite (baseline) | 0.476 | 0.603 | 0.613 |
| random_forest | 0.462 | 0.358 | 0.371 |
| gradient_boosting | 0.458 | 0.431 | 0.453 |

**`logistic_no_selection` (no Stage 1/2 pruning at all, just every non-`prev_week_*` raw
column) again beat every selected-feature candidate**, same as it roughly tied the best
candidate in Run 1. This is the second run in a row where skipping feature selection entirely
outperforms it on precision — worth taking seriously as a signal that Stage 1/2 selection, as
currently scored, is not clearly adding value here (see "Next steps").

## Is logistic_regression (the production model) actually the best model? (`scripts/compare_models_on_holdout.py`)

`scripts/evaluate_models.py` picks a production model by highest **walk-forward mean
precision** among candidates, then only refits and scores *that one model* on the true 2025
holdout — so it was never actually verified that the walk-forward winner is also the true
holdout winner. `scripts/compare_models_on_holdout.py` closes that gap: it refits every
baseline + candidate model (same feature set, same walk-forward-selected threshold) on the
identical training seasons and scores all nine on the same never-touched 2025 holdout.

| model | walk-forward rank | walk-forward mean precision | **holdout rank** | **holdout precision** | holdout n flagged / 215 |
|---|---|---|---|---|---|
| logistic_no_selection (baseline) | 1 | 0.540 | **1** | **0.625** | 24 (11%) |
| lightgbm | 4 | 0.479 | 2 | 0.526 | 116 (54%) |
| always_favorite (baseline) | 6 | 0.476 | 3 | 0.515 | 136 (63%) |
| random_forest | 7 | 0.462 | 4 | 0.505 | 103 (48%) |
| **logistic_regression (production)** | **2** | **0.492** | **5** | **0.473** | 110 (51%) |
| catboost | 3 | 0.480 | 6 | 0.471 | 87 (40%) |
| gradient_boosting | 8 | 0.458 | 7 | 0.470 | 115 (53%) |
| xgboost | 5 | 0.476 | 8 | 0.430 | 114 (53%) |
| majority_class (baseline) | 9 | 0.000 | 9 | 0.000 | 0 |

**No — `logistic_regression`, the deployed production model, is not the best model. It ranks
5th of 9 on the true 2025 holdout**, despite ranking 2nd by the walk-forward selection metric
that actually chose it. The model that wins on *both* walk-forward mean precision AND the true
holdout is `logistic_no_selection` — plain logistic regression on all 712 raw (non-`prev_week_*`)
columns, with **no feature selection at all** — at 0.625 holdout precision, well clear of
everything else.

**Two important caveats on that "winner," both visible in the table above:**

1. **It only flagged 24 of 215 games (11% coverage)** — barely above the 10% coverage floor. A
   precision estimate from 24 games has much wider uncertainty than one from ~110-140 games
   (most other models' coverage). `lightgbm`'s 0.526 precision on 116 flagged games is a
   meaningfully more stable second-place result than the raw ranking suggests.
2. It's structurally excluded from `evaluate_models.py`'s production-selection pool by design —
   it exists as a diagnostic baseline (does feature selection help at all?), not a deployable
   candidate, specifically *because* using all 712 raw columns defeats this project's
   dimensionality-reduction goal even when it wins on precision. That design choice is now the
   central tension this build has surfaced: **the more interpretable, smaller, selected-feature
   models are consistently worse than "use everything," on both evaluation views, across both
   Run 1 and Run 2.**

Also notable: `xgboost` actually performed *worse than random* on the true holdout (ROC-AUC
0.427, below the 0.5 chance line) despite being solidly mid-pack on walk-forward — a concrete
example of walk-forward rank not transferring to the holdout at all for some models.
`always_favorite`'s `log_loss` column in the raw CSV is enormous (16.4) and should be ignored —
it's a hard 0/1 classifier, not a real probability model, so log_loss on its degenerate
predictions isn't a meaningful number.

## Run 3: engineered differentials + trend, replacing raw home/away pairs (694 candidate columns)

Tested a specific hypothesis: the feature set only ever expresses `home_X` and `away_X` as
separate levels — nothing directly represents the **matchup** (how much better is the home team
on metric X) or **recent form** (is a team trending up or down vs. its own season baseline).
`config/data.yaml`'s new `feature_representation: differential` toggle
(`src/cfb_spread_model/feature_engineering.py`) replaces the raw `avg_all`/`avg3`/non-temporal
`home_*`/`away_*` pairs with `diff_<transform>_<base> = home_* - away_*` (354 columns) and
`trend_<side>_<base> = <side>_avg3 - <side>_avg_all` (336 columns) — 694 candidate columns total,
run through the exact same Stage 1/2 selection, training, and evaluation pipeline as every prior
run. `prev_week_*` and context columns (`spread`, `home_favored`, etc.) are unaffected either way.

**Walk-forward selection picked `xgboost`** (threshold=0.70, mean_precision=0.530 — the highest
walk-forward number in Run 3). **On the true 2025 holdout, it flagged zero games.** Not low
precision — zero games ever cleared a 0.70 predicted probability on the holdout season at all.
The model the selection process rated best was completely non-functional on fresh data.

Full holdout comparison, all 9 models refit and scored the same way as the "Is
`logistic_regression` the best model?" section above:

| model | walk-forward rank | walk-forward mean precision | **holdout rank** | **holdout precision** | holdout n flagged / 215 |
|---|---|---|---|---|---|
| gradient_boosting | 6 | 0.484 | **1** | **0.564** | 78 (36%) |
| catboost | 4 | 0.508 | 2 | 0.542 | 59 (27%) |
| lightgbm | 7 | 0.478 | 3 | 0.518 | 83 (39%) |
| always_favorite (baseline) | 8 | 0.476 | 4 | 0.515 | 136 (63%) |
| random_forest | 3 | 0.508 | 5 | 0.484 | 64 (30%) |
| logistic_regression | 5 | 0.486 | 6 | 0.472 | 36 (17%) |
| logistic_no_selection (baseline) | 2 | 0.518 | 7 | 0.431 | 58 (27%) |
| **xgboost (the walk-forward winner)** | **1** | **0.530** | **8 (tied last)** | **0.000** | **0 (0%)** |
| majority_class (baseline) | 9 | 0.000 | 8 (tied last) | 0.000 | 0 |

**Spearman rank correlation between walk-forward rank and true holdout rank: -0.667.** For
comparison, Run 2 (raw_dual)'s equivalent correlation was **+0.405** — weak-positive, itself
nothing to be confident in, but at least pointed the right direction. Under the differential
representation, walk-forward rank is not just unreliable, it's **actively anti-correlated** with
what actually happens on fresh data: the model walk-forward liked *most* (`xgboost`) did *worst*
(tied for last) on holdout, and one of the models it liked *least* (`gradient_boosting`, ranked
6th of 8 real candidates) did *best*.

**This is reported as a negative result for the differential representation as currently
built, not a new best model.** `gradient_boosting`'s 0.564 is the best individual holdout
number seen anywhere in this project (ahead of the 0.536 notebook-reproduction benchmark and the
0.473 current production model) — but it was **not** the walk-forward pick, and promoting it to
production now, on the strength of one 215-game holdout look, would repeat the exact
test-set-peeking mistake this document already diagnosed in the notebook's original methodology
("Why does the notebook look better?" above). Treat it as an interesting data point pending a
second true holdout season, not a result to act on.

Also notable: `logistic_no_selection` — the strongest baseline in every prior run (0.535 pooled
walk-forward / 0.625 holdout under `raw_dual`) — got **worse** under the differential
representation (0.517 pooled walk-forward / 0.431 holdout). The representation change did not
uniformly help even the model family that had been winning throughout this project; if anything
it looks like it hurt the previously-best baseline while helping a couple of previously-weaker
candidates (`gradient_boosting`, `catboost`) — inconsistent enough that "differential features
are better" cannot be concluded from this run.

**Operational note**: this run's selected model (`xgboost`, 0 holdout precision) was saved as
`outputs/models/best_model_20260807.pkl` and would have silently become the artifact
`scripts/generate_week_predictions.py` loads next (newest date wins). It was deleted after this
analysis, `config/data.yaml` was reverted to `feature_representation: raw_dual`, and the
existing Run 2 production model (`outputs/models/best_model_20260806.pkl`, `logistic_regression`)
is once again the one that would actually be used.

## Overfitting or insufficient signal? (`scripts/analyze_train_vs_holdout.py`)

Scored the production model on its own training rows (2,080 games), pooled walk-forward OOF
(1,207 games), and the true 2025 holdout (215 games), to separate two very different failure
modes that look similar from precision alone: **overfitting** (the model fits training data
well but that fit doesn't transfer) vs. **insufficient signal** (the model doesn't even fit its
own training data well, so there's nothing to overfit).

| split | n | precision | recall | ROC-AUC | avg. precision | log_loss |
|---|---|---|---|---|---|---|
| train | 2,080 | 0.570 | 0.494 | **0.613** | 0.582 | **0.671** |
| walk-forward OOF (pooled) | 1,207 | 0.491 | 0.462 | 0.515 | 0.491 | 0.726 |
| 2025 holdout | 215 | 0.473 | 0.491 | 0.506 | 0.551 | 0.701 |

**The dominant story is insufficient signal, not overfitting.** The clearest single number:
**training log_loss (0.671) is barely better than the trivial "always predict the base rate"
baseline (0.692)** — a 0.021 improvement, on the data the model was fit on. A model that had
found strong exploitable patterns in training data would show training log_loss well below that
trivial baseline (and training ROC-AUC well above 0.613, likely 0.80+). This one isn't finding
much to memorize in the first place.

That said, there IS a real, consistent gap on top of that weak ceiling: ROC-AUC drops from 0.613
(train) to ~0.51 (both OOF and holdout, essentially chance) — a ~0.10 point gap, and precision
drops ~0.08-0.10 points the same way. So it's not that nothing generalizes at all: what little
apparent signal exists in training data erodes to statistical noise out of sample. **Read
together: the ceiling on how predictive this feature set can be is low (~0.61 ROC-AUC at best,
on training data), and even that modest ceiling doesn't fully survive contact with new data.**
Both things are true; insufficient signal is the bigger piece of the story, with a secondary,
real generalization gap layered on top of an already-weak fit.

This is consistent with — and helps explain — the rank-calibration finding below (near-zero
monotonicity): a model whose training-set ROC-AUC is only 0.61 was never going to produce a
sharply-ranked, trustworthy probability score in the first place.

## Rank calibration: are the top predictions winners, and do the bottom predictions call the other side? (`scripts/analyze_rank_calibration.py`)

Sorted games by predicted probability of `home_covered` into buckets and checked whether the
**highest**-probability bucket actually covers at a high rate, and whether the **lowest**-
probability bucket correctly identifies the other side (home does NOT cover) at a high rate —
i.e., whether the model's own confidence ranking means anything end to end, not just whether the
threshold it picked happens to work.

**Walk-forward OOF, pooled across all 5 seasons (1,207 games, 10 deciles)** —
`outputs/calibration/walk_forward_buckets.csv`:

| bucket (low → high predicted prob) | mean predicted | actual cover rate |
|---|---|---|
| 1 | 0.270 | 0.479 |
| 2 | 0.356 | 0.405 |
| 3 | 0.398 | 0.425 |
| 4 | 0.437 | 0.529 |
| 5 | 0.471 | 0.504 |
| 6 | 0.503 | 0.533 |
| 7 | 0.539 | 0.496 |
| 8 | 0.577 | 0.492 |
| 9 | 0.624 | **0.537** |
| 10 (highest) | 0.723 | **0.438** |

- **Monotonicity (Spearman rank correlation between bucket order and actual rate): 0.394** —
  weak positive, far from the +1 a well-calibrated ranking would show.
- **The single highest-confidence decile (mean predicted 0.72) actually covers at 43.8% — below
  the 48.4% base rate**, and below several middle deciles. The model's most confident "home
  covers" predictions are, in this pooled 5-season view, *worse* than a coin flip, not better.
- The lowest decile (mean predicted 0.27) correctly calls "home does NOT cover" 52.1% of the
  time — a lift of only +0.5 points over the base rate, essentially no better than chance.

**2025 final holdout (215 games, 5 buckets)** — `outputs/calibration/holdout_buckets.csv`:

| bucket | mean predicted | actual cover rate |
|---|---|---|
| 1 (lowest) | 0.391 | 0.512 |
| 2 | 0.462 | 0.488 |
| 3 | 0.503 | 0.465 |
| 4 | 0.550 | 0.442 |
| 5 (highest) | 0.633 | **0.558** |

- **Monotonicity: 0.000** — literally zero rank correlation. Rates decline steadily from bucket
  1 through 4, then jump back up at bucket 5; there is no consistent trend a bettor could act on
  by "trusting the model more" as predicted probability moves further from 0.5.
- The top bucket does show a real lift here (+6.5 points over the 49.3% base rate), but buckets
  2-4 actively go the *wrong* direction (higher predicted probability, lower actual rate) before
  the reversal at the top bucket — the opposite of monotonic.

### Honest conclusion

**The model's predicted probability does not behave as a reliable confidence signal.** A
bettor using this model to size bets by "how far the probability is from 0.5" would be acting on
noise for most of the distribution — the walk-forward view's top decile underperforming its own
base rate is the clearest single piece of evidence. The threshold-based precision numbers
reported elsewhere in this document (0.492 walk-forward, 0.473 holdout) are legitimate at that
*specific* threshold, but they should not be read as implying "and it gets more reliable the
higher the probability goes" — this analysis shows that's not true here. This is a genuine
limitation to flag before this model informs any real decision, not just a modeling detail.

## Feature importance (`scripts/explain_model.py`, Run 2's production model)

Two measures computed on the true 2025 holdout: gain-based (unavailable for logistic
regression — it has no `feature_importances_`, only `coef_`, so this fell back gracefully to
permutation-only, exactly as designed) and permutation importance scored with the same
precision-at-coverage-floor metric used throughout this project.

**By temporal-transform category** (`outputs/model_comparison/feature_importance_by_category_20260806.csv`):

| category | n features | % of features | % of permutation-importance mass |
|---|---|---|---|
| `avg_all` (season-to-date average) | 31 | 41.3% | **54.1%** |
| `avg3` (trailing 3-game average) | 39 | 52.0% | 42.2% |
| non-temporal (talent/coaching/returning production) | 4 | 5.3% | 3.7% |
| context (`conference_game`) | 1 | 1.3% | 0.0% |
| `prev_week_*` | 0 | 0% | 0% |

The `prev_week_*` exclusion is confirmed working end-to-end: zero single-game predictors reach
the production model. Importance mass leans toward `avg_all` (season-to-date, the most-smoothed
signal) over `avg3` (3-game), consistent with the hypothesis that smoother signals are more
reliable than noisier ones — though this run used logistic regression on 75 features rather
than a apples-to-apples rerun of the same model family as Run 1, so this isn't a fully
controlled comparison.

**Top individual features by permutation importance**
(`outputs/model_comparison/feature_importance_20260806.csv`, full ranking there):

1. `home_Total_Offense_Success_avg_all`
2. `away_Offense_Success_Rate_avg_all`
3. `home_Total_Defense_Touchdown_Drives_avg_all`
4. `home_Defense_Total_Run_Plays_avg_all`
5. `home_fumbles_lost_allowed_avg3`

Season-to-date offensive/defensive success-rate metrics dominate the top of the list, well
above any single individually-important special-teams or turnover-margin column.

## Other findings worth flagging

- **RFECV's selected feature count was highly unstable across folds** in both runs (Run 1: 10,
  470, 641, 326, 82, 330 features across six folds; Run 2 similarly wide) compared to the
  permutation-importance sweep's much tighter range. This instability is reported, not hidden —
  it suggests RFECV's greedy elimination is sensitive to fold-specific noise in this feature
  space, and the permutation-importance sweep is the more trustworthy of the two Stage 2
  methods here. See `docs/feature_selection_methodology.md`.
- Stage 1 correlation pruning is a modest cut in both runs (~22-30% of candidate columns), not
  the much larger reduction a naive "3x temporal redundancy" assumption would predict.

## What this means for next steps

- **Differential/trend features (Run 3) are not a validated improvement — the walk-forward
  selection process became actively unreliable under that representation** (rank correlation
  with true holdout performance went from a weak +0.405 to -0.667), even though one candidate
  model's holdout number looked good in isolation. Before trying this representation again:
  (a) get a second true holdout season so a promising model can be checked twice, not once, (b)
  investigate *why* `xgboost` collapsed to zero holdout coverage under this representation
  specifically — likely worth checking whether `diff_*`/`trend_*` columns have a different scale/
  distribution shift between training-era seasons and 2025 than the raw columns did, and (c)
  don't conflate "one model did well" with "the representation works" — the whole point of
  tracking rank correlation is to catch exactly this gap.
- **The production model is not the best-performing model on true holdout data — worth fixing
  the selection process, not just noting it.** `logistic_no_selection` beating everything, twice
  now (walk-forward AND true holdout), on both runs, is strong enough evidence that
  `evaluate_models.py`'s exclusion of it from the candidate pool deserves revisiting — either
  promote "all features, plain regularized logistic regression" to a real candidate, or
  investigate why every selected-feature model underperforms it before trusting any of their
  precision numbers for a real decision. Also worth running `compare_models_on_holdout.py`-style
  full comparisons routinely, not just for the winner — this run's biggest surprise (`xgboost`
  scoring below chance on holdout, `logistic_regression` landing 5th of 9) would never have
  surfaced from `evaluate_models.py`'s output alone.
- **The train-vs-holdout gap (`docs/project_story.md`'s "Overfitting or insufficient signal?"
  section) says the priority is finding MORE signal, not fighting overfitting.** Regularizing
  harder, simplifying the model further, or adding more data-hungry safeguards would treat the
  wrong problem — training log_loss is already barely better than the trivial base-rate
  baseline, so there isn't much of a strong in-sample fit to protect against overfitting in the
  first place. The likely lever is new information the current ~1,000 engineered columns don't
  carry (line movement, injury reports, weather, personnel changes) rather than better
  regularization or more selection discipline on the existing feature family.
- **The rank-calibration finding (near-zero/weak monotonicity, top decile underperforming its
  own base rate in the walk-forward view) is arguably the most important limitation in this
  whole document.** Before this model informs any real decision, it should not be treated as
  producing a usable "confidence" score beyond the single threshold it was tuned at. Worth
  investigating: whether isotonic/Platt calibration on top of the existing model's raw scores
  fixes this, or whether it's a symptom of the same "features may not carry much signal" issue
  `logistic_no_selection`'s performance already points at.
- **Do not replace `../Python Scripts/CFB_Gambling_Model.ipynb` / `Week_Predictions.ipynb`** on
  the strength of either run — neither beats the documented baseline, and the baseline's own
  headline number needs to be treated with real skepticism (it was never walk-forward
  validated). All three should be considered unproven at a genuinely useful precision level.
- **`logistic_no_selection` beating every selected-feature candidate, twice, plus the notebook's
  independently-selected 52 features also beating this project's production model by ~4.5
  points (0.536 vs. 0.491, pooled walk-forward), is the most actionable finding here.** This
  project's Stage 1/2 selection hasn't beaten "use everything" OR a completely different
  selection process run by someone else on this same data — two independent pieces of evidence
  pointing the same direction. Consider either (a) treating "everything, plain logistic
  regression" as the real candidate pool baseline going forward rather than a diagnostic
  baseline, or (b) revisiting the Stage 1/2 selection objective itself — it optimizes precision
  at a coverage floor on a small inner-CV validation slice, which may be too noisy a target to
  reliably beat "use everything and let L2 regularization sort it out." Either way, **0.536
  (pooled walk-forward) is now the number to beat**, not the previous in-project baselines.
- ~~Re-run the current notebook's exact 52-feature model through this project's walk-forward
  harness~~ — **done**, see "Why does the notebook look better?" above
  (`scripts/replicate_notebook_features.py`): 0.529 reproduced vs. 0.569 reported, with the gap
  best explained by test-set reuse during the notebook's own feature/threshold search.
- Given the `avg_all` > `avg3` importance-mass split observed here, a natural next experiment:
  try excluding `avg3` as well (keep only `avg_all` + non-temporal + context), or the reverse,
  to see whether a single clean temporal transform per metric beats offering the model all of
  them at once.
- The coverage floor (`min_coverage_floor: 0.10`) is still fairly permissive — several models in
  both runs land at 30-50%+ mean coverage, well above the floor. A stricter floor (fewer, more
  selective bets) is a natural next experiment.
