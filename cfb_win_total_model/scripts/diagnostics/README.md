# scripts/diagnostics/

Investigation into prediction compression and the 2024-2025 out-of-sample accuracy decline
(`docs/diagnostics_compression_report.md`, `docs/project_story.md`). Every script here is
**evaluate-only**: none of them ever write to `outputs/model_comparison/selected_model.json`,
`outputs/models/final_model.joblib`, or `outputs/predictions/predicted_win_totals_<season>.csv`
— nothing in this directory changes the shipped model. All new artifacts land under
`outputs/diagnostics_compression/`.

Run the core suite with one command:

```bash
python scripts/diagnostics/run_all.py
```

That covers the three read-mostly diagnostic stages (fast, no retraining). The follow-up
tuning experiments below it are run standalone since they retrain models and take longer.

## Scripts, in the order this investigation actually happened

| Script | What it does | Runtime |
|---|---|---|
| `compute_compression_diagnostics.py` | Core diagnostics from existing artifacts: std/range comparison, both-direction calibration and regression slopes, train-vs-val (underfitting check), regularization/hyperparameter-grid boundary check, conference/season/`n_games`-endogeneity breakdowns. No retraining. | ~15s |
| `feature_experiment.py` | Tests 7 candidate new features/interactions (talent-vs-schedule, QB continuity, etc.) via identical walk-forward CV. Result: no measurable improvement — ruled out "weak features" as the cause. | ~2 min |
| `win_probability_sum_prototype.py` | Tests summing per-game win probabilities (Elo/CFBD fields) as an alternative to direct season-win regression. Flags an important in-season-information caveat in its own output. | ~10s |
| `run_all.py` | Runs the three scripts above in sequence. | ~3 min |
| `variance_aware_retune.py` | The core fix attempt: adds `modeling.evaluation.variance_aware_score` as an alternative GridSearchCV objective (MAE + a penalty for under-spread predictions) and widens the hyperparameter grids for gradient_boosting/ridge/elasticnet. Confirms widening the grid alone does nothing; the objective is what matters. | ~8 min |
| `penalty_weight_sweep.py` | Sweeps the variance-aware objective's `penalty_weight` across {0.1, 0.25, 0.5, 1.0, 2.0} to trace the MAE-vs-compression tradeoff curve. Finds gradient_boosting saturates by `pw≈1.0`. | ~13 min |
| `penalty_weight_single_check.py [pw]` | Re-checks one `penalty_weight` value (arg, default 5.0) and additionally saves fold pipelines so a train-vs-val (over/underfitting) comparison can be run. Used for pw=0.5 and pw=5.0 in the report. | ~4 min |
| `pw_final_holdout_check.py [pw]` | Refits under a given `penalty_weight` on the *true* final-holdout training set and scores on the real 2025 season (not walk-forward OOF) — the check that matters before ever promoting a fix. Result: none of the tested weights beat the shipped model on the true holdout, despite working in cross-validation. | ~2 min |
| `plot_r2_vs_transfers.py` | Renders `outputs/diagnostics_compression/plots/oof_r2_vs_transfers_by_season.png` — out-of-sample R² alongside average transfers/team by season. | ~5s |
| `plot_fold_table.py` | Renders `outputs/diagnostics_compression/plots/walk_forward_fold_table.png` — the walk-forward fold table (train seasons, # train seasons, OOF R²) highlighting that the two folds with the *most* training data have the *worst* R². | ~2s |

## Headline result

Every fix tested here — widened grids, a variance-aware tuning objective at multiple penalty
strengths, a simpler lasso model — improved cross-validated (walk-forward) metrics but **none**
beat the shipped model on the true 2025 holdout. Out-of-sample R² has declined for three
straight seasons even as training data grew each year, which points to real non-stationarity
in the sport (transfer-portal churn, conference realignment) rather than a fixable modeling
artifact. See `docs/project_story.md` for the full narrative and `docs/diagnostics_compression_report.md`
for the underlying technical detail.
