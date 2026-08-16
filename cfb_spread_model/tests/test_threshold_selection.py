from __future__ import annotations

import pandas as pd

from cfb_spread_model.modeling.threshold_selection import evaluate_threshold_grid, select_best_threshold_per_model


def _make_oof_df() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022]:
        for i in range(20):
            y_true = 1 if i < 8 else 0
            y_score = 0.9 - i * 0.03
            rows.append({"model_name": "model_a", "fold_validation_season": season, "y_true": y_true, "y_score": y_score})
    return pd.DataFrame(rows)


def test_evaluate_threshold_grid_produces_one_row_per_model_fold_threshold():
    oof_df = _make_oof_df()
    thresholds = [0.5, 0.7, 0.9]
    grid = evaluate_threshold_grid(oof_df, thresholds)
    n_model_folds = oof_df[["model_name", "fold_validation_season"]].drop_duplicates().shape[0]
    assert len(grid) == len(thresholds) * n_model_folds


def test_select_best_threshold_respects_coverage_floor():
    oof_df = _make_oof_df()
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    grid = evaluate_threshold_grid(oof_df, thresholds)
    best = select_best_threshold_per_model(grid, min_coverage_floor=0.3)
    row = best.iloc[0]
    if row["meets_floor_every_fold"]:
        assert row["min_coverage"] >= 0.3


def test_select_best_threshold_flags_when_floor_unreachable():
    oof_df = _make_oof_df()
    grid = evaluate_threshold_grid(oof_df, [0.5, 0.9])
    best = select_best_threshold_per_model(grid, min_coverage_floor=0.99)
    assert best.iloc[0]["meets_floor_every_fold"] == False  # noqa: E712
