from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_rb_rushing_model.modeling.evaluation import (
    evaluate_predictions,
    out_of_fold_residuals,
    prediction_interval_from_residuals,
    walk_forward_results,
)


def test_evaluate_predictions_perfect_prediction_is_zero_mae():
    y = np.array([10.0, 50.0, 0.0, 120.0])
    metrics = evaluate_predictions(y, y)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_evaluate_predictions_mean_bias_sign():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([15.0, 15.0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["mean_bias"] == 5


def test_prediction_interval_from_residuals_widens_with_more_extreme_levels():
    point = np.array([50.0])
    residuals = np.array([-30, -20, -10, 0, 10, 20, 30])
    lo_narrow, hi_narrow = prediction_interval_from_residuals(point, residuals, (0.25, 0.75))
    lo_wide, hi_wide = prediction_interval_from_residuals(point, residuals, (0.05, 0.95))
    assert lo_wide[0] <= lo_narrow[0]
    assert hi_wide[0] >= hi_narrow[0]


def test_out_of_fold_residuals_matches_manual_diff():
    oof_df = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [12, 18, 33]})
    resid = out_of_fold_residuals(oof_df)
    np.testing.assert_array_equal(resid, np.array([-2, 2, -3]))


def test_walk_forward_results_groups_by_model_and_fold():
    oof_df = pd.DataFrame(
        {
            "model_name": ["a", "a", "b", "b"],
            "fold_validation_season": [2022, 2023, 2022, 2023],
            "y_true": [10, 20, 10, 20],
            "y_pred": [10, 20, 15, 25],
        }
    )
    result = walk_forward_results(oof_df)
    assert set(result["model_name"]) == {"a", "b"}
    a_2022_mae = result[(result["model_name"] == "a") & (result["fold_validation_season"] == 2022)]["mae"].iloc[0]
    assert a_2022_mae == 0
