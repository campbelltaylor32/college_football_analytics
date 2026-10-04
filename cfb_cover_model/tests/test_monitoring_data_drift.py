import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.monitoring.data_drift import run_data_drift_report


def test_no_drift_when_distributions_match(tmp_path, rng):
    ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(5, 2, 50)})

    summary = run_data_drift_report(ref, cur, ["a", "b"], output_dir=tmp_path, label="no_drift")

    assert summary["n_features_compared"] == 2
    assert summary["dataset_drift"] is False
    assert summary["drifted_columns"] == []
    assert (tmp_path / "no_drift_data_drift.json").exists()
    assert (tmp_path / "no_drift_data_drift.html").exists()


def test_drift_detected_when_distribution_shifts(tmp_path, rng):
    ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
    cur = pd.DataFrame({"a": rng.normal(10, 1, 50), "b": rng.normal(5, 2, 50)})  # "a" shifted, "b" unchanged

    summary = run_data_drift_report(ref, cur, ["a", "b"], output_dir=tmp_path, label="shifted")

    assert summary["dataset_drift"] is True
    assert "a" in summary["drifted_columns"]
    assert summary["share_drifted_columns"] > 0


def test_reference_and_current_are_restricted_to_feature_columns(tmp_path, rng):
    """Bookkeeping columns present in either frame (game_id, home_covered, ...) must never
    leak into the comparison - only feature_columns should ever be passed to Evidently."""
    ref = pd.DataFrame({"a": rng.normal(0, 1, 100), "game_id": np.arange(100), "home_covered": rng.integers(0, 2, 100)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 30), "game_id": np.arange(30), "home_covered": rng.integers(0, 2, 30)})

    summary = run_data_drift_report(ref, cur, ["a"], output_dir=tmp_path, label="restricted")
    assert summary["n_features_compared"] == 1
