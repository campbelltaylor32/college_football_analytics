import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.data import load_predictors_df, load_raw_joined


def _write_predictors_csv(path, game_ids):
    n = len(game_ids)
    df = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": [2020] * n,
            "week": [5] * n,
            "neutral_site": [("TRUE" if i % 2 == 0 else "FALSE") for i in range(n)],
            "conference_game": [("FALSE" if i % 2 == 0 else "TRUE") for i in range(n)],
            "spread": [7.0] * n,
            "home_talent": [500.0] * n,
        }
    )
    df.to_csv(path, index=False)


def _write_results_csv(path, game_ids):
    df = pd.DataFrame(
        {
            "game_id": game_ids,
            "home_points": [21] * len(game_ids),
            "away_points": [14] * len(game_ids),
            "home_minus_away": [7] * len(game_ids),
            "spread": [-7.0] * len(game_ids),
        }
    )
    df.to_csv(path, index=False)


def test_bool_columns_parsed_as_python_bool(tmp_path):
    predictors_path = tmp_path / "predictors.csv"
    _write_predictors_csv(predictors_path, [1, 2, 3, 4])
    cfg = {"paths": {"predictors_csv": str(predictors_path)}}

    df = load_predictors_df(cfg)
    assert df["neutral_site"].dtype == bool
    assert df["conference_game"].dtype == bool
    assert df.loc[0, "neutral_site"] is True or df.loc[0, "neutral_site"] == True  # noqa: E712


def test_load_raw_joined_matches_every_row(tmp_path):
    game_ids = [101, 102, 103, 104]
    predictors_path = tmp_path / "predictors.csv"
    results_path = tmp_path / "results.csv"
    _write_predictors_csv(predictors_path, game_ids)
    _write_results_csv(results_path, game_ids)
    cfg = {"paths": {"predictors_csv": str(predictors_path), "results_csv": str(results_path)}}

    merged = load_raw_joined(cfg)
    assert len(merged) == len(game_ids)
    assert "signed_spread" in merged.columns
    assert "home_minus_away" in merged.columns
    assert set(merged["game_id"]) == set(game_ids)


def test_load_raw_joined_raises_when_a_predictors_row_has_no_match(tmp_path):
    predictors_path = tmp_path / "predictors.csv"
    results_path = tmp_path / "results.csv"
    _write_predictors_csv(predictors_path, [201, 202, 203])
    _write_results_csv(results_path, [201, 202])  # missing 203 on purpose
    cfg = {"paths": {"predictors_csv": str(predictors_path), "results_csv": str(results_path)}}

    with pytest.raises(ValueError):
        load_raw_joined(cfg)
