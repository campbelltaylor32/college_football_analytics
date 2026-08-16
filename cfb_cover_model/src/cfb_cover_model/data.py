"""Load the two source CSVs and join them into a single raw frame.

The predictors CSV (produced by ../R Scripts/Merge_Predictors_CFB_Historical.R) carries the
engineered feature columns plus an *absolute-value* spread and the R layer's own
(push-mislabeled) home_covered label. The results CSV (produced by
../R Scripts/Full_CFB_Game_Outcome_Historical.R) carries the final score and a *signed* spread
for the same games, keyed by game_id. Joining them here - rather than trusting the predictors
file's home_covered - is what lets targets.py detect pushes and build the continuous
cover_margin target correctly.
"""
from __future__ import annotations

import pandas as pd

from cfb_cover_model.config import load_data_config, resolve_path

BOOL_COLUMNS = ["neutral_site", "conference_game"]


def load_predictors_df(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_data_config()
    path = resolve_path(cfg["paths"]["predictors_csv"])
    df = pd.read_csv(path)
    for col in BOOL_COLUMNS:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    return df


def load_results_df(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_data_config()
    path = resolve_path(cfg["paths"]["results_csv"])
    keep = [
        "game_id",
        "home_points",
        "away_points",
        "home_minus_away",
        "spread",  # signed: negative means home favored, matches formatted_spread's sign
    ]
    df = pd.read_csv(path, usecols=keep)
    df = df.rename(columns={"spread": "signed_spread"})
    return df


def load_raw_joined(cfg: dict | None = None) -> pd.DataFrame:
    """Predictors frame with home_points/away_points/home_minus_away/signed_spread attached.

    Inner join on game_id: every predictors row is expected to have a matching results row
    (verified: 2,386/2,386 in the current data), so an inner join that silently drops rows
    would indicate a real data problem worth investigating, not something to work around here.
    """
    cfg = cfg or load_data_config()
    predictors = load_predictors_df(cfg)
    results = load_results_df(cfg)

    n_before = len(predictors)
    merged = predictors.merge(results, on="game_id", how="inner", validate="one_to_one")
    if len(merged) != n_before:
        raise ValueError(
            f"Expected every predictors row to join to a results row; "
            f"{n_before - len(merged)} of {n_before} rows were dropped."
        )
    return merged


def load_week_predictors_df(week_csv_path) -> pd.DataFrame:
    """Load a single week's CFB_Pred_Week_<N>.csv for inference (no results join - future
    games have no final score)."""
    df = pd.read_csv(week_csv_path)
    for col in BOOL_COLUMNS:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    return df
