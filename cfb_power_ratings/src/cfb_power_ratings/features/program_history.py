"""The strongest single preseason signal: a team's own recent actual on-field strength.
Trailing-season SRS ratings (individually, plus a rolling mean) let the model weight recency
itself rather than being handed a single pre-averaged number. Built directly from
srs.build_historical_srs_table's output -- this module has no DB access of its own.
"""
from __future__ import annotations

import pandas as pd


def build_program_history_features(historical_srs: pd.DataFrame, seasons: list[int], trailing_seasons: int = 3) -> pd.DataFrame:
    """`historical_srs` is srs.build_historical_srs_table's output (team, season, srs) --
    covering at least `trailing_seasons` years before the earliest year in `seasons`, or the
    lag columns for those early rows will be NaN (expected and handled by the model's
    NaN-tolerant estimator, not an error)."""
    wide = historical_srs.pivot(index="team", columns="season", values="srs")

    rows = []
    for season in seasons:
        row = pd.DataFrame({"team": wide.index})
        row["season"] = season
        for lag in range(1, trailing_seasons + 1):
            lag_season = season - lag
            row[f"srs_lag{lag}"] = wide[lag_season].values if lag_season in wide.columns else float("nan")
        lag_cols = [f"srs_lag{lag}" for lag in range(1, trailing_seasons + 1)]
        row["srs_trailing_mean"] = row[lag_cols].mean(axis=1, skipna=True)
        rows.append(row)

    return pd.concat(rows, ignore_index=True)
