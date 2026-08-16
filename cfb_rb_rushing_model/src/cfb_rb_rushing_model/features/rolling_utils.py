"""Shared two-step compute-then-lag helper, used identically by rushing_workload.py (player
grain), team_offense_context.py (team grain), and opponent_defense_context.py (defense grain).

Step 1: compute trailing-window (`avg{window}`) and cumulative season-to-date (`avg_all`)
aggregates over each entity's own rows sorted by date, CURRENT row included.
Step 2: `.shift(1)` every rolled column by entity group -- only the `_lag1` columns are ever
used as a feature or joined onto a prediction row.

Kept as one shared helper (not duplicated three times) specifically so a leakage test can
independently recompute "value at row i" and assert equality to "raw rolling value through
row i-1" against ONE code path, not three -- see tests/test_feature_shifting.py.
"""

from __future__ import annotations

import pandas as pd


def compute_rolling_and_lag(
    df: pd.DataFrame, group_cols: list[str], sort_col: str, value_cols: list[str], window: int
) -> pd.DataFrame:
    """Adds, for every column in value_cols: `{col}_avg{window}`, `{col}_avg_all`, and their
    `_lag1` counterparts. Only the `_lag1` columns are safe to use as model features."""
    df = df.sort_values(group_cols + [sort_col]).reset_index(drop=True).copy()
    grouped = df.groupby(group_cols, group_keys=False)

    rolled_cols = []
    for col in value_cols:
        avg_w_col = f"{col}_avg{window}"
        avg_all_col = f"{col}_avg_all"
        df[avg_w_col] = grouped[col].transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        df[avg_all_col] = grouped[col].transform(lambda s: s.expanding(min_periods=1).mean())
        rolled_cols += [avg_w_col, avg_all_col]

    grouped_again = df.groupby(group_cols, group_keys=False)
    for col in rolled_cols:
        df[f"{col}_lag1"] = grouped_again[col].shift(1)

    return df


def attach_games_played_lag1(
    df: pd.DataFrame, group_cols: list[str], sort_col: str, out_col: str = "games_played_lag1"
) -> pd.DataFrame:
    """Count of this entity's own PRIOR rows (games) -- 0 for a true first-ever recorded
    row/game. Used both as a cold-start feature and by eligibility.py's fallback gate."""
    df = df.sort_values(group_cols + [sort_col]).reset_index(drop=True).copy()
    df[out_col] = df.groupby(group_cols).cumcount()
    return df
