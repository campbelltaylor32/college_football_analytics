"""How the game is going for this offense, from its EARLIER calls in the same game only.

Every feature is a cumulative sum minus the current row (or a shift by one), grouped by
(game_id, pos_team) over snap order -- the current play's outcome can never leak in.
Garbage-time calls count as history here; they are real snaps the play caller saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = ["game_id", "pos_team"]


def _prior_cumsum(s: pd.Series, by: list[pd.Series]) -> pd.Series:
    return s.groupby(by).cumsum() - s


def in_game_features(calls: pd.DataFrame) -> pd.DataFrame:
    """calls must be in snap order (plays.build_calls guarantees it)."""
    c = calls
    keys = [c[k] for k in GROUP]
    drive_keys = keys + [c["drive_number"]]

    is_pass = c["is_pass"].astype(float)
    is_run = 1.0 - is_pass
    success = c["success"].fillna(0).astype(float)
    epa = c["EPA"].fillna(0.0)
    yards = c["yards_gained"].fillna(0.0)

    out = pd.DataFrame(index=c.index)
    out["g_calls_so_far"] = c.groupby(GROUP).cumcount()
    out["g_passes_so_far"] = _prior_cumsum(is_pass, keys)
    out["g_runs_so_far"] = out["g_calls_so_far"] - out["g_passes_so_far"]
    out["g_run_success_sum"] = _prior_cumsum(success * is_run, keys)
    out["g_pass_success_sum"] = _prior_cumsum(success * is_pass, keys)
    out["g_run_epa_sum"] = _prior_cumsum(epa * is_run, keys)
    out["g_pass_epa_sum"] = _prior_cumsum(epa * is_pass, keys)
    out["g_rush_yards_sum"] = _prior_cumsum(yards * is_run, keys)

    prev = c.groupby(GROUP)[["is_pass", "success", "yards_gained", "drive_number"]].shift(1)
    out["prev_call_pass"] = prev["is_pass"].fillna(-1)          # -1 = first call of the game
    out["prev_call_success"] = prev["success"].fillna(-1)
    out["prev_call_yards"] = prev["yards_gained"].fillna(0)
    out["prev_call_same_drive"] = (prev["drive_number"] == c["drive_number"]).astype(int)

    # Length of the run of identical calls ending at the previous play.
    change = (c["is_pass"] != c.groupby(GROUP)["is_pass"].shift(1)).astype(int)
    run_id = change.groupby(keys).cumsum()
    streak_through_current = c.groupby(GROUP + [run_id.rename("_run")]).cumcount() + 1
    out["prev_streak_len"] = streak_through_current.groupby(keys).shift(1).fillna(0)

    out["drive_calls_so_far"] = c.groupby(GROUP + ["drive_number"]).cumcount()
    out["drive_passes_so_far"] = _prior_cumsum(is_pass, drive_keys)
    return out


def shrink_in_game(df: pd.DataFrame, k_rate: float, k_eff: float) -> pd.DataFrame:
    """Blend the raw in-game sums with priors (columns already on df):
      pass rate     -> season-to-date overall pass rate (std_pr_overall)
      run/pass EPA, success -> offense's season-to-date efficiency (off_std_*)"""
    out = pd.DataFrame(index=df.index)
    n, runs, passes = df["g_calls_so_far"], df["g_runs_so_far"], df["g_passes_so_far"]
    out["g_pass_rate"] = (df["g_passes_so_far"] + k_rate * df["std_pr_overall"]) / (n + k_rate)
    out["g_pass_rate_vs_std"] = out["g_pass_rate"] - df["std_pr_overall"]
    out["g_run_sr"] = (df["g_run_success_sum"] + k_eff * df["off_std_run_sr"]) / (runs + k_eff)
    out["g_pass_sr"] = (df["g_pass_success_sum"] + k_eff * df["off_std_pass_sr"]) / (passes + k_eff)
    out["g_run_epa"] = (df["g_run_epa_sum"] + k_eff * df["off_std_run_epa"]) / (runs + k_eff)
    out["g_pass_epa"] = (df["g_pass_epa_sum"] + k_eff * df["off_std_pass_epa"]) / (passes + k_eff)
    out["g_pass_minus_run_epa"] = out["g_pass_epa"] - out["g_run_epa"]
    out["g_ypc"] = np.where(runs > 0, df["g_rush_yards_sum"] / runs.clip(lower=1), np.nan)
    return out
