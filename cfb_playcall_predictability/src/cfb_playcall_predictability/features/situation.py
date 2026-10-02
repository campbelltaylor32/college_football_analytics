"""Pre-snap game situation: down & distance, field position, clock, score, timeouts, venue,
and the pregame market (spread / total) from the offense's perspective."""

from __future__ import annotations

import numpy as np
import pandas as pd


def situation_features(calls: pd.DataFrame) -> pd.DataFrame:
    c = calls
    out = pd.DataFrame(index=c.index)
    out["down"] = c["down"].astype(int)
    out["distance"] = c["distance"].clip(1, 30)
    out["yards_to_goal"] = c["yards_to_goal"]
    out["goal_to_go"] = (c["distance"] >= c["yards_to_goal"]).astype(int)
    out["red_zone"] = (c["yards_to_goal"] <= 20).astype(int)
    out["period"] = c["period"]
    out["half_secs_rem"] = c["TimeSecsRem"].clip(0, 1800)
    out["game_secs_rem"] = c["adj_TimeSecsRem"].clip(0, 3600)
    out["two_minute"] = (out["half_secs_rem"] <= 120).astype(int)
    out["score_diff"] = c["pos_score_diff_start"]
    # The same deficit matters more late: scale by the share of the game that has elapsed.
    out["score_diff_x_elapsed"] = out["score_diff"] * (1 - out["game_secs_rem"] / 3600)
    out["score_diff_per_poss_left"] = out["score_diff"] / np.sqrt(out["game_secs_rem"] / 150 + 1)
    out["wp_before"] = c["wp_before"]
    out["off_timeouts"] = c["pos_team_timeouts_rem_before"].clip(0, 3)
    out["def_timeouts"] = c["def_pos_team_timeouts_rem_before"].clip(0, 3)
    out["receives_2h_kickoff"] = c["pos_team_receives_2H_kickoff"]
    out["drive_play_number"] = c["drive_play_number"].clip(upper=30)
    out["is_home"] = (c["pos_team"] == c["home"]).astype(int)
    return out


def market_features(calls: pd.DataFrame) -> pd.DataFrame:
    """Expects `spread` and `over_under` merged on. CFBD's spread is the home line
    (negative = home favored), so the offense's expected margin is -spread at home."""
    is_home = calls["pos_team"] == calls["home"]
    out = pd.DataFrame(index=calls.index)
    out["exp_margin"] = np.where(is_home, -calls["spread"], calls["spread"])
    out["game_total"] = calls["over_under"]
    out["implied_team_total"] = (out["game_total"] + out["exp_margin"]) / 2
    return out
