"""Team tendencies (pass rate by situation bucket) and team / opponent efficiency, each as a
prior-season and a season-to-date feature. Built from non-garbage-time calls only."""

from __future__ import annotations

import pandas as pd

from cfb_playcall_predictability.config import FeaturesConfig
from cfb_playcall_predictability.features.rates import prior_and_season_to_date

EFFICIENCY_METRICS = ["run_epa", "pass_epa", "run_sr", "pass_sr"]


def bucket_masks(calls: pd.DataFrame, cfg: FeaturesConfig) -> dict[str, pd.Series]:
    down, dist = calls["down"], calls["distance"]
    margin = calls["pos_score_diff_start"]
    early = down.isin([1, 2])
    masks = {
        "overall": pd.Series(True, index=calls.index),
        "early_down_neutral": early & (margin.abs() <= cfg.neutral_margin) & (calls["period"] <= 3),
        "first_and_10": (down == 1) & (dist == 10),
        "second_long": (down == 2) & (dist >= cfg.long_distance),
        "second_short": (down == 2) & (dist <= cfg.short_distance),
        "third_long": (down == 3) & (dist >= cfg.long_distance),
        "third_short": (down == 3) & (dist <= cfg.short_distance),
        "red_zone": calls["yards_to_goal"] <= 20,
        "trailing": margin <= -cfg.lead_margin,
        "leading": margin >= cfg.lead_margin,
    }
    return {b: masks[b] for b in cfg.tendency_buckets}


def _efficiency_columns(calls: pd.DataFrame) -> pd.DataFrame:
    """n_/sum_ columns for run and pass EPA / success, one row per call."""
    run, pas = calls["is_pass"] == 0, calls["is_pass"] == 1
    has_epa = calls["EPA"].notna()
    epa = calls["EPA"].fillna(0.0)
    succ = calls["success"].fillna(0)
    return pd.DataFrame({
        "n_run_epa": (run & has_epa).astype(int), "sum_run_epa": epa.where(run, 0.0),
        "n_pass_epa": (pas & has_epa).astype(int), "sum_pass_epa": epa.where(pas, 0.0),
        "n_run_sr": run.astype(int), "sum_run_sr": succ.where(run, 0),
        "n_pass_sr": pas.astype(int), "sum_pass_sr": succ.where(pas, 0),
    }, index=calls.index)


def _aggregate_to_team_games(per_call: pd.DataFrame, team_col: str, calls: pd.DataFrame,
                             team_games: pd.DataFrame) -> pd.DataFrame:
    """Sum per-call n_/sum_ columns to (game_id, team), left-joined onto the full schedule so
    games without play-by-play still occupy their game_number slot (with zero counts)."""
    per_call = per_call.assign(game_id=calls["game_id"].to_numpy(), team=calls[team_col].to_numpy())
    agg = per_call.groupby(["game_id", "team"]).sum().reset_index()
    tg = team_games[["game_id", "team", "season", "game_number"]].merge(agg, on=["game_id", "team"], how="left")
    count_cols = [c for c in agg.columns if c.startswith(("n_", "sum_"))]
    tg[count_cols] = tg[count_cols].fillna(0)
    return tg


def team_tendencies(calls: pd.DataFrame, team_games: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Per (game_id, offense): prior_/std_ pass rate for every bucket, plus the offense's own
    run/pass EPA and success rate."""
    calls = calls[~calls["garbage_time"]]
    cols = {}
    for bucket, mask in bucket_masks(calls, cfg).items():
        cols[f"n_pr_{bucket}"] = mask.astype(int)
        cols[f"sum_pr_{bucket}"] = (mask & (calls["is_pass"] == 1)).astype(int)
    per_call = pd.concat([pd.DataFrame(cols, index=calls.index), _efficiency_columns(calls)], axis=1)

    tg = _aggregate_to_team_games(per_call, "pos_team", calls, team_games)
    metrics = [f"pr_{b}" for b in cfg.tendency_buckets]
    rates = prior_and_season_to_date(tg, metrics, cfg.shrinkage["prior_season_to_league"],
                                     cfg.shrinkage["season_to_date_to_prior"])
    eff = prior_and_season_to_date(tg, EFFICIENCY_METRICS, cfg.shrinkage["prior_season_to_league"],
                                   cfg.shrinkage["defense_to_prior"])
    eff = eff.rename(columns=lambda c: c if c in ("game_id", "team", "season", "game_number") else f"off_{c}")
    out = rates.merge(eff.drop(columns=["season", "game_number"]), on=["game_id", "team"])
    keep = ["game_id", "team", "std_n_pr_overall"] + [
        c for c in out.columns if c.startswith(("prior_", "std_", "off_prior_", "off_std_"))
        and not c.startswith(("std_n_", "off_std_n_"))
    ]
    return out[keep].rename(columns={"std_n_pr_overall": "std_calls_so_far"})


def opponent_defense(calls: pd.DataFrame, team_games: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Per (game_id, defense): run/pass EPA and success rate allowed, and the pass rate
    offenses have used against it, prior season + season-to-date."""
    calls = calls[~calls["garbage_time"]]
    per_call = _efficiency_columns(calls).assign(
        n_pass_rate_faced=1, sum_pass_rate_faced=calls["is_pass"].to_numpy()
    )
    tg = _aggregate_to_team_games(per_call, "def_pos_team", calls, team_games)
    metrics = EFFICIENCY_METRICS + ["pass_rate_faced"]
    d = prior_and_season_to_date(tg, metrics, cfg.shrinkage["prior_season_to_league"],
                                 cfg.shrinkage["defense_to_prior"])
    keep = [c for c in d.columns if c.startswith(("prior_", "std_")) and not c.startswith("std_n_")]
    return d[["game_id", "team"] + keep].rename(
        columns={"team": "def_pos_team", **{c: f"def_{c}" for c in keep}}
    )
