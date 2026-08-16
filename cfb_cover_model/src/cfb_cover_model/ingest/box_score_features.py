"""Port of R Scripts/Full_CFB_Game_Outcome_Historical.R's box-score cleanup and engineered-
ratio block (section 1.8-1.9 of the ported spec). Input: cfbd_client.fetch_game_team_stats's
one-row-per-(game,team) output, which - unlike cfbfastR's R-side flattening - does not yet
have the opponent's stats merged in as "_allowed" columns. That merge happens here first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fields that arrive as "X-Y" strings and must be split into two numeric columns. Verified
# against the real historical CSV (see docs/api_ingestion.md): R's naming is NOT simply
# "{offense_name}"/"{offense_name}_allowed" for every field - completion_attempts and
# total_penalties_yards use different names on the allowed/defensive side entirely, not
# just an "_allowed" suffix, so each suffix gets its own explicit (num, den) pair.
_EFF_SPLITS = {
    "third_down_eff": {"": ("third_down_conversion", "third_down_attempts"), "_allowed": ("third_down_conversion_allowed", "third_down_attempts_allowed")},
    "fourth_down_eff": {"": ("fourth_down_conversion", "fourth_down_attempts"), "_allowed": ("fourth_down_conversion_allowed", "fourth_down_attempts_allowed")},
    "total_penalties_yards": {"": ("total_penalties", "penalty_yards"), "_allowed": ("penalties_allowed", "penalty_yards_allowed")},
    "completion_attempts": {"": ("completions", "attempted_passes"), "_allowed": ("completions_against", "completion_attempts_against")},
}


def add_allowed_columns(stats: pd.DataFrame) -> pd.DataFrame:
    """R's cfbd_game_team_stats() already returns each team's row merged with their
    opponent's stats as "_allowed" columns - the raw CFBD API doesn't, so we do the
    self-join here: for each (game_id, team) row, attach the other team's row in the same
    game, every stat renamed with an "_allowed" suffix. Games without exactly two teams in
    the pull (an incomplete fetch) are skipped rather than guessed at."""
    id_cols = {"game_id", "team", "home_away", "conference", "week", "year"}
    stat_cols = [c for c in stats.columns if c not in id_cols]

    rows = []
    by_game = {gid: g for gid, g in stats.groupby("game_id")}
    for game_id, g in by_game.items():
        if len(g) != 2:
            continue  # incomplete pull for this game - skip rather than guess
        for i in range(2):
            own = g.iloc[i]
            opp = g.iloc[1 - i]
            row = own.to_dict()
            for c in stat_cols:
                row[f"{c}_allowed"] = opp[c]
            rows.append(row)
    return pd.DataFrame(rows)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_possession_time(series: pd.Series) -> pd.Series:
    """"MM:SS" string -> decimal minutes (MM + SS/60), matching R's regex-based parse."""
    parts = series.astype(str).str.extract(r"^(\d+):(\d+)$")
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    return minutes + seconds / 60


def clean_box_score(raw_stats: pd.DataFrame) -> pd.DataFrame:
    """Full port of R section 1.8: split compound fields, parse possession_time, coerce
    to numeric. Expects raw_stats to already have "_allowed" columns (see
    add_allowed_columns)."""
    df = raw_stats.copy()

    for eff_col, suffix_map in _EFF_SPLITS.items():
        for suffix, (num_name, den_name) in suffix_map.items():
            col = f"{eff_col}{suffix}"
            if col not in df.columns:
                continue
            parts = df[col].astype(str).str.split("-", n=1, expand=True)
            df[num_name] = _to_numeric(parts[0])
            df[den_name] = _to_numeric(parts[1]) if parts.shape[1] > 1 else np.nan
            df = df.drop(columns=[col])

    if "possession_time" in df.columns:
        df["possession_time"] = _parse_possession_time(df["possession_time"])
    if "possession_time_allowed" in df.columns:
        df["possession_time_allowed"] = _parse_possession_time(df["possession_time_allowed"])

    non_numeric = {"game_id", "team", "home_away", "conference", "week", "year"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = _to_numeric(df[col])

    df[[c for c in df.columns if c not in non_numeric]] = df[
        [c for c in df.columns if c not in non_numeric]
    ].fillna(0)

    return df


def add_engineered_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Port of R section 1.9's mutate() block - engineered ratio/margin columns computed
    from the box score. Divisions that hit a 0 denominator produce NaN here (matching R's
    silent NaN/Inf-producing behavior), left for the caller's fill policy to resolve."""
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["third_down_percentage_offense"] = df["third_down_conversion"] / df["third_down_attempts"]
        df["fourth_down_percentage_offense"] = df["fourth_down_conversion"] / df["fourth_down_attempts"]

        df["pressure_percentage"] = df["qb_hurries"] / df["completion_attempts_against"]
        df["sack_percentage"] = df["sacks"] / df["completion_attempts_against"]

        df["pressure_percentage_allowed"] = df["qb_hurries_allowed"] / df["attempted_passes"]
        df["sack_percentage_allowed"] = df["sacks_allowed"] / df["attempted_passes"]

        df["interception_rate_offense"] = df["passes_intercepted"] / df["attempted_passes"]
        df["intercetpion_rate_defense"] = df["interceptions"] / df["completion_attempts_against"]

        df["point_differential"] = df["points"] - df["points_allowed"]
        df["possession_time_difference"] = df["possession_time"] - df["possession_time_allowed"]
        df["turnover_margin"] = df["turnovers"] - df["turnovers_allowed"]
        df["penalty_yard_margin"] = df["penalty_yards"] - df["penalty_yards_allowed"]

        df["total_plays"] = df["rushing_attempts"] + df["attempted_passes"]
        df["rush_percentage"] = df["rushing_attempts"] / df["total_plays"]
        df["yards_per_play"] = df["total_yards"] / df["total_plays"]

        df["total_plays_against"] = df["rushing_attempts_allowed"] + df["completion_attempts_against"]
        df["rush_percentage_against"] = df["rushing_attempts_allowed"] / df["total_plays_against"]
        df["yards_per_play_allowed"] = df["total_yards_allowed"] / df["total_plays_against"]

    return df


def consensus_or_average_spread(betting_lines: pd.DataFrame) -> pd.DataFrame:
    """Port of R section 1.3: prefer the "consensus" provider's line; where no consensus
    line exists for a game, average across whatever providers are available."""
    consensus = betting_lines.loc[betting_lines["provider"].str.lower() == "consensus"].copy()
    consensus = consensus[["game_id", "home_team", "away_team", "spread", "over_under", "formatted_spread"]]

    fallback = (
        betting_lines.groupby(["game_id", "home_team", "away_team"])
        .agg(
            spread=("spread", "mean"),
            over_under=("over_under", "mean"),
            formatted_spread=("formatted_spread", "first"),
        )
        .reset_index()
    )
    fallback_only = fallback.merge(
        consensus[["game_id", "home_team", "away_team"]], on=["game_id", "home_team", "away_team"], how="left", indicator=True
    )
    fallback_only = fallback_only[fallback_only["_merge"] == "left_only"].drop(columns="_merge")

    return pd.concat([consensus, fallback_only], ignore_index=True)
