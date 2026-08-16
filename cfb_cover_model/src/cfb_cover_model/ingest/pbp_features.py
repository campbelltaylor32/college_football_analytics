"""Port of the offense_epa/defense_epa summarise() blocks from
../R Scripts/Full_CFB_Game_Outcome_Historical.R (section 1.10 of the ported spec) -
EPA/success-rate/explosiveness/first-down/drive-percentage aggregation from play-by-play
data, per (team, week, year).

Deliberately does NOT replicate the R source's Offense_EPA_per_Run/Defense_EPA_per_Run
duplicate-assignment bug (that column silently held pass-play EPA, not run-play EPA, in
the historical data) - this computes both *_EPA_per_Run and *_EPA_per_Pass correctly. This
doesn't affect the trained model either way: config/data.yaml's known_bad_base_stats
already excludes both buggy-named historical columns from candidate features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUN_PLAY_TYPES = {"Rush", "Rushing Touchdown"}
PASS_PLAY_TYPES = {
    "Sack", "Passing Touchdown", "Interception Return",
    "Interception Return Touchdown", "Pass Incompletion", "Pass Reception",
}
VALID_SNAP_TYPES = RUN_PLAY_TYPES | PASS_PLAY_TYPES
EXPLOSIVE_YARDS_THRESHOLD = 20


def _success(row: pd.Series) -> bool:
    """Standard down-and-distance "success" definition (Football Outsiders / SP+
    convention, which cfbfastR's own `success` column follows): a play succeeds if it
    gains at least 50% of yards-to-go on 1st down, 70% on 2nd down, or 100% (i.e.
    converts) on 3rd/4th down."""
    down, distance, gained = row["down"], row["distance"], row["yards_gained"]
    if pd.isna(down) or pd.isna(distance) or distance <= 0:
        return False
    if down == 1:
        return gained >= 0.5 * distance
    if down == 2:
        return gained >= 0.7 * distance
    return gained >= distance  # 3rd/4th down: must convert


def add_success_column(plays: pd.DataFrame) -> pd.DataFrame:
    plays = plays.copy()
    plays["success"] = plays.apply(_success, axis=1)
    return plays


def _aggregate_one_side(plays: pd.DataFrame, drives: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """team_col is "pos_team" for offense or "def_pos_team" for defense; prefix is
    "Offense"/"Total_Offense" or "Defense"/"Total_Defense" matching the R column names."""
    valid = plays[plays["play_type"].isin(VALID_SNAP_TYPES)].copy()
    is_run = valid["play_type"].isin(RUN_PLAY_TYPES)
    is_pass = valid["play_type"].isin(PASS_PLAY_TYPES)
    is_explosive = valid["yards_gained"] >= EXPLOSIVE_YARDS_THRESHOLD
    is_first_down = valid["down"] == 1

    valid["_is_run"] = is_run
    valid["_is_pass"] = is_pass
    valid["_is_explosive"] = is_explosive
    valid["_is_first_down"] = is_first_down

    rows = []
    for (team, week, year), g in valid.groupby([team_col, "week", "year"]):
        total_plays = g["id_play"].nunique()
        run_plays = g.loc[g["_is_run"], "id_play"].nunique()
        pass_plays = g.loc[g["_is_pass"], "id_play"].nunique()
        first_down_plays = g.loc[g["_is_first_down"], "id_play"].nunique()
        first_down_pass_plays = g.loc[g["_is_first_down"] & g["_is_pass"], "id_play"].nunique()

        total_epa = g["EPA"].sum(skipna=True)
        epa_run = g.loc[g["_is_run"], "EPA"].sum(skipna=True)
        epa_pass = g.loc[g["_is_pass"], "EPA"].sum(skipna=True)

        total_success = g["success"].sum()
        run_success = g.loc[g["_is_run"], "success"].sum()
        pass_success = g.loc[g["_is_pass"], "success"].sum()
        first_down_success = g.loc[g["_is_first_down"], "success"].sum()
        first_down_run_success = g.loc[g["_is_first_down"] & g["_is_run"], "success"].sum()
        first_down_pass_success = g.loc[g["_is_first_down"] & g["_is_pass"], "success"].sum()

        total_explosives = g.loc[g["_is_explosive"], "id_play"].nunique()
        run_explosives = g.loc[g["_is_explosive"] & g["_is_run"], "id_play"].nunique()
        pass_explosives = g.loc[g["_is_explosive"] & g["_is_pass"], "id_play"].nunique()

        third_down_distance = g.loc[g["down"] == 3, "distance"]
        avg_3rd_down_distance = third_down_distance.mean() if len(third_down_distance) else np.nan

        drive_side_col = "offense" if team_col == "pos_team" else "defense"
        team_drives = drives[
            (drives[drive_side_col] == team) & (drives["week"] == week) & (drives["year"] == year)
        ]
        n_drives = team_drives["drive_id"].nunique()
        n_scoring_drives = team_drives.loc[team_drives["scoring"] == True, "drive_id"].nunique()  # noqa: E712
        n_td_drives = team_drives.loc[team_drives["drive_result"] == "TD", "drive_id"].nunique()

        with np.errstate(divide="ignore", invalid="ignore"):
            row = {
                team_col: team, "week": week, "year": year,
                f"Total_{prefix}_Drives": n_drives,
                f"Total_{prefix}_Plays": total_plays,
                f"{prefix}_Total_Run_Plays": run_plays,
                f"{prefix}_Total_Pass_Plays": pass_plays,
                f"{prefix}_Pass_Rate": pass_plays / total_plays if total_plays else np.nan,
                f"{prefix}_Run_Rate": run_plays / total_plays if total_plays else np.nan,
                f"{prefix}_first_down_pass_rate": first_down_pass_plays / first_down_plays if first_down_plays else np.nan,
                f"{prefix}_Avg_3rd_Down_Distance": avg_3rd_down_distance,
                f"Total_{prefix}_EPA": total_epa,
                f"{prefix}_EPA_per_Play": total_epa / total_plays if total_plays else np.nan,
                f"Total_{prefix}_Success": total_success,
                f"{prefix}_Success_Rate": total_success / total_plays if total_plays else np.nan,
                f"Total_{prefix}_EPA_Run": epa_run,
                f"{prefix}_EPA_per_Run": epa_run / run_plays if run_plays else np.nan,
                f"Total_{prefix}_Run_Success": run_success,
                f"{prefix}_Run_Success_Rate": run_success / run_plays if run_plays else np.nan,
                f"Total_{prefix}_EPA_Pass": epa_pass,
                f"{prefix}_EPA_per_Pass": epa_pass / pass_plays if pass_plays else np.nan,
                f"Total_{prefix}_Pass_Success": pass_success,
                f"{prefix}_Pass_Success_Rate": pass_success / pass_plays if pass_plays else np.nan,
                f"Total_{prefix}_Explosives": total_explosives,
                f"Total_{prefix}_Explosive_Rate": total_explosives / total_plays if total_plays else np.nan,
                f"Total_{prefix}_Run_Explosives": run_explosives,
                f"Total_{prefix}_Run_Explosive_Rate": run_explosives / run_plays if run_plays else np.nan,
                f"Total_{prefix}_Pass_Explosives": pass_explosives,
                f"Total_{prefix}_Pass_Explosive_Rate": pass_explosives / pass_plays if pass_plays else np.nan,
                f"{prefix}_First_Down_Success": first_down_success,
                f"{prefix}_First_Down_Success_Rate": first_down_success / first_down_plays if first_down_plays else np.nan,
                f"{prefix}_First_Down_Run_Success": first_down_run_success,
                f"{prefix}_First_Down_Run_Success_Rate": (
                    first_down_run_success / g.loc[g["_is_first_down"] & g["_is_run"], "id_play"].nunique()
                    if g.loc[g["_is_first_down"] & g["_is_run"], "id_play"].nunique() else np.nan
                ),
                f"{prefix}_First_Down_Pass_Success": first_down_pass_success,
                f"{prefix}_First_Down_Pass_Success_Rate": first_down_pass_success / first_down_pass_plays if first_down_pass_plays else np.nan,
                f"Total_{prefix}_Scoring_Drives": n_scoring_drives,
                f"Total_{prefix}_Touchdown_Drives": n_td_drives,
                f"{prefix}_Scoring_Drive_Percentage": n_scoring_drives / n_drives if n_drives else np.nan,
                f"{prefix}_Touchdown_Drive_Percentage": n_td_drives / n_drives if n_drives else np.nan,
            }
        rows.append(row)

    return pd.DataFrame(rows).rename(columns={team_col: "team"})


def compute_epa_features(plays: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (team, week, year) with both Offense_* and Defense_* columns -
    the equivalent of R's `merge(offense_epa, defense_epa, by=c("team","week","year"))`."""
    plays = add_success_column(plays)
    offense = _aggregate_one_side(plays, drives, "pos_team", "Offense")
    defense = _aggregate_one_side(plays, drives, "def_pos_team", "Defense")
    return offense.merge(defense, on=["team", "week", "year"], how="outer")
