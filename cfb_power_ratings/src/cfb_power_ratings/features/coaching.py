"""Preseason coaching features: who's the head coach entering season t (known before kickoff),
their tenure at that school, their career win% entering the season (built only from seasons
strictly before t -- never that season's own games), and whether this is a coaching change
year. Ported conceptually from cfb_win_total_model's coaching.py category against this
project's own simpler feature set (no SP+ inputs -- this project doesn't use them elsewhere).

`coaches` has one row per coach per school-STINT (a mid-season change produces two rows for
that season). The "preseason" coach for a (school, season) is the row with the earliest
`hire_date` that season -- the one who started it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_power_ratings.database import run_query


def build_coaching_features(engine, seasons: list[int]) -> pd.DataFrame:
    coaches = run_query(
        "SELECT first_name, last_name, hire_date, school, season, games, wins, losses FROM coaches",
        engine=engine,
    )
    if coaches.empty:
        return pd.DataFrame(columns=["team", "season", "coach_tenure_years", "coach_career_win_pct_prior", "coaching_change"])

    coaches["coach_name"] = (coaches["first_name"].fillna("") + " " + coaches["last_name"].fillna("")).str.strip()
    coaches = coaches[coaches["coach_name"] != ""]

    # Career win% entering each season -- cumulative BEFORE that season only (cumsum through
    # season t minus season t's own contribution), same "prior seasons only" discipline every
    # sibling project's leakage rules require.
    season_totals = (
        coaches.groupby(["coach_name", "season"], as_index=False)
        .agg(season_games=("games", "sum"), season_wins=("wins", "sum"))
        .sort_values(["coach_name", "season"])
    )
    season_totals["career_games_prior"] = (
        season_totals.groupby("coach_name")["season_games"].cumsum() - season_totals["season_games"]
    )
    season_totals["career_wins_prior"] = (
        season_totals.groupby("coach_name")["season_wins"].cumsum() - season_totals["season_wins"]
    )
    season_totals["coach_career_win_pct_prior"] = (
        season_totals["career_wins_prior"] / season_totals["career_games_prior"].replace(0, np.nan)
    )

    # Primary (season-starting) coach per (school, season): earliest hire_date that season.
    coaches_sorted = coaches.sort_values(["school", "season", "hire_date"], na_position="last")
    primary = coaches_sorted.groupby(["school", "season"], as_index=False).first()
    primary = primary.rename(columns={"school": "team"})[["team", "season", "coach_name", "hire_date"]]
    primary["coach_tenure_years"] = primary["season"] - pd.to_datetime(primary["hire_date"], errors="coerce").dt.year

    primary = primary.sort_values(["team", "season"])
    prev_coach = primary.groupby("team")["coach_name"].shift(1)
    primary["coaching_change"] = (primary["coach_name"] != prev_coach).astype(int)
    # First season a school appears has no prior coach to compare against -- not a "change",
    # just missing history; mark as NaN (unknown) rather than a false-positive change.
    first_season_mask = primary.groupby("team")["season"].transform("min") == primary["season"]
    primary.loc[first_season_mask, "coaching_change"] = np.nan

    out = primary.merge(season_totals[["coach_name", "season", "coach_career_win_pct_prior"]], on=["coach_name", "season"], how="left")
    out = out[out["season"].isin(seasons)]
    return out[["team", "season", "coach_tenure_years", "coach_career_win_pct_prior", "coaching_change"]]
