"""Preseason feature: how old/experienced is the roster, tested two independent ways.

`team_rosters.year` is meant to be an eligibility class (1=FR..4=SR, 5/6=grad/medical) but is
significantly corrupted for older seasons -- verified live: in 2015, 81.5% of rows have `year`
equal to the season itself (e.g. 2015) rather than a plausible class value; this improves over
time (2026 is clean). It's also the *only* class-related field CFBD's roster endpoint returns at
all -- there's no separate age/eligibility field to fall back to.

Two signals, built independently so either can be kept or dropped on its own merits after
retraining (see docs/methodology.md for the honest before/after comparison):

1. `class_avg`/`class_valid_row_share` -- straight from `year`, but filtered to plausible rows
   (1-6, excluding the season-equals-year corruption pattern) and gated to NaN when too few
   valid rows exist for a team-season to trust the average.
2. `avg_roster_experience`/`veteran_roster_share` -- a `year`-independent alternative: how many
   prior seasons a given `athlete_id` has appeared in `team_rosters` at all (any team, so a
   transfer's prior experience still counts), which `athlete_id` (100% populated, verified) makes
   possible without ever touching the corrupted field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_power_ratings.database import run_query

MIN_CLASS_YEAR = 1
MAX_CLASS_YEAR = 6
VETERAN_TENURE_THRESHOLD = 3


def _build_class_features(engine, seasons: list[int], min_valid_class_rows: int) -> pd.DataFrame:
    roster = run_query(
        "SELECT season, team, athlete_id, year FROM team_rosters WHERE season IN :seasons",
        params={"seasons": tuple(seasons)}, engine=engine,
    )
    if roster.empty:
        return pd.DataFrame(columns=["team", "season", "class_avg", "class_valid_row_share"])

    is_valid = (
        (roster["year"] >= MIN_CLASS_YEAR)
        & (roster["year"] <= MAX_CLASS_YEAR)
        & (roster["year"] != roster["season"])
    )
    roster["valid_class"] = roster["year"].where(is_valid)

    out = roster.groupby(["team", "season"]).agg(
        class_avg=("valid_class", "mean"),
        n_total=("athlete_id", "size"),
        n_valid=("valid_class", "count"),
    ).reset_index()
    out["class_valid_row_share"] = out["n_valid"] / out["n_total"]
    out.loc[out["n_valid"] < min_valid_class_rows, "class_avg"] = np.nan

    return out[["team", "season", "class_avg", "class_valid_row_share"]]


def _build_tenure_features(engine, seasons: list[int], tenure_lookback_seasons: int) -> pd.DataFrame:
    lookback_start = min(seasons) - tenure_lookback_seasons
    all_rosters = run_query(
        "SELECT season, team, athlete_id FROM team_rosters WHERE season BETWEEN :start AND :end",
        params={"start": lookback_start, "end": max(seasons)}, engine=engine,
    )
    if all_rosters.empty:
        return pd.DataFrame(columns=["team", "season", "avg_roster_experience", "veteran_roster_share"])

    # Every season a given athlete_id appears in *any* team's roster -- a transfer's prior
    # experience still counts, since this measures general football experience, not tenure at
    # one specific school (roster_turnover.py already covers the school-continuity angle).
    athlete_seasons: dict[str, set[int]] = all_rosters.groupby("athlete_id")["season"].apply(set).to_dict()

    target = all_rosters[all_rosters["season"].isin(seasons)].copy()
    target["tenure"] = target.apply(
        lambda row: sum(1 for s in athlete_seasons.get(row["athlete_id"], ()) if s < row["season"]),
        axis=1,
    )

    out = target.groupby(["team", "season"]).agg(
        avg_roster_experience=("tenure", "mean"),
        veteran_roster_share=("tenure", lambda s: (s >= VETERAN_TENURE_THRESHOLD).mean()),
    ).reset_index()
    return out


def build_roster_experience_features(
    engine, seasons: list[int], min_valid_class_rows: int = 15, tenure_lookback_seasons: int = 5
) -> pd.DataFrame:
    """One row per (team, season): class_avg, class_valid_row_share, avg_roster_experience,
    veteran_roster_share. All computed from season t's own roster -- preseason-safe, same
    justification roster_turnover.py already uses for reading season-t team_rosters directly."""
    class_features = _build_class_features(engine, seasons, min_valid_class_rows)
    tenure_features = _build_tenure_features(engine, seasons, tenure_lookback_seasons)
    return class_features.merge(tenure_features, on=["team", "season"], how="outer")
