"""Preseason talent features: the raw team-talent composite (team_talent) plus a *corrected*
blue-chip ratio, porting the join-fix from
cfb_talent_distribution/build_corrected_blue_chip_ratio.R rather than the diluted version the
shared R pipeline still computes.

The shared pipeline's blue_chip_ratio divides by the FULL roster (walk-ons included, ~150
players) and only matches recruits via athlete_id, which cfb_talent_distribution found
understates real-world figures by roughly a third (Alabama 2021: 54% vs. the ~86% publicly
reported). The fix here: match roster players to a recruiting record via EITHER athlete_id OR
the roster's own recruit_ids link (union of both, since each method catches players the other
misses), and divide by the count of *matched* players only, not the full walk-on-inclusive
roster.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb_power_ratings.database import run_query


def _parse_recruit_ids(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    return [str(r) for r in raw if r not in (None, "", "0")]


def _corrected_blue_chip_ratio(engine, seasons: list[int], min_matched: int) -> pd.DataFrame:
    roster = run_query(
        "SELECT season AS year, team, athlete_id, recruit_ids FROM team_rosters WHERE season IN :seasons",
        params={"seasons": tuple(seasons)}, engine=engine,
    )
    if roster.empty:
        return pd.DataFrame(columns=["team", "season", "blue_chip_ratio", "n_matched_recruits", "avg_recruit_rating"])

    recruits = run_query("SELECT recruit_id, athlete_id, stars, rating FROM recruiting_players", engine=engine)
    by_athlete = (
        recruits.dropna(subset=["athlete_id"])
        .drop_duplicates("athlete_id")
        .set_index("athlete_id")[["stars", "rating"]]
    )
    by_recruit = recruits.drop_duplicates("recruit_id").set_index("recruit_id")[["stars", "rating"]]

    roster = roster.reset_index(drop=True)
    roster["roster_row"] = roster.index
    roster["recruit_ids_list"] = roster["recruit_ids"].apply(_parse_recruit_ids)

    stars_a = roster["athlete_id"].map(by_athlete["stars"])
    rating_a = roster["athlete_id"].map(by_athlete["rating"])

    # A roster row's recruit_ids list is usually 0 or 1 entries; take the best (max stars)
    # match across every id in the list, mirroring the R script's group_by(roster_row) max().
    exploded = roster[["roster_row", "recruit_ids_list"]].explode("recruit_ids_list")
    stars_b = exploded["recruit_ids_list"].map(by_recruit["stars"])
    rating_b = exploded["recruit_ids_list"].map(by_recruit["rating"])
    best_b = (
        pd.DataFrame({"roster_row": exploded["roster_row"], "stars_b": stars_b, "rating_b": rating_b})
        .groupby("roster_row").max()
    )

    def _best_of(a, b):
        return np.nan if pd.isna(a) and pd.isna(b) else np.nanmax([a, b])

    matched_stars = stars_a.combine(roster["roster_row"].map(best_b["stars_b"]), _best_of)
    matched_rating = rating_a.combine(roster["roster_row"].map(best_b["rating_b"]), _best_of)

    roster["stars"] = matched_stars
    roster["rating"] = matched_rating
    matched = roster.dropna(subset=["stars"])

    out = matched.groupby(["team", "year"]).agg(
        n_matched_recruits=("stars", "size"),
        blue_chip_ratio=("stars", lambda s: (s >= 4).sum() / len(s)),
        avg_recruit_rating=("rating", "mean"),
    ).reset_index().rename(columns={"year": "season"})

    out.loc[out["n_matched_recruits"] < min_matched, "blue_chip_ratio"] = np.nan
    return out


def build_talent_recruiting_features(engine, seasons: list[int], min_matched_recruits: int = 20) -> pd.DataFrame:
    """One row per (team, season): talent_composite, blue_chip_ratio, n_matched_recruits,
    avg_recruit_rating. All preseason-known for season `season` (talent/recruiting data is
    published well before kickoff)."""
    talent = run_query(
        "SELECT season, school AS team, talent AS talent_composite FROM team_talent WHERE season IN :seasons",
        params={"seasons": tuple(seasons)}, engine=engine,
    )
    bcr = _corrected_blue_chip_ratio(engine, seasons, min_matched_recruits)
    # outer, not left: team_talent can be empty for a season CFBD hasn't published its talent
    # composite for yet (e.g. 2026 as of this writing, verified live) while blue_chip_ratio is
    # still fully computable from team_rosters/recruiting_players alone -- a left-on-talent
    # merge would silently drop bcr's real rows too whenever talent has none.
    return talent.merge(bcr, on=["team", "season"], how="outer")
