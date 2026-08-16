"""Name-to-roster resolution: maps `plays.rusher_player_name` (a free-text string, no stable
athlete_id) onto `team_rosters.athlete_id` for the configured position(s) (RB for this
project).

Deliberately done in pandas, not a SQL CONCAT(...)-join -- a naive join against the full
`plays` table has no supporting index and is slow at scale (confirmed during planning).
Instead: resolve only the small number of DISTINCT (name, team, season) combinations (a few
thousand rows per season, not hundreds of thousands), then merge the resolution back onto the
full row set by the caller.

Verified live during planning (2022-2024 seasons, decomposed by resolved position): of all
rush-play rows league-wide, ~68-71% exact-match to an RB, ~19% to a QB (scrambles/sneaks --
correctly excluded here, not a resolution failure), ~5% to another position, and only ~6-9%
fail to resolve to ANY position at all pre-normalization. The normalized fallback pass below
is expected to recover a meaningful share of that 6-9% (suffixes, accents, punctuation).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

RESOLUTION_COLUMNS = ["rusher_player_name", "pos_team", "season", "athlete_id", "roster_match_method"]


def normalize_name(name: str, suffixes_to_strip: list[str]) -> str:
    """Strip accents (Unicode NFKD), punctuation, and trailing generational suffixes;
    collapse whitespace; casefold. Used only for the fallback pass, after an exact match has
    already failed."""
    if not isinstance(name, str):
        return ""
    stripped_accents = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    no_punct = re.sub(r"[.'’]", "", stripped_accents)
    tokens = [t for t in re.split(r"\s+", no_punct.strip()) if t]
    suffix_set = {s.lower().rstrip(".") for s in suffixes_to_strip}
    tokens = [t for t in tokens if t.lower().rstrip(".") not in suffix_set]
    return " ".join(tokens).casefold()


def _pull_rosters(engine: Engine, seasons: list[int], positions: list[str] | None) -> pd.DataFrame:
    """positions=None pulls the full roster with no position filter -- used by
    data_validation.check_player_resolution_match_rate to compute the "resolved to ANY
    position" denominator, which separates genuine resolution failures from rows that
    correctly don't match the RB-only roster (QB scrambles, other positions' occasional
    carries)."""
    if not seasons:
        return pd.DataFrame(columns=["athlete_id", "team", "season", "position", "full_name"])
    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    pos_filter = ""
    if positions is not None:
        pos_placeholders = ", ".join(f":p{i}" for i in range(len(positions)))
        params.update({f"p{i}": p for i, p in enumerate(positions)})
        pos_filter = f"AND position IN ({pos_placeholders})"
    sql = f"""
        SELECT athlete_id, team, season, position, first_name, last_name
        FROM team_rosters
        WHERE season IN ({season_placeholders}) {pos_filter}
    """
    rosters = run_query(sql, params=params, engine=engine)
    rosters["full_name"] = (rosters["first_name"].fillna("") + " " + rosters["last_name"].fillna("")).str.strip()
    return rosters


def _dedupe_ambiguous(merged: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, set]:
    """A (name, team, season) key that resolves to >1 distinct athlete_id (two same-named
    players on one roster) is ambiguous -- flagged and excluded rather than guessed."""
    nunique = merged.dropna(subset=["athlete_id"]).groupby(key_cols)["athlete_id"].nunique()
    ambiguous_keys = set(nunique[nunique > 1].index)
    return merged, ambiguous_keys


def resolve_players(
    engine: Engine,
    distinct_names: pd.DataFrame,
    seasons: list[int],
    positions: list[str] | None,
    name_suffixes_to_strip: list[str],
) -> pd.DataFrame:
    """distinct_names: DataFrame with columns [rusher_player_name, pos_team, season]
    (typically already deduplicated by the caller). Returns one row per input row with
    athlete_id (nullable) and roster_match_method in {'exact','normalized','ambiguous','unmatched'}.
    positions=None matches against the full roster regardless of position (see _pull_rosters).
    """
    if distinct_names.empty:
        return pd.DataFrame(columns=RESOLUTION_COLUMNS)

    names = distinct_names.rename(columns={"rusher_player_name": "full_name", "pos_team": "team"}).copy()
    rosters = _pull_rosters(engine, seasons, positions)

    # --- Pass 1: exact match on (team, season, full_name) ---
    exact = names.merge(
        rosters[["team", "season", "full_name", "athlete_id"]], on=["team", "season", "full_name"], how="left"
    )
    exact, ambiguous_keys_exact = _dedupe_ambiguous(exact, ["team", "season", "full_name"])
    exact["roster_match_method"] = exact.apply(
        lambda r: "ambiguous" if (r["team"], r["season"], r["full_name"]) in ambiguous_keys_exact
        else ("exact" if pd.notna(r["athlete_id"]) else "unmatched"),
        axis=1,
    )
    exact.loc[exact["roster_match_method"] == "ambiguous", "athlete_id"] = None

    # --- Pass 2: normalized fallback, only for Pass-1 misses ---
    missed_mask = exact["roster_match_method"] == "unmatched"
    if missed_mask.any() and not rosters.empty:
        rosters = rosters.copy()
        rosters["norm_name"] = rosters["full_name"].apply(lambda n: normalize_name(n, name_suffixes_to_strip))

        missed = exact.loc[missed_mask, ["team", "season", "full_name"]].copy()
        missed["norm_name"] = missed["full_name"].apply(lambda n: normalize_name(n, name_suffixes_to_strip))

        norm_merge = missed.merge(
            rosters[["team", "season", "norm_name", "athlete_id"]], on=["team", "season", "norm_name"], how="left"
        )
        norm_merge, ambiguous_keys_norm = _dedupe_ambiguous(norm_merge, ["team", "season", "norm_name"])
        norm_merge["roster_match_method"] = norm_merge.apply(
            lambda r: "ambiguous" if (r["team"], r["season"], r["norm_name"]) in ambiguous_keys_norm
            else ("normalized" if pd.notna(r["athlete_id"]) else "unmatched"),
            axis=1,
        )
        norm_merge.loc[norm_merge["roster_match_method"] == "ambiguous", "athlete_id"] = None
        norm_merge = norm_merge.drop_duplicates(subset=["team", "season", "full_name"])

        norm_lookup = norm_merge.set_index(["team", "season", "full_name"])[["athlete_id", "roster_match_method"]]
        for (team, season, full_name), row in norm_lookup.iterrows():
            idx = exact[(exact["team"] == team) & (exact["season"] == season) & (exact["full_name"] == full_name)].index
            exact.loc[idx, "athlete_id"] = row["athlete_id"]
            exact.loc[idx, "roster_match_method"] = row["roster_match_method"]

    result = exact.rename(columns={"full_name": "rusher_player_name", "team": "pos_team"})
    return result[RESOLUTION_COLUMNS]


def match_rate_summary(resolved: pd.DataFrame) -> pd.DataFrame:
    """Per-season resolution-method breakdown, used by data_validation.check_player_resolution_match_rate."""
    if resolved.empty:
        return pd.DataFrame(columns=["season", "roster_match_method", "n", "share"])
    counts = resolved.groupby(["season", "roster_match_method"]).size().rename("n").reset_index()
    totals = counts.groupby("season")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts
