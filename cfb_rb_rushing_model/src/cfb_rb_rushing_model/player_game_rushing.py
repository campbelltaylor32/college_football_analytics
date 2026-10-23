"""Raw player-game rushing table, built fresh from `plays` -- no such table exists anywhere
else in the DB or the legacy CSV pipeline. Keyed by (rusher_player_name, pos_team, season,
week, game_id) -- name-keyed, not athlete_id-keyed. Resolution to athlete_id happens in
player_resolution.py, deliberately kept separate so this module has no roster-schema
knowledge and can be unit-tested independently.

Verified live (see docs/assumptions_and_limitations.md): `yards_gained` is 0% NULL on rush
plays across every season 2014-2025, while `yds_rushed` has a small but nonzero NULL rate
(up to 43 rows in 2024) -- yards_gained is used for rushing_yards for this reason.

`plays.rusher_player_name` is mostly NULL from 2025 week 9 on (99% of 2026): cfbfastR doesn't
parse the newer play-text format (`#28 S.Alexander rush middle for 7 yards ...`).
recover_rusher_names() fills those rows from play_text before aggregation, matching the
jersey number + first initial + last name to `team_rosters`, so everything downstream still
sees a full roster name.
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.player_resolution import normalize_name
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

RAW_COLUMNS = [
    "rusher_player_name", "pos_team", "season", "week", "game_id",
    "carries", "rushing_yards", "yards_per_carry", "success_rate",
    "explosive_runs", "explosive_run_rate", "stuffed_run_rate",
    "red_zone_carries", "avg_epa_per_rush", "first_down_rate",
]


def _pull_rush_plays(engine: Engine, seasons: list[int], rush_play_types: list[str]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    type_placeholders = ", ".join(f":t{i}" for i in range(len(rush_play_types)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    params.update({f"t{i}": t for i, t in enumerate(rush_play_types)})
    sql = f"""
        SELECT rusher_player_name, play_text, pos_team, season, week, game_id,
               yards_gained, success, rz_play, stuffed_run, epa, down, distance
        FROM plays
        WHERE season IN ({season_placeholders})
          AND play_type IN ({type_placeholders})
          AND pos_team IS NOT NULL
    """
    return run_query(sql, params=params, engine=engine)


def pull_jersey_rosters(engine: Engine, seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["season", "team", "jersey", "first_name", "last_name"])
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    sql = f"""
        SELECT season, team, jersey, first_name, last_name
        FROM team_rosters WHERE season IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


# "(11:48) Shotgun #7 A.Flores rush middle for 19 yards ..." / "#6 G.Modest Jr. rush ..."
_JERSEY_RE = re.compile(r"#(\d{1,2})\s+([A-Z][A-Za-z]?)\.\s?(.+?)\s+(?:rush|run|scramble)\b")
# "Angel Flores 50 Yd Run (Caden VanHuis PAT blocked)" / "Devin Mockobee run for 3 yds"
_FULL_NAME_RE = re.compile(r"^(?:\(\d{1,2}:\d{2}\)\s*)?([A-Z][\w'.\-]+(?:\s+[A-Z][\w'.\-]+)+?)\s+(?:\d+\s+Yd\s+Run|run\s+for)\b")

_DEFAULT_SUFFIXES = ["jr", "sr", "ii", "iii", "iv", "v"]


def recover_rusher_names(raw: pd.DataFrame, rosters: pd.DataFrame, suffixes: list[str] = _DEFAULT_SUFFIXES) -> pd.DataFrame:
    """Fill NULL rusher_player_name from play_text, then put every name in the roster's
    spelling. Jersey-format plays are matched to the roster on (season, team, jersey) and
    accepted only when the first initial and the normalized last name agree (two players can
    share a number, e.g. an RB and an LB); if the jersey doesn't match, a unique (season, team,
    initial, last name) roster row is used. Full-name formats are taken as written, and
    anything else stays NULL.

    The canonicalization matters once recovered and cfbfastR-parsed rows mix within a game:
    "Marcellous Hawkins" (recovered), "Marcellous Hawkins Jr." and "M. Hawkins Jr." are one
    player and must aggregate to one player-game row, not three."""
    out = raw.copy()
    missing = out["rusher_player_name"].isna() & out["play_text"].notna()

    r = rosters.dropna(subset=["last_name"]).copy()
    r["first_name"] = r["first_name"].fillna("")
    r["full_name"] = (r["first_name"] + " " + r["last_name"]).str.strip()
    r["initial"] = r["first_name"].str[:1].str.upper()
    r["norm_last"] = r["last_name"].apply(lambda n: normalize_name(n, suffixes))
    r["jersey"] = pd.to_numeric(r["jersey"], errors="coerce")

    by_jersey: dict = {}
    for row in r.dropna(subset=["jersey"]).itertuples():
        by_jersey.setdefault((row.season, row.team, int(row.jersey)), []).append(row)
    by_name: dict = {}
    by_full: dict = {}
    for row in r.itertuples():
        by_name.setdefault((row.season, row.team, row.initial, row.norm_last), []).append(row.full_name)
        by_full.setdefault((row.season, row.team, normalize_name(row.full_name, suffixes)), set()).add(row.full_name)

    def recover(season, team, text):
        m = _JERSEY_RE.search(text)
        if m:
            jersey, initial, last = int(m.group(1)), m.group(2)[0].upper(), normalize_name(m.group(3).replace(",", " "), suffixes)
            hits = {c.full_name for c in by_jersey.get((season, team, jersey), [])
                    if c.initial == initial and c.norm_last == last}
            if len(hits) != 1:
                hits = set(by_name.get((season, team, initial, last), []))
            return hits.pop() if len(hits) == 1 else None
        m = _FULL_NAME_RE.search(text)
        return m.group(1) if m else None

    def canonical(season, team, name):
        hits = by_full.get((season, team, normalize_name(name, suffixes)), set())
        if len(hits) == 1:
            return next(iter(hits))
        tokens = normalize_name(name, suffixes).split()
        if len(tokens) >= 2 and len(tokens[0]) <= 2:  # abbreviated first name: "M. Hawkins Jr."
            abbrev = set(by_name.get((season, team, tokens[0][0].upper(), " ".join(tokens[1:])), []))
            if len(abbrev) == 1:
                return abbrev.pop()
        return name

    if missing.any():
        sub = out.loc[missing, ["season", "pos_team", "play_text"]]
        out.loc[missing, "rusher_player_name"] = [recover(s, t, x) for s, t, x in sub.itertuples(index=False)]
    named = out["rusher_player_name"].notna()
    keys = out.loc[named, ["season", "pos_team", "rusher_player_name"]].drop_duplicates()
    mapping = {k: canonical(*k) for k in keys.itertuples(index=False, name=None)}
    out.loc[named, "rusher_player_name"] = [mapping[k] for k in out.loc[named, ["season", "pos_team", "rusher_player_name"]].itertuples(index=False, name=None)]
    return out


def build_raw_player_game_rushing(
    engine: Engine, seasons: list[int], rush_play_types: list[str], explosive_run_yard_threshold: int
) -> pd.DataFrame:
    """One row per (rusher_player_name, pos_team, season, week, game_id). `first_down_rate`
    is a documented approximation (yards_gained >= distance) -- `plays` has no explicit
    first-down-achieved flag, so this proxy slightly overcounts penalty-aided first downs and
    undercounts spot-adjustment edge cases (see docs/data_dictionary.md)."""
    raw = _pull_rush_plays(engine, seasons, rush_play_types)
    if raw.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    raw = recover_rusher_names(raw, pull_jersey_rosters(engine, seasons))
    raw = raw[raw["rusher_player_name"].notna()].copy()

    raw["is_explosive"] = raw["yards_gained"] >= explosive_run_yard_threshold
    raw["is_first_down"] = raw["yards_gained"] >= raw["distance"].fillna(10)

    grouped = raw.groupby(["rusher_player_name", "pos_team", "season", "week", "game_id"], as_index=False).agg(
        carries=("yards_gained", "count"),
        rushing_yards=("yards_gained", "sum"),
        success_rate=("success", "mean"),
        explosive_runs=("is_explosive", "sum"),
        stuffed_run_rate=("stuffed_run", "mean"),
        red_zone_carries=("rz_play", "sum"),
        avg_epa_per_rush=("epa", "mean"),
        first_down_rate=("is_first_down", "mean"),
    )
    grouped["yards_per_carry"] = grouped["rushing_yards"] / grouped["carries"]
    grouped["explosive_run_rate"] = grouped["explosive_runs"] / grouped["carries"]
    grouped["red_zone_carries"] = grouped["red_zone_carries"].astype(int)
    grouped["explosive_runs"] = grouped["explosive_runs"].astype(int)

    return grouped[RAW_COLUMNS]


def describe_features() -> list[dict]:
    base = {
        "source_table": "plays (rush/rushing-touchdown play_types only)",
        "source_season": "realized game outcome -- only ever used as raw material for rolling/lagged features, never a same-game feature",
        "category": "player_game_rushing",
        "known_before_kickoff": False,
        "missing_value_treatment": "n/a -- this is the raw realized table, not a feature table",
    }
    names = [
        ("carries", "Rush attempts in this player-game"),
        ("rushing_yards", "Sum of yards_gained across this player-game's rush attempts"),
        ("yards_per_carry", "rushing_yards / carries"),
        ("success_rate", "Mean of plays.success across this player-game's carries"),
        ("explosive_runs", "Count of carries with yards_gained >= config explosive_run_yard_threshold"),
        ("explosive_run_rate", "explosive_runs / carries"),
        ("stuffed_run_rate", "Mean of plays.stuffed_run across this player-game's carries"),
        ("red_zone_carries", "Count of carries flagged plays.rz_play"),
        ("avg_epa_per_rush", "Mean of plays.epa across this player-game's carries"),
        ("first_down_rate", "Approximation: share of carries with yards_gained >= distance (plays has no explicit first-down-achieved flag)"),
    ]
    return [{**base, "feature_name": n, "description": d, "transformation": "n/a", "expected_direction": "n/a"} for n, d in names]
