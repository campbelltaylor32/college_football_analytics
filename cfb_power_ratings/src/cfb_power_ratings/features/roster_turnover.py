"""Preseason roster-turnover features: net transfer-portal talent gained/lost entering season
t. Upgrades cfb_win_total_model's roster-diff inference with the dedicated transfer-portal
endpoint cfb_transfer_portal_flow already validated (real origin/destination/rating per move,
not inferred from which names disappear from a roster) -- but that endpoint has 0 rows before
2021 (confirmed live and in cfb_transfer_portal_flow/README.md), so seasons before
`transfer_portal_start_season` fall back to the roster-diff method against `team_rosters`.

Both paths are preseason-safe: a transfer's portal entry and a roster's season-t composition
are both known before that season's games are played.
"""
from __future__ import annotations

import pandas as pd

from cfb_power_ratings.database import run_query


def _transfer_portal_features(client, season: int) -> pd.DataFrame:
    import cfbd

    api = cfbd.PlayersApi(client)
    entries = api.get_transfer_portal(year=season)
    rows = [
        {"origin": e.origin, "destination": e.destination, "rating": e.rating}
        for e in entries if e.origin and e.destination
    ]
    if not rows:
        return pd.DataFrame(columns=["team", "season", "transfers_in", "transfers_out", "net_transfer_rating"])
    df = pd.DataFrame(rows)

    incoming = df.groupby("destination").agg(transfers_in=("destination", "size"), rating_in=("rating", "sum"))
    outgoing = df.groupby("origin").agg(transfers_out=("origin", "size"), rating_out=("rating", "sum"))
    incoming.index.name = outgoing.index.name = "team"
    combined = incoming.join(outgoing, how="outer").fillna(0)
    combined["net_transfer_rating"] = combined["rating_in"] - combined["rating_out"]
    combined = combined.reset_index()
    combined["season"] = season
    return combined[["team", "season", "transfers_in", "transfers_out", "net_transfer_rating"]]


def _roster_diff_features(engine, season: int) -> pd.DataFrame:
    """Fallback for seasons before the transfer portal endpoint has data: infer turnover from
    which athlete_ids leave/join each team's roster year over year, valued by recruiting
    rating where available (unrated -- true walk-ons or unrated recruits -- contribute to the
    count but not the rating sum)."""
    prev = run_query(
        "SELECT athlete_id, team FROM team_rosters WHERE season = :season",
        params={"season": season - 1}, engine=engine,
    )
    cur = run_query(
        "SELECT athlete_id, team FROM team_rosters WHERE season = :season",
        params={"season": season}, engine=engine,
    )
    if cur.empty:
        return pd.DataFrame(columns=["team", "season", "transfers_in", "transfers_out", "net_transfer_rating"])

    recruits = run_query(
        "SELECT athlete_id, rating FROM recruiting_players WHERE athlete_id IS NOT NULL",
        engine=engine,
    ).dropna(subset=["athlete_id"]).drop_duplicates("athlete_id").set_index("athlete_id")["rating"]

    prev_by_athlete = prev.set_index("athlete_id")["team"] if not prev.empty else pd.Series(dtype=object)
    cur_by_athlete = cur.set_index("athlete_id")["team"]

    # "In" = on this team's current roster, not on ANY team's roster last season (new arrival,
    # transfer or true freshman alike -- can't distinguish without the portal endpoint, hence
    # this fallback only applies pre-2021).
    arrivals = cur[~cur["athlete_id"].isin(prev_by_athlete.index)].copy()
    arrivals["rating"] = arrivals["athlete_id"].map(recruits)
    in_agg = arrivals.groupby("team").agg(transfers_in=("athlete_id", "size"), rating_in=("rating", "sum"))

    departures = prev[~prev["athlete_id"].isin(cur_by_athlete.index)].copy() if not prev.empty else prev
    if not departures.empty:
        departures["rating"] = departures["athlete_id"].map(recruits)
        out_agg = departures.groupby("team").agg(transfers_out=("athlete_id", "size"), rating_out=("rating", "sum"))
    else:
        out_agg = pd.DataFrame(columns=["transfers_out", "rating_out"])

    combined = in_agg.join(out_agg, how="outer").fillna(0)
    combined["net_transfer_rating"] = combined["rating_in"] - combined["rating_out"]
    combined = combined.reset_index().rename(columns={"index": "team"})
    combined["season"] = season
    return combined[["team", "season", "transfers_in", "transfers_out", "net_transfer_rating"]]


def build_roster_turnover_features(
    engine, seasons: list[int], transfer_portal_start_season: int = 2021, client=None
) -> pd.DataFrame:
    frames = []
    portal_seasons = [s for s in seasons if s >= transfer_portal_start_season]
    diff_seasons = [s for s in seasons if s < transfer_portal_start_season]

    if portal_seasons:
        if client is None:
            from cfb_power_ratings.cfbd_client import get_client

            client = get_client()
        for season in portal_seasons:
            frames.append(_transfer_portal_features(client, season))

    for season in diff_seasons:
        frames.append(_roster_diff_features(engine, season))

    if not frames:
        return pd.DataFrame(columns=["team", "season", "transfers_in", "transfers_out", "net_transfer_rating"])
    return pd.concat(frames, ignore_index=True)
