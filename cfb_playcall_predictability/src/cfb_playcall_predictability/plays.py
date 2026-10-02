"""Load the cached play-by-play (scripts/pull_pbp.R) and turn it into one row per offensive
run/pass call, in snap order.

Why the cfbfastR release and not the MySQL `plays` table: see scripts/pull_pbp.R -- `plays`
has no score/timeout columns, and its ids can't be joined to the release's exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_playcall_predictability.config import DataConfig
from cfb_playcall_predictability.utils.logging import get_logger
from cfb_playcall_predictability.utils.paths import DATA_INTERIM_DIR

logger = get_logger(__name__)


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Concatenate the cached release files, regular season only (the MySQL `games` table --
    neutral site, betting lines, FBS division -- only holds regular-season games), in snap
    order. game_play_number repeats for a kickoff and an accompanying penalty, so the file's
    own row order breaks ties."""
    frames = []
    for season in seasons:
        path = DATA_INTERIM_DIR / f"pbp_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run scripts/pull_pbp.R first")
        df = pd.read_parquet(path)
        df = df[df["season_type"] == "regular"].copy()
        df["season"] = season
        df["row_order"] = np.arange(len(df))
        frames.append(df)
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp.sort_values(["season", "game_id", "game_play_number", "row_order"], kind="stable")
    return pbp.reset_index(drop=True)


def label_calls(pbp: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """1.0 = pass call (dropbacks, sacks included), 0.0 = run call, NaN = not a run/pass decision.

    Fumble and safety plays can follow either call, so for those play types the label comes
    from cfbfastR's own rush/pass flags (it parses the play text). Scrambles are logged by the
    source as rushes and can't be told apart -- a known, documented limitation."""
    play_type = pbp["play_type"]
    label = pd.Series(np.nan, index=pbp.index)
    label[play_type.isin(cfg.pass_play_types)] = 1.0
    label[play_type.isin(cfg.run_play_types)] = 0.0

    flagged = play_type.isin(cfg.flag_labeled_play_types)
    label[flagged & (pbp["pass"] == 1)] = 1.0
    label[flagged & (pbp["pass"] != 1) & (pbp["rush"] == 1)] = 0.0
    return label


def is_garbage_time(pbp: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """Score margin past the per-quarter threshold (same rule as cfb_ryan_day_offense)."""
    margin = pbp["pos_score_diff_start"].abs()
    threshold = pbp["period"].map(cfg.garbage_time_margin)
    return (margin > threshold).fillna(False)


def build_calls(pbp: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """One row per run/pass call. Keeps garbage-time snaps (so in-game "so far" features see the
    whole game) but flags them; the model trains and is scored on non-garbage calls only."""
    df = pbp.copy()
    df["is_pass"] = label_calls(df, cfg)

    text = df["play_text"].fillna("").str.lower()
    keep = (
        df["is_pass"].notna()
        & ~text.str.contains(cfg.exclude_text_pattern, regex=True)
        & df["down"].between(1, 4)
        & (df["period"] <= cfg.max_period)
        & df["distance"].notna()
        & df["yards_to_goal"].between(1, 99)
        & df["pos_score_diff_start"].notna()
    )
    calls = df[keep].copy()
    calls["garbage_time"] = is_garbage_time(calls, cfg)
    calls["is_pass"] = calls["is_pass"].astype(int)
    calls["pos_score_diff_start"] = calls["pos_score_diff_start"] + 0.0  # -0.0 -> 0.0
    calls["play_seq"] = calls.groupby("game_id").cumcount()

    logger.info(
        f"{len(calls):,} run/pass calls from {len(df):,} plays "
        f"({calls['garbage_time'].mean():.1%} garbage time, pass rate {calls['is_pass'].mean():.1%})"
    )
    return calls.reset_index(drop=True)
