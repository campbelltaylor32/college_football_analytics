"""Push-aware label construction.

The R layer's home_covered treats an exact push (home_minus_away == -signed_spread) as
"did not cover" because every branch of its ifelse() uses strict >/>= comparisons that never
evaluate true on the exact number. Standard ATS practice is to exclude pushes entirely - no
money changes hands on a push, so it is neither a win nor a loss for either side of the bet.
This module recomputes the label from the joined, signed data rather than trusting the R
layer's column, and adds the continuous cover_margin target for the regression track.
"""
from __future__ import annotations

import pandas as pd


def add_push_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Requires home_minus_away and signed_spread columns (see data.load_raw_joined).

    Adds:
      is_push       bool  - True if the game landed exactly on the spread
      cover_margin  float - home_minus_away + signed_spread; positive means home covered,
                             negative means home did not cover, exactly zero means push
      home_covered  int   - 1 if cover_margin > 0, 0 if cover_margin < 0; undefined (NaN)
                             for pushes, which callers should filter via is_push before
                             using this column for training or evaluation
    """
    out = df.copy()
    out["cover_margin"] = out["home_minus_away"] + out["signed_spread"]
    out["is_push"] = out["cover_margin"] == 0
    out["home_covered"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[~out["is_push"], "home_covered"] = (
        out.loc[~out["is_push"], "cover_margin"] > 0
    ).astype(int)
    return out


def drop_pushes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop push rows for training/evaluation. Call after add_push_and_targets."""
    return df.loc[~df["is_push"]].reset_index(drop=True)
