"""Port of the talent/blue-chip-ratio, coach cumulative-record + lag-by-year, and
returning-production logic from ../R Scripts/Full_CFB_Game_Outcome_Historical.R (sections
1.5-1.7) and Merge_Predictors_CFB_Historical.R (sections 2.1-2.3).

Note: CFBD's /coaches endpoint only returns a coach's season record for the specific
`year` requested (verified against the live API - it does NOT return full career history
in one call, unlike what a first read of the R source might suggest), so
compute_coach_cumulative_record expects a multi-year-concatenated coaches DataFrame -
fetching that range is pipeline.py's job, not this module's.
"""
from __future__ import annotations

import pandas as pd


def compute_blue_chip_ratio(roster: pd.DataFrame, recruits: pd.DataFrame) -> pd.DataFrame:
    """Join each season's roster to recruiting data by athlete_id (matching R's
    merge(tot_roster, tot_recruits, by="athlete_id") - note this joins on athlete_id alone,
    not athlete_id + year, so a recruit's rating attaches to every season they appear on a
    roster, not just their recruiting class year - replicated here intentionally, not a bug
    to "fix", since it matches the historical data's actual behavior)."""
    merged = roster.merge(recruits, on="athlete_id", how="left")
    out = (
        merged.groupby(["team", "year"])
        .apply(
            lambda g: pd.Series(
                {
                    "blue_chip_ratio": (g["stars"] >= 4).sum() / g["athlete_id"].nunique(),
                    "avg_player_rating": g["rating"].mean(),
                }
            )
        )
        .reset_index()
    )
    return out.fillna(0)


def merge_talent(talent: pd.DataFrame, blue_chip: pd.DataFrame) -> pd.DataFrame:
    merged = talent.merge(blue_chip, on=["year", "team"], how="left")
    return merged.fillna(0)


def compute_coach_cumulative_record(coaches_multi_year: pd.DataFrame) -> pd.DataFrame:
    """coaches_multi_year: concatenated per-year pulls (see module docstring), columns
    Name, team, year, games, wins. Returns career cumulative Total_Games_Coached /
    Winning_Percentage as of and including each year - the lag-by-year shift (avoiding
    lookahead) is a separate step, see lag_coach_by_year."""
    df = coaches_multi_year.sort_values("year").copy()
    grouped = df.groupby("Name", sort=False)
    df["Total_Games_Coached"] = grouped["games"].cumsum()
    total_wins = grouped["wins"].cumsum()
    df["Winning_Percentage"] = total_wins / df["Total_Games_Coached"]
    return df[["Name", "year", "team", "Total_Games_Coached", "Winning_Percentage"]]


def lag_coach_by_year(coach_record: pd.DataFrame) -> pd.DataFrame:
    """Port of R's `arrange(desc(year)) %>% mutate(year = lag(year)) %>% na.omit()`: within
    each coach, sort years descending, shift the *year label* back by one position (so each
    row's stats - accumulated through year Y - become labeled as belonging to year Y+1),
    then drop the row that received no shifted label (the most recent year, which would
    otherwise falsely claim to be lookahead-free for a season beyond what's been coached)."""
    df = coach_record.sort_values(["Name", "year"], ascending=[True, False]).copy()
    df["year"] = df.groupby("Name", sort=False)["year"].shift(1)
    return df.dropna(subset=["year"])


def merge_talent_coach_returning(
    talent: pd.DataFrame, coach_lagged: pd.DataFrame, returning: pd.DataFrame, min_year: int
) -> pd.DataFrame:
    """Port of R sections 2.2-2.3: full outer join talent+coach on (year,team), de-duplicate,
    filter to min_year, outer join returning production."""
    merged = talent.merge(coach_lagged, on=["year", "team"], how="outer")
    merged = merged.drop_duplicates()
    merged = merged[merged["year"] >= min_year]
    merged = merged.drop_duplicates(subset=["year", "team"], keep="first")

    returning = returning.rename(columns={"season": "year"}).drop(columns=["conference"], errors="ignore")
    merged = merged.merge(returning, on=["year", "team"], how="outer")
    return merged
