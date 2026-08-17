"""Direct port of the `returning_production` table (CFBD's own preseason-known metric of how
much of a team's prior production is walking back in the door) -- no correction needed, unlike
talent_recruiting.py's blue-chip ratio."""
from __future__ import annotations

import pandas as pd

from cfb_power_ratings.database import run_query


def build_returning_production_features(engine, seasons: list[int]) -> pd.DataFrame:
    return run_query(
        """
        SELECT season, team, total_ppa, percent_ppa, percent_passing_ppa,
               percent_receiving_ppa, percent_rushing_ppa, usage_pct
        FROM returning_production
        WHERE season IN :seasons
        """,
        params={"seasons": tuple(seasons)}, engine=engine,
    )
