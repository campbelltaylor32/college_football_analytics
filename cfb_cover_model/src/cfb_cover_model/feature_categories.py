"""Domain taxonomy for the ~160 base game stats (plus talent/coaching/returning-production/
context columns), used to roll individual-feature importance up into something readable.

The sibling cfb_spread_model project rolls features up by *temporal transform*
(avg_all/avg3/prev_week) - uninformative here, since this project's winning config is
~100% prev_week already (see docs/feature_selection_methodology.md). This module instead
classifies by *domain*: what kind of football stat is this, independent of which transform
window it was computed over. Rules are keyword-based and priority-ordered (first match
wins) - built against the exact base-stat name list documented in docs/data_dictionary.md.
"""
from __future__ import annotations

import re

PREFIXES = ("home_", "away_", "diff_")  # "diff_" covers the differential home/away
# representation (feature_engineering.py::apply_home_away_representation) - lets this
# taxonomy classify diff_X columns from an avg_all_only/differential config the same way
# as home_X/away_X columns from a raw_dual config, so cross-config comparisons in
# scripts/analyze_feature_stability.py can match on base_stat regardless of representation.

# (category, keywords) - checked in order, first match wins. Keywords are matched
# case-insensitively as substrings against the base stat name (prefix/transform already
# stripped by parse_column).
_DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    # "intercetpion" is a verbatim upstream R-layer typo (see docs/data_dictionary.md) -
    # matched alongside the correct spelling so that column isn't dropped to uncategorized.
    ("turnover_penalty", ("fumble", "intercept", "intercetpion", "turnover", "penalt")),
    ("possession_time", ("possession",)),
    ("explosiveness", ("explosive",)),
    ("pass_rush_pressure", ("sack", "pressure", "qb_hurr", "tackle", "deflect")),
    ("down_conversion", ("third_down", "fourth_down", "3rd_down", "first_down")),
    ("drive_efficiency", ("drive",)),
    ("play_volume_mix", ("pass_rate", "run_rate", "rush_percentage", "plays")),
    ("epa_success_rate", ("epa", "success")),
    ("special_teams", ("kick_return", "punt_return", "kicking_points")),
    ("talent", ("talent", "blue_chip", "player_rating")),
    ("coaching", ("games_coached", "winning_percentage")),
    ("returning_production", ("_ppa", "usage")),
    (
        "box_score_scoring_yardage",
        (
            "points", "yards", "touchdown", "_tds", "completion", "attempted_passes",
            "rushing_attempts", "first_downs",
        ),
    ),
    ("context_market", ("spread", "week", "neutral_site", "conference_game", "favored")),
]

_TEMPORAL_SUFFIX_RE = re.compile(r"^(?:prev_week_)?(?P<stat>.+?)(?:_avg_all|_avg3)?$")

_TRAILING_TRANSFORM_SUFFIXES = ("prev_week", "avg_all", "avg3")

# Only used to tag the 4 returning_production columns kept by
# engineered_features.py::consolidate_returning_production as "engineered" for
# analysis/reporting purposes - it does not affect which columns exist (that's decided by
# engineered_features.py itself, this is read-only labeling).
RETURNING_PRODUCTION_KEEP_HINT = frozenset(
    {"rushing_usage", "receiving_usage", "percent_rushing_ppa", "total_rushing_ppa"}
)


def _strip_trailing_transform(body: str) -> tuple[str, str | None]:
    """For engineered columns, the transform is a trailing suffix (..._prev_week) rather
    than the leading prev_week_ prefix / trailing _avg_all|_avg3 used by raw columns -
    see engineered_features.py's naming (matchup_adj_<stat>_<transform>,
    special_teams_net_score_<transform>)."""
    for t in _TRAILING_TRANSFORM_SUFFIXES:
        if body == t or body.endswith(f"_{t}"):
            stat = body[: -(len(t) + 1)] if body != t else ""
            return stat, t
    return body, None


def parse_column(column_name: str) -> dict:
    """Split a candidate-feature column name into its component parts.

    Returns {"side": "home"|"away"|None, "base_stat": str, "transform": "prev_week"|
    "avg_all"|"avg3"|None, "offense_defense": "offense"|"defense"|None,
    "domain": <category>, "engineered": None|"matchup_adjustment"|"special_teams_composite"|
    "returning_production_consolidated"}.
    """
    side = next((p[:-1] for p in PREFIXES if column_name.startswith(p)), None)
    remainder = column_name[len(side) + 1:] if side else column_name

    engineered = None
    if remainder.startswith("matchup_adj_"):
        engineered = "matchup_adjustment"
        stat, transform = _strip_trailing_transform(remainder[len("matchup_adj_"):])
    elif remainder.startswith("special_teams_net_score"):
        engineered = "special_teams_composite"
        stat, transform = "special_teams_net_score", remainder[len("special_teams_net_score") :].lstrip("_") or None
    elif remainder.startswith("prev_week_"):
        transform, stat = "prev_week", remainder[len("prev_week_"):]
    elif remainder.endswith("_avg_all"):
        transform, stat = "avg_all", remainder[: -len("_avg_all")]
    elif remainder.endswith("_avg3"):
        transform, stat = "avg3", remainder[: -len("_avg3")]
    else:
        transform, stat = None, remainder
        if stat in RETURNING_PRODUCTION_KEEP_HINT:
            engineered = "returning_production_consolidated"

    lower_stat = stat.lower()
    if "defense" in lower_stat or lower_stat.endswith("_allowed") or lower_stat.endswith("_against"):
        offense_defense = "defense"
    elif "offense" in lower_stat:
        offense_defense = "offense"
    else:
        offense_defense = None

    if engineered == "special_teams_composite":
        domain = "special_teams"
    else:
        domain = "uncategorized"
        for category, keywords in _DOMAIN_RULES:
            if any(kw in lower_stat for kw in keywords):
                domain = category
                break

    return {
        "column": column_name,
        "side": side,
        "base_stat": stat,
        "transform": transform,
        "offense_defense": offense_defense,
        "domain": domain,
        "engineered": engineered,
    }


def categorize_by_domain(base_stat_name_or_column: str) -> str:
    """Convenience wrapper - just the domain category for a base stat name or full column."""
    return parse_column(base_stat_name_or_column)["domain"]
