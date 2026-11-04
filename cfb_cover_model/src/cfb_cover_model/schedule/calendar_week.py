"""Resolve "which CFB week just finished, and which one is next" from the CFBD calendar,
so scripts/run_weekly.py never needs a hand-bumped week number (the manual-bump convention
the rest of this repo still uses - see docs/serving_and_monitoring.md section 8).

`cfbd.GamesApi.get_calendar(year)` returns one `CalendarWeek` per week with `week`,
`season_type` (REGULAR / POSTSEASON), and tz-aware `first_game_start` / `last_game_start`
boundaries. This model is regular-season only (same scope as the training data), so
POSTSEASON weeks are ignored entirely.

Despite the field names, CFBD's first_game_start / last_game_start are the week's *window*
(Monday ~08:00 UTC -> next Monday ~07:59 UTC), not real kickoffs. So "the window has opened"
is NOT "the week has been played": on the Tuesday picks run, the current week's window is
already open while its games are days away. The rule:

  a week is complete once its Saturday slate is done - ref >= window start + 6 days
                   (Sunday ~08:00 UTC) - or its window has closed
  completed_week = the highest REGULAR week that is complete
  upcoming_week  = completed_week + 1, or None if the regular season is over

So a Sunday results run and the following Tuesday picks run both resolve the same
(completed=C, upcoming=C+1). Caveat: the handful of Sunday/Monday games some weeks have
(week 1, holiday weekends) won't be in that Sunday's retrain; the next run picks them up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Window opens Monday ~08:00 UTC; +6 days is Sunday ~08:00 UTC, after Saturday's late games.
WEEK_COMPLETE_AFTER = timedelta(days=6)


@dataclass(frozen=True)
class ResolvedWeeks:
    season: int
    completed_week: int | None
    upcoming_week: int | None
    in_season: bool
    note: str


def _season_type_name(week) -> str:
    st = getattr(week, "season_type", "")
    return str(getattr(st, "value", st)).lower()


def _regular_weeks(client, season: int) -> list:
    import cfbd

    api = cfbd.GamesApi(client)
    weeks = api.get_calendar(year=season) or []
    regular = [
        w for w in weeks
        if _season_type_name(w) == "regular"
        and w.first_game_start is not None
        and w.last_game_start is not None
    ]
    regular.sort(key=lambda w: w.week)
    return regular


def resolve_weeks(
    client,
    *,
    season: int | None = None,
    ref: datetime | None = None,
) -> ResolvedWeeks:
    """`season` defaults to $PRODUCTION_SEASON, else `ref`'s calendar year. `ref` defaults
    to now (UTC); a naive `ref` is assumed UTC."""
    if ref is None:
        ref = datetime.now(timezone.utc)
    elif ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    if season is None:
        env_season = os.environ.get("PRODUCTION_SEASON")
        season = int(env_season) if env_season else ref.year

    regular = _regular_weeks(client, season)
    if not regular:
        return ResolvedWeeks(
            season, None, None, False,
            f"CFBD calendar has no dated regular-season weeks for {season} yet",
        )

    first, last = regular[0], regular[-1]

    if ref < first.first_game_start:
        return ResolvedWeeks(
            season, None, None, False,
            f"season {season} has not started "
            f"(week 1 kicks off {first.first_game_start:%Y-%m-%d})",
        )

    if ref >= last.last_game_start:
        return ResolvedWeeks(
            season, last.week, None, False,
            f"season {season} regular schedule is complete (through week {last.week})",
        )

    complete = [
        w.week for w in regular
        if ref >= w.first_game_start + WEEK_COMPLETE_AFTER or ref >= w.last_game_start
    ]
    if not complete:
        return ResolvedWeeks(
            season, None, first.week, True,
            f"season {season}: week {first.week} in progress, no completed week yet",
        )
    completed_week = max(complete)
    upcoming_week = completed_week + 1 if completed_week < last.week else None
    return ResolvedWeeks(
        season, completed_week, upcoming_week, True,
        f"season {season}: week {completed_week} complete, "
        f"upcoming week {upcoming_week if upcoming_week is not None else '(none - regular season ending)'}",
    )
