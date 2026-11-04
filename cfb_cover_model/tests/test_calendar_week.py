import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.schedule import calendar_week


def _wk(week, start, end):
    return SimpleNamespace(
        week=week,
        season_type="regular",
        first_game_start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        last_game_start=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )


# Compact fake calendar: weekly windows run Mon 07:00 UTC -> next Mon 06:59 UTC, so the only
# Sunday inside week N's window is the one *after* week N's Saturday slate.
FAKE_WEEKS = [
    _wk(1, "2025-08-25T07:00", "2025-09-01T06:59"),
    _wk(2, "2025-09-01T07:00", "2025-09-08T06:59"),
    _wk(3, "2025-09-08T07:00", "2025-09-15T06:59"),
    _wk(4, "2025-09-15T07:00", "2025-09-22T06:59"),
    _wk(5, "2025-09-22T07:00", "2025-09-29T06:59"),
]


@pytest.fixture
def patched_calendar(monkeypatch):
    monkeypatch.setattr(calendar_week, "_regular_weeks", lambda client, season: FAKE_WEEKS)


def _resolve(ref, season=2025):
    return calendar_week.resolve_weeks(None, season=season, ref=ref)


def test_preseason_is_not_in_season(patched_calendar):
    r = _resolve(datetime(2025, 8, 1, tzinfo=timezone.utc))
    assert not r.in_season
    assert r.completed_week is None and r.upcoming_week is None


def test_sunday_after_week_3(patched_calendar):
    r = _resolve(datetime(2025, 9, 14, 15, tzinfo=timezone.utc))
    assert r.in_season
    assert r.completed_week == 3
    assert r.upcoming_week == 4


def test_first_sunday_of_season(patched_calendar):
    # Sunday inside week 1's window, after week 1's games
    r = _resolve(datetime(2025, 8, 31, 15, tzinfo=timezone.utc))
    assert r.completed_week == 1 and r.upcoming_week == 2


def test_final_regular_week_has_no_upcoming(patched_calendar):
    # Sunday after week 5 (the last week) - week 5 is complete, nothing left to pick
    r = _resolve(datetime(2025, 9, 28, 15, tzinfo=timezone.utc))
    assert r.in_season
    assert r.completed_week == 5
    assert r.upcoming_week is None


def test_postseason_after_last_week_window(patched_calendar):
    r = _resolve(datetime(2025, 12, 1, tzinfo=timezone.utc))
    assert not r.in_season
    assert r.completed_week == 5
    assert r.upcoming_week is None


def test_empty_calendar_is_not_in_season(monkeypatch):
    monkeypatch.setattr(calendar_week, "_regular_weeks", lambda client, season: [])
    r = calendar_week.resolve_weeks(None, season=2099, ref=datetime(2099, 9, 1, tzinfo=timezone.utc))
    assert not r.in_season and r.completed_week is None


def test_naive_ref_treated_as_utc(patched_calendar):
    r = calendar_week.resolve_weeks(None, season=2025, ref=datetime(2025, 9, 14, 15))
    assert r.completed_week == 3


def test_season_defaults_from_env(patched_calendar, monkeypatch):
    monkeypatch.setenv("PRODUCTION_SEASON", "2025")
    r = calendar_week.resolve_weeks(None, ref=datetime(2025, 9, 14, 15, tzinfo=timezone.utc))
    assert r.season == 2025 and r.completed_week == 3


def test_season_defaults_from_ref_year_without_env(patched_calendar, monkeypatch):
    monkeypatch.delenv("PRODUCTION_SEASON", raising=False)
    r = calendar_week.resolve_weeks(None, ref=datetime(2025, 9, 14, 15, tzinfo=timezone.utc))
    assert r.season == 2025


def test_tuesday_picks_run_does_not_skip_a_week(patched_calendar):
    # Tuesday inside week 4's window: week 4's games haven't been played, so week 3 is the last
    # completed week and the picks are for week 4 (regression: this used to resolve 4 -> 5).
    r = _resolve(datetime(2025, 9, 16, 20, tzinfo=timezone.utc))
    assert r.completed_week == 3 and r.upcoming_week == 4


def test_sunday_and_following_tuesday_agree(patched_calendar):
    sunday = _resolve(datetime(2025, 9, 14, 19, tzinfo=timezone.utc))
    tuesday = _resolve(datetime(2025, 9, 16, 20, tzinfo=timezone.utc))
    assert (sunday.completed_week, sunday.upcoming_week) == (tuesday.completed_week, tuesday.upcoming_week) == (3, 4)


def test_saturday_does_not_count_its_own_week_complete(patched_calendar):
    r = _resolve(datetime(2025, 9, 13, 23, tzinfo=timezone.utc))
    assert r.completed_week == 2 and r.upcoming_week == 3


def test_week_one_before_first_sunday_has_no_completed_week(patched_calendar):
    r = _resolve(datetime(2025, 8, 28, 1, tzinfo=timezone.utc))
    assert r.in_season and r.completed_week is None and r.upcoming_week == 1
