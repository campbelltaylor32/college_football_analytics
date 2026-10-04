"""Step-sequencing / guard tests for scripts/run_weekly.py - every CFBD-touching or
model-fitting call is stubbed, so this only exercises the wrapper's own branching."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

run_weekly = importlib.import_module("run_weekly")


@pytest.fixture
def calls():
    return {"ingest": [], "retrain": [], "predict": [], "pred_drift": [], "data_drift": []}


@pytest.fixture
def stub(monkeypatch, calls):
    monkeypatch.setattr(run_weekly.cfbd_client, "get_client", lambda: object())

    def _ingest(season, weeks, *a, **k):
        calls["ingest"].append((season, tuple(weeks)))
        return pd.DataFrame({"game_id": [1, 2], "week": [weeks[-1], weeks[-1]]})

    monkeypatch.setattr(run_weekly.ingest_and_update_history, "run", _ingest)

    def _retrain(season, week, *a, **k):
        calls["retrain"].append((season, week))
        return SimpleNamespace(version=f"v{season}w{week}_stub", training_row_count=1400)

    monkeypatch.setattr(run_weekly.train_production_artifact, "run", _retrain)

    def _predict(season, week, *a, **k):
        calls["predict"].append((season, week))
        return Path(f"/tmp/live_{season}_week_{week}.csv")

    monkeypatch.setattr(run_weekly, "score_and_save_upcoming_week", _predict)
    monkeypatch.setattr(run_weekly, "load_latest_artifact", lambda: object())

    fake_monitoring = SimpleNamespace(
        run_prediction_drift_check=lambda s, w, f, v: calls["pred_drift"].append((s, w))
        or {"precision_this_week": 0.6, "rolling_mean_precision": 0.6, "alert": False},
        run_data_drift_check=lambda s, w, live: calls["data_drift"].append((s, w))
        or {"dataset_drift": False},
    )
    monkeypatch.setattr(run_weekly, "_monitoring", lambda: fake_monitoring)
    return calls


def test_midseason_runs_every_step(stub):
    code = run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9)
    assert code == 0
    assert stub["ingest"] == [(2025, tuple(range(1, 9)))]  # whole season-to-date, not just wk 8
    assert stub["retrain"] == [(2025, 8)]
    assert stub["predict"] == [(2025, 9)]
    assert stub["pred_drift"] == [(2025, 8)]
    assert stub["data_drift"] == [(2025, 9)]


def test_early_season_skips_retrain_and_predict(stub):
    # completed week 2 < MIN_WEEK_HISTORICAL (3); upcoming week 3 < MIN_WEEK_LIVE (4)
    code = run_weekly.run_weekly(season=2025, completed_week=2, upcoming_week=3)
    assert code == 0
    assert stub["ingest"] == [(2025, (1, 2))]
    assert stub["retrain"] == []
    assert stub["predict"] == []


def test_predict_skipped_when_no_artifact(stub, monkeypatch):
    monkeypatch.setattr(run_weekly, "load_latest_artifact", lambda: None)
    run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9)
    assert stub["retrain"] == [(2025, 8)]
    assert stub["predict"] == []


def test_no_upcoming_week_skips_predict(stub):
    run_weekly.run_weekly(season=2025, completed_week=15, upcoming_week=None)
    assert stub["retrain"] == [(2025, 15)]
    assert stub["predict"] == []
    assert stub["data_drift"] == []


def test_ingest_failure_still_attempts_retrain(stub, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("CFBD down")

    monkeypatch.setattr(run_weekly.ingest_and_update_history, "run", _boom)
    code = run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9)
    assert code == 1  # hard failure flagged
    assert stub["retrain"] == [(2025, 8)]  # but retrain still ran
    assert stub["predict"] == [(2025, 9)]


def test_out_of_season_is_a_noop(stub, monkeypatch):
    monkeypatch.setattr(
        run_weekly, "resolve_weeks",
        lambda client, season=None: SimpleNamespace(
            season=2025, completed_week=None, upcoming_week=None, in_season=False, note="offseason"
        ),
    )
    code = run_weekly.run_weekly(season=None, completed_week=None, upcoming_week=None)
    assert code == 0
    assert stub["ingest"] == [] and stub["retrain"] == []


def test_auto_resolves_when_no_override(stub, monkeypatch):
    monkeypatch.setattr(
        run_weekly, "resolve_weeks",
        lambda client, season=None: SimpleNamespace(
            season=2025, completed_week=7, upcoming_week=8, in_season=True, note="w7"
        ),
    )
    run_weekly.run_weekly(season=None, completed_week=None, upcoming_week=None)
    assert stub["ingest"] == [(2025, tuple(range(1, 8)))]
    assert stub["retrain"] == [(2025, 7)]
    assert stub["predict"] == [(2025, 8)]


def test_only_sunday_results_subset(stub):
    code = run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9, only={"ingest", "monitor"})
    assert code == 0
    assert stub["ingest"] == [(2025, tuple(range(1, 9)))]
    assert stub["retrain"] == [] and stub["predict"] == []
    assert stub["pred_drift"] == [(2025, 8)]


def test_only_tuesday_picks_subset(stub):
    code = run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9, only={"ingest", "retrain", "score"})
    assert code == 0
    assert stub["retrain"] == [(2025, 8)]
    assert stub["predict"] == [(2025, 9)]
    assert stub["pred_drift"] == [] and stub["data_drift"] == []


def test_only_retrain_score_without_ingest(stub):
    code = run_weekly.run_weekly(season=2025, completed_week=8, upcoming_week=9, only={"retrain", "score"})
    assert code == 0
    assert stub["ingest"] == []
    assert stub["retrain"] == [(2025, 8)] and stub["predict"] == [(2025, 9)]
