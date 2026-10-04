import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from cfb_cover_model.api.app import create_app
from cfb_cover_model.api.artifact_store import ArtifactStore
from cfb_cover_model.serving import scoring


def _fake_build_feature_frame(df, feature_columns, transforms, representation, data_cfg):
    """Same rationale as tests/test_serving_scoring.py's fake - isolates the API/routing
    layer from the real feature-engineering pipeline, which is already exercised directly
    against real data by scripts/generate_weekly_predictions.py and is out of scope here."""
    week_df = pd.DataFrame(
        {"game_id": [1, 2], "home_team": ["A", "B"], "away_team": ["X", "Y"], "season": [2025, 2025], "week": [10, 10], "spread": [-3.5, 7.0]}
    )
    week_variant = pd.DataFrame(
        {"diff_feature_a": [2.0, -2.0], "diff_feature_b": [1.0, -1.0], "diff_feature_c": [0.5, -0.5], "diff_feature_d": [0.1, -0.1]}
    )
    return week_df, week_variant, list(week_variant.columns)


@pytest.fixture(autouse=True)
def _patch_build_feature_frame(monkeypatch):
    monkeypatch.setattr(scoring, "build_feature_frame", _fake_build_feature_frame)


@pytest.fixture
def empty_store():
    return ArtifactStore(output_dir=Path("/tmp/nonexistent_cfb_cover_model_test_dir"))


@pytest.fixture
def loaded_store(tiny_production_artifact):
    store = ArtifactStore(output_dir=Path("/tmp/nonexistent_cfb_cover_model_test_dir"))
    store.replace(tiny_production_artifact)
    return store


def _client(store):
    app = create_app(artifact_store=store)
    return TestClient(app)


def test_health_always_ok_even_with_no_artifact(empty_store):
    r = _client(empty_store).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_is_false_with_no_artifact(empty_store):
    r = _client(empty_store).get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def test_ready_is_true_with_loaded_artifact(loaded_store, tiny_production_artifact):
    r = _client(loaded_store).get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["artifact_version"] == tiny_production_artifact.version


def test_predictions_503_with_no_artifact(empty_store):
    r = _client(empty_store).get("/predictions/week/10", params={"season": 2025, "file": "/dev/null"})
    assert r.status_code == 503


def test_predictions_week_returns_expected_schema(loaded_store, tmp_path):
    csv_path = tmp_path / "week.csv"
    pd.DataFrame(
        {"game_id": [1, 2], "season": [None, None], "week": [10, 10], "home_team": ["A", "B"], "away_team": ["X", "Y"], "spread": [-3.5, 7.0], "neutral_site": [False, False], "conference_game": [True, False]}
    ).to_csv(csv_path, index=False)

    r = _client(loaded_store).get("/predictions/week/10", params={"season": 2025, "file": str(csv_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == 2025
    assert body["week"] == 10
    assert body["mode"] == "csv"
    assert body["n_games"] == 2
    assert len(body["predictions"]) == 2
    assert body["n_agreement_bets"] == sum(p["agreement_bet"] for p in body["predictions"])


def test_predictions_week_404_when_csv_missing(loaded_store, tmp_path):
    r = _client(loaded_store).get("/predictions/week/10", params={"season": 2025, "file": str(tmp_path / "missing.csv")})
    assert r.status_code == 404


def test_admin_retrain_requires_token(loaded_store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret")
    r = _client(loaded_store).post("/admin/retrain", json={"season": 2025, "week": 10})
    assert r.status_code == 401


def test_admin_retrain_rejects_wrong_token(loaded_store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret")
    r = _client(loaded_store).post("/admin/retrain", json={"season": 2025, "week": 10}, headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


def test_admin_retrain_500_when_token_not_configured(loaded_store, monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    r = _client(loaded_store).post("/admin/retrain", json={"season": 2025, "week": 10}, headers={"X-Admin-Token": "anything"})
    assert r.status_code == 500


def test_admin_retrain_hot_swaps_artifact(loaded_store, monkeypatch, tiny_production_artifact):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret")

    class _FakeTrainProductionArtifact:
        @staticmethod
        def run(season, week):
            new = tiny_production_artifact
            new.version = f"v{season}w{week}_retrained"
            return new

    monkeypatch.setitem(sys.modules, "train_production_artifact", _FakeTrainProductionArtifact())

    client = _client(loaded_store)
    r = client.post("/admin/retrain", json={"season": 2099, "week": 1}, headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200
    assert r.json()["version"] == "v2099w1_retrained"

    r2 = client.get("/ready")
    assert r2.json()["artifact_version"] == "v2099w1_retrained"
