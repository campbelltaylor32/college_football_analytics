import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.serving.artifact import load_artifact, load_latest_artifact, save_artifact


def test_save_and_load_round_trip_preserves_predictions(tmp_path, tiny_production_artifact):
    artifact = tiny_production_artifact
    X = artifact.reference_training_frame

    pre_save_classifier_proba = artifact.classifier.predict_proba(X)[:, 1]
    pre_save_regressor_proba = artifact.regressor.predict_proba(X)[:, 1]

    version_dir = save_artifact(artifact, output_dir=tmp_path)
    assert version_dir == tmp_path / artifact.version
    for f in ("classifier.joblib", "regressor.joblib", "selected_columns.json", "reference_training_features.parquet", "metadata.json"):
        assert (version_dir / f).exists()

    loaded = load_artifact(version_dir)
    np.testing.assert_allclose(loaded.classifier.predict_proba(X)[:, 1], pre_save_classifier_proba)
    np.testing.assert_allclose(loaded.regressor.predict_proba(X)[:, 1], pre_save_regressor_proba)


def test_save_and_load_round_trip_preserves_metadata(tmp_path, tiny_production_artifact):
    artifact = tiny_production_artifact
    version_dir = save_artifact(artifact, output_dir=tmp_path)
    loaded = load_artifact(version_dir)

    assert loaded.version == artifact.version
    assert loaded.season_through == artifact.season_through
    assert loaded.week_through == artifact.week_through
    assert loaded.selected_columns == artifact.selected_columns
    assert loaded.feature_columns == artifact.feature_columns
    assert loaded.transforms == artifact.transforms
    assert loaded.representation == artifact.representation
    assert loaded.classifier_threshold == artifact.classifier_threshold
    assert loaded.regressor_threshold == artifact.regressor_threshold
    assert loaded.training_row_count == artifact.training_row_count
    assert loaded.git_commit == artifact.git_commit


def test_latest_pointer_resolves_to_most_recently_saved(tmp_path, tiny_production_artifact):
    artifact = tiny_production_artifact
    save_artifact(artifact, output_dir=tmp_path)

    latest = load_latest_artifact(output_dir=tmp_path)
    assert latest is not None
    assert latest.version == artifact.version


def test_load_latest_artifact_returns_none_when_never_trained(tmp_path):
    assert load_latest_artifact(output_dir=tmp_path) is None


def test_load_latest_artifact_returns_none_when_pointer_targets_missing_dir(tmp_path):
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "latest.json").write_text(json.dumps({"version": "v_does_not_exist"}))
    assert load_latest_artifact(output_dir=tmp_path) is None
