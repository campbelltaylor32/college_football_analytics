"""Trains and persists the dual-model ("agreement") production signal - logistic_regression
(Track A classifier) + xgboost_regressor (Track B regression-to-probability) - as a versioned,
on-disk artifact, so the API can load a fitted model instead of refitting on every request.

Before this module existed, nothing was ever persisted: scripts/generate_weekly_predictions.py
refit both models from scratch on every run (cheap given ~1,400 training rows, but not what an
API serving repeated requests wants). train_production_artifact() is a literal extraction of
that script's history-load -> transform -> fit_feature_set("reduced") -> classifier/regressor
.fit() logic; scoring.score_week() is the matching extraction of its week-scoring half. See
docs/final_writeup_2026.md for why this specific model pair is the recommended signal, and
docs/serving_and_monitoring.md for the end-to-end serving story.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set, fit_feature_set
from cfb_cover_model.modeling.splits import get_eligible_frame

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet"
EXTENDED_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "extended_history.parquet"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = PROJECT_ROOT / "outputs" / "feature_analysis" / "winning_feature_config.json"
THRESHOLD_TABLE_PATH = PROJECT_ROOT / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
PRODUCTION_DIR = PROJECT_ROOT / "outputs" / "models" / "production"

CLASSIFIER_MODEL = "logistic_regression"
REGRESSOR_MODEL = "xgboost_regressor"

# scripts/ isn't an installed package (no __init__.py) - make_track_a_specs/make_track_b_specs
# live in scripts/train_models.py and generate_weekly_predictions.py already reaches them via
# this same sys.path trick. Mirrored here rather than duplicating the spec-building logic.
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@dataclass
class ProductionArtifact:
    version: str
    trained_at: datetime
    season_through: int
    week_through: int
    classifier: object
    regressor: object
    selected_columns: list[str]
    feature_columns: list[str]
    transforms: list[str]
    representation: str
    classifier_threshold: float
    regressor_threshold: float
    reference_training_frame: pd.DataFrame
    training_row_count: int
    git_commit: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)


def _current_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _library_versions() -> dict[str, str]:
    versions = {}
    for mod_name in ("pandas", "numpy", "sklearn", "xgboost"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def train_production_artifact(
    data_cfg: dict,
    features_cfg: dict,
    modeling_cfg: dict,
    random_state: int,
    *,
    season_through: int,
    week_through: int,
    dataset_path: Path = DATASET_PATH,
    extended_history_path: Path = EXTENDED_HISTORY_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    winning_config_path: Path = WINNING_CONFIG_PATH,
    threshold_table_path: Path = THRESHOLD_TABLE_PATH,
) -> ProductionArtifact:
    """Refit the dual-model agreement signal on all eligible history and return a
    serializable artifact. `season_through`/`week_through` are labels only (the most recent
    completed week folded into this training run) - they do not filter the history, which
    already includes every eligible row through whatever ingest_and_update_history.py /
    the R pipeline has written to disk."""
    from train_models import make_track_a_specs, make_track_b_specs  # noqa: E402

    threshold_table = pd.read_csv(threshold_table_path).set_index("model_name")["threshold"].to_dict()
    for name in (CLASSIFIER_MODEL, REGRESSOR_MODEL):
        if name not in threshold_table:
            raise ValueError(f"No walk-forward-tuned threshold recorded for {name!r} in {threshold_table_path}")

    frame = pd.read_parquet(dataset_path)
    feature_columns = json.loads(Path(feature_columns_path).read_text())
    winning_cfg = json.loads(Path(winning_config_path).read_text())

    all_history = get_eligible_frame(frame, data_cfg)
    if Path(extended_history_path).exists():
        extended = pd.read_parquet(extended_history_path)
        extended = get_eligible_frame(extended, data_cfg)
        missing = [c for c in feature_columns if c not in extended.columns]
        if missing:
            raise ValueError(
                f"extended_history.parquet is missing {len(missing)} feature column(s) that "
                f"modeling_dataset.parquet has ({missing[:5]}...) - re-run "
                "scripts/validate_against_r_pipeline.py before trusting this history for training."
            )
        keep_cols = sorted(
            set(feature_columns) | {"game_id", "season", "week", "home_team", "away_team", "home_covered", "cover_margin", "home_favored"}
        )
        all_history = pd.concat([all_history[keep_cols], extended[keep_cols]], ignore_index=True)

    train_variant, train_cols = build_transform_variant(all_history, feature_columns, winning_cfg["transforms"])
    train_variant, train_cols = apply_home_away_representation(train_variant, train_cols, winning_cfg["representation"])
    train_variant = train_variant.reset_index(drop=True)
    train_variant.index = all_history.index
    y_train = all_history["home_covered"]
    cover_margin_train = all_history["cover_margin"]
    season_train = all_history["season"]

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}
    classifier_spec = spec_lookup[CLASSIFIER_MODEL]
    regressor_spec = spec_lookup[REGRESSOR_MODEL]
    assert classifier_spec["feature_set_mode"] == regressor_spec["feature_set_mode"] == "reduced"

    selected_columns, _report = fit_feature_set(
        train_variant[train_cols], y_train, season_train, "reduced", features_cfg, random_state
    )
    X_train = apply_feature_set(train_variant[train_cols], "reduced", selected_columns)

    classifier = classifier_spec["builder"]()
    classifier.fit(X_train, y_train)

    regressor = regressor_spec["builder"]()
    regressor.fit(X_train, cover_margin_train)

    version = f"v{season_through}w{week_through}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    return ProductionArtifact(
        version=version,
        trained_at=datetime.now(timezone.utc),
        season_through=season_through,
        week_through=week_through,
        classifier=classifier,
        regressor=regressor,
        selected_columns=list(selected_columns),
        feature_columns=list(feature_columns),
        transforms=list(winning_cfg["transforms"]),
        representation=winning_cfg["representation"],
        classifier_threshold=float(threshold_table[CLASSIFIER_MODEL]),
        regressor_threshold=float(threshold_table[REGRESSOR_MODEL]),
        reference_training_frame=X_train.reset_index(drop=True),
        training_row_count=len(X_train),
        git_commit=_current_git_commit(),
        library_versions=_library_versions(),
    )


def save_artifact(artifact: ProductionArtifact, output_dir: Path = PRODUCTION_DIR) -> Path:
    version_dir = Path(output_dir) / artifact.version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact.classifier, version_dir / "classifier.joblib")
    joblib.dump(artifact.regressor, version_dir / "regressor.joblib")
    (version_dir / "selected_columns.json").write_text(json.dumps(artifact.selected_columns, indent=2))
    artifact.reference_training_frame.to_parquet(version_dir / "reference_training_features.parquet")

    metadata = {
        "version": artifact.version,
        "trained_at": artifact.trained_at.isoformat(),
        "season_through": artifact.season_through,
        "week_through": artifact.week_through,
        "feature_columns": artifact.feature_columns,
        "transforms": artifact.transforms,
        "representation": artifact.representation,
        "classifier_threshold": artifact.classifier_threshold,
        "regressor_threshold": artifact.regressor_threshold,
        "training_row_count": artifact.training_row_count,
        "git_commit": artifact.git_commit,
        "library_versions": artifact.library_versions,
        "classifier_model_name": CLASSIFIER_MODEL,
        "regressor_model_name": REGRESSOR_MODEL,
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_pointer = output_dir / "latest.json"
    tmp_pointer = output_dir / ".latest.json.tmp"
    tmp_pointer.write_text(json.dumps({"version": artifact.version}, indent=2))
    tmp_pointer.replace(latest_pointer)  # atomic swap - no reader ever sees a half-written pointer

    return version_dir


def load_artifact(version_dir: Path) -> ProductionArtifact:
    version_dir = Path(version_dir)
    metadata = json.loads((version_dir / "metadata.json").read_text())
    installed_versions = _library_versions()
    trained_versions = metadata.get("library_versions", {})
    mismatched = {
        lib: (trained_versions[lib], installed_versions[lib])
        for lib in trained_versions
        if lib in installed_versions and trained_versions[lib] != installed_versions[lib]
    }
    if mismatched:
        print(
            f"WARNING: artifact {metadata['version']!r} was trained with different library "
            f"versions than are currently installed - unpickling may behave unexpectedly: "
            f"{mismatched}",
            file=sys.stderr,
        )

    return ProductionArtifact(
        version=metadata["version"],
        trained_at=datetime.fromisoformat(metadata["trained_at"]),
        season_through=metadata["season_through"],
        week_through=metadata["week_through"],
        classifier=joblib.load(version_dir / "classifier.joblib"),
        regressor=joblib.load(version_dir / "regressor.joblib"),
        selected_columns=json.loads((version_dir / "selected_columns.json").read_text()),
        feature_columns=metadata["feature_columns"],
        transforms=metadata["transforms"],
        representation=metadata["representation"],
        classifier_threshold=metadata["classifier_threshold"],
        regressor_threshold=metadata["regressor_threshold"],
        reference_training_frame=pd.read_parquet(version_dir / "reference_training_features.parquet"),
        training_row_count=metadata["training_row_count"],
        git_commit=metadata.get("git_commit"),
        library_versions=trained_versions,
    )


def load_latest_artifact(output_dir: Path = PRODUCTION_DIR) -> ProductionArtifact | None:
    """Returns None (not an exception) when no artifact has ever been trained yet - callers
    (the API's startup lifespan, in particular) need to distinguish "fresh clone, not trained
    yet" from a real error, so they can still start and serve /health while returning a clear
    503 from endpoints that need a model."""
    output_dir = Path(output_dir)
    latest_pointer = output_dir / "latest.json"
    if not latest_pointer.exists():
        return None
    version = json.loads(latest_pointer.read_text())["version"]
    version_dir = output_dir / version
    if not version_dir.exists():
        return None
    return load_artifact(version_dir)


def resolve_default_configs() -> tuple[dict, dict, dict, int]:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    return data_cfg, features_cfg, modeling_cfg, modeling_cfg["random_state"]
