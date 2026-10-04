"""In-process holder for the currently-loaded ProductionArtifact. The API is the single writer
of outputs/models/production/ (see docs/serving_and_monitoring.md) - a retrain happens via
POST /admin/retrain, which trains a new artifact and calls replace() in the same request, so
the hot-swap is atomic from every other request's point of view. maybe_reload_from_disk() is a
defense-in-depth fallback for the case where something wrote a new artifact outside the API
(e.g. a human running scripts/train_production_artifact.py directly on the host).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from cfb_cover_model.serving.artifact import PRODUCTION_DIR, ProductionArtifact, load_artifact, load_latest_artifact


class ArtifactStore:
    def __init__(self, output_dir: Path = PRODUCTION_DIR):
        self.output_dir = Path(output_dir)
        self._current: ProductionArtifact | None = None
        self._latest_pointer_version: str | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> ProductionArtifact | None:
        return self._current

    def load_latest_if_present(self) -> None:
        with self._lock:
            artifact = load_latest_artifact(self.output_dir)
            self._current = artifact
            self._latest_pointer_version = artifact.version if artifact else None

    def replace(self, new_artifact: ProductionArtifact) -> None:
        with self._lock:
            self._current = new_artifact
            self._latest_pointer_version = new_artifact.version

    def maybe_reload_from_disk(self) -> bool:
        """Cheap check: has latest.json's recorded version changed since we last loaded?
        Returns True if a reload happened. Safe to call on every request - the common case
        (no change) is just one small JSON read."""
        pointer_path = self.output_dir / "latest.json"
        if not pointer_path.exists():
            return False
        try:
            on_disk_version = json.loads(pointer_path.read_text())["version"]
        except (json.JSONDecodeError, KeyError, OSError):
            return False
        if on_disk_version == self._latest_pointer_version:
            return False
        with self._lock:
            version_dir = self.output_dir / on_disk_version
            if not version_dir.exists():
                return False
            self._current = load_artifact(version_dir)
            self._latest_pointer_version = on_disk_version
        return True
