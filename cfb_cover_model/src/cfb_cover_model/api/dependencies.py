from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request

from cfb_cover_model.api.artifact_store import ArtifactStore
from cfb_cover_model.serving.artifact import ProductionArtifact


def get_artifact_store(request: Request) -> ArtifactStore:
    return request.app.state.artifact_store


def require_artifact(request: Request) -> ProductionArtifact:
    store: ArtifactStore = request.app.state.artifact_store
    store.maybe_reload_from_disk()
    if store.current is None:
        raise HTTPException(
            status_code=503,
            detail="No production artifact found - run `python scripts/train_production_artifact.py "
            "--season <Y> --week <N>` at least once before requesting predictions.",
        )
    return store.current


def require_admin_token(x_admin_token: str = Header(default="")) -> None:
    expected = os.environ.get("ADMIN_API_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_TOKEN is not set on the server - refusing to allow retrain requests.",
        )
    import secrets

    if not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token header.")
