from __future__ import annotations

from fastapi import APIRouter, Depends

from cfb_cover_model.api.artifact_store import ArtifactStore
from cfb_cover_model.api.dependencies import get_artifact_store
from cfb_cover_model.api.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only - never depends on an artifact being loaded, so this always returns 200
    once the process is up, even on a fresh clone before the first training run."""
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
def ready(store: ArtifactStore = Depends(get_artifact_store)) -> ReadinessResponse:
    store.maybe_reload_from_disk()
    artifact = store.current
    if artifact is None:
        return ReadinessResponse(ready=False)
    return ReadinessResponse(
        ready=True,
        artifact_version=artifact.version,
        trained_at=artifact.trained_at,
        training_row_count=artifact.training_row_count,
    )
