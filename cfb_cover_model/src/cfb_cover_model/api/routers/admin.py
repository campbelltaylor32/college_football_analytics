from __future__ import annotations

from fastapi import APIRouter, Depends

from cfb_cover_model.api.artifact_store import ArtifactStore
from cfb_cover_model.api.dependencies import get_artifact_store, require_admin_token
from cfb_cover_model.api.schemas import RetrainRequest, RetrainResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(require_admin_token)])
def retrain(body: RetrainRequest, store: ArtifactStore = Depends(get_artifact_store)) -> RetrainResponse:
    """Synchronous, in-process retrain - blocks this worker for the full training duration
    (tens of seconds given ~1,400 rows + XGBoost + embedded elastic-net selection). Acceptable
    for a local, single-operator, weekly-cadence deployment; not safe under concurrent traffic.
    The API is the only writer of outputs/models/production/ by design (see
    docs/serving_and_monitoring.md) - scripts/run_scheduler.py calls this endpoint over HTTP
    rather than running scripts/train_production_artifact.py directly, so the hot-swap below
    always happens in the same process/request that persisted the new artifact to disk."""
    import train_production_artifact  # scripts/ - see serving/artifact.py's sys.path note

    artifact = train_production_artifact.run(body.season, body.week)
    store.replace(artifact)

    return RetrainResponse(
        version=artifact.version,
        trained_at=artifact.trained_at,
        training_row_count=artifact.training_row_count,
    )
