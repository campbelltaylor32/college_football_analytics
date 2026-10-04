from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from cfb_cover_model.api.artifact_store import ArtifactStore
from cfb_cover_model.api.routers import admin, health, monitoring, predictions


def create_app(artifact_store: ArtifactStore | None = None) -> FastAPI:
    """`artifact_store` is only ever passed explicitly by tests, to inject a fixture artifact
    without touching outputs/models/production/ on disk. Real deployments always call this
    with no arguments and let the lifespan below load whatever scripts/train_production_artifact.py
    last wrote."""
    store = artifact_store if artifact_store is not None else ArtifactStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if artifact_store is None:
            store.load_latest_if_present()
        yield

    app = FastAPI(
        title="cfb_cover_model inference API",
        description="Serves the dual-model (logistic_regression + xgboost_regressor agreement) "
        "spread-cover prediction signal. See docs/serving_and_monitoring.md.",
        lifespan=lifespan,
    )
    app.state.artifact_store = store

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(monitoring.router)
    app.include_router(admin.router)

    return app
