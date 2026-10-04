from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from cfb_cover_model.api.schemas import MonitoringStatusResponse
from cfb_cover_model.monitoring.data_drift import DATA_DRIFT_DIR
from cfb_cover_model.monitoring.prediction_drift import PREDICTION_DRIFT_DIR

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@router.get("/status", response_model=MonitoringStatusResponse)
def monitoring_status(
    season: int = Query(...),
    week: int = Query(..., description="Week the reports were generated for"),
) -> MonitoringStatusResponse:
    data_drift = _read_json(DATA_DRIFT_DIR / f"{season}_week_{week}_data_drift.json")
    prediction_drift = _read_json(PREDICTION_DRIFT_DIR / f"{season}_week_{week}_prediction_drift.json")

    if data_drift is None and prediction_drift is None:
        raise HTTPException(
            status_code=404,
            detail=f"No monitoring reports found for season={season} week={week} - has "
            "scripts/run_weekly_monitoring.py been run for this week yet?",
        )

    return MonitoringStatusResponse(season=season, week=week, data_drift=data_drift, prediction_drift=prediction_drift)
