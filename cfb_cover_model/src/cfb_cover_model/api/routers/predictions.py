from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from cfb_cover_model.api.dependencies import require_artifact
from cfb_cover_model.api.schemas import WeekPredictionsResponse
from cfb_cover_model.config import load_data_config, resolve_path
from cfb_cover_model.data import load_week_predictors_df
from cfb_cover_model.serving.artifact import ProductionArtifact
from cfb_cover_model.serving.scoring import score_week

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _default_week_csv_path(data_cfg: dict, week: int) -> Path:
    """Mirrors scripts/generate_weekly_predictions.py::resolve_week_file's --week convention."""
    return resolve_path(data_cfg["paths"]["predictors_csv"]).parent / f"CFB_Pred_Week_{week}.csv"


@router.get("/week/{week}", response_model=WeekPredictionsResponse)
def predict_week(
    week: int,
    season: int = Query(..., description="Season year. Always required: CFB_Pred_Week_<N>.csv's own "
                         "season column is unreliable (observed all-NaN in real weekly files), so the "
                         "caller's value is authoritative and is what's returned, not whatever (if "
                         "anything) is in the scored frame."),
    file: str | None = Query(default=None, description="Explicit CSV path override"),
    live: bool = Query(default=False, description="Pull features directly from the CFBD API instead of a CSV"),
    artifact: ProductionArtifact = Depends(require_artifact),
) -> WeekPredictionsResponse:
    data_cfg = load_data_config()

    if live:
        from cfb_cover_model.ingest import cfbd_client, pipeline

        client = cfbd_client.get_client()
        week_df_raw = pipeline.build_current_week_rows(client, season, week)
        mode = "live"
    else:
        path = Path(file).resolve() if file else _default_week_csv_path(data_cfg, week)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Week file not found: {path}")
        week_df_raw = load_week_predictors_df(path)
        mode = "csv"

    if week_df_raw.empty:
        raise HTTPException(status_code=422, detail=f"No games found for season={season} week={week}")

    out = score_week(artifact, week_df_raw, data_cfg)
    out["season"] = season  # authoritative over the scored frame's own (possibly unreliable) column
    predictions = out.to_dict(orient="records")

    return WeekPredictionsResponse(
        artifact_version=artifact.version,
        season=season,
        week=week,
        mode=mode,
        generated_at=datetime.now(timezone.utc),
        n_games=len(out),
        n_agreement_bets=int(out["agreement_bet"].sum()),
        predictions=predictions,
    )
