from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GamePrediction(BaseModel):
    """One game's dual-model prediction. Probabilities are documented as poorly calibrated
    (Brier 0.257 on the 2025 holdout, see docs/final_writeup_2026.md) - treat the *_flag /
    agreement_bet booleans as the primary signal, probabilities as secondary context only."""

    game_id: int
    home_team: str
    away_team: str
    season: int
    week: int
    spread: float
    logistic_regression_probability: float
    logistic_regression_flag: bool
    xgboost_regressor_probability: float
    xgboost_regressor_flag: bool
    agreement_bet: bool
    avg_probability: float
    artifact_version: str


class WeekPredictionsResponse(BaseModel):
    artifact_version: str
    season: int
    week: int
    mode: Literal["csv", "live"]
    generated_at: datetime
    n_games: int
    n_agreement_bets: int
    predictions: list[GamePrediction]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    artifact_version: str | None = None
    trained_at: datetime | None = None
    training_row_count: int | None = None


class MonitoringStatusResponse(BaseModel):
    season: int
    week: int
    data_drift: dict | None = None
    prediction_drift: dict | None = None


class RetrainRequest(BaseModel):
    season: int = Field(..., description="Most recent completed season")
    week: int = Field(..., description="Most recent completed week")


class RetrainResponse(BaseModel):
    version: str
    trained_at: datetime
    training_row_count: int
