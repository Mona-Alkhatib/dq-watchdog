"""Pydantic types used across the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Metric(str, Enum):
    ROW_COUNT = "row_count"
    NULL_RATE = "null_rate"
    FRESHNESS_LAG_SECONDS = "freshness_lag_seconds"
    SCHEMA_HASH_CHANGES = "schema_hash_changes"
    DUPLICATE_RATE = "duplicate_rate"


class AnomalyType(str, Enum):
    POINT_SPIKE = "point_spike"
    LEVEL_SHIFT = "level_shift"
    GRADUAL_DRIFT = "gradual_drift"
    MISSING_WINDOW = "missing_window"


class MetricPoint(BaseModel):
    timestamp: datetime
    row_count: int = Field(ge=0)
    null_rate: float = Field(ge=0.0, le=1.0)
    freshness_lag_seconds: int = Field(ge=0)
    schema_hash_changes: int = Field(ge=0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)


class PlantedAnomaly(BaseModel):
    timestamp: datetime
    metric: Metric
    type: AnomalyType
    duration_hours: int = 1


class ScoreRequest(BaseModel):
    window: list[list[float]] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    iforest_score: float
    autoencoder_score: float
    is_anomaly: bool
    model_versions: dict[str, str]
