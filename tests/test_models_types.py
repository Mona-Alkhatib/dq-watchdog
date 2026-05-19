from datetime import datetime

import pytest
from pydantic import ValidationError

from watchdog.models_types import (
    AnomalyType,
    Metric,
    MetricPoint,
    ScoreRequest,
    ScoreResponse,
)


def test_anomaly_type_values():
    assert {t.value for t in AnomalyType} == {
        "point_spike", "level_shift", "gradual_drift", "missing_window"
    }


def test_metric_values():
    assert {m.value for m in Metric} == {
        "row_count", "null_rate", "freshness_lag_seconds", "schema_hash_changes", "duplicate_rate"
    }


def test_metric_point_basic():
    pt = MetricPoint(
        timestamp=datetime(2025, 1, 1, 0, 0, 0),
        row_count=100,
        null_rate=0.01,
        freshness_lag_seconds=120,
        schema_hash_changes=0,
        duplicate_rate=0.001,
    )
    assert pt.row_count == 100


def test_metric_point_rejects_negative_row_count():
    with pytest.raises(ValidationError):
        MetricPoint(
            timestamp=datetime(2025, 1, 1),
            row_count=-1,
            null_rate=0.01,
            freshness_lag_seconds=120,
            schema_hash_changes=0,
            duplicate_rate=0.001,
        )


def test_metric_point_rejects_null_rate_outside_unit_interval():
    with pytest.raises(ValidationError):
        MetricPoint(
            timestamp=datetime(2025, 1, 1),
            row_count=100,
            null_rate=1.5,
            freshness_lag_seconds=120,
            schema_hash_changes=0,
            duplicate_rate=0.001,
        )


def test_score_request_window_must_be_non_empty():
    with pytest.raises(ValidationError):
        ScoreRequest(window=[])


def test_score_response_basic():
    r = ScoreResponse(
        iforest_score=0.6,
        autoencoder_score=0.8,
        is_anomaly=True,
        model_versions={"iforest": "run_abc", "autoencoder": "run_xyz"},
    )
    assert r.is_anomaly is True
