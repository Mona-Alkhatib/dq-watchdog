from datetime import datetime, timedelta

import pandas as pd

from watchdog.generate import generate_timeseries
from watchdog.models_types import AnomalyType, Metric


def test_generate_produces_expected_row_count():
    df, _ = generate_timeseries(seed=42, num_days=2, hours_per_day=24, anomalies=[])
    assert len(df) == 48


def test_generate_includes_all_five_metrics():
    df, _ = generate_timeseries(seed=42, num_days=1, anomalies=[])
    assert {col for col in df.columns} >= {m.value for m in Metric} | {"timestamp"}


def test_generate_is_deterministic_given_seed():
    df1, _ = generate_timeseries(seed=7, num_days=2, anomalies=[])
    df2, _ = generate_timeseries(seed=7, num_days=2, anomalies=[])
    assert df1.equals(df2)


def test_generate_baselines_in_expected_ranges():
    df, _ = generate_timeseries(seed=42, num_days=5, anomalies=[])
    assert df["row_count"].min() > 0
    assert (df["null_rate"] >= 0).all() and (df["null_rate"] <= 1).all()
    assert (df["duplicate_rate"] >= 0).all() and (df["duplicate_rate"] <= 1).all()


def test_generate_injects_point_spike_at_labeled_timestamp():
    spike_at = datetime(2025, 1, 2, 12, 0, 0)
    df, labels = generate_timeseries(
        seed=42,
        num_days=5,
        anomalies=[
            {"timestamp": spike_at, "metric": "null_rate", "type": "point_spike"}
        ],
    )
    row = df[df["timestamp"] == spike_at].iloc[0]
    assert row["null_rate"] > 0.3
    assert any(lbl["type"] == "point_spike" for lbl in labels)


def test_generate_injects_level_shift_persists_for_duration():
    shift_at = datetime(2025, 1, 2, 0, 0, 0)
    df, _ = generate_timeseries(
        seed=42,
        num_days=5,
        anomalies=[
            {
                "timestamp": shift_at,
                "metric": "row_count",
                "type": "level_shift",
                "duration_hours": 6,
            }
        ],
    )
    pre = df[df["timestamp"] < shift_at]["row_count"].mean()
    during = df[
        (df["timestamp"] >= shift_at)
        & (df["timestamp"] < shift_at + timedelta(hours=6))
    ]["row_count"].mean()
    assert during > pre * 1.3 or during < pre * 0.7


def test_generate_missing_window_zeroes_metric():
    miss_at = datetime(2025, 1, 2, 0, 0, 0)
    df, _ = generate_timeseries(
        seed=42,
        num_days=5,
        anomalies=[
            {
                "timestamp": miss_at,
                "metric": "row_count",
                "type": "missing_window",
                "duration_hours": 3,
            }
        ],
    )
    window = df[
        (df["timestamp"] >= miss_at)
        & (df["timestamp"] < miss_at + timedelta(hours=3))
    ]
    assert (window["row_count"] == 0).all()
