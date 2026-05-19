"""Synthetic timeseries generator for data quality metrics.

Emits 5 metrics per hour with realistic baselines + daily seasonality
+ Gaussian noise. Optionally injects anomalies at specific timestamps.

Deterministic given a seed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

START = datetime(2025, 1, 1, 0, 0, 0)

BASELINES = {
    "row_count": {"mean": 50_000, "amplitude": 15_000, "noise_sd": 2_000},
    "null_rate": {"mean": 0.02, "amplitude": 0.005, "noise_sd": 0.003},
    "freshness_lag_seconds": {"mean": 600, "amplitude": 300, "noise_sd": 60},
    "schema_hash_changes": {"mean": 0, "amplitude": 0, "noise_sd": 0},
    "duplicate_rate": {"mean": 0.002, "amplitude": 0.0005, "noise_sd": 0.0003},
}


def _baseline_series(rng: np.random.Generator, n: int, params: dict) -> np.ndarray:
    t = np.arange(n)
    seasonal = params["amplitude"] * np.sin(2 * np.pi * t / 24)
    noise = rng.normal(0, params["noise_sd"], size=n)
    return params["mean"] + seasonal + noise


def _clip(values: np.ndarray, lo: float, hi: float | None) -> np.ndarray:
    return np.clip(values, lo, hi)


def _apply_anomaly(
    df: pd.DataFrame,
    rng: np.random.Generator,
    timestamp: datetime,
    metric: str,
    anomaly_type: str,
    duration_hours: int,
) -> None:
    idx = df.index[df["timestamp"] >= timestamp]
    if len(idx) == 0:
        return
    start = idx[0]
    end = min(start + duration_hours, len(df))

    if anomaly_type == "point_spike":
        baseline = df.at[start, metric]
        if metric == "null_rate":
            df.at[start, metric] = 0.5
        else:
            df.at[start, metric] = baseline * 10
    elif anomaly_type == "level_shift":
        scale = 1.6 if rng.random() < 0.5 else 0.4
        new_vals = df.loc[start : end - 1, metric] * scale
        df.loc[start : end - 1, metric] = new_vals.astype(df[metric].dtype)
    elif anomaly_type == "gradual_drift":
        steps = end - start
        trend = np.linspace(1.0, 2.0, steps)
        new_vals = df.loc[start : end - 1, metric] * trend
        df.loc[start : end - 1, metric] = new_vals.astype(df[metric].dtype)
    elif anomaly_type == "missing_window":
        df.loc[start : end - 1, metric] = 0


def generate_timeseries(
    *,
    seed: int,
    num_days: int = 30,
    hours_per_day: int = 24,
    anomalies: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n = num_days * hours_per_day
    timestamps = [START + timedelta(hours=i) for i in range(n)]
    df = pd.DataFrame({"timestamp": timestamps})

    for metric, params in BASELINES.items():
        series = _baseline_series(rng, n, params)
        if metric in {"null_rate", "duplicate_rate"}:
            series = _clip(series, 0.0, 1.0)
        elif metric == "row_count":
            series = _clip(series, 0, None).astype(int)
        elif metric == "freshness_lag_seconds":
            series = _clip(series, 0, None).astype(int)
        elif metric == "schema_hash_changes":
            series = np.zeros(n, dtype=int)
        df[metric] = series

    labels: list[dict[str, Any]] = []
    for a in anomalies or []:
        _apply_anomaly(
            df,
            rng,
            timestamp=a["timestamp"],
            metric=a["metric"],
            anomaly_type=a["type"],
            duration_hours=int(a.get("duration_hours", 1)),
        )
        labels.append({
            "timestamp": a["timestamp"].isoformat() if isinstance(a["timestamp"], datetime) else a["timestamp"],
            "metric": a["metric"],
            "type": a["type"],
            "duration_hours": int(a.get("duration_hours", 1)),
        })

    return df, labels
