"""Eval metrics: precision/recall/F1/AUC-PR per anomaly type.

Anomaly labels are dilated by a tolerance window so a near-miss in
time counts as a true positive (real on-call doesn't fire at the
exact second the spike happens).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def label_windows(
    timestamps: pd.DatetimeIndex | pd.Series,
    anomalies: list[dict[str, Any]],
    tolerance_hours: int = 1,
) -> np.ndarray:
    """Return a boolean array marking which timestamps fall inside any
    anomaly window (with ±tolerance)."""
    ts = pd.to_datetime(pd.Index(timestamps))
    labels = np.zeros(len(ts), dtype=bool)
    tol = timedelta(hours=tolerance_hours)
    for a in anomalies:
        start = pd.to_datetime(a["timestamp"])
        duration = timedelta(hours=int(a.get("duration_hours", 1)))
        window_start = start - tol
        window_end = start + duration + tol
        in_window = (ts >= window_start) & (ts < window_end)
        labels |= np.asarray(in_window)
    return labels


def compute_metrics(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_pred, average="binary", zero_division=0
    )
    auc_pr = float(average_precision_score(y_true.astype(int), scores))
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_pr": auc_pr,
        "n_positives": int(y_true.sum()),
        "n_predicted": int(y_pred.sum()),
    }


def per_type_metrics(
    *,
    timestamps: pd.DatetimeIndex | pd.Series,
    scores: np.ndarray,
    anomalies: list[dict[str, Any]],
    threshold: float,
    tolerance_hours: int = 1,
) -> dict[str, dict[str, float]]:
    """Compute metrics for each anomaly type separately by filtering
    the labels to that type only."""
    by_type: dict[str, dict[str, float]] = {}
    types = {a["type"] for a in anomalies}
    for atype in sorted(types):
        type_anoms = [a for a in anomalies if a["type"] == atype]
        y_true = label_windows(timestamps, type_anoms, tolerance_hours)
        by_type[atype] = compute_metrics(y_true=y_true, scores=scores, threshold=threshold)
    return by_type
