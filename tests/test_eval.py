import numpy as np
import pandas as pd

from watchdog.eval import compute_metrics, label_windows


def test_label_windows_marks_anomaly_windows():
    timestamps = pd.date_range("2025-01-01", periods=10, freq="h")
    anomalies = [
        {"timestamp": "2025-01-01T05:00:00", "type": "point_spike", "duration_hours": 1}
    ]
    labels = label_windows(timestamps, anomalies, tolerance_hours=1)
    assert labels[4] and labels[5] and labels[6]
    assert not labels[0] and not labels[9]


def test_compute_metrics_basic_precision_recall():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6])
    metrics = compute_metrics(y_true=y_true, scores=scores, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc_pr"] > 0.95


def test_compute_metrics_with_misses():
    y_true = np.array([0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.4, 0.3])
    metrics = compute_metrics(y_true=y_true, scores=scores, threshold=0.5)
    assert abs(metrics["recall"] - (1 / 3)) < 1e-6
