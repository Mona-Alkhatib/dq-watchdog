"""Eval harness: precision/recall/F1/AUC-PR per model per anomaly type.

Slow — trains both models. Skipped unless WATCHDOG_RUN_EVALS=1.
"""
from __future__ import annotations

from datetime import datetime

import mlflow
import numpy as np
import pytest

from evals.conftest import RUN_EVALS, load_fixture
from watchdog.drift import compute_psi
from watchdog.eval import per_type_metrics
from watchdog.features import FeatureBuilder
from watchdog.models.autoencoder import AutoencoderModel
from watchdog.models.iforest import IForestModel

F1_THRESHOLD = 0.70
AUC_THRESHOLD = 0.80


@pytest.mark.skipif(not RUN_EVALS, reason="set WATCHDOG_RUN_EVALS=1 to run")
def test_benchmark_v1_both_models(tmp_path, monkeypatch):
    tracking = f"sqlite:///{tmp_path / 'mlruns.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    mlflow.set_tracking_uri(tracking)

    train_df, test_df, labels = load_fixture("benchmark_v1")

    fb = FeatureBuilder(window_size=24)
    fb.fit(train_df)
    X_train_seq = fb.transform_sequence(train_df)
    X_train_flat = fb.transform_flat(train_df)

    cut = int(len(X_train_seq) * 0.8)
    val_seq = X_train_seq[cut:]
    val_flat = X_train_flat[cut:]
    X_train_seq = X_train_seq[:cut]
    X_train_flat = X_train_flat[:cut]

    iforest = IForestModel(n_estimators=200, random_state=7)
    iforest.fit(X_train_flat, val_flat)
    autoencoder = AutoencoderModel(hidden_dim=32, epochs=15, batch_size=64, random_state=7)
    autoencoder.fit(X_train_seq, val_seq)

    test_seq = fb.transform_sequence(test_df)
    test_flat = fb.transform_flat(test_df)
    timestamps = test_df["timestamp"].iloc[fb.window_size - 1 :].reset_index(drop=True)

    iforest_scores = iforest.score(test_flat)
    ae_scores = autoencoder.score(test_seq)

    per_type_if = per_type_metrics(
        timestamps=timestamps,
        scores=iforest_scores,
        anomalies=labels,
        threshold=0.5,
    )
    per_type_ae = per_type_metrics(
        timestamps=timestamps,
        scores=ae_scores,
        anomalies=labels,
        threshold=0.5,
    )

    iforest_f1 = np.mean([m["f1"] for m in per_type_if.values()])
    ae_f1 = np.mean([m["f1"] for m in per_type_ae.values()])
    iforest_auc = np.mean([m["auc_pr"] for m in per_type_if.values()])
    ae_auc = np.mean([m["auc_pr"] for m in per_type_ae.values()])

    print(f"\n=== benchmark_v1 ===")
    print(f"IForest    F1={iforest_f1:.3f}  AUC-PR={iforest_auc:.3f}")
    print(f"Autoenc.   F1={ae_f1:.3f}  AUC-PR={ae_auc:.3f}")
    print("Per-type (F1):")
    for t in per_type_if:
        print(f"  {t}: IForest={per_type_if[t]['f1']:.3f}, AE={per_type_ae[t]['f1']:.3f}")

    assert iforest_f1 >= F1_THRESHOLD or ae_f1 >= F1_THRESHOLD, (
        f"Both models below F1 threshold ({F1_THRESHOLD}): "
        f"IF={iforest_f1:.3f}, AE={ae_f1:.3f}"
    )
    assert iforest_auc >= AUC_THRESHOLD or ae_auc >= AUC_THRESHOLD, (
        f"Both models below AUC-PR threshold ({AUC_THRESHOLD}): "
        f"IF={iforest_auc:.3f}, AE={ae_auc:.3f}"
    )


@pytest.mark.skipif(not RUN_EVALS, reason="set WATCHDOG_RUN_EVALS=1 to run")
def test_drift_detected_within_five_days(tmp_path, monkeypatch):
    """Generate a drifted stream and assert PSI crosses 0.2 within 5 days."""
    train_df, _, _ = load_fixture("benchmark_v1")

    fb = FeatureBuilder(window_size=24)
    fb.fit(train_df)
    iforest = IForestModel(n_estimators=100, random_state=11)
    iforest.fit(fb.transform_flat(train_df), fb.transform_flat(train_df))

    baseline_scores = iforest.score(fb.transform_flat(train_df))

    drifted_scores_by_day: list[float] = []
    days_until_drift_detected: int | None = None
    for day in range(30):
        per_day = train_df.iloc[: 24].copy()
        per_day.loc[:, "row_count"] = per_day["row_count"] * (1 + 0.10 * day)
        per_day.loc[:, "null_rate"] = (per_day["null_rate"] * (1 + 0.10 * day)).clip(0, 1)
        per_day.loc[:, "freshness_lag_seconds"] = per_day["freshness_lag_seconds"] * (1 + 0.10 * day)
        if len(per_day) < fb.window_size:
            continue
        day_scores = iforest.score(fb.transform_flat(per_day))
        drifted_scores_by_day.extend(day_scores.tolist())
        psi = compute_psi(
            baseline=baseline_scores,
            production=np.array(drifted_scores_by_day),
        )
        if psi >= 0.2 and days_until_drift_detected is None:
            days_until_drift_detected = day + 1
            break

    assert days_until_drift_detected is not None and days_until_drift_detected <= 5, (
        f"Drift not detected within 5 days. days_until_detected={days_until_drift_detected}"
    )
