"""Train both models and log to MLflow.

Each model gets its own MLflow run. Both runs log:
- params (hyperparameters)
- metrics (training time, calibration min/max)
- artifacts (model file + the shared FeatureBuilder)
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from watchdog.features import FeatureBuilder
from watchdog.models.autoencoder import AutoencoderModel
from watchdog.models.iforest import IForestModel

EXPERIMENT_NAME = "dq-watchdog"


def _ensure_experiment() -> None:
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def _train_one(
    model_name: str,
    model: Any,
    X_train: Any,
    X_val: Any,
    feature_builder: FeatureBuilder,
    params: dict[str, Any],
) -> dict[str, Any]:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("window_size", feature_builder.window_size)

        t0 = time.perf_counter()
        model.fit(X_train, X_val)
        elapsed = time.perf_counter() - t0

        mlflow.log_metric("training_seconds", elapsed)

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            model_path = tdp / f"{model_name}.pkl"
            scaler_path = tdp / "feature_builder.pkl"
            model.save(model_path)
            feature_builder.save(scaler_path)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(scaler_path))

        return {"run_id": run.info.run_id, "training_seconds": elapsed}


def train_both(
    *,
    df: pd.DataFrame,
    window_size: int = 24,
    iforest_params: dict[str, Any] | None = None,
    autoencoder_params: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    val_fraction: float = 0.2,
) -> dict[str, dict[str, Any]]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment()

    n = len(df)
    cut = int(n * (1 - val_fraction))
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]

    fb = FeatureBuilder(window_size=window_size)
    fb.fit(train_df)
    X_train_seq = fb.transform_sequence(train_df)
    X_val_seq = fb.transform_sequence(val_df)
    X_train_flat = fb.transform_flat(train_df)
    X_val_flat = fb.transform_flat(val_df)

    iforest_params = iforest_params or {}
    autoencoder_params = autoencoder_params or {}

    return {
        "iforest": _train_one(
            "iforest", IForestModel(**iforest_params),
            X_train_flat, X_val_flat, fb, iforest_params,
        ),
        "autoencoder": _train_one(
            "autoencoder", AutoencoderModel(**autoencoder_params),
            X_train_seq, X_val_seq, fb, autoencoder_params,
        ),
    }
