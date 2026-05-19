from datetime import datetime

import mlflow
import pandas as pd

from watchdog.generate import generate_timeseries
from watchdog.train import train_both


def test_train_both_logs_two_mlflow_runs(tmp_path, monkeypatch):
    tracking = f"sqlite:///{tmp_path / 'mlruns.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    mlflow.set_tracking_uri(tracking)

    df, _ = generate_timeseries(seed=11, num_days=4)
    result = train_both(
        df=df,
        window_size=12,
        iforest_params={"n_estimators": 10},
        autoencoder_params={"hidden_dim": 4, "epochs": 2, "batch_size": 8},
        tracking_uri=tracking,
    )

    assert "iforest" in result and "autoencoder" in result
    assert result["iforest"]["run_id"]
    assert result["autoencoder"]["run_id"]

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking)
    iforest_run = client.get_run(result["iforest"]["run_id"])
    assert iforest_run.data.params["model"] == "iforest"
    assert "training_seconds" in iforest_run.data.metrics
