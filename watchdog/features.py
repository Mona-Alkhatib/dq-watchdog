"""Windowing + normalization for data quality metric timeseries.

FeatureBuilder is fit on training data only. It exposes two transforms:
- transform_flat: (n - W + 1, W * F) for classical models (Isolation Forest)
- transform_sequence: (n - W + 1, W, F) for sequence models (LSTM)

Persists to a single pickle file alongside the model in MLflow.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

METRIC_COLUMNS = [
    "row_count",
    "null_rate",
    "freshness_lag_seconds",
    "schema_hash_changes",
    "duplicate_rate",
]


class FeatureBuilder:
    def __init__(self, window_size: int = 24) -> None:
        self.window_size = window_size
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df[METRIC_COLUMNS].to_numpy(dtype=float))
        self._fitted = True

    def _scaled_values(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        return self.scaler.transform(df[METRIC_COLUMNS].to_numpy(dtype=float))

    def transform_sequence(self, df: pd.DataFrame) -> np.ndarray:
        scaled = self._scaled_values(df)
        n = len(scaled)
        if n < self.window_size:
            raise ValueError(f"need at least {self.window_size} rows, got {n}")
        return np.stack(
            [scaled[i : i + self.window_size] for i in range(n - self.window_size + 1)]
        )

    def transform_flat(self, df: pd.DataFrame) -> np.ndarray:
        seq = self.transform_sequence(df)
        return seq.reshape(seq.shape[0], -1)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> FeatureBuilder:
        with Path(path).open("rb") as f:
            return pickle.load(f)
