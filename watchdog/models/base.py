"""Shared Protocol for anomaly models.

Both Isolation Forest and LSTM Autoencoder implement this interface so
the rest of the pipeline (training, eval, service) is model-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class AnomalyModel(Protocol):
    name: str

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> None: ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]. Higher = more anomalous."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> AnomalyModel: ...
