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


def calibrate(raw: np.ndarray, cal_min: float, cal_max: float) -> np.ndarray:
    """Min-max scale raw anomaly scores into [0, 1], clipped.

    If the calibration range is degenerate (min == max), returns zeros.
    Shared by IForestModel and AutoencoderModel — both calibrate raw
    scores against a validation split using this identical mapping.
    """
    if cal_max == cal_min:
        return np.zeros_like(raw)
    return np.clip((raw - cal_min) / (cal_max - cal_min), 0.0, 1.0)
