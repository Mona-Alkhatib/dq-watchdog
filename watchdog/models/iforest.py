"""Isolation Forest anomaly model.

decision_function returns a value where higher = more normal. We
negate and min-max scale on the validation split so output scores
live in [0, 1] with 1 = most anomalous.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


class IForestModel:
    name = "iforest"

    def __init__(self, n_estimators: int = 200, random_state: int = 0) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._calibration_min: float = 0.0
        self._calibration_max: float = 1.0

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> None:
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.random_state,
        )
        self._model.fit(X_train)
        raw_val = -self._model.decision_function(X_val)
        self._calibration_min = float(np.min(raw_val))
        self._calibration_max = float(np.max(raw_val))

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fit")
        raw = -self._model.decision_function(X)
        if self._calibration_max == self._calibration_min:
            return np.zeros_like(raw)
        scaled = (raw - self._calibration_min) / (
            self._calibration_max - self._calibration_min
        )
        return np.clip(scaled, 0.0, 1.0)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "IForestModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)
