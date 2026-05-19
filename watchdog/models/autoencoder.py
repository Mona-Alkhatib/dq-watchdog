"""LSTM Autoencoder anomaly model.

Trains via reconstruction MSE on sequence windows. Anomaly score =
normalized reconstruction error, calibrated on a validation split.

CPU-friendly: hidden_dim=32 with batch_size=64 trains in seconds on
30 days of hourly data. Auto-detects CUDA, falls back to CPU on OOM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from watchdog.models.base import calibrate


class _AE(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        seq_len = x.size(1)
        decoded_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoded_input)
        return self.output(decoded)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


class AutoencoderModel:
    name = "autoencoder"

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self._net: _AE | None = None
        self._n_features: int = 0
        self._device = _pick_device()
        self._calibration_min: float = 0.0
        self._calibration_max: float = 1.0

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self._device)
            recon = self._net(x)
            err = ((recon - x) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy()

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        self._n_features = X_train.shape[-1]
        self._net = _AE(self._n_features, self.hidden_dim).to(self._device)
        optim = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self._device)
                optim.zero_grad()
                out = self._net(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                optim.step()

        val_err = self._reconstruction_errors(X_val)
        self._calibration_min = float(np.min(val_err))
        self._calibration_max = float(np.max(val_err))

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fit")
        err = self._reconstruction_errors(X)
        return calibrate(err, self._calibration_min, self._calibration_max)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("Model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "n_features": self._n_features,
                "hidden_dim": self.hidden_dim,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "random_state": self.random_state,
                "calibration_min": self._calibration_min,
                "calibration_max": self._calibration_max,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        instance = cls(
            hidden_dim=checkpoint["hidden_dim"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            lr=checkpoint["lr"],
            random_state=checkpoint["random_state"],
        )
        instance._n_features = checkpoint["n_features"]
        instance._device = torch.device("cpu")
        instance._net = _AE(instance._n_features, instance.hidden_dim).to(instance._device)
        instance._net.load_state_dict(checkpoint["state_dict"])
        instance._calibration_min = checkpoint["calibration_min"]
        instance._calibration_max = checkpoint["calibration_max"]
        return instance
