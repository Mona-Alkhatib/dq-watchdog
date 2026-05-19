"""Population Stability Index (PSI).

PSI = sum over bins of (p_prod - p_base) * ln(p_prod / p_base).

Thresholds:
  PSI < 0.1   → stable
  0.1 ≤ PSI < 0.2 → monitor
  PSI ≥ 0.2   → drift (warning)

Empty bins use ε-substitution (replace 0 with 1e-6) so the log is finite.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

PSI_MONITOR = 0.1
PSI_DRIFT = 0.2
EPSILON = 1e-6


def compute_psi(
    *,
    baseline: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    base_hist, _ = np.histogram(baseline, bins=bin_edges)
    prod_hist, _ = np.histogram(production, bins=bin_edges)

    base_pct = base_hist / max(base_hist.sum(), 1)
    prod_pct = prod_hist / max(prod_hist.sum(), 1)

    base_pct = np.where(base_pct == 0, EPSILON, base_pct)
    prod_pct = np.where(prod_pct == 0, EPSILON, prod_pct)

    psi = float(np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct)))
    return psi


def classify_psi(psi: float) -> Literal["stable", "monitor", "drift"]:
    if psi < PSI_MONITOR:
        return "stable"
    if psi < PSI_DRIFT:
        return "monitor"
    return "drift"
