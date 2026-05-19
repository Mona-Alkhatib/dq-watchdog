import numpy as np

from watchdog.drift import classify_psi, compute_psi


def test_psi_zero_for_identical_distributions():
    baseline = np.random.default_rng(0).beta(2, 5, size=1000)
    production = np.random.default_rng(0).beta(2, 5, size=1000)
    psi = compute_psi(baseline=baseline, production=production, n_bins=10)
    assert psi < 0.05


def test_psi_significant_for_shifted_distributions():
    baseline = np.random.default_rng(0).beta(2, 5, size=1000)
    production = np.random.default_rng(1).beta(5, 2, size=1000)
    psi = compute_psi(baseline=baseline, production=production, n_bins=10)
    assert psi > 0.3


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "monitor"
    assert classify_psi(0.30) == "drift"


def test_psi_handles_empty_bins_with_epsilon():
    baseline = np.linspace(0, 1, 200)
    production = np.array([0.5] * 200)
    psi = compute_psi(baseline=baseline, production=production, n_bins=10)
    assert psi > 0
    assert np.isfinite(psi)
