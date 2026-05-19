import numpy as np
import pandas as pd

from watchdog.features import FeatureBuilder


def _df(n=48):
    return pd.DataFrame({
        "row_count": np.arange(n) * 10.0,
        "null_rate": np.linspace(0, 0.1, n),
        "freshness_lag_seconds": np.arange(n) * 1.0,
        "schema_hash_changes": np.zeros(n),
        "duplicate_rate": np.linspace(0, 0.01, n),
    })


def test_windowing_produces_expected_shape():
    fb = FeatureBuilder(window_size=24)
    fb.fit(_df(48))
    sequences = fb.transform_sequence(_df(48))
    # n - W + 1 = 48 - 24 + 1 = 25 windows
    assert sequences.shape == (25, 24, 5)


def test_flat_transform_flattens_window():
    fb = FeatureBuilder(window_size=24)
    fb.fit(_df(48))
    flat = fb.transform_flat(_df(48))
    assert flat.shape == (25, 24 * 5)


def test_scaler_fitted_on_train_only():
    fb = FeatureBuilder(window_size=12)
    train = _df(48)
    test = _df(48) * 5  # different scale
    fb.fit(train)
    train_seq = fb.transform_sequence(train)
    test_seq = fb.transform_sequence(test)
    assert abs(train_seq.mean()) < 1.0
    assert abs(test_seq.mean()) > 1.0


def test_save_and_load_round_trip(tmp_path):
    fb = FeatureBuilder(window_size=12)
    fb.fit(_df(48))
    p = tmp_path / "fb.pkl"
    fb.save(p)
    loaded = FeatureBuilder.load(p)
    a = fb.transform_flat(_df(48))
    b = loaded.transform_flat(_df(48))
    assert np.allclose(a, b)
