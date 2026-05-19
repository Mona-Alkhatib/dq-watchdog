import numpy as np

from watchdog.models.autoencoder import AutoencoderModel


def _rand_seq(rows=30, window=12, features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, window, features)).astype(np.float32)


def test_autoencoder_fit_score_returns_unit_interval():
    m = AutoencoderModel(hidden_dim=8, epochs=2, batch_size=8, random_state=0)
    X_train = _rand_seq(40)
    X_val = _rand_seq(10, seed=1)
    m.fit(X_train, X_val)
    scores = m.score(_rand_seq(5, seed=2))
    assert scores.shape == (5,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_autoencoder_save_and_load(tmp_path):
    m = AutoencoderModel(hidden_dim=4, epochs=2, batch_size=8, random_state=0)
    m.fit(_rand_seq(40), _rand_seq(10, seed=1))
    p = tmp_path / "ae.pt"
    m.save(p)
    loaded = AutoencoderModel.load(p)
    X_test = _rand_seq(3, seed=7)
    a = m.score(X_test)
    b = loaded.score(X_test)
    assert np.allclose(a, b, atol=1e-5)


def test_autoencoder_name():
    assert AutoencoderModel().name == "autoencoder"
