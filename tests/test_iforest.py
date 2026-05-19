import numpy as np

from watchdog.models.iforest import IForestModel


def _rand_X(rows=50, dim=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, dim))


def test_iforest_fit_score_returns_array_in_unit_interval():
    m = IForestModel(n_estimators=50, random_state=0)
    X_train = _rand_X(50)
    X_val = _rand_X(20)
    m.fit(X_train, X_val)
    scores = m.score(_rand_X(10))
    assert scores.shape == (10,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_iforest_is_deterministic_with_seed():
    a = IForestModel(n_estimators=50, random_state=42)
    b = IForestModel(n_estimators=50, random_state=42)
    X_train, X_val = _rand_X(50), _rand_X(20)
    a.fit(X_train, X_val)
    b.fit(X_train, X_val)
    X_test = _rand_X(10, seed=99)
    assert np.allclose(a.score(X_test), b.score(X_test))


def test_iforest_save_and_load(tmp_path):
    m = IForestModel(n_estimators=20, random_state=0)
    m.fit(_rand_X(50), _rand_X(20))
    p = tmp_path / "iforest.pkl"
    m.save(p)
    loaded = IForestModel.load(p)
    X_test = _rand_X(5, seed=7)
    assert np.allclose(m.score(X_test), loaded.score(X_test))


def test_iforest_name():
    assert IForestModel().name == "iforest"
