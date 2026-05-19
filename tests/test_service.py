import numpy as np
from fastapi.testclient import TestClient

from watchdog.service import ServiceState, build_app


class _StubModel:
    name = "stub"

    def __init__(self, return_score: float) -> None:
        self.return_score = return_score
        self.version = "test_run_id"

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.return_score)


def _state(iforest_score=0.3, ae_score=0.7, anomaly_threshold=0.5) -> ServiceState:
    return ServiceState(
        iforest=_StubModel(iforest_score),
        autoencoder=_StubModel(ae_score),
        feature_builder=None,
        anomaly_threshold=anomaly_threshold,
        request_log_path=None,
        baseline_scores={"iforest": np.zeros(10), "autoencoder": np.zeros(10)},
    )


def test_healthz_returns_ok():
    app = build_app(_state())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_score_returns_both_scores():
    app = build_app(_state(iforest_score=0.2, ae_score=0.8))
    client = TestClient(app)
    payload = {"window": [[0.0] * 5] * 24}
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["iforest_score"] == 0.2
    assert body["autoencoder_score"] == 0.8
    assert body["is_anomaly"] is True  # 0.8 > 0.5


def test_score_rejects_empty_window():
    app = build_app(_state())
    client = TestClient(app)
    r = client.post("/score", json={"window": []})
    assert r.status_code == 422


def test_admin_drift_with_no_traffic():
    app = build_app(_state())
    client = TestClient(app)
    r = client.get("/admin/drift")
    assert r.status_code == 200
    assert r.json()["status"] == "no_traffic"
