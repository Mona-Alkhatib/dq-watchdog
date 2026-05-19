"""FastAPI service for anomaly scoring + drift inspection.

build_app(state) returns a FastAPI app configured with two models and
their shared baseline score distributions. Request scores are logged
to a SQLite (or in-memory list during tests) and used to compute PSI
on demand.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from watchdog.drift import classify_psi, compute_psi
from watchdog.models_types import ScoreRequest, ScoreResponse


@dataclass
class ServiceState:
    iforest: Any
    autoencoder: Any
    feature_builder: Any
    anomaly_threshold: float = 0.5
    request_log_path: Path | None = None
    baseline_scores: dict[str, np.ndarray] = field(default_factory=dict)
    in_memory_log: list[dict[str, float]] = field(default_factory=list)

    def log_scores(self, iforest_score: float, ae_score: float) -> None:
        if self.request_log_path is None:
            self.in_memory_log.append(
                {"iforest_score": iforest_score, "autoencoder_score": ae_score}
            )
            return
        con = sqlite3.connect(str(self.request_log_path))
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS scores ("
                "  ts INTEGER DEFAULT (strftime('%s','now')),"
                "  iforest REAL, autoencoder REAL)"
            )
            con.execute(
                "INSERT INTO scores (iforest, autoencoder) VALUES (?, ?)",
                (iforest_score, ae_score),
            )
            con.commit()
        finally:
            con.close()

    def recent_scores(self) -> dict[str, np.ndarray]:
        if self.request_log_path is None:
            rows = [(r["iforest_score"], r["autoencoder_score"]) for r in self.in_memory_log]
        else:
            con = sqlite3.connect(str(self.request_log_path))
            try:
                rows = con.execute("SELECT iforest, autoencoder FROM scores").fetchall()
            finally:
                con.close()
        if not rows:
            return {}
        arr = np.array(rows)
        return {"iforest": arr[:, 0], "autoencoder": arr[:, 1]}


def build_app(state: ServiceState) -> FastAPI:
    app = FastAPI(title="dq-watchdog")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        window = np.array(req.window, dtype=float)
        if window.ndim != 2:
            raise HTTPException(status_code=400, detail="window must be 2D")

        flat = window.flatten().reshape(1, -1)
        seq = window.reshape(1, *window.shape)

        iforest_score = float(state.iforest.score(flat)[0])
        ae_score = float(state.autoencoder.score(seq)[0])
        state.log_scores(iforest_score, ae_score)

        max_score = max(iforest_score, ae_score)
        return ScoreResponse(
            iforest_score=iforest_score,
            autoencoder_score=ae_score,
            is_anomaly=max_score >= state.anomaly_threshold,
            model_versions={
                "iforest": getattr(state.iforest, "version", "unknown"),
                "autoencoder": getattr(state.autoencoder, "version", "unknown"),
            },
        )

    @app.get("/admin/drift")
    def admin_drift() -> dict[str, Any]:
        recent = state.recent_scores()
        if not recent:
            return {"status": "no_traffic"}

        out: dict[str, Any] = {"status": "ok", "models": {}}
        for model_name, scores in recent.items():
            baseline = state.baseline_scores.get(model_name)
            if baseline is None or len(baseline) == 0:
                out["models"][model_name] = {"psi": None, "reason": "no_baseline"}
                continue
            psi = compute_psi(baseline=baseline, production=scores)
            out["models"][model_name] = {
                "psi": psi,
                "classification": classify_psi(psi),
                "n_observations": int(len(scores)),
            }
        return out

    return app


def load_from_mlflow() -> ServiceState:
    """Load both models from their most recent MLflow runs.

    Used by app_factory() at uvicorn startup. Raises if no runs exist —
    the operator gets a clear error pointing at `watchdog train`.
    """
    import mlflow

    from watchdog.features import FeatureBuilder
    from watchdog.models.autoencoder import AutoencoderModel
    from watchdog.models.iforest import IForestModel
    from watchdog.train import EXPERIMENT_NAME

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found — run `watchdog train` first"
        )

    client = mlflow.tracking.MlflowClient()

    def latest(model_name: str):
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"params.model = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"No MLflow runs found for model='{model_name}'")
        return runs[0]

    iforest_run = latest("iforest")
    ae_run = latest("autoencoder")

    iforest_path = mlflow.artifacts.download_artifacts(
        run_id=iforest_run.info.run_id, artifact_path="iforest.pkl"
    )
    ae_path = mlflow.artifacts.download_artifacts(
        run_id=ae_run.info.run_id, artifact_path="autoencoder.pkl"
    )
    fb_path = mlflow.artifacts.download_artifacts(
        run_id=iforest_run.info.run_id, artifact_path="feature_builder.pkl"
    )

    iforest = IForestModel.load(Path(iforest_path))
    iforest.version = iforest_run.info.run_id  # type: ignore[attr-defined]
    autoencoder = AutoencoderModel.load(Path(ae_path))
    autoencoder.version = ae_run.info.run_id  # type: ignore[attr-defined]
    fb = FeatureBuilder.load(Path(fb_path))

    return ServiceState(
        iforest=iforest,
        autoencoder=autoencoder,
        feature_builder=fb,
        request_log_path=Path("request_log.db"),
        baseline_scores={},  # v1: populated as production traffic accumulates
    )


def app_factory() -> FastAPI:
    """Used by uvicorn with --factory at module path watchdog.service:app_factory."""
    return build_app(load_from_mlflow())
