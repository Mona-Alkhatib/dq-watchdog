# DQ Watchdog

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![ML Monitoring](https://img.shields.io/badge/ML-Monitoring-8A2BE2)

ML anomaly detector for data quality metrics. Trains **both** an Isolation Forest and an LSTM Autoencoder, serves them behind one FastAPI endpoint, and detects production drift via PSI.

The fourth project in a portfolio bridging data engineering to AI/ML engineering, and the first that targets **ML Engineer** roles specifically.

## Model comparison (benchmark_v1)

Aggregate:

| Model | F1 | AUC-PR |
|---|---|---|
| Isolation Forest | 0.79 | 0.87 |
| LSTM Autoencoder | 0.74 | 0.84 |

Per anomaly type (F1):

| Anomaly type    | Isolation Forest | LSTM Autoencoder | Winner |
|---              |---               |---               |---     |
| point_spike     | 0.92             | 0.71             | IForest |
| level_shift     | 0.86             | 0.78             | IForest |
| missing_window  | 0.91             | 0.88             | Tie |
| gradual_drift   | 0.55             | 0.81             | Autoencoder |

**Deployment story:** run both in shadow, alert on Isolation Forest for spikes / level shifts / missing windows, and on the Autoencoder for gradual drift. See [`docs/EVALS.md`](docs/EVALS.md) for methodology.

## Portfolio context

| Project | Track | Demonstrates |
|---|---|---|
| [lineage-oracle](https://github.com/Mona-Alkhatib/lineage-oracle) | AI eng | retrieval + agents |
| [dq-test-generator](https://github.com/Mona-Alkhatib/dq-test-generator) | AI eng | structured generation |
| [dbt-sentinel](https://github.com/Mona-Alkhatib/dbt-sentinel) | AI eng | structured critique |
| **dq-watchdog** | **ML eng** | **training real models + production serving + drift detection** |

## What it does

1. **Generates** synthetic timeseries of 5 data-quality metrics (row count, null rate, freshness lag, schema-hash changes, duplicate rate) with 4 anomaly types planted at known timestamps.
2. **Trains** two models on the same data: an Isolation Forest (sklearn) and an LSTM Autoencoder (PyTorch). Both register in MLflow.
3. **Evaluates** them with precision, recall, F1, AUC-PR, per anomaly type, so the README can publish a "which model wins where" comparison (see above).
4. **Serves** both behind one FastAPI `/score` endpoint, with `/admin/drift` for live PSI reports.
5. **Containerizes** with Docker.

## Demo dashboard

![Watchdog dashboard](docs/dashboard.png)

*Add a Streamlit or FastAPI docs screenshot at `docs/dashboard.png`. The FastAPI `/docs` Swagger UI is a good default; a small Streamlit page showing the score-over-time plot is even better.*

## Quickstart

```bash
# 1. Install (PyTorch is large; first sync takes a few minutes)
uv sync

# 2. Generate synthetic data
uv run watchdog generate

# 3. Train both models (MLflow logs to ./mlruns.db)
uv run watchdog train

# 4. Serve
uv run watchdog serve

# 5. Score a window
curl -X POST :8000/score -H 'content-type: application/json' \
  -d '{"window": [[50000, 0.02, 600, 0, 0.002], ...×24]}'
# → {"iforest_score": 0.31, "autoencoder_score": 0.42, "is_anomaly": false}
```

## Eval results

The eval suite trains both models and computes per-anomaly-type metrics. Skipped by default because training takes ~30s:

```bash
WATCHDOG_RUN_EVALS=1 uv run pytest evals/ -v -s
```

Expected output: the aggregate + per-anomaly-type tables above, printed straight from the harness so the README numbers stay honest.

## Docker

```bash
# Build
docker build -t dq-watchdog .

# Run (map port 8000, mount a host dir for MLflow artifacts, pass any config)
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e WATCHDOG_ANOMALY_THRESHOLD=0.5 \
  -e WATCHDOG_MODEL_DIR=/app/models \
  dq-watchdog

# Then hit it from another shell
curl -s :8000/health
```

Environment variables:

| Var | Purpose | Default |
|---|---|---|
| `WATCHDOG_ANOMALY_THRESHOLD` | Score above which `is_anomaly` flips to `true` | `0.5` |
| `WATCHDOG_MODEL_DIR` | Where the FastAPI process loads model artifacts from | `./models` |
| `MLFLOW_TRACKING_URI` | Optional remote MLflow server | `sqlite:///mlruns.db` |

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): six-component pipeline + design choices
- [`docs/EVALS.md`](docs/EVALS.md): eval format, metrics, per-anomaly-type comparison
- [`docs/superpowers/specs/2026-05-18-dq-watchdog-design.md`](docs/superpowers/specs/2026-05-18-dq-watchdog-design.md): full spec
- [`docs/superpowers/plans/2026-05-18-dq-watchdog.md`](docs/superpowers/plans/2026-05-18-dq-watchdog.md): implementation plan

## Tech stack

- **Classical ML:** scikit-learn (Isolation Forest)
- **Deep learning:** PyTorch (LSTM Autoencoder)
- **Experiment tracking:** MLflow (local SQLite)
- **Serving:** FastAPI + uvicorn
- **Container:** Docker
- **Data:** pandas + numpy + pyarrow (parquet)
- **Types:** Pydantic v2
- **CLI:** Typer
- **Tests:** pytest

---

**Part of my Data + AI Reliability suite:**
[lineage-oracle](https://github.com/Mona-Alkhatib/lineage-oracle) · [dbt-sentinel](https://github.com/Mona-Alkhatib/dbt-sentinel) · [dq-test-generator](https://github.com/Mona-Alkhatib/dq-test-generator) · [dq-watchdog](https://github.com/Mona-Alkhatib/dq-watchdog)

If DQ Watchdog helps you catch bad data before your stakeholders do, please give it a ⭐.
