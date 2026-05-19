# DQ Watchdog

ML anomaly detector for data quality metrics. Trains **both** an Isolation Forest and an LSTM Autoencoder, serves them behind one FastAPI endpoint, and detects production drift via PSI.

The fourth project in a portfolio bridging data engineering to AI/ML engineering — and the first that targets **ML Engineer** roles specifically.

| Project | Track | Demonstrates |
|---|---|---|
| [lineage-oracle](https://github.com/Mona-Alkhatib/lineage-oracle) | AI eng | retrieval + agents |
| [dq-test-generator](https://github.com/Mona-Alkhatib/dq-test-generator) | AI eng | structured generation |
| [dbt-sentinel](https://github.com/Mona-Alkhatib/dbt-sentinel) | AI eng | structured critique |
| **dq-watchdog** | **ML eng** | **training real models + production serving + drift detection** |

## What it does

1. **Generates** synthetic timeseries of 5 data-quality metrics (row count, null rate, freshness lag, schema-hash changes, duplicate rate) with 4 anomaly types planted at known timestamps.
2. **Trains** two models on the same data: an Isolation Forest (sklearn) and an LSTM Autoencoder (PyTorch). Both register in MLflow.
3. **Evaluates** them with precision, recall, F1, AUC-PR — **per anomaly type**, so the README can publish a "which model wins where" comparison.
4. **Serves** both behind one FastAPI `/score` endpoint, with `/admin/drift` for live PSI reports.
5. **Containerizes** with Docker.

## Quickstart

```bash
# 1. Install (PyTorch is large — first sync takes a few minutes)
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

Expected output: an aggregate table comparing both models' F1 and AUC-PR overall, and a per-anomaly-type breakdown — the "which model wins on which type" story.

## Docker

```bash
docker build -t dq-watchdog .
docker run -p 8000:8000 dq-watchdog
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — six-component pipeline + design choices
- [`docs/EVALS.md`](docs/EVALS.md) — eval format, metrics, per-anomaly-type comparison
- [`docs/superpowers/specs/2026-05-18-dq-watchdog-design.md`](docs/superpowers/specs/2026-05-18-dq-watchdog-design.md) — full spec
- [`docs/superpowers/plans/2026-05-18-dq-watchdog.md`](docs/superpowers/plans/2026-05-18-dq-watchdog.md) — implementation plan

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
