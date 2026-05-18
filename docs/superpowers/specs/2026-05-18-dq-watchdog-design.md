# DQ Watchdog — Design Spec

**Date:** 2026-05-18
**Author:** Mona Alkhatib (with Claude Code)
**Status:** Approved (brainstorming → writing-plans)

---

## 1. Vision

**DQ Watchdog** is an ML service that detects anomalies in data-quality metric timeseries — row counts, null rates, freshness lag, schema-hash changes, duplicate rates. It trains **both an Isolation Forest and an LSTM Autoencoder** on synthetic data with planted anomalies, serves both via FastAPI, and detects production drift via PSI.

It is the fourth project in a portfolio bridging data engineering to AI / ML engineering, and the first that targets **ML Engineer** roles specifically:

| Project | Track | What it demonstrates |
|---|---|---|
| Lineage Oracle | AI eng | retrieval + agents |
| DQ Test Generator | AI eng | structured generation |
| dbt Sentinel | AI eng | structured critique |
| **DQ Watchdog** | **ML eng** | **training real models + production serving + drift detection** |

### Example flow

```
$ watchdog generate              # produce synthetic data
$ watchdog train                 # train both models, log to MLflow
$ watchdog eval                  # precision / recall / F1 per model per anomaly type
$ watchdog serve                 # FastAPI on :8000
$ curl -X POST :8000/score -d '{"window": [...]}'
  {"iforest_score": 0.82, "autoencoder_score": 0.91, "is_anomaly": true}
$ watchdog drift                 # PSI report on logged scores
```

---

## 2. Goals & non-goals

### Goals (v1)

1. Generate synthetic timeseries of 5 DQ metrics with 4 anomaly types planted at known timestamps.
2. Train two models — Isolation Forest (sklearn) and LSTM Autoencoder (PyTorch) — both tracked in MLflow.
3. Evaluate both models with precision, recall, F1, AUC-PR, and time-to-detect — **broken down per anomaly type**.
4. Serve both models via FastAPI behind one `/score` endpoint; ship a Dockerfile.
5. Drift detection — PSI on logged score distributions; expose `/admin/drift` and `watchdog drift` CLI.
6. README with a comparison table making the "which model wins on which anomaly type" story explicit.

### Non-goals (v1)

- Real production deployment (cloud hosting, k8s). Docker is the deployment unit.
- Online retraining loop. Trigger by hand for v1.
- Model registry stages (staging → production promotion). MLflow tracking only; serve from `latest`.
- A web UI for drift / scores. CLI + JSON endpoints only.
- Real datasets (NAB, public benchmarks). Synthetic only for v1.
- Bring-your-own-warehouse. Watchdog runs against its own generator output.

---

## 3. Architecture

Six components in a clear pipeline. Training is deterministic-by-seed. Serving is stateless beyond the SQLite request log used for drift.

```
1. Generator  →  2. Features  →  3. Models           →  4. Eval
                                  (IForest + LSTM AE)     (per-type metrics)
                                       ↓ MLflow
                                  5. Service  ↔  6. Drift
                                  (FastAPI)      (PSI on logged scores)
```

### 3.1 Generator — `watchdog/generate.py`

**Inputs:** seed, num_days (default 30), metrics-per-day-hours (default 24).

**Output:** parquet file + ground-truth labels JSON.

**Five metrics emitted per timestamp:**
- `row_count` — integer, typical 10k–100k with daily seasonality
- `null_rate` — float in [0, 1], typical 0.01–0.05
- `freshness_lag_seconds` — integer, typical 60–3600
- `schema_hash_changes` — integer, typical 0
- `duplicate_rate` — float in [0, 1], typical < 0.005

Each metric has a configurable baseline + daily seasonal pattern + Gaussian noise.

**Four anomaly types injected at known timestamps:**

| Type | Effect | Difficulty |
|---|---|---|
| `point_spike` | single-timestamp outlier (10×–100× normal) | easy |
| `level_shift` | sustained step change over N hours | medium |
| `gradual_drift` | linear trend over N hours | hard for IF |
| `missing_window` | zeros / nulls for N hours | medium |

Generation is deterministic given a seed so eval results are reproducible.

### 3.2 Features — `watchdog/features.py`

- **Windowing:** at each timestamp t, build a feature window covering `t-W` to `t-1` (default W=24 hours).
- **Normalization:** per-metric `StandardScaler` fit on training data only; reused for inference.
- **Flat output (for IForest):** the window flattened to a 1D vector (length W × 5).
- **Sequence output (for autoencoder):** the window as a `(W, 5)` tensor.

The scaler is persisted alongside each model in MLflow so production scoring uses the same normalization as training.

### 3.3 Models

**`watchdog/models/iforest.py`** — Isolation Forest
- `sklearn.ensemble.IsolationForest`, `n_estimators=200`, `contamination='auto'`.
- Trained on flat windows.
- Anomaly score: `-decision_function(...)` rescaled to `[0, 1]` via min-max on a validation split.

**`watchdog/models/autoencoder.py`** — LSTM Autoencoder (PyTorch)
- Encoder: 1-layer LSTM, hidden_dim=32. Decoder: 1-layer LSTM symmetric.
- Loss: MSE on reconstruction.
- Optimizer: Adam, lr=1e-3, batch_size=64, epochs=30 (early-stopped on val loss).
- Anomaly score: per-window MSE, rescaled to `[0, 1]` via percentile-based calibration on a validation split.

Both models implement a common interface:
```python
class AnomalyModel(Protocol):
    def fit(self, X_train, X_val) -> None: ...
    def score(self, X) -> np.ndarray: ...  # returns scores in [0, 1]
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "AnomalyModel": ...
```

### 3.4 Training orchestration — `watchdog/train.py`

`watchdog train` runs:
1. Generate (or reuse cached) training data.
2. Fit scaler on train split.
3. Build features for train + val.
4. Train each model with `mlflow.start_run()`:
   - Log params (hyperparams, anomaly-type distribution in train data)
   - Log metrics during training (val loss for autoencoder per epoch)
   - Log artifacts (the saved model + scaler)
5. Print run IDs.

MLflow defaults to `./mlruns` (SQLite-backed local store) — no external service required.

### 3.5 Eval — `watchdog/eval.py`

`watchdog eval` runs:
1. Generate a held-out test set (different seed from train).
2. Load each model + scaler from MLflow `latest`.
3. Score every test window with each model.
4. Compare scores against ground-truth labels using a configurable threshold (chosen on val split to maximize F1).
5. Compute precision, recall, F1, AUC-PR, time-to-detect — **per model AND per anomaly type**.
6. Write `eval_report.json` and `eval_report.md` (the markdown is what we publish in the README).

### 3.6 Service — `watchdog/service.py`

**FastAPI app** with three endpoints:

- `POST /score` — request body: `{"window": [[...5 metrics × W hours...]], "metadata": {...}}`. Response: `{"iforest_score": float, "autoencoder_score": float, "is_anomaly": bool, "model_versions": {...}}`.
- `GET /healthz` — `{"status": "ok"}`. Returns 503 if any model fails to load.
- `GET /admin/drift` — current PSI report (see §3.7).

Both models load at startup from the **most recent successful MLflow run** for each model name (or from `models/` on disk as a fallback if MLflow is unreachable). The scaler loads as a co-located artifact in the same run.

Every `/score` request is logged to a local SQLite (`request_log.db`) with timestamp + input hash + scores. This is the data source for drift detection.

**Containerized.** Dockerfile uses `python:3.11-slim`, installs project + deps, exposes 8000, runs `uvicorn watchdog.service:app`.

### 3.7 Drift Detector — `watchdog/drift.py`

**Population Stability Index (PSI)** on the score distribution.

- Baseline: training-time score distribution (logged once at train time, saved as a histogram).
- Production: histogram of recent `/score` outputs (last 24h by default).
- Compute PSI on each model's scores independently.

**Thresholds:**
- `PSI < 0.1` — stable
- `0.1 ≤ PSI < 0.2` — monitor
- `PSI ≥ 0.2` — significant drift (warning)

Surfaced via `GET /admin/drift` (JSON) and `watchdog drift` (text).

### 3.8 CLI — `watchdog/cli.py`

Five Typer subcommands:

```
watchdog generate    # produce synthetic data
watchdog train       # train both models + log to MLflow
watchdog eval        # run eval on test set, write report
watchdog serve       # start FastAPI server (uvicorn)
watchdog drift       # print drift report
```

---

## 4. Eval strategy

The differentiator: per-anomaly-type model comparison.

### 4.1 Fixture format

```
evals/fixtures/benchmark_v1/
├── train.parquet           # clean training data
├── test.parquet            # test data with planted anomalies
└── labels.json             # ground truth
```

`labels.json`:
```json
{
  "seed": 42,
  "anomalies": [
    {"timestamp": "2025-04-12T03:00:00", "metric": "null_rate", "type": "point_spike"},
    {"timestamp": "2025-04-15T00:00:00", "metric": "row_count", "type": "level_shift", "duration_hours": 24},
    ...
  ]
}
```

### 4.2 Metrics

| Metric | Definition | v1 Threshold (per model, aggregate) |
|---|---|---|
| **Precision** | TP / (TP + FP) | ≥ 0.70 |
| **Recall** | TP / (TP + FN) | ≥ 0.70 |
| **F1** | 2·P·R / (P+R) | ≥ 0.70 |
| **AUC-PR** | area under precision-recall curve | ≥ 0.80 |
| **Time-to-detect** | hours between anomaly start and first alert | ≤ 3 hours (for level shifts and gradual drift) |

A timestamp counts as a true positive if the model emits an anomaly score above threshold AND a ground-truth anomaly is active within ±1 hour.

### 4.3 Per-anomaly-type comparison

Report is a markdown table:

| Anomaly type | IForest P | IForest R | IForest F1 | AE P | AE R | AE F1 | Winner |
|---|---|---|---|---|---|---|---|
| point_spike | … | … | … | … | … | … | … |
| level_shift | … | … | … | … | … | … | … |
| gradual_drift | … | … | … | … | … | … | … |
| missing_window | … | … | … | … | … | … | … |

The README extracts the takeaway: which model wins where, and why.

### 4.4 Drift eval

Separate, lighter:
- Generate a "drifted" stream where the input distribution shifts slowly over 30 days.
- Replay it through `/score`.
- Assert PSI crosses 0.2 within ≤ 5 days of the drift starting.

### 4.5 Runner

`pytest evals/` — auto-skips unless `WATCHDOG_RUN_EVALS=1` is set, since model evals take minutes. Each fixture is one parametrized case.

---

## 5. Project layout

```
dq-watchdog/
├── watchdog/                    # library — single source of truth
│   ├── __init__.py
│   ├── generate.py              # synthetic data generator
│   ├── features.py              # windowing + StandardScaler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # AnomalyModel Protocol
│   │   ├── iforest.py           # Isolation Forest
│   │   └── autoencoder.py       # LSTM Autoencoder
│   ├── train.py                 # orchestration → MLflow logging
│   ├── eval.py                  # precision/recall + per-type breakdown
│   ├── drift.py                 # PSI
│   ├── service.py               # FastAPI app
│   └── cli.py                   # Typer CLI
├── evals/
│   ├── fixtures/benchmark_v1/   # train.parquet, test.parquet, labels.json
│   ├── conftest.py
│   └── test_evals.py
├── tests/                       # unit tests, mirrors watchdog/
├── Dockerfile
├── pyproject.toml               # uv + ruff + hatchling
├── .env.example                 # MLFLOW_TRACKING_URI (optional)
├── .gitignore
├── README.md
└── docs/
    ├── ARCHITECTURE.md
    ├── EVALS.md
    └── superpowers/
        ├── specs/2026-05-18-dq-watchdog-design.md
        └── plans/2026-05-18-dq-watchdog.md  (next)
```

**Responsibility split:**
- `generate.py` — pure (seeded RNG, no I/O beyond writing files)
- `features.py` — pure transformations on arrays
- `models/iforest.py`, `models/autoencoder.py` — implement the `AnomalyModel` Protocol
- `train.py` — orchestrates; only place that talks to MLflow at training time
- `eval.py` — pure metrics
- `drift.py` — pure PSI math
- `service.py` — only file that runs FastAPI / writes to request_log.db
- `cli.py` — Typer glue

---

## 6. Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Matches other projects |
| Classical ML | scikit-learn | Industry standard for tabular |
| Deep learning | PyTorch | Industry standard; better for sequence models than Keras |
| Experiment tracking | MLflow (local SQLite) | Standard MLOps idiom, zero infra |
| Serving | FastAPI + uvicorn | Standard production-grade Python API |
| Containerization | Docker (`python:3.11-slim`) | Deployment-ready unit |
| Data | pandas + numpy + pyarrow (parquet) | Standard tabular stack |
| Schema validation | Pydantic v2 | Same as other projects, used at request boundary |
| CLI | Typer | Same as other projects |
| Test framework | pytest | Same |
| Package mgr | uv | Same |
| Lint/format | ruff | Same |
| Build backend | hatchling | Same |

No LLM API in this project. No Anthropic dependency.

---

## 7. Error handling

- **`watchdog serve` with no trained models:** refuse to start, point at `watchdog train`. Never serve a placeholder.
- **`/score` with malformed payload:** Pydantic returns 422 with field-level errors.
- **`/score` with wrong window size:** explicit 400 with the expected shape.
- **MLflow store unreachable at serve time:** fall back to disk-loaded models from `models/` (if present); log a warning.
- **Autoencoder CUDA OOM during training:** automatic fallback to CPU with a logged warning. Dataset sizes are small enough that CPU training takes minutes.
- **Drift detector with zero requests logged:** return `{"status": "no_traffic"}`, not an error.
- **PSI undefined buckets (zero in baseline or production):** use the standard ε-substitution (replace 0 with 1e-6) before computing the log term.

---

## 8. Testing strategy

### 8.1 Unit tests (`tests/`)

- `test_generate.py` — deterministic given seed; injected anomalies land at the labeled timestamps; baseline metrics stay within expected ranges.
- `test_features.py` — window shapes; scaler fit-only-on-train invariant; round-trip stability.
- `test_iforest.py` — train on small fixtures returns scores in `[0, 1]`; deterministic with a seed.
- `test_autoencoder.py` — small model trains for 2 epochs without crashing; scores are non-negative reals.
- `test_drift.py` — PSI on hand-crafted distributions equals known values (textbook examples).
- `test_service.py` — FastAPI test client: `/healthz`, `/score` happy path, malformed payload 422, missing window 400.
- `test_cli.py` — `--help` and basic command shape.

Target ≥ 90% line coverage on `watchdog/`.

### 8.2 Integration test

End-to-end "smoke" test that runs `generate → train (tiny configs) → eval (tiny test set) → service.startup`. Asserts wiring; numerical thresholds are loose.

### 8.3 Eval harness (`evals/`)

The full precision/recall + drift evals from §4. Slow (~minutes). Skipped unless `WATCHDOG_RUN_EVALS=1` is set.

---

## 9. Open questions / future work

- **Online retraining loop:** scheduled `watchdog train` triggered by drift detection. v2.
- **Model registry stages:** promote runs to `staging` / `production`. Real ML platform pattern. v2.
- **Real-warehouse data:** integration with DuckDB / Snowflake `information_schema` as a feature source. v2.
- **Multi-table support:** v1 trains per-metric, not per-table. Multi-table cross-correlation features are v2.
- **A/B test framework:** support routing a fraction of traffic to a candidate model. v3.
- **Public benchmark integration:** NAB / Yahoo S5 datasets as additional eval fixtures. v2.

---

## 10. Success criteria for v1

The project is "shippable to portfolio" when:

1. `uv sync && watchdog generate && watchdog train && watchdog eval` works from a fresh clone.
2. `watchdog serve` starts on port 8000; `/healthz` returns 200; `/score` returns valid responses on a real window.
3. `docker build . && docker run -p 8000:8000 ...` works.
4. Eval suite passes thresholds: every model hits aggregate P, R, F1 ≥ 0.70 and AUC-PR ≥ 0.80.
5. Per-anomaly-type comparison table is published in the README.
6. Drift eval passes: PSI crosses 0.2 within 5 days on the drifted stream.
7. README, ARCHITECTURE.md, EVALS.md committed.
8. Repo deployed to https://github.com/Mona-Alkhatib/dq-watchdog.
