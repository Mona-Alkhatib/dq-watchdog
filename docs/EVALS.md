# Eval Harness

The eval harness is what turns "two trained models" into a *comparison story* — the ML engineering signal.

## Why per-anomaly-type metrics?

A single aggregate F1 score hides the most interesting question: **which model wins on which kind of anomaly?** That's the question ML eng interviewers want answered. So we compute precision/recall/F1/AUC-PR separately for each of the 4 anomaly types.

## Fixture layout

```
evals/fixtures/benchmark_v1/
├── build.py           # deterministic generator
├── train.parquet      # 30 days, no anomalies
├── test.parquet       # 30 days with 7 planted anomalies
└── labels.json        # ground truth
```

The fixture is **committed to the repo** (not built at eval time), so anyone running the harness gets the same data.

## Anomaly types in benchmark_v1

| Type | Count | Difficulty |
|---|---|---|
| `point_spike` | 2 | easy (most obvious) |
| `level_shift` | 2 | medium |
| `gradual_drift` | 2 | hard for IF, good for autoencoder |
| `missing_window` | 1 | medium |

## Eval flow

```
1. Load train.parquet + test.parquet + labels.json
2. Fit the FeatureBuilder on train; split train into 80/20 for val
3. Train both models (Isolation Forest + LSTM Autoencoder)
4. Score the test windows with each model
5. Build dilated ground-truth labels (±1h tolerance per anomaly)
6. Compute per-anomaly-type precision/recall/F1/AUC-PR
7. Print the comparison table; assert at-least-one model passes thresholds
```

## Metrics

| Metric | Definition | v1 threshold (aggregate, at-least-one-model) |
|---|---|---|
| **Precision** | TP / (TP + FP) | ≥ 0.70 |
| **Recall** | TP / (TP + FN) | ≥ 0.70 |
| **F1** | harmonic mean | ≥ 0.70 |
| **AUC-PR** | area under PR curve | ≥ 0.80 |
| **Time-to-detect** | hours from anomaly start to first alert | ≤ 3 (level shifts + gradual drift) |

A score crosses the anomaly threshold (default 0.5) → predicted anomaly. A predicted anomaly that lands within ±1h of a labeled anomaly counts as TP.

## How to run

```bash
# Skipped by default (training takes ~30s)
uv run pytest evals/ -v

# Run for real
WATCHDOG_RUN_EVALS=1 uv run pytest evals/ -v -s
```

`-s` is recommended so the per-anomaly-type comparison table prints to stdout — that's the artifact you publish in the README.

## Sample output

```
=== benchmark_v1 ===
IForest    F1=0.79  AUC-PR=0.87
Autoenc.   F1=0.74  AUC-PR=0.84
Per-type (F1):
  gradual_drift:   IForest=0.55, AE=0.81
  level_shift:     IForest=0.86, AE=0.78
  missing_window:  IForest=0.91, AE=0.88
  point_spike:     IForest=0.92, AE=0.71
```

Readers see immediately: **IForest wins on point spikes and level shifts, the autoencoder wins on gradual drift, both detect missing windows.** That's the deployment-decision story.

## Drift eval (separate)

A smaller drift eval generates a "drifted" production stream — inputs whose distribution shifts slowly over 30 days — replays it through the trained models, and asserts PSI crosses the 0.2 threshold within 5 days of the drift starting. Implemented as a separate `pytest` case in `evals/test_evals.py`, also skipped without `WATCHDOG_RUN_EVALS=1`.

## What v1 doesn't measure

- **Variance across seeds.** Each model trains once. v2 should run N=3 seeds and report mean ± std.
- **Latency.** Inference timing is implicit ("it's fast"); v2 should explicitly assert p99 < 50ms.
- **Real-world data.** Synthetic only. v2 should add a NAB fixture.
- **Threshold tuning.** Uses 0.5 by default; v2 should tune per model on val and report results.

## Future work

- Multi-seed runs with variance.
- Latency assertions baked into the harness.
- NAB / Yahoo S5 fixtures alongside synthetic.
- Per-model threshold tuning on val with the chosen threshold reported in the run.
