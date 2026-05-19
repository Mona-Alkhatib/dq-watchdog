"""Generate the benchmark_v1 fixture deterministically.

Run once with: uv run python evals/fixtures/benchmark_v1/build.py
The output parquet + labels are committed to the repo.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from watchdog.generate import generate_timeseries

OUT = Path(__file__).parent

TRAIN_SEED = 1001
TEST_SEED = 2002

TEST_ANOMALIES = [
    {"timestamp": datetime(2025, 1, 5, 14, 0, 0), "metric": "null_rate", "type": "point_spike"},
    {"timestamp": datetime(2025, 1, 8, 0, 0, 0), "metric": "duplicate_rate", "type": "point_spike"},
    {
        "timestamp": datetime(2025, 1, 12, 0, 0, 0),
        "metric": "row_count",
        "type": "level_shift",
        "duration_hours": 24,
    },
    {
        "timestamp": datetime(2025, 1, 18, 0, 0, 0),
        "metric": "freshness_lag_seconds",
        "type": "level_shift",
        "duration_hours": 12,
    },
    {
        "timestamp": datetime(2025, 1, 22, 6, 0, 0),
        "metric": "null_rate",
        "type": "gradual_drift",
        "duration_hours": 48,
    },
    {
        "timestamp": datetime(2025, 1, 26, 0, 0, 0),
        "metric": "freshness_lag_seconds",
        "type": "gradual_drift",
        "duration_hours": 36,
    },
    {
        "timestamp": datetime(2025, 1, 28, 4, 0, 0),
        "metric": "row_count",
        "type": "missing_window",
        "duration_hours": 6,
    },
]


def main() -> None:
    train_df, _ = generate_timeseries(seed=TRAIN_SEED, num_days=30, anomalies=[])
    train_df.to_parquet(OUT / "train.parquet")

    test_df, labels = generate_timeseries(
        seed=TEST_SEED, num_days=30, anomalies=TEST_ANOMALIES
    )
    test_df.to_parquet(OUT / "test.parquet")
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2))

    print(f"Wrote {len(train_df)} train rows, {len(test_df)} test rows, {len(labels)} anomalies")


if __name__ == "__main__":
    main()
