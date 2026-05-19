"""Shared helpers for the slow eval harness."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
RUN_EVALS = os.environ.get("WATCHDOG_RUN_EVALS") == "1"


def load_fixture(name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base = FIXTURES / name
    train = pd.read_parquet(base / "train.parquet")
    test = pd.read_parquet(base / "test.parquet")
    labels = json.loads((base / "labels.json").read_text())
    return train, test, labels
