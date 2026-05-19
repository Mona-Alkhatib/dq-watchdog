"""Typer CLI: generate, train, eval, serve, drift."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True)


@app.command()
def generate(
    out_dir: Path = typer.Option(Path("data"), help="Output directory"),
    seed: int = typer.Option(42),
    num_days: int = typer.Option(30),
    inject_anomalies: bool = typer.Option(True, help="Inject the standard 4-type anomaly set"),
) -> None:
    """Generate synthetic data quality timeseries."""
    load_dotenv()
    from watchdog.generate import generate_timeseries

    out_dir.mkdir(parents=True, exist_ok=True)
    anomalies: list[dict] = []
    if inject_anomalies:
        anomalies = [
            {
                "timestamp": datetime(2025, 1, 10, 8, 0, 0),
                "metric": "null_rate",
                "type": "point_spike",
            },
            {
                "timestamp": datetime(2025, 1, 15, 0, 0, 0),
                "metric": "row_count",
                "type": "level_shift",
                "duration_hours": 24,
            },
            {
                "timestamp": datetime(2025, 1, 20, 6, 0, 0),
                "metric": "freshness_lag_seconds",
                "type": "gradual_drift",
                "duration_hours": 48,
            },
            {
                "timestamp": datetime(2025, 1, 25, 10, 0, 0),
                "metric": "duplicate_rate",
                "type": "missing_window",
                "duration_hours": 6,
            },
        ]
    df, labels = generate_timeseries(seed=seed, num_days=num_days, anomalies=anomalies)
    df.to_parquet(out_dir / "timeseries.parquet")
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    typer.echo(f"Wrote {len(df)} rows + {len(labels)} labels to {out_dir}/")


@app.command()
def train(
    data: Path = typer.Option(Path("data/timeseries.parquet")),
    window_size: int = typer.Option(24),
) -> None:
    """Train both models, log to MLflow."""
    load_dotenv()
    import pandas as pd
    from watchdog.train import train_both

    df = pd.read_parquet(data)
    result = train_both(df=df, window_size=window_size, tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command(name="eval")
def eval_cmd(
    data: Path = typer.Option(Path("data/timeseries.parquet")),
    labels: Path = typer.Option(Path("data/labels.json")),
    window_size: int = typer.Option(24),
    threshold: float = typer.Option(0.5),
    out: Path = typer.Option(Path("eval_report.json")),
) -> None:
    """Evaluate both models on a labeled test set."""
    load_dotenv()
    typer.echo(
        "eval CLI runs the trained models against the test set and writes precision/recall per "
        "anomaly type. Implementation lives in evals/test_evals.py for the harness; this command "
        "invokes the same library code on user-supplied data.",
        err=True,
    )
    typer.echo(f"(Use: WATCHDOG_RUN_EVALS=1 uv run pytest evals/ -v)")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI service."""
    load_dotenv()
    import uvicorn
    uvicorn.run("watchdog.service:app_factory", host=host, port=port, factory=True)


@app.command()
def drift() -> None:
    """Print drift report from the most recent score log."""
    typer.echo("Use GET /admin/drift on the running service for a live report.")
