from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_aggregate_benchmark_uses_exact_prefix_boundaries(tmp_path: Path) -> None:
    history_csv = tmp_path / "outputs/history/experiment_history.csv"
    _write_csv(
        history_csv,
        [
            {
                "run_id": "run_demo_a",
                "run_name": "benchmark-lock-demo-seed-11",
                "profile": "small",
                "model_name": "logreg",
                "model_type": "classical",
                "model_family": "linear",
                "val_top1": 0.20,
                "test_top1": 0.18,
                "val_top5": 0.40,
                "test_top5": 0.36,
                "fit_seconds": 1.0,
            },
            {
                "run_id": "run_demo_b",
                "run_name": "benchmark-lock-demo-seed-42",
                "profile": "small",
                "model_name": "logreg",
                "model_type": "classical",
                "model_family": "linear",
                "val_top1": 0.22,
                "test_top1": 0.19,
                "val_top5": 0.41,
                "test_top5": 0.37,
                "fit_seconds": 1.1,
            },
            {
                "run_id": "run_demo2_a",
                "run_name": "benchmark-lock-demo2-seed-11",
                "profile": "small",
                "model_name": "logreg",
                "model_type": "classical",
                "model_family": "linear",
                "val_top1": 0.91,
                "test_top1": 0.88,
                "val_top5": 0.99,
                "test_top5": 0.98,
                "fit_seconds": 0.2,
            },
        ],
    )

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs/history"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_benchmark.py",
            "--benchmark-id",
            "demo",
            "--history-csv",
            str(history_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "benchmark_manifest=" in completed.stdout
    rows_path = output_dir / "benchmark_lock_demo_rows.csv"
    with rows_path.open("r", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    assert len(rows) == 2
    assert {row["run_name"] for row in rows} == {
        "benchmark-lock-demo-seed-11",
        "benchmark-lock-demo-seed-42",
    }
