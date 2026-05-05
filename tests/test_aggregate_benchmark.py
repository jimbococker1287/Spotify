from __future__ import annotations

import csv
import json
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


def test_aggregate_benchmark_fails_research_grade_lock_without_deep_comparator(tmp_path: Path) -> None:
    history_csv = tmp_path / "outputs/history/experiment_history.csv"
    _write_csv(
        history_csv,
        [
            {
                "run_id": "run_demo_a",
                "run_name": "benchmark-lock-demo-seed-11",
                "profile": "small",
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "model_family": "retrieval",
                "val_top1": 0.24,
                "test_top1": 0.21,
                "val_top5": 0.48,
                "test_top5": 0.43,
                "fit_seconds": 1.0,
            },
            {
                "run_id": "run_demo_b",
                "run_name": "benchmark-lock-demo-seed-42",
                "profile": "small",
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "model_family": "retrieval",
                "val_top1": 0.25,
                "test_top1": 0.22,
                "val_top5": 0.49,
                "test_top5": 0.44,
                "fit_seconds": 1.1,
            },
            {
                "run_id": "run_demo_c",
                "run_name": "benchmark-lock-demo-seed-77",
                "profile": "small",
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "model_family": "retrieval",
                "val_top1": 0.23,
                "test_top1": 0.20,
                "val_top5": 0.47,
                "test_top5": 0.42,
                "fit_seconds": 1.2,
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
            "--declared-deep-models",
            "gru_artist",
            "--declared-classical-models",
            "logreg",
            "--retrieval-enabled",
            "--research-grade",
            "--fail-not-comparison-ready",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "benchmark_not_comparison_ready=" in completed.stderr
    manifest_path = output_dir / "benchmark_lock_demo_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["comparison_ready"] is False
    assert payload["model_class_mix"]["declared"]["expected_model_classes"] == ["candidate", "classical", "deep"]
    assert payload["comparator_guard"]["deep_comparator_ready"] is False
