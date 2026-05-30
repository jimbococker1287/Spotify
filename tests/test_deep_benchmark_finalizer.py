from __future__ import annotations

import json
from pathlib import Path

from spotify.deep_benchmark_finalizer import build_deep_benchmark_summary, main, write_deep_benchmark_summary


def _write_history(run_dir: Path) -> None:
    (run_dir / "histories.json").write_text(
        json.dumps(
            {
                "dense": {
                    "artist_output_sparse_categorical_accuracy": [0.1, 0.2],
                    "val_artist_output_sparse_categorical_accuracy": [0.15, 0.25],
                    "val_artist_output_top_5": [0.4, 0.5],
                    "loss": [4.0, 3.0],
                    "val_loss": [5.0, 4.0],
                },
                "gru": {
                    "sparse_categorical_accuracy": [0.2, 0.3],
                    "val_sparse_categorical_accuracy": [0.22, 0.21],
                    "val_top_5": [0.55, 0.53],
                    "loss": [3.0, 2.8],
                    "val_loss": [4.2, 4.4],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        "\n".join(
            [
                "2026-01-01 | INFO | [TEST] dense: Top-1=0.1234 | Top-5=0.5678",
                "2026-01-01 | INFO | [TEST] gru: Top-1=0.2222 | Top-5=0.6666",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "best_dense.keras").write_text("model", encoding="utf-8")
    (run_dir / "best_gru.keras").write_text("model", encoding="utf-8")


def test_build_deep_benchmark_summary_from_partial_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "20260101_000000_everything-all-deep"
    run_dir.mkdir(parents=True)
    _write_history(run_dir)

    summary = build_deep_benchmark_summary(run_dir=run_dir, expected_models=("dense", "gru"))

    assert summary["status"] == "deep_complete_pipeline_incomplete"
    assert summary["observed_model_count"] == 2
    assert summary["artifact_model_count"] == 2
    assert summary["test_metric_model_count"] == 2
    assert summary["best_by_val_top1"]["model_name"] == "dense"
    assert summary["best_by_test_top1"]["model_name"] == "gru"
    dense = next(row for row in summary["rows"] if row["model_name"] == "dense")
    assert dense["best_epoch"] == 2
    assert dense["best_val_top1"] == 0.25
    assert dense["test_top1"] == 0.1234
    assert round(dense["val_test_top1_gap"], 4) == 0.1266
    assert round(dense["test_to_val_top1_ratio"], 4) == 0.4936
    assert summary["generalization"]["large_gap_model_count"] == 0


def test_write_deep_benchmark_summary_mirrors_latest(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260101_000000_everything-all-deep"
    run_dir.mkdir(parents=True)
    _write_history(run_dir)

    paths = write_deep_benchmark_summary(run_dir=run_dir, outputs_dir=outputs_dir)

    assert run_dir / "analysis" / "deep_benchmark" / "deep_benchmark_summary.json" in paths
    assert (outputs_dir / "analysis" / "deep_benchmark" / "latest_deep_benchmark_summary.md").exists()
    latest = json.loads(
        (outputs_dir / "analysis" / "deep_benchmark" / "latest_deep_benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["run_id"] == run_dir.name
    assert latest["source_summary_dir"].endswith("analysis/deep_benchmark")


def test_cli_finds_run_by_name(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "runs" / "20260101_000000_named-run"
    run_dir.mkdir(parents=True)
    _write_history(run_dir)

    assert main(["--outputs-dir", str(outputs_dir), "--run-name", "named-run", "--expected-models", "dense,gru"]) == 0
    assert (run_dir / "analysis" / "deep_benchmark" / "deep_benchmark_summary.csv").exists()
    summary = json.loads(
        (run_dir / "analysis" / "deep_benchmark" / "deep_benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "deep_complete_pipeline_incomplete"
    assert summary["expected_model_count"] == 2
