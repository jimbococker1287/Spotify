from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace
import logging

import pandas as pd

from spotify.evaluation import _build_label_lookup
from spotify.pipeline_postrun_reporting import write_research_artifacts
from spotify.reporting import persist_to_sqlite, restore_deep_reporting_artifacts, save_deep_reporting_artifacts
from spotify.run_timing import RunPhaseRecorder


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_build_label_lookup_returns_sorted_first_seen_names() -> None:
    frame = pd.DataFrame(
        {
            "artist_label": [2, 0, 2, 1],
            "master_metadata_album_artist_name": ["Artist C", "Artist A", "Artist C Alt", "Artist B"],
        }
    )

    assert _build_label_lookup(frame) == {
        0: "Artist A",
        1: "Artist B",
        2: "Artist C",
    }


def test_persist_to_sqlite_writes_expected_summary_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts" / "metrics.sqlite"
    frame = pd.DataFrame({"artist_label": [0, 1], "play_count": [4, 2]})
    histories = {
        "demo_model": SimpleNamespace(
            history={
                "artist_output_sparse_categorical_accuracy": [0.30, 0.45],
                "val_artist_output_sparse_categorical_accuracy": [0.25, 0.40],
                "val_artist_output_top_5": [0.50, 0.70],
                "loss": [1.2, 0.8],
                "val_loss": [1.3, 0.9],
            }
        )
    }

    persisted = persist_to_sqlite(
        frame,
        histories=histories,
        cpu_usage=[10.0, 20.0, 30.0],
        gpu_usage=[5.0, 15.0],
        db_path=db_path,
    )

    assert persisted == db_path
    with sqlite3.connect(db_path) as conn:
        final_accuracy = conn.execute("SELECT model, val_top1, val_top5 FROM final_accuracy").fetchall()
        learning_curves = conn.execute(
            "SELECT model, epoch, train_artist_acc, val_artist_acc, val_artist_top5, train_loss, val_loss "
            "FROM learning_curves ORDER BY epoch"
        ).fetchall()
        utilization = conn.execute(
            "SELECT timestamp, cpu_usage, gpu_usage FROM utilization ORDER BY timestamp"
        ).fetchall()

    assert final_accuracy == [("demo_model", 0.40, 0.70)]
    assert learning_curves == [
        ("demo_model", 1, 0.30, 0.25, 0.50, 1.2, 1.3),
        ("demo_model", 2, 0.45, 0.40, 0.70, 0.8, 0.9),
    ]
    assert utilization == [
        (0, 10.0, 5.0),
        (1, 20.0, 15.0),
        (2, 30.0, None),
    ]


def test_deep_reporting_cache_round_trip_restores_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_CACHE_DEEP_REPORTING", "1")

    output_dir = tmp_path / "run"
    cache_root = tmp_path / "cache"
    db_path = output_dir / "spotify_training.db"
    histories = {
        "demo_model": SimpleNamespace(
            history={
                "loss": [1.0, 0.8],
                "val_loss": [1.1, 0.9],
                "val_artist_output_sparse_categorical_accuracy": [0.3, 0.4],
                "val_artist_output_top_5": [0.7, 0.8],
            }
        )
    }
    cpu_usage = [10.0, 20.0]
    gpu_usage = [5.0, 15.0]

    output_dir.mkdir(parents=True, exist_ok=True)
    original_payloads = {
        output_dir / "model_comparison.png": b"model-comparison",
        output_dir / "demo_model_learning_curve.png": b"learning-curve",
        output_dir / "histories.json": b"{\"histories\":true}",
        output_dir / "utilization.png": b"utilization",
        db_path: b"sqlite-artifact",
    }
    for path, payload in original_payloads.items():
        path.write_bytes(payload)

    save_deep_reporting_artifacts(
        histories=histories,
        cpu_usage=cpu_usage,
        gpu_usage=gpu_usage,
        output_dir=output_dir,
        db_path=db_path,
        cache_root=cache_root,
        cache_fingerprint="prepared123",
    )

    for path in original_payloads:
        path.unlink()

    restored = restore_deep_reporting_artifacts(
        histories=histories,
        cpu_usage=cpu_usage,
        gpu_usage=gpu_usage,
        output_dir=output_dir,
        db_path=db_path,
        cache_root=cache_root,
        cache_fingerprint="prepared123",
    )

    assert restored is not None
    model_comparison_path, learning_paths, histories_path, utilization_path, restored_db_path = restored
    assert model_comparison_path == output_dir / "model_comparison.png"
    assert learning_paths == [output_dir / "demo_model_learning_curve.png"]
    assert histories_path == output_dir / "histories.json"
    assert utilization_path == output_dir / "utilization.png"
    assert restored_db_path == db_path
    for path, payload in original_payloads.items():
        assert path.read_bytes() == payload


def test_write_research_artifacts_reuses_cached_analysis_summaries(tmp_path: Path) -> None:
    counters = {
        "benchmark_protocol": 0,
        "experiment_registry": 0,
        "ablation": 0,
        "significance": 0,
    }

    def _write_file(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _write_benchmark_protocol(*, output_dir: Path, run_id: str, **_kwargs):
        counters["benchmark_protocol"] += 1
        return [
            _write_file(output_dir / "benchmark_protocol.json", f"json-{run_id}"),
            _write_file(output_dir / "benchmark_protocol.md", f"md-{run_id}"),
        ]

    def _write_experiment_registry(*, output_dir: Path, run_id: str, **_kwargs):
        counters["experiment_registry"] += 1
        return _write_file(output_dir / "experiment_registry.json", f"registry-{run_id}")

    def _write_ablation_summary(*, output_dir: Path, **_kwargs):
        counters["ablation"] += 1
        return [
            _write_file(output_dir / "ablation_summary.csv", "csv"),
            _write_file(output_dir / "ablation_summary.json", "json"),
        ]

    def _write_significance_summary(*, output_dir: Path, **_kwargs):
        counters["significance"] += 1
        return [
            _write_file(output_dir / "backtest_significance.csv", "csv"),
            _write_file(output_dir / "backtest_significance.json", "json"),
        ]

    def _run_once(run_dir: Path, run_id: str) -> list[Path]:
        artifact_paths: list[Path] = []
        write_research_artifacts(
            artifact_paths=artifact_paths,
            backtest_rows=[{"model_name": "dense", "fold": 1, "top1": 0.55}],
            cache_info_payload={"fingerprint": "prepared123"},
            config=SimpleNamespace(output_dir=tmp_path / "outputs", profile="full"),
            data=SimpleNamespace(),
            logger=_logger(f"research-{run_id}"),
            phase_recorder=RunPhaseRecorder(run_id=run_id),
            result_rows=[{"model_name": "dense", "model_type": "deep", "val_top1": 0.61, "test_top1": 0.58}],
            run_dir=run_dir,
            run_id=run_id,
            write_ablation_summary=_write_ablation_summary,
            write_benchmark_protocol=_write_benchmark_protocol,
            write_experiment_registry=_write_experiment_registry,
            write_significance_summary=_write_significance_summary,
        )
        return artifact_paths

    first_run_dir = tmp_path / "outputs" / "runs" / "run_a"
    second_run_dir = tmp_path / "outputs" / "runs" / "run_b"
    _run_once(first_run_dir, "run_a")
    _run_once(second_run_dir, "run_b")

    assert counters == {
        "benchmark_protocol": 2,
        "experiment_registry": 2,
        "ablation": 1,
        "significance": 1,
    }
    assert (second_run_dir / "analysis" / "ablation_summary.csv").exists()
    assert (second_run_dir / "analysis" / "ablation_summary.json").exists()
    assert (second_run_dir / "analysis" / "backtest_significance.csv").exists()
    assert (second_run_dir / "analysis" / "backtest_significance.json").exists()
