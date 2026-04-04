from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pandas as pd

from spotify.evaluation import _build_label_lookup
from spotify.reporting import persist_to_sqlite, restore_deep_reporting_artifacts, save_deep_reporting_artifacts


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
