from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pandas as pd

from spotify.evaluation import _build_label_lookup
from spotify.reporting import persist_to_sqlite


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
