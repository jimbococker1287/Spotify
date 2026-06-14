from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.recommender_expansion_lab import (
    ExpansionRunConfig,
    _evaluate_popularity,
    build_recommender_expansion_lab,
)
from spotify.track_level_data import build_track_level_dataset, split_track_level_examples


def _logger() -> logging.Logger:
    logger = logging.getLogger("spotify.test.recommender_expansion_lab")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _write_history(path: Path) -> None:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for session in range(8):
        start = base + pd.Timedelta(hours=session * 2)
        for position, track in enumerate(("a", "b", f"tail-{session}", "c")):
            rows.append(
                {
                    "ts": (start + pd.Timedelta(minutes=position * 5)).isoformat(),
                    "spotify_track_uri": f"spotify:track:{track}",
                    "master_metadata_track_name": track,
                    "master_metadata_album_artist_name": f"artist-{track}",
                    "ms_played": 180_000,
                    "skipped": position == 2,
                    "reason_end": "fwdbtn" if position == 2 else "trackdone",
                }
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "Streaming_History_Audio_2026_0.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def test_popularity_baseline_reports_recall_and_catalog_diagnostics() -> None:
    timestamps: list[pd.Timestamp] = []
    tracks: list[str] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for session in range(6):
        start = base + pd.Timedelta(hours=session * 2)
        timestamps.extend(start + pd.Timedelta(minutes=offset) for offset in (0, 5, 10))
        tracks.extend(
            [
                "spotify:track:a",
                "spotify:track:b",
                "spotify:track:c",
            ]
        )
    rows = pd.DataFrame(
        {
            "ts": timestamps,
            "spotify_track_uri": tracks,
        }
    )
    dataset = build_track_level_dataset(rows)
    splits = split_track_level_examples(dataset)

    result = _evaluate_popularity(
        splits.train,
        splits.validation,
        k=2,
        limit=100,
    )

    assert result["status"] == "complete"
    assert result["model"] == "track_popularity"
    assert result["evaluated_examples"] == len(splits.validation)
    assert result["exclude_seen"] is False
    assert 0.0 <= float(result["recall_at_k"]) <= 1.0
    assert 0.0 <= float(result["catalog_coverage"]) <= 1.0


def test_build_recommender_expansion_lab_writes_durable_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    _write_history(raw_dir)

    paths = build_recommender_expansion_lab(
        config=ExpansionRunConfig(
            raw_data_dir=raw_dir,
            output_dir=output_dir,
            max_history=16,
            evaluation_k=5,
            evaluation_limit=100,
        ),
        logger=_logger(),
    )

    root = output_dir / "analysis" / "recommender_expansion"
    manifest = json.loads((root / "expansion_manifest.json").read_text(encoding="utf-8"))
    continuation = (root / "CONTINUE_HERE.md").read_text(encoding="utf-8")

    assert len(paths) == 5
    assert manifest["status"] == "implementation_ready"
    assert manifest["dataset"]["unique_tracks"] == 11
    assert manifest["dataset"]["session_count"] == 8
    assert manifest["dataset"]["train_examples"] > 0
    assert manifest["dataset"]["validation_examples"] > 0
    assert manifest["dataset"]["test_examples"] > 0
    assert manifest["baseline"]["status"] == "complete"
    assert manifest["capability_count"] == 11
    assert "Next Training Pass" in continuation
    assert "`meantime_tisasrec`" in continuation


def test_lab_carries_completed_training_into_handoff(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    _write_history(raw_dir)
    training_dir = output_dir / "analysis" / "recommender_expansion" / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "retrieval_results": [
                    {
                        "model_name": "ease",
                        "k": 5,
                        "recall_at_k": 0.2,
                        "target_catalog_coverage": 0.4,
                    }
                ],
                "neural_results": [
                    {
                        "model_name": "meantime",
                        "validation": {"k": 5, "recall_at_k": 0.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    build_recommender_expansion_lab(
        config=ExpansionRunConfig(
            raw_data_dir=raw_dir,
            output_dir=output_dir,
            max_history=16,
            evaluation_k=5,
            evaluation_limit=100,
        ),
        logger=_logger(),
    )

    root = output_dir / "analysis" / "recommender_expansion"
    manifest = json.loads((root / "expansion_manifest.json").read_text(encoding="utf-8"))
    continuation = (root / "CONTINUE_HERE.md").read_text(encoding="utf-8")

    assert manifest["training"]["status"] == "complete"
    assert "Latest Training Pass" in continuation
    assert "`ease`: Recall@5 `0.200000`" in continuation
    assert "Train DCN-V2" in continuation
