from __future__ import annotations

import json
import logging

from spotify.data import (
    discover_streaming_files,
    discover_technical_log_files,
    load_or_prepare_training_data,
    load_streaming_history,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("test-data-discovery")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    return logger


def test_discover_streaming_files_supports_nested_export_folder(tmp_path) -> None:
    export_dir = tmp_path / "Spotify Extended Streaming History"
    export_dir.mkdir()
    for name in (
        "Streaming_History_Audio_2024_7.json",
        "Streaming_History_Audio_2024-2025_8.json",
        "Streaming_History_Audio_2025-2026_9.json",
        "Streaming_History_Video_2018-2026.json",
    ):
        (export_dir / name).write_text("[]", encoding="utf-8")

    discovered = discover_streaming_files(tmp_path, include_video=True, logger=_logger())

    assert [path.name for path in discovered] == [
        "Streaming_History_Audio_2024-2025_8.json",
        "Streaming_History_Audio_2024_7.json",
        "Streaming_History_Audio_2025-2026_9.json",
        "Streaming_History_Video_2018-2026.json",
    ]


def test_discover_streaming_files_prefers_largest_history_group(tmp_path) -> None:
    root_audio = tmp_path / "Streaming_History_Audio_2024_7.json"
    root_audio.write_text("[]", encoding="utf-8")

    export_dir = tmp_path / "Spotify Extended Streaming History"
    export_dir.mkdir()
    for name in (
        "Streaming_History_Audio_2024_6.json",
        "Streaming_History_Audio_2024_7.json",
        "Streaming_History_Audio_2024-2025_8.json",
    ):
        (export_dir / name).write_text("[]", encoding="utf-8")

    discovered = discover_streaming_files(tmp_path, include_video=False, logger=_logger())

    assert [path.parent for path in discovered] == [export_dir, export_dir, export_dir]
    assert [path.name for path in discovered] == [
        "Streaming_History_Audio_2024-2025_8.json",
        "Streaming_History_Audio_2024_6.json",
        "Streaming_History_Audio_2024_7.json",
    ]


def test_discover_streaming_files_ignores_account_data_exports(tmp_path) -> None:
    account_dir = tmp_path / "Spotify Account Data"
    account_dir.mkdir()
    (account_dir / "StreamingHistory_music_0.json").write_text("[]", encoding="utf-8")

    export_dir = tmp_path / "Spotify Extended Streaming History"
    export_dir.mkdir()
    (export_dir / "Streaming_History_Audio_2025-2026_9.json").write_text("[]", encoding="utf-8")

    discovered = discover_streaming_files(tmp_path, include_video=False, logger=_logger())

    assert [path.name for path in discovered] == ["Streaming_History_Audio_2025-2026_9.json"]


def test_load_streaming_history_uses_parallel_shards_and_records_stats(tmp_path, monkeypatch) -> None:
    export_dir = tmp_path / "Spotify Extended Streaming History"
    export_dir.mkdir()
    payloads = {
        "Streaming_History_Audio_2024_0.json": [
            {"ts": "2026-01-01T00:00:00Z", "master_metadata_album_artist_name": "A"},
            {"ts": "2026-01-01T00:05:00Z", "master_metadata_album_artist_name": "B"},
        ],
        "Streaming_History_Audio_2025_1.json": [
            {"ts": "2026-01-01T00:10:00Z", "master_metadata_album_artist_name": "C"}
        ],
        "Streaming_History_Audio_2026_2.json": [
            {"ts": "2026-01-01T00:15:00Z", "master_metadata_album_artist_name": "D"}
        ],
    }
    for filename, rows in payloads.items():
        (export_dir / filename).write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setenv("SPOTIFY_HISTORY_LOAD_WORKERS", "2")

    frame = load_streaming_history(tmp_path, include_video=False, logger=_logger())

    assert frame["master_metadata_album_artist_name"].tolist() == ["A", "B", "C", "D"]
    stats = frame.attrs["spotify_load_stats"]
    assert stats["file_count"] == 3
    assert stats["record_count"] == 4
    assert stats["worker_count"] == 2
    assert stats["rows_per_second"] > 0
    assert len(stats["files"]) == 3


def test_prepared_cache_metadata_records_ingestion_stats(tmp_path, monkeypatch) -> None:
    export_dir = tmp_path / "Spotify Extended Streaming History"
    export_dir.mkdir()
    rows = [
        {
            "ts": f"2026-01-01T00:{idx:02d}:00Z",
            "master_metadata_album_artist_name": ["A", "B", "C", "D"][idx % 4],
            "platform": "ios",
            "reason_start": "trackdone",
            "reason_end": "trackdone",
            "shuffle": False,
            "skipped": idx % 2 == 0,
            "offline": False,
            "incognito_mode": False,
        }
        for idx in range(10)
    ]
    (export_dir / "Streaming_History_Audio_2026_0.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setenv("SPOTIFY_CACHE_PREPARED", "1")

    _prepared, cache_info = load_or_prepare_training_data(
        data_dir=tmp_path,
        include_video=False,
        enable_spotify_features=False,
        max_artists=4,
        sequence_length=2,
        scaler_path=tmp_path / "outputs" / "context_scaler.joblib",
        cache_root=tmp_path / "outputs" / "cache" / "prepared_data",
        logger=_logger(),
    )

    assert cache_info.enabled is True
    assert cache_info.hit is False
    assert cache_info.metadata_path is not None
    metadata = json.loads(cache_info.metadata_path.read_text(encoding="utf-8"))
    assert metadata["load_stats"]["record_count"] == len(rows)
    assert metadata["load_stats"]["file_count"] == 1
    assert metadata["feature_stats"]["input_rows"] == len(rows)
    assert metadata["feature_stats"]["vectorized_recent_artist_unique_ratio"] is True


def test_discover_technical_log_files_supports_nested_export_folder(tmp_path) -> None:
    export_dir = tmp_path / "Spotify Technical Log Information"
    export_dir.mkdir()
    for name in (
        "ConnectionInfo.json",
        "AudioStreamingSettingsReport.json",
        "PlaybackError.json",
    ):
        (export_dir / name).write_text("[]", encoding="utf-8")

    discovered = discover_technical_log_files(tmp_path, logger=_logger())

    assert [path.name for path in discovered] == [
        "ConnectionInfo.json",
        "AudioStreamingSettingsReport.json",
        "PlaybackError.json",
    ]


def test_discover_technical_log_files_prefers_largest_group(tmp_path) -> None:
    root_dir = tmp_path / "Spotify Technical Log Information"
    root_dir.mkdir()
    (root_dir / "ConnectionInfo.json").write_text("[]", encoding="utf-8")

    nested_dir = tmp_path / "Exports" / "Spotify Technical Log Information"
    nested_dir.mkdir(parents=True)
    for name in (
        "ConnectionInfo.json",
        "PlaybackError.json",
        "Stutter.json",
    ):
        (nested_dir / name).write_text("[]", encoding="utf-8")

    discovered = discover_technical_log_files(tmp_path, logger=_logger())

    assert [path.parent for path in discovered] == [nested_dir, nested_dir, nested_dir]
    assert [path.name for path in discovered] == [
        "ConnectionInfo.json",
        "PlaybackError.json",
        "Stutter.json",
    ]
