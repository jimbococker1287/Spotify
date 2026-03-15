from __future__ import annotations

import logging

from spotify.data import discover_streaming_files


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
