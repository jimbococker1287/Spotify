from __future__ import annotations

import json
import logging
from pathlib import Path

from spotify.taste_os_state import TasteOSStateStore


def test_taste_os_state_store_records_feedback_and_sessions(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "analysis" / "taste_os_service"
    logger = logging.getLogger("spotify.test.taste_os_state")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    store = TasteOSStateStore(
        db_path=output_dir / "taste_os_state.sqlite3",
        logger=logger,
        legacy_feedback_memory_path=output_dir / "feedback_memory.json",
        legacy_feedback_event_log_path=output_dir / "feedback_events.jsonl",
        legacy_session_history_path=output_dir / "session_history.jsonl",
    )

    event = store.record_feedback(
        timestamp="2026-05-02T12:00:00Z",
        session_id="taste-os-123",
        artist_name="Artist C",
        signal="like",
        notes="great fit",
    )
    store.record_session_history(
        {
            "session_id": "taste-os-123",
            "created_at": "2026-05-02T12:00:00Z",
            "mode": "focus",
            "scenario": "steady",
            "top_artist": "Artist C",
            "adaptive_replans": 1,
            "safe_route_steps": 2,
            "effective_recent_artists": ["Artist A", "Artist B"],
            "used_feedback_memory": True,
            "persisted": False,
        }
    )

    assert event["artist_name"] == "Artist C"
    assert store.feedback_store()["event_count"] == 1
    assert store.recent_feedback(limit=1)[0]["signal"] == "like"
    assert store.recent_sessions(limit=1)[0]["mode"] == "focus"
    assert (output_dir / "feedback_memory.json").exists()
    assert (output_dir / "feedback_events.jsonl").exists()
    assert (output_dir / "session_history.jsonl").exists()


def test_taste_os_state_store_migrates_legacy_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "analysis" / "taste_os_service"
    output_dir.mkdir(parents=True)
    feedback_memory_path = output_dir / "feedback_memory.json"
    feedback_events_path = output_dir / "feedback_events.jsonl"
    session_history_path = output_dir / "session_history.jsonl"

    feedback_memory_path.write_text(
        json.dumps(
            {
                "artists": {
                    "artist-d": {
                        "artist_name": "Artist D",
                        "like_count": 2,
                        "repeat_count": 1,
                        "skip_count": 0,
                        "dislike_count": 0,
                        "last_signal": "repeat",
                        "last_session_id": "taste-os-seed",
                        "updated_at": "2026-05-02T11:00:00Z",
                    }
                },
                "event_count": 3,
                "updated_at": "2026-05-02T11:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    feedback_events_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-02T11:00:00Z",
                "session_id": "taste-os-seed",
                "artist_name": "Artist D",
                "signal": "repeat",
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session_history_path.write_text(
        json.dumps(
            {
                "session_id": "taste-os-seed",
                "created_at": "2026-05-02T11:00:00Z",
                "mode": "discovery",
                "scenario": "steady",
                "top_artist": "Artist D",
                "adaptive_replans": 0,
                "safe_route_steps": 1,
                "effective_recent_artists": ["Artist D"],
                "used_feedback_memory": True,
                "persisted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    logger = logging.getLogger("spotify.test.taste_os_state.migrate")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    store = TasteOSStateStore(
        db_path=output_dir / "taste_os_state.sqlite3",
        logger=logger,
        legacy_feedback_memory_path=feedback_memory_path,
        legacy_feedback_event_log_path=feedback_events_path,
        legacy_session_history_path=session_history_path,
    )

    feedback_store = store.feedback_store()
    artists = feedback_store.get("artists", {})
    assert isinstance(artists, dict)
    assert any(str(row.get("artist_name", "")) == "Artist D" for row in artists.values())
    assert feedback_store["event_count"] == 1
    assert store.recent_feedback(limit=1)[0]["artist_name"] == "Artist D"
    assert store.recent_sessions(limit=1)[0]["session_id"] == "taste-os-seed"


def test_taste_os_state_store_supports_database_url(tmp_path: Path) -> None:
    db_path = tmp_path / "outputs" / "analysis" / "taste_os_service" / "state.sqlite3"
    logger = logging.getLogger("spotify.test.taste_os_state.database_url")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    store = TasteOSStateStore(
        db_path=None,
        database_url=f"sqlite:///{db_path.as_posix()}",
        logger=logger,
    )

    store.record_feedback(
        timestamp="2026-05-02T12:30:00Z",
        session_id="taste-os-url",
        artist_name="Artist E",
        signal="repeat",
        notes=None,
    )

    health = store.health_payload()
    assert health["backend"] == "sqlite"
    assert health["reachable"] is True
    assert store.feedback_store()["event_count"] == 1
