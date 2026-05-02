from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, cast


def _read_json_document(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, object]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _artist_key(name: str) -> str:
    return str(name).strip().casefold()


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _coerce_recent_artists(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


class TasteOSStateStore:
    def __init__(
        self,
        *,
        db_path: Path,
        logger: logging.Logger,
        legacy_feedback_memory_path: Path | None = None,
        legacy_feedback_event_log_path: Path | None = None,
        legacy_session_history_path: Path | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.logger = logger
        self.legacy_feedback_memory_path = legacy_feedback_memory_path
        self.legacy_feedback_event_log_path = legacy_feedback_event_log_path
        self.legacy_session_history_path = legacy_session_history_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._migrate_legacy_if_needed()
        self.sync_legacy_snapshots()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_artist_stats (
                    artist_key TEXT PRIMARY KEY,
                    artist_name TEXT NOT NULL,
                    like_count INTEGER NOT NULL DEFAULT 0,
                    repeat_count INTEGER NOT NULL DEFAULT 0,
                    skip_count INTEGER NOT NULL DEFAULT 0,
                    dislike_count INTEGER NOT NULL DEFAULT 0,
                    last_signal TEXT NOT NULL DEFAULT '',
                    last_session_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    artist_name TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT '',
                    scenario TEXT NOT NULL DEFAULT '',
                    top_artist TEXT NOT NULL DEFAULT '',
                    adaptive_replans INTEGER NOT NULL DEFAULT 0,
                    safe_route_steps INTEGER NOT NULL DEFAULT 0,
                    effective_recent_artists_json TEXT NOT NULL DEFAULT '[]',
                    used_feedback_memory INTEGER NOT NULL DEFAULT 0,
                    persisted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_events_timestamp ON feedback_events(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_history_created_at ON session_history(created_at DESC)"
            )

    def _table_count(self, table_name: str) -> int:
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        return int((row["count"] if row is not None else 0) or 0)

    def _migrate_legacy_if_needed(self) -> None:
        has_state = any(
            (
                self._table_count("feedback_artist_stats") > 0,
                self._table_count("feedback_events") > 0,
                self._table_count("session_history") > 0,
            )
        )
        if has_state:
            return

        migrated = False
        memory_rows = _read_json_document(self.legacy_feedback_memory_path) if self.legacy_feedback_memory_path else {}
        event_rows = _read_jsonl_rows(self.legacy_feedback_event_log_path) if self.legacy_feedback_event_log_path else []
        session_rows = _read_jsonl_rows(self.legacy_session_history_path) if self.legacy_session_history_path else []

        with self._connect() as conn:
            artists_obj = memory_rows.get("artists", {})
            artists = cast(dict[str, object], artists_obj) if isinstance(artists_obj, dict) else {}
            for raw_key, raw_stats in artists.items():
                if not isinstance(raw_stats, dict):
                    continue
                artist_name = str(raw_stats.get("artist_name") or raw_key).strip()
                if not artist_name:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feedback_artist_stats (
                        artist_key, artist_name, like_count, repeat_count, skip_count, dislike_count,
                        last_signal, last_session_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _artist_key(artist_name),
                        artist_name,
                        _coerce_int(raw_stats.get("like_count", 0)),
                        _coerce_int(raw_stats.get("repeat_count", 0)),
                        _coerce_int(raw_stats.get("skip_count", 0)),
                        _coerce_int(raw_stats.get("dislike_count", 0)),
                        str(raw_stats.get("last_signal", "")),
                        str(raw_stats.get("last_session_id", "")),
                        str(raw_stats.get("updated_at", "")),
                    ),
                )
                migrated = True

            for row in event_rows:
                conn.execute(
                    """
                    INSERT INTO feedback_events (timestamp, session_id, artist_name, signal, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("timestamp", "")),
                        str(row.get("session_id", "")),
                        str(row.get("artist_name", "")),
                        str(row.get("signal", "")),
                        str(row.get("notes", "")),
                    ),
                )
                migrated = True

            for row in session_rows:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO session_history (
                        session_id, created_at, mode, scenario, top_artist, adaptive_replans,
                        safe_route_steps, effective_recent_artists_json, used_feedback_memory, persisted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("session_id", "")),
                        str(row.get("created_at", "")),
                        str(row.get("mode", "")),
                        str(row.get("scenario", "")),
                        str(row.get("top_artist", "")),
                        _coerce_int(row.get("adaptive_replans", 0)),
                        _coerce_int(row.get("safe_route_steps", 0)),
                        json.dumps(_coerce_recent_artists(row.get("effective_recent_artists"))),
                        1 if bool(row.get("used_feedback_memory")) else 0,
                        1 if bool(row.get("persisted")) else 0,
                    ),
                )
                migrated = True

        if migrated:
            self.logger.info("Migrated legacy Taste OS state into %s", self.db_path)

    def feedback_store(self) -> dict[str, object]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT artist_key, artist_name, like_count, repeat_count, skip_count, dislike_count,
                       last_signal, last_session_id, updated_at
                FROM feedback_artist_stats
                ORDER BY updated_at DESC, artist_name ASC
                """
            ).fetchall()
            summary = conn.execute(
                """
                SELECT COUNT(*) AS event_count, COALESCE(MAX(timestamp), '') AS updated_at
                FROM feedback_events
                """
            ).fetchone()

        artists: dict[str, dict[str, object]] = {}
        for row in rows:
            artist_name = str(row["artist_name"])
            artists[str(row["artist_key"])] = {
                "artist_name": artist_name,
                "like_count": int(row["like_count"] or 0),
                "repeat_count": int(row["repeat_count"] or 0),
                "skip_count": int(row["skip_count"] or 0),
                "dislike_count": int(row["dislike_count"] or 0),
                "last_signal": str(row["last_signal"] or ""),
                "last_session_id": str(row["last_session_id"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
        return {
            "artists": artists,
            "event_count": int((summary["event_count"] if summary is not None else 0) or 0),
            "updated_at": str((summary["updated_at"] if summary is not None else "") or ""),
        }

    def recent_feedback(self, *, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, session_id, artist_name, signal, notes
                FROM feedback_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "timestamp": str(row["timestamp"] or ""),
                "session_id": str(row["session_id"] or ""),
                "artist_name": str(row["artist_name"] or ""),
                "signal": str(row["signal"] or ""),
                "notes": str(row["notes"] or ""),
            }
            for row in rows
        ]

    def recent_sessions(self, *, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, created_at, mode, scenario, top_artist, adaptive_replans,
                       safe_route_steps, effective_recent_artists_json, used_feedback_memory, persisted
                FROM session_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            try:
                effective_recent_artists = json.loads(str(row["effective_recent_artists_json"] or "[]"))
            except json.JSONDecodeError:
                effective_recent_artists = []
            result.append(
                {
                    "session_id": str(row["session_id"] or ""),
                    "created_at": str(row["created_at"] or ""),
                    "mode": str(row["mode"] or ""),
                    "scenario": str(row["scenario"] or ""),
                    "top_artist": str(row["top_artist"] or ""),
                    "adaptive_replans": int(row["adaptive_replans"] or 0),
                    "safe_route_steps": int(row["safe_route_steps"] or 0),
                    "effective_recent_artists": _coerce_recent_artists(effective_recent_artists),
                    "used_feedback_memory": bool(row["used_feedback_memory"]),
                    "persisted": bool(row["persisted"]),
                }
            )
        return result

    def record_feedback(
        self,
        *,
        timestamp: str,
        session_id: str,
        artist_name: str,
        signal: str,
        notes: str | None,
    ) -> dict[str, object]:
        normalized_artist_name = str(artist_name).strip()
        normalized_signal = str(signal).strip().lower()
        counter_name = f"{normalized_signal}_count"
        artist_row_key = _artist_key(normalized_artist_name)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events (timestamp, session_id, artist_name, signal, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(timestamp),
                    str(session_id),
                    normalized_artist_name,
                    normalized_signal,
                    str(notes or ""),
                ),
            )
            row = conn.execute(
                """
                SELECT like_count, repeat_count, skip_count, dislike_count
                FROM feedback_artist_stats
                WHERE artist_key = ?
                """,
                (artist_row_key,),
            ).fetchone()
            counts = {
                "like_count": int((row["like_count"] if row is not None else 0) or 0),
                "repeat_count": int((row["repeat_count"] if row is not None else 0) or 0),
                "skip_count": int((row["skip_count"] if row is not None else 0) or 0),
                "dislike_count": int((row["dislike_count"] if row is not None else 0) or 0),
            }
            counts[counter_name] = int(counts.get(counter_name, 0)) + 1
            conn.execute(
                """
                INSERT INTO feedback_artist_stats (
                    artist_key, artist_name, like_count, repeat_count, skip_count, dislike_count,
                    last_signal, last_session_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artist_key) DO UPDATE SET
                    artist_name=excluded.artist_name,
                    like_count=excluded.like_count,
                    repeat_count=excluded.repeat_count,
                    skip_count=excluded.skip_count,
                    dislike_count=excluded.dislike_count,
                    last_signal=excluded.last_signal,
                    last_session_id=excluded.last_session_id,
                    updated_at=excluded.updated_at
                """,
                (
                    artist_row_key,
                    normalized_artist_name,
                    int(counts["like_count"]),
                    int(counts["repeat_count"]),
                    int(counts["skip_count"]),
                    int(counts["dislike_count"]),
                    normalized_signal,
                    str(session_id),
                    str(timestamp),
                ),
            )
        self.sync_legacy_snapshots()
        return {
            "timestamp": str(timestamp),
            "session_id": str(session_id),
            "artist_name": normalized_artist_name,
            "signal": normalized_signal,
            "notes": str(notes or ""),
        }

    def record_session_history(self, row: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_history (
                    session_id, created_at, mode, scenario, top_artist, adaptive_replans,
                    safe_route_steps, effective_recent_artists_json, used_feedback_memory, persisted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("session_id", "")),
                    str(row.get("created_at", "")),
                    str(row.get("mode", "")),
                    str(row.get("scenario", "")),
                    str(row.get("top_artist", "")),
                    _coerce_int(row.get("adaptive_replans", 0)),
                    _coerce_int(row.get("safe_route_steps", 0)),
                    json.dumps(_coerce_recent_artists(row.get("effective_recent_artists")), sort_keys=True),
                    1 if bool(row.get("used_feedback_memory")) else 0,
                    1 if bool(row.get("persisted")) else 0,
                ),
            )
        self.sync_legacy_snapshots()

    def sync_legacy_snapshots(self) -> None:
        if self.legacy_feedback_memory_path is not None:
            self.legacy_feedback_memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.legacy_feedback_memory_path.write_text(
                json.dumps(self.feedback_store(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if self.legacy_feedback_event_log_path is not None:
            _write_jsonl_rows(self.legacy_feedback_event_log_path, list(reversed(self.recent_feedback(limit=10_000_000))))
        if self.legacy_session_history_path is not None:
            _write_jsonl_rows(self.legacy_session_history_path, list(reversed(self.recent_sessions(limit=10_000_000))))
