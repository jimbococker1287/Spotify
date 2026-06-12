from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine

DEFAULT_SESSION_EVENT_LIMIT = 20
MAX_SESSION_EVENT_LIMIT = 100


def _read_json_document(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl_rows(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
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


def _database_url_from_path(path: Path) -> str:
    return f"sqlite:///{path.expanduser().resolve().as_posix()}"


def _sqlite_path_from_database_url(database_url: str) -> Path | None:
    parsed = urlsplit(database_url)
    scheme = parsed.scheme.strip().lower()
    if not scheme.startswith("sqlite"):
        return None
    if not parsed.path:
        return None
    return Path(parsed.path).expanduser()


def _database_backend_name(database_url: str) -> str:
    scheme = urlsplit(database_url).scheme.strip().lower()
    if not scheme:
        return "sqlite"
    return scheme.split("+", 1)[0]


def _redact_database_url(database_url: str) -> str:
    if not database_url:
        return ""
    parts = urlsplit(database_url)
    netloc = parts.netloc
    if "@" in netloc:
        auth, host = netloc.rsplit("@", 1)
        user = auth.split(":", 1)[0].strip()
        netloc = f"{user}:***@{host}" if user else f"***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class ActiveSessionNotFoundError(LookupError):
    pass


class StaleSessionVersionError(RuntimeError):
    def __init__(self, *, expected_version: int, current_version: int) -> None:
        self.expected_version = int(expected_version)
        self.current_version = int(current_version)
        super().__init__(
            f"Session version {self.expected_version} is stale; current version is {self.current_version}."
        )


class SessionEventConflictError(RuntimeError):
    pass


class TasteOSStateStore:
    def __init__(
        self,
        *,
        db_path: Path | None = None,
        logger: logging.Logger,
        database_url: str | None = None,
        legacy_feedback_memory_path: Path | None = None,
        legacy_feedback_event_log_path: Path | None = None,
        legacy_session_history_path: Path | None = None,
    ) -> None:
        if db_path is None and not str(database_url or "").strip():
            raise ValueError("TasteOSStateStore requires either db_path or database_url.")

        self.logger = logger
        self.legacy_feedback_memory_path = legacy_feedback_memory_path
        self.legacy_feedback_event_log_path = legacy_feedback_event_log_path
        self.legacy_session_history_path = legacy_session_history_path

        self.db_path = db_path.expanduser().resolve() if db_path is not None else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_database_url = str(database_url or "").strip()
        if not resolved_database_url:
            assert self.db_path is not None
            resolved_database_url = _database_url_from_path(self.db_path)
        self.database_url = resolved_database_url
        self.database_url_redacted = _redact_database_url(resolved_database_url)
        self.backend_name = _database_backend_name(resolved_database_url)
        if self.backend_name == "sqlite" and self.db_path is None:
            sqlite_path = _sqlite_path_from_database_url(resolved_database_url)
            if sqlite_path is not None:
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        connect_args: dict[str, object] = {}
        if self.backend_name == "sqlite":
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(
            resolved_database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        if self.backend_name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def _configure_sqlite(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                finally:
                    cursor.close()

        self.metadata = MetaData()
        self.feedback_artist_stats = Table(
            "feedback_artist_stats",
            self.metadata,
            Column("artist_key", String(512), primary_key=True),
            Column("artist_name", String(512), nullable=False),
            Column("like_count", Integer, nullable=False, default=0),
            Column("repeat_count", Integer, nullable=False, default=0),
            Column("skip_count", Integer, nullable=False, default=0),
            Column("dislike_count", Integer, nullable=False, default=0),
            Column("last_signal", String(64), nullable=False, default=""),
            Column("last_session_id", String(256), nullable=False, default=""),
            Column("updated_at", String(64), nullable=False, default=""),
        )
        self.feedback_events = Table(
            "feedback_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", String(64), nullable=False),
            Column("session_id", String(256), nullable=False),
            Column("artist_name", String(512), nullable=False),
            Column("signal", String(64), nullable=False),
            Column("notes", Text, nullable=False, default=""),
            Index("idx_feedback_events_timestamp", "timestamp"),
        )
        self.session_history = Table(
            "session_history",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(256), nullable=False, unique=True),
            Column("created_at", String(64), nullable=False),
            Column("mode", String(128), nullable=False, default=""),
            Column("scenario", String(128), nullable=False, default=""),
            Column("top_artist", String(512), nullable=False, default=""),
            Column("adaptive_replans", Integer, nullable=False, default=0),
            Column("safe_route_steps", Integer, nullable=False, default=0),
            Column("effective_recent_artists_json", Text, nullable=False, default="[]"),
            Column("used_feedback_memory", Integer, nullable=False, default=0),
            Column("persisted", Integer, nullable=False, default=0),
            Index("idx_session_history_created_at", "created_at"),
        )
        self.active_sessions = Table(
            "active_sessions",
            self.metadata,
            Column("session_id", String(256), primary_key=True),
            Column("created_at", String(64), nullable=False),
            Column("updated_at", String(64), nullable=False),
            Column("version", Integer, nullable=False, default=0),
            Column("request_json", Text, nullable=False),
            Column("plan_json", Text, nullable=False),
            Column("adaptation_json", Text, nullable=False, default="{}"),
            Index("idx_active_sessions_updated_at", "updated_at"),
        )
        self.session_events = Table(
            "session_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(256), nullable=False),
            Column("event_id", String(256), nullable=False),
            Column("timestamp", String(64), nullable=False),
            Column("event_type", String(64), nullable=False),
            Column("artist_name", String(512), nullable=False),
            Column("expected_version", Integer, nullable=False),
            Column("resulting_version", Integer, nullable=False),
            Column("request_json", Text, nullable=False),
            Column("response_json", Text, nullable=False),
            Column("durable_feedback", Integer, nullable=False, default=0),
            UniqueConstraint("session_id", "event_id", name="uq_session_events_session_event"),
            Index("idx_session_events_session_id", "session_id"),
            Index("idx_session_events_timestamp", "timestamp"),
        )

        self._initialize()
        self._migrate_legacy_if_needed()
        self.sync_legacy_snapshots()

    def _initialize(self) -> None:
        self.metadata.create_all(self.engine)

    def _table_count(self, table: Table) -> int:
        with self.engine.connect() as conn:
            count = conn.execute(select(func.count()).select_from(table)).scalar_one()
        return int(count or 0)

    def _migrate_legacy_if_needed(self) -> None:
        has_state = any(
            (
                self._table_count(self.feedback_artist_stats) > 0,
                self._table_count(self.feedback_events) > 0,
                self._table_count(self.session_history) > 0,
            )
        )
        if has_state:
            return

        migrated = False
        memory_rows = _read_json_document(self.legacy_feedback_memory_path)
        event_rows = _read_jsonl_rows(self.legacy_feedback_event_log_path)
        session_rows = _read_jsonl_rows(self.legacy_session_history_path)

        with self.engine.begin() as conn:
            artists_obj = memory_rows.get("artists", {})
            artists = artists_obj if isinstance(artists_obj, dict) else {}
            for raw_key, raw_stats in artists.items():
                if not isinstance(raw_stats, dict):
                    continue
                artist_name = str(raw_stats.get("artist_name") or raw_key).strip()
                if not artist_name:
                    continue
                conn.execute(
                    self.feedback_artist_stats.insert().values(
                        artist_key=_artist_key(artist_name),
                        artist_name=artist_name,
                        like_count=_coerce_int(raw_stats.get("like_count", 0)),
                        repeat_count=_coerce_int(raw_stats.get("repeat_count", 0)),
                        skip_count=_coerce_int(raw_stats.get("skip_count", 0)),
                        dislike_count=_coerce_int(raw_stats.get("dislike_count", 0)),
                        last_signal=str(raw_stats.get("last_signal", "")),
                        last_session_id=str(raw_stats.get("last_session_id", "")),
                        updated_at=str(raw_stats.get("updated_at", "")),
                    )
                )
                migrated = True

            for row in event_rows:
                conn.execute(
                    self.feedback_events.insert().values(
                        timestamp=str(row.get("timestamp", "")),
                        session_id=str(row.get("session_id", "")),
                        artist_name=str(row.get("artist_name", "")),
                        signal=str(row.get("signal", "")),
                        notes=str(row.get("notes", "")),
                    )
                )
                migrated = True

            for row in session_rows:
                conn.execute(
                    self.session_history.insert().values(
                        session_id=str(row.get("session_id", "")),
                        created_at=str(row.get("created_at", "")),
                        mode=str(row.get("mode", "")),
                        scenario=str(row.get("scenario", "")),
                        top_artist=str(row.get("top_artist", "")),
                        adaptive_replans=_coerce_int(row.get("adaptive_replans", 0)),
                        safe_route_steps=_coerce_int(row.get("safe_route_steps", 0)),
                        effective_recent_artists_json=json.dumps(_coerce_recent_artists(row.get("effective_recent_artists"))),
                        used_feedback_memory=1 if bool(row.get("used_feedback_memory")) else 0,
                        persisted=1 if bool(row.get("persisted")) else 0,
                    )
                )
                migrated = True

        if migrated:
            self.logger.info("Migrated legacy Taste OS state into %s", self.database_url_redacted)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            self.logger.exception("Taste OS state store ping failed")
            return False

    def health_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "database_url": self.database_url_redacted,
            "db_path": str(self.db_path) if self.db_path is not None else "",
            "legacy_snapshot_export": any(
                path is not None
                for path in (
                    self.legacy_feedback_memory_path,
                    self.legacy_feedback_event_log_path,
                    self.legacy_session_history_path,
                )
            ),
            "reachable": self.ping(),
        }

    def feedback_store(self) -> dict[str, object]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    self.feedback_artist_stats.c.artist_key,
                    self.feedback_artist_stats.c.artist_name,
                    self.feedback_artist_stats.c.like_count,
                    self.feedback_artist_stats.c.repeat_count,
                    self.feedback_artist_stats.c.skip_count,
                    self.feedback_artist_stats.c.dislike_count,
                    self.feedback_artist_stats.c.last_signal,
                    self.feedback_artist_stats.c.last_session_id,
                    self.feedback_artist_stats.c.updated_at,
                ).order_by(
                    self.feedback_artist_stats.c.updated_at.desc(),
                    self.feedback_artist_stats.c.artist_name.asc(),
                )
            ).mappings().all()
            summary = conn.execute(
                select(
                    func.count().label("event_count"),
                    func.coalesce(func.max(self.feedback_events.c.timestamp), "").label("updated_at"),
                ).select_from(self.feedback_events)
            ).mappings().one()

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
            "event_count": int(summary["event_count"] or 0),
            "updated_at": str(summary["updated_at"] or ""),
        }

    def recent_feedback(self, *, limit: int) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    self.feedback_events.c.timestamp,
                    self.feedback_events.c.session_id,
                    self.feedback_events.c.artist_name,
                    self.feedback_events.c.signal,
                    self.feedback_events.c.notes,
                )
                .order_by(self.feedback_events.c.id.desc())
                .limit(int(limit))
            ).mappings().all()
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
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    self.session_history.c.session_id,
                    self.session_history.c.created_at,
                    self.session_history.c.mode,
                    self.session_history.c.scenario,
                    self.session_history.c.top_artist,
                    self.session_history.c.adaptive_replans,
                    self.session_history.c.safe_route_steps,
                    self.session_history.c.effective_recent_artists_json,
                    self.session_history.c.used_feedback_memory,
                    self.session_history.c.persisted,
                )
                .order_by(self.session_history.c.id.desc())
                .limit(int(limit))
            ).mappings().all()
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

    @staticmethod
    def _decode_json_object(raw: object) -> dict[str, object]:
        try:
            parsed = json.loads(str(raw or "{}"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _json_text(payload: dict[str, object]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def active_session(self, session_id: str) -> dict[str, object]:
        normalized_session_id = str(session_id).strip()
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    self.active_sessions.c.session_id,
                    self.active_sessions.c.created_at,
                    self.active_sessions.c.updated_at,
                    self.active_sessions.c.version,
                    self.active_sessions.c.request_json,
                    self.active_sessions.c.plan_json,
                    self.active_sessions.c.adaptation_json,
                ).where(self.active_sessions.c.session_id == normalized_session_id)
            ).mappings().first()
        if row is None:
            raise ActiveSessionNotFoundError(f"Active Taste OS session not found: {normalized_session_id}")
        return {
            "session_id": str(row["session_id"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "version": int(row["version"] or 0),
            "request": self._decode_json_object(row["request_json"]),
            "plan": self._decode_json_object(row["plan_json"]),
            "adaptation": self._decode_json_object(row["adaptation_json"]),
        }

    def create_active_session(
        self,
        *,
        session_id: str,
        created_at: str,
        request: dict[str, object],
        plan: dict[str, object],
        adaptation: dict[str, object],
    ) -> dict[str, object]:
        normalized_session_id = str(session_id).strip()
        with self.engine.begin() as conn:
            conn.execute(
                self.active_sessions.insert().values(
                    session_id=normalized_session_id,
                    created_at=str(created_at),
                    updated_at=str(created_at),
                    version=0,
                    request_json=self._json_text(request),
                    plan_json=self._json_text(plan),
                    adaptation_json=self._json_text(adaptation),
                )
            )
        return self.active_session(normalized_session_id)

    def session_event(self, *, session_id: str, event_id: str) -> dict[str, object] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    self.session_events.c.session_id,
                    self.session_events.c.event_id,
                    self.session_events.c.event_type,
                    self.session_events.c.artist_name,
                    self.session_events.c.expected_version,
                    self.session_events.c.resulting_version,
                    self.session_events.c.request_json,
                    self.session_events.c.response_json,
                    self.session_events.c.durable_feedback,
                ).where(
                    self.session_events.c.session_id == str(session_id).strip(),
                    self.session_events.c.event_id == str(event_id).strip(),
                )
            ).mappings().first()
        if row is None:
            return None
        return {
            "session_id": str(row["session_id"]),
            "event_id": str(row["event_id"]),
            "event_type": str(row["event_type"]),
            "artist_name": str(row["artist_name"]),
            "expected_version": int(row["expected_version"]),
            "resulting_version": int(row["resulting_version"]),
            "request": self._decode_json_object(row["request_json"]),
            "response": self._decode_json_object(row["response_json"]),
            "durable_feedback": bool(row["durable_feedback"]),
        }

    def list_session_events(
        self,
        *,
        session_id: str,
        limit: int = DEFAULT_SESSION_EVENT_LIMIT,
    ) -> list[dict[str, object]]:
        bounded_limit = min(MAX_SESSION_EVENT_LIMIT, max(0, int(limit)))
        if bounded_limit <= 0:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    self.session_events.c.session_id,
                    self.session_events.c.event_id,
                    self.session_events.c.timestamp,
                    self.session_events.c.event_type,
                    self.session_events.c.artist_name,
                    self.session_events.c.expected_version,
                    self.session_events.c.resulting_version,
                    self.session_events.c.response_json,
                    self.session_events.c.durable_feedback,
                )
                .where(self.session_events.c.session_id == str(session_id).strip())
                .order_by(self.session_events.c.id.desc())
                .limit(bounded_limit)
            ).mappings().all()

        events: list[dict[str, object]] = []
        for row in reversed(rows):
            response = self._decode_json_object(row["response_json"])
            events.append(
                {
                    "session_id": str(row["session_id"]),
                    "event_id": str(row["event_id"]),
                    "timestamp": str(row["timestamp"]),
                    "event_type": str(row["event_type"]),
                    "artist_name": str(row["artist_name"]),
                    "expected_version": int(row["expected_version"]),
                    "resulting_version": int(row["resulting_version"]),
                    "durable_feedback": bool(row["durable_feedback"]),
                    "why_this_changed": str(response.get("why_this_changed", "")),
                }
            )
        return events

    def _record_feedback_in_transaction(
        self,
        conn: Any,
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
        conn.execute(
            self.feedback_events.insert().values(
                timestamp=str(timestamp),
                session_id=str(session_id),
                artist_name=normalized_artist_name,
                signal=normalized_signal,
                notes=str(notes or ""),
            )
        )
        row = conn.execute(
            select(
                self.feedback_artist_stats.c.like_count,
                self.feedback_artist_stats.c.repeat_count,
                self.feedback_artist_stats.c.skip_count,
                self.feedback_artist_stats.c.dislike_count,
            ).where(self.feedback_artist_stats.c.artist_key == artist_row_key)
        ).mappings().first()
        counts = {
            "like_count": int((row["like_count"] if row is not None else 0) or 0),
            "repeat_count": int((row["repeat_count"] if row is not None else 0) or 0),
            "skip_count": int((row["skip_count"] if row is not None else 0) or 0),
            "dislike_count": int((row["dislike_count"] if row is not None else 0) or 0),
        }
        counts[counter_name] = int(counts.get(counter_name, 0)) + 1
        values = {
            "artist_name": normalized_artist_name,
            "like_count": int(counts["like_count"]),
            "repeat_count": int(counts["repeat_count"]),
            "skip_count": int(counts["skip_count"]),
            "dislike_count": int(counts["dislike_count"]),
            "last_signal": normalized_signal,
            "last_session_id": str(session_id),
            "updated_at": str(timestamp),
        }
        if row is None:
            conn.execute(
                self.feedback_artist_stats.insert().values(
                    artist_key=artist_row_key,
                    **values,
                )
            )
        else:
            conn.execute(
                self.feedback_artist_stats.update()
                .where(self.feedback_artist_stats.c.artist_key == artist_row_key)
                .values(**values)
            )
        return {
            "timestamp": str(timestamp),
            "session_id": str(session_id),
            "artist_name": normalized_artist_name,
            "signal": normalized_signal,
            "notes": str(notes or ""),
        }

    def apply_session_event(
        self,
        *,
        timestamp: str,
        session_id: str,
        event_id: str,
        event_type: str,
        artist_name: str,
        expected_version: int,
        request: dict[str, object],
        response: dict[str, object],
        updated_request: dict[str, object],
        adaptation: dict[str, object],
        durable_feedback: bool,
    ) -> tuple[dict[str, object], bool]:
        normalized_session_id = str(session_id).strip()
        normalized_event_id = str(event_id).strip()
        normalized_event_type = str(event_type).strip().lower()
        normalized_artist_name = str(artist_name).strip()
        request_text = self._json_text(request)

        with self.engine.begin() as conn:
            existing_event = conn.execute(
                select(
                    self.session_events.c.event_type,
                    self.session_events.c.artist_name,
                    self.session_events.c.expected_version,
                    self.session_events.c.request_json,
                    self.session_events.c.response_json,
                ).where(
                    self.session_events.c.session_id == normalized_session_id,
                    self.session_events.c.event_id == normalized_event_id,
                )
            ).mappings().first()
            if existing_event is not None:
                if str(existing_event["request_json"]) != request_text:
                    raise SessionEventConflictError(
                        f"event_id {normalized_event_id!r} was already used with a different payload."
                    )
                return self._decode_json_object(existing_event["response_json"]), True

            current_version = conn.execute(
                select(self.active_sessions.c.version).where(
                    self.active_sessions.c.session_id == normalized_session_id
                )
            ).scalar_one_or_none()
            if current_version is None:
                raise ActiveSessionNotFoundError(f"Active Taste OS session not found: {normalized_session_id}")
            if int(current_version) != int(expected_version):
                raise StaleSessionVersionError(
                    expected_version=int(expected_version),
                    current_version=int(current_version),
                )

            resulting_version = int(expected_version) + 1
            update_result = conn.execute(
                self.active_sessions.update()
                .where(
                    self.active_sessions.c.session_id == normalized_session_id,
                    self.active_sessions.c.version == int(expected_version),
                )
                .values(
                    updated_at=str(timestamp),
                    version=resulting_version,
                    request_json=self._json_text(updated_request),
                    plan_json=self._json_text(response),
                    adaptation_json=self._json_text(adaptation),
                )
            )
            if update_result.rowcount != 1:
                latest_version = conn.execute(
                    select(self.active_sessions.c.version).where(
                        self.active_sessions.c.session_id == normalized_session_id
                    )
                ).scalar_one()
                raise StaleSessionVersionError(
                    expected_version=int(expected_version),
                    current_version=int(latest_version),
                )

            conn.execute(
                self.session_events.insert().values(
                    session_id=normalized_session_id,
                    event_id=normalized_event_id,
                    timestamp=str(timestamp),
                    event_type=normalized_event_type,
                    artist_name=normalized_artist_name,
                    expected_version=int(expected_version),
                    resulting_version=resulting_version,
                    request_json=request_text,
                    response_json=self._json_text(response),
                    durable_feedback=1 if durable_feedback else 0,
                )
            )
            if durable_feedback:
                self._record_feedback_in_transaction(
                    conn,
                    timestamp=timestamp,
                    session_id=normalized_session_id,
                    artist_name=normalized_artist_name,
                    signal=normalized_event_type,
                    notes=f"Live session event {normalized_event_id}",
                )

        if durable_feedback:
            self.sync_legacy_snapshots()
        return response, False

    def record_feedback(
        self,
        *,
        timestamp: str,
        session_id: str,
        artist_name: str,
        signal: str,
        notes: str | None,
    ) -> dict[str, object]:
        with self.engine.begin() as conn:
            event_row = self._record_feedback_in_transaction(
                conn,
                timestamp=timestamp,
                session_id=session_id,
                artist_name=artist_name,
                signal=signal,
                notes=notes,
            )
        self.sync_legacy_snapshots()
        return event_row

    def record_session_history(self, row: dict[str, object]) -> None:
        session_id = str(row.get("session_id", ""))
        values = {
            "created_at": str(row.get("created_at", "")),
            "mode": str(row.get("mode", "")),
            "scenario": str(row.get("scenario", "")),
            "top_artist": str(row.get("top_artist", "")),
            "adaptive_replans": _coerce_int(row.get("adaptive_replans", 0)),
            "safe_route_steps": _coerce_int(row.get("safe_route_steps", 0)),
            "effective_recent_artists_json": json.dumps(
                _coerce_recent_artists(row.get("effective_recent_artists")),
                sort_keys=True,
            ),
            "used_feedback_memory": 1 if bool(row.get("used_feedback_memory")) else 0,
            "persisted": 1 if bool(row.get("persisted")) else 0,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(self.session_history.c.id).where(self.session_history.c.session_id == session_id)
            ).scalar_one_or_none()
            if existing is None:
                conn.execute(
                    self.session_history.insert().values(
                        session_id=session_id,
                        **values,
                    )
                )
            else:
                conn.execute(
                    self.session_history.update()
                    .where(self.session_history.c.session_id == session_id)
                    .values(**values)
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
