from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, TypedDict
from urllib.parse import quote, unquote

from .taste_os_demo import MODE_CONFIGS, SCENARIOS

MAX_REQUEST_BYTES = 1_000_000
DEFAULT_MAX_TOP_K = 20
DEFAULT_HISTORY_LIMIT = 8
FEEDBACK_SIGNALS = {"like", "dislike", "repeat", "skip"}


class RequestValidationError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)
        self.details = details or {}


class TasteOSRequestPayload(TypedDict):
    mode: str
    scenario: str
    top_k: int
    include_video: bool
    recent_artists: list[str] | None
    persist_artifacts: bool
    use_feedback_memory: bool


class TasteOSFeedbackPayload(TypedDict):
    session_id: str
    artist_name: str
    signal: str
    notes: str | None


def parse_taste_os_args() -> argparse.Namespace:
    env_max_top_k_raw = os.getenv("SPOTIFY_TASTE_OS_MAX_TOP_K", str(DEFAULT_MAX_TOP_K)).strip()
    try:
        env_max_top_k = max(1, int(env_max_top_k_raw))
    except ValueError:
        env_max_top_k = DEFAULT_MAX_TOP_K

    parser = argparse.ArgumentParser(
        prog="python -m spotify.taste_os_service",
        description="Serve Taste OS session plans over HTTP.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis/taste_os_service",
        help="Directory for optional persisted session artifacts.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8010, help="HTTP port.")
    parser.add_argument("--max-top-k", type=int, default=env_max_top_k, help="Maximum top_k value accepted by the API.")
    parser.add_argument(
        "--auth-token",
        type=str,
        default=os.getenv("SPOTIFY_TASTE_OS_AUTH_TOKEN"),
        help="Optional API token. When set, POST /taste-os/session requires Authorization: Bearer <token>.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files by default when rebuilding request context.",
    )
    return parser.parse_args()


def error_payload(code: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def artist_key(name: str) -> str:
    return str(name).strip().casefold()


def load_json_document(path: Path, *, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return dict(default)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return parsed if isinstance(parsed, dict) else dict(default)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, object]]:
    if limit <= 0 or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, object]] = []
    for line in lines[-limit:]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return list(reversed(rows))


def guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".markdown"}:
        return "text/markdown; charset=utf-8"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    return "text/plain; charset=utf-8"


def read_json_payload(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_type = handler.headers.get("Content-Type", "")
    if content_type and "application/json" not in content_type.lower():
        raise RequestValidationError(
            status_code=415,
            code="unsupported_media_type",
            message="Content-Type must be application/json.",
        )

    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except ValueError as exc:
        raise RequestValidationError(
            status_code=400,
            code="invalid_content_length",
            message="Content-Length header must be an integer.",
        ) from exc

    if length < 0:
        raise RequestValidationError(
            status_code=400,
            code="invalid_content_length",
            message="Content-Length must be non-negative.",
        )
    if length > MAX_REQUEST_BYTES:
        raise RequestValidationError(
            status_code=413,
            code="payload_too_large",
            message=f"Payload exceeds limit of {MAX_REQUEST_BYTES} bytes.",
        )

    try:
        raw = handler.rfile.read(length) if length > 0 else b"{}"
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RequestValidationError(
            status_code=400,
            code="invalid_json",
            message="Request body must be valid JSON.",
        ) from exc

    if not isinstance(parsed, dict):
        raise RequestValidationError(
            status_code=400,
            code="invalid_payload",
            message="JSON payload must be an object.",
        )
    return parsed


def _normalize_recent_artists(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = [part.strip() for part in value.split("|") if part.strip()]
        return parsed or None
    if not isinstance(value, list):
        raise RequestValidationError(
            status_code=422,
            code="invalid_recent_artists",
            message="recent_artists must be a list of strings or a pipe-separated string.",
        )
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RequestValidationError(
                status_code=422,
                code="invalid_recent_artists",
                message="recent_artists list must contain only strings.",
            )
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned or None


def _normalize_top_k(value: object, *, max_top_k: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestValidationError(
            status_code=422,
            code="invalid_top_k",
            message="top_k must be an integer.",
        )
    if value < 1:
        raise RequestValidationError(
            status_code=422,
            code="invalid_top_k",
            message="top_k must be at least 1.",
        )
    if value > max_top_k:
        raise RequestValidationError(
            status_code=422,
            code="invalid_top_k",
            message=f"top_k cannot exceed {max_top_k}.",
        )
    return int(value)


def _normalize_optional_bool(value: object, *, default: bool, field_name: str) -> bool:
    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise RequestValidationError(
            status_code=422,
            code=f"invalid_{field_name}",
            message=f"{field_name} must be a boolean.",
        )
    return bool(value)


def _normalize_choice(value: object, *, choices: set[str], default: str, field_name: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise RequestValidationError(
            status_code=422,
            code=f"invalid_{field_name}",
            message=f"{field_name} must be a string.",
        )
    normalized = value.strip().lower()
    if normalized not in choices:
        raise RequestValidationError(
            status_code=422,
            code=f"invalid_{field_name}",
            message=f"{field_name} must be one of: {', '.join(sorted(choices))}.",
        )
    return normalized


def normalize_taste_os_payload(
    payload: dict[str, object],
    *,
    default_include_video: bool,
    max_top_k: int,
) -> TasteOSRequestPayload:
    allowed_keys = {
        "mode",
        "scenario",
        "top_k",
        "include_video",
        "recent_artists",
        "persist_artifacts",
        "use_feedback_memory",
    }
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise RequestValidationError(
            status_code=400,
            code="unknown_fields",
            message="Payload contains unknown fields.",
            details={"fields": unknown_keys},
        )

    mode = _normalize_choice(
        payload.get("mode"),
        choices=set(MODE_CONFIGS),
        default="focus",
        field_name="mode",
    )
    scenario = _normalize_choice(
        payload.get("scenario"),
        choices=set(SCENARIOS),
        default="steady",
        field_name="scenario",
    )
    top_k = _normalize_top_k(payload.get("top_k", 5), max_top_k=max_top_k)
    include_video_raw = payload.get("include_video", default_include_video)
    if not isinstance(include_video_raw, bool):
        raise RequestValidationError(
            status_code=422,
            code="invalid_include_video",
            message="include_video must be a boolean.",
        )
    include_video = bool(include_video_raw)
    recent_artists = _normalize_recent_artists(payload.get("recent_artists"))
    persist_artifacts = _normalize_optional_bool(
        payload.get("persist_artifacts"),
        default=False,
        field_name="persist_artifacts",
    )
    use_feedback_memory = _normalize_optional_bool(
        payload.get("use_feedback_memory"),
        default=True,
        field_name="use_feedback_memory",
    )
    return {
        "mode": mode,
        "scenario": scenario,
        "top_k": top_k,
        "include_video": include_video,
        "recent_artists": recent_artists,
        "persist_artifacts": persist_artifacts,
        "use_feedback_memory": use_feedback_memory,
    }


def normalize_taste_os_feedback_payload(payload: dict[str, object]) -> TasteOSFeedbackPayload:
    allowed_keys = {"session_id", "artist_name", "signal", "notes"}
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise RequestValidationError(
            status_code=400,
            code="unknown_fields",
            message="Payload contains unknown fields.",
            details={"fields": unknown_keys},
        )

    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise RequestValidationError(
            status_code=422,
            code="invalid_session_id",
            message="session_id must be a non-empty string.",
        )

    artist_name = str(payload.get("artist_name", "")).strip()
    if not artist_name:
        raise RequestValidationError(
            status_code=422,
            code="invalid_artist_name",
            message="artist_name must be a non-empty string.",
        )

    signal = str(payload.get("signal", "")).strip().lower()
    if signal not in FEEDBACK_SIGNALS:
        raise RequestValidationError(
            status_code=422,
            code="invalid_signal",
            message=f"signal must be one of: {', '.join(sorted(FEEDBACK_SIGNALS))}.",
        )

    notes_raw = payload.get("notes")
    if notes_raw is None:
        notes = None
    elif not isinstance(notes_raw, str):
        raise RequestValidationError(
            status_code=422,
            code="invalid_notes",
            message="notes must be a string when provided.",
        )
    else:
        notes = notes_raw.strip() or None

    return {
        "session_id": session_id,
        "artist_name": artist_name,
        "signal": signal,
        "notes": notes,
    }


def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    value = auth_header.strip()
    if not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


def is_authorized_request(headers: Any, expected_token: str | None) -> bool:
    if not expected_token:
        return True
    bearer = _extract_bearer_token(headers.get("Authorization"))
    api_key = headers.get("X-API-Key")
    provided = bearer or (str(api_key).strip() if api_key is not None else "")
    if not provided:
        return False
    return secrets.compare_digest(provided, expected_token)


def build_taste_os_handler(service: Any, *, page_renderer: Callable[[Any], str]):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            status_code: int,
            body: bytes,
            *,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_html(
            self,
            status_code: int,
            body: str,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            payload = body.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self,
            status_code: int,
            payload: dict[str, object],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_error(
            self,
            status_code: int,
            *,
            code: str,
            message: str,
            details: dict[str, object] | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self._send_json(
                status_code,
                error_payload(code=code, message=message, details=details),
                extra_headers=extra_headers,
            )

        def log_message(self, fmt: str, *args) -> None:
            service.logger.info("HTTP %s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path in {"", "/", "/taste-os"}:
                self._send_html(200, page_renderer(service))
                return
            if path.startswith("/taste-os/artifacts/"):
                relative_path = unquote(path.removeprefix("/taste-os/artifacts/"))
                try:
                    artifact_path = service.resolve_artifact_path(relative_path)
                    self._send_bytes(
                        200,
                        artifact_path.read_bytes(),
                        content_type=guess_content_type(artifact_path),
                    )
                except FileNotFoundError as exc:
                    self._send_error(404, code="artifact_not_found", message=str(exc))
                return
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model_name": service.model_name,
                        "model_type": service.model_type,
                        "run_dir": str(service.run_dir),
                        "max_top_k": service.max_top_k,
                        "requires_auth": bool(service.auth_token),
                    },
                )
                return
            if path == "/taste-os/catalog":
                self._send_json(
                    200,
                    {
                        "modes": [
                            {"name": mode.name, "description": mode.description, "planned_horizon": int(mode.horizon)}
                            for mode in MODE_CONFIGS.values()
                        ],
                        "scenarios": [
                            {"name": scenario.name, "description": scenario.description, "event_count": len(scenario.events)}
                            for scenario in SCENARIOS.values()
                        ],
                    },
                )
                return
            if path == "/taste-os/history":
                self._send_json(200, service.history_snapshot())
                return
            self._send_error(404, code="not_found", message="Resource not found.")

        def do_POST(self) -> None:  # noqa: N802
            request_path = self.path.rstrip("/")
            if request_path not in {"/taste-os/session", "/taste-os/feedback"}:
                self._send_error(404, code="not_found", message="Resource not found.")
                return

            if not is_authorized_request(self.headers, service.auth_token):
                self._send_error(
                    401,
                    code="unauthorized",
                    message="Missing or invalid API token.",
                    extra_headers={"WWW-Authenticate": "Bearer"},
                )
                return

            try:
                payload = read_json_payload(self)
                if request_path == "/taste-os/session":
                    normalized = normalize_taste_os_payload(
                        payload,
                        default_include_video=service.include_video,
                        max_top_k=service.max_top_k,
                    )
                else:
                    normalized = normalize_taste_os_feedback_payload(payload)
            except RequestValidationError as exc:
                self._send_error(
                    exc.status_code,
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                )
                return

            try:
                if request_path == "/taste-os/session":
                    session_request = normalized
                    assert isinstance(session_request, dict)
                    result = service.plan_session(
                        mode=str(session_request["mode"]),
                        scenario=str(session_request["scenario"]),
                        top_k=int(session_request["top_k"]),
                        recent_artists=session_request["recent_artists"],
                        include_video=bool(session_request["include_video"]),
                        persist_artifacts=bool(session_request["persist_artifacts"]),
                        use_feedback_memory=bool(session_request["use_feedback_memory"]),
                    )
                else:
                    feedback_request = normalized
                    assert isinstance(feedback_request, dict)
                    result = service.record_feedback(
                        session_id=str(feedback_request["session_id"]),
                        artist_name=str(feedback_request["artist_name"]),
                        signal=str(feedback_request["signal"]),
                        notes=feedback_request["notes"],
                    )
                self._send_json(200, result)
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                self._send_error(
                    422,
                    code="taste_os_input_error",
                    message=str(exc),
                )
            except Exception:
                service.logger.exception("Unhandled Taste OS planning failure")
                self._send_error(
                    500,
                    code="internal_error",
                    message="Taste OS planning failed due to an internal server error.",
                )

    return Handler


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MAX_TOP_K",
    "RequestValidationError",
    "TasteOSFeedbackPayload",
    "TasteOSRequestPayload",
    "append_jsonl",
    "artist_key",
    "build_taste_os_handler",
    "guess_content_type",
    "is_authorized_request",
    "load_json_document",
    "normalize_taste_os_feedback_payload",
    "normalize_taste_os_payload",
    "parse_taste_os_args",
    "read_jsonl_tail",
    "utc_now_iso",
]
