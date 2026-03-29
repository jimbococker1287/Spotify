from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import logging
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote, unquote

import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .digital_twin import ListenerDigitalTwinArtifact
from .env import load_local_env
from .multimodal import MultimodalArtistSpace
from .predict_next import (
    PredictionInputContext,
    _prepare_inputs,
    load_prediction_input_context,
    prediction_source_signature,
)
from .safe_policy import SafeBanditPolicyArtifact
from .serving import load_predictor, resolve_model_row
from .taste_os_demo import MODE_CONFIGS, SCENARIOS, _load_artifact, build_taste_os_demo_payload, write_taste_os_demo_artifacts

MAX_REQUEST_BYTES = 1_000_000
DEFAULT_MAX_TOP_K = 20
DEFAULT_HISTORY_LIMIT = 8
FEEDBACK_SIGNALS = {"like", "dislike", "repeat", "skip"}


@dataclass(frozen=True)
class _TasteOSContextCacheEntry:
    signature: tuple[tuple[str, int, int], ...]
    context: PredictionInputContext


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


def _parse_args() -> argparse.Namespace:
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


def _error_payload(code: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artist_key(name: str) -> str:
    return str(name).strip().casefold()


def _load_json_document(path: Path, *, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return dict(default)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return parsed if isinstance(parsed, dict) else dict(default)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, object]]:
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


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".markdown"}:
        return "text/markdown; charset=utf-8"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    return "text/plain; charset=utf-8"


def _read_json_payload(handler: BaseHTTPRequestHandler) -> dict[str, object]:
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


def _taste_os_page_html(service: "TasteOSService") -> str:
    config = {
        "modelName": service.model_name,
        "modelType": service.model_type,
        "runDir": str(service.run_dir),
        "outputDir": str(service.output_dir),
        "maxTopK": int(service.max_top_k),
        "defaultTopK": min(5, int(service.max_top_k)),
        "requiresAuth": bool(service.auth_token),
    }
    config_json = json.dumps(config)
    model_label = escape(f"{service.model_name} [{service.model_type}]")
    run_dir = escape(str(service.run_dir))
    output_dir = escape(str(service.output_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Taste OS Session Studio</title>
  <style>
    :root {{
      --paper: #f4efe5;
      --paper-strong: #fbf8f1;
      --ink: #1e1d1a;
      --muted: #625b4f;
      --line: rgba(30, 29, 26, 0.12);
      --accent: #b64f2f;
      --accent-soft: rgba(182, 79, 47, 0.12);
      --accent-deep: #8e3419;
      --olive: #687347;
      --olive-soft: rgba(104, 115, 71, 0.12);
      --sky: #cadfd9;
      --card-shadow: 0 18px 48px rgba(58, 46, 36, 0.10);
      --mono: "IBM Plex Mono", "SFMono-Regular", "Menlo", monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Avenir Next", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182, 79, 47, 0.18), transparent 28%),
        radial-gradient(circle at right 12%, rgba(104, 115, 71, 0.16), transparent 22%),
        linear-gradient(180deg, #f7f2e8 0%, #efe7d9 54%, #ece2d2 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1320px, calc(100vw - 32px));
      margin: 24px auto 40px;
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
      gap: 20px;
    }}
    .panel, .stage {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: rgba(251, 248, 241, 0.88);
      box-shadow: var(--card-shadow);
      backdrop-filter: blur(12px);
    }}
    .panel {{
      padding: 24px;
      position: sticky;
      top: 24px;
      align-self: start;
    }}
    .stage {{
      padding: 26px;
      overflow: hidden;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-deep);
      font: 600 12px/1 var(--mono);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1, h2, h3 {{
      font-family: var(--serif);
      font-weight: 700;
      margin: 0;
    }}
    h1 {{
      font-size: clamp(2.3rem, 4vw, 4rem);
      line-height: 0.96;
      margin-top: 14px;
      max-width: 10ch;
    }}
    h2 {{
      font-size: 1.55rem;
      margin-bottom: 10px;
    }}
    p, li, label, input, textarea, button {{
      font-size: 0.98rem;
      line-height: 1.45;
    }}
    .lede {{
      margin: 14px 0 24px;
      color: var(--muted);
      max-width: 34ch;
    }}
    .meta {{
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(202, 223, 217, 0.32), rgba(251, 248, 241, 0.92));
      border: 1px solid rgba(104, 115, 71, 0.14);
    }}
    .meta-row {{
      display: grid;
      gap: 4px;
    }}
    .meta-label {{
      color: var(--muted);
      font: 600 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .meta-value {{
      font-weight: 600;
      word-break: break-word;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .choices {{
      display: grid;
      gap: 10px;
    }}
    .choice-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }}
    .choice {{
      padding: 14px 14px 12px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      text-align: left;
    }}
    .choice:hover {{ transform: translateY(-2px); }}
    .choice.active {{
      border-color: rgba(182, 79, 47, 0.55);
      background: linear-gradient(180deg, rgba(182, 79, 47, 0.12), rgba(255, 255, 255, 0.9));
    }}
    .choice strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.98rem;
    }}
    .choice span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .field {{
      display: grid;
      gap: 7px;
    }}
    .field label {{
      font-weight: 600;
    }}
    input[type="number"], textarea {{
      width: 100%;
      border: 1px solid rgba(30, 29, 26, 0.15);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      font-family: var(--sans);
    }}
    textarea {{
      min-height: 92px;
      resize: vertical;
    }}
    .inline-options {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .inline-options label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    button.primary {{
      border: none;
      border-radius: 999px;
      padding: 13px 20px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
      color: white;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 14px 28px rgba(182, 79, 47, 0.20);
    }}
    button.primary:hover {{ transform: translateY(-1px); }}
    .hint {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .stage-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 22px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .summary-card {{
      padding: 16px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(202, 223, 217, 0.22));
      animation: rise 300ms ease;
    }}
    .summary-card .label {{
      color: var(--muted);
      font: 600 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .summary-card .value {{
      font-family: var(--serif);
      font-size: 1.45rem;
      line-height: 1.0;
    }}
    .result-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .result-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.78);
      animation: rise 320ms ease;
    }}
    .result-card.wide {{
      grid-column: 1 / -1;
    }}
    .result-card ol, .result-card ul {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    .history-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.72);
      animation: rise 340ms ease;
    }}
    .mini-card ul, .mini-card ol {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    .pill-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(104, 115, 71, 0.12);
      color: var(--olive);
      font: 600 12px/1 var(--mono);
    }}
    .pill.warn {{
      background: rgba(182, 79, 47, 0.12);
      color: var(--accent-deep);
    }}
    .candidate-list li, .plan-list li, .transcript-list li {{
      margin-bottom: 10px;
    }}
    .feedback-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .feedback-button {{
      border: 1px solid rgba(30, 29, 26, 0.12);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      cursor: pointer;
      font: 600 12px/1 var(--mono);
    }}
    .feedback-button:hover {{
      border-color: rgba(182, 79, 47, 0.35);
      color: var(--accent-deep);
    }}
    .metric-line {{
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .artifact-links a {{
      color: var(--accent-deep);
      text-decoration: none;
      font-weight: 600;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px;
      border-radius: 22px;
      border: 1px dashed rgba(30, 29, 26, 0.18);
      background: rgba(255, 255, 255, 0.54);
    }}
    .status {{
      min-height: 1.3em;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 0.82rem;
    }}
    .status.error {{
      color: #9b2314;
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 960px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .panel {{
        position: static;
      }}
      .result-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel">
      <div class="eyebrow">Taste OS Studio</div>
      <h1>Shape a listening session before it starts drifting.</h1>
      <p class="lede">This browser surface sits on top of the Taste OS planner, so we can steer a session, stress it with an event, and inspect the explanation, guardrails, and recovery path in one place.</p>
      <div class="meta">
        <div class="meta-row">
          <div class="meta-label">Serveable model</div>
          <div class="meta-value">{model_label}</div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Run</div>
          <div class="meta-value">{run_dir}</div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Artifact output</div>
          <div class="meta-value">{output_dir}</div>
        </div>
      </div>
      <div class="stack">
        <section class="choices">
          <h2>Mode</h2>
          <div id="mode-grid" class="choice-grid"></div>
        </section>
        <section class="choices">
          <h2>Scenario</h2>
          <div id="scenario-grid" class="choice-grid"></div>
        </section>
        <div class="field">
          <label for="top-k">Top candidates</label>
          <input id="top-k" type="number" min="1" max="{int(service.max_top_k)}" value="{min(5, int(service.max_top_k))}">
        </div>
        <div class="field">
          <label for="recent-artists">Recent artists</label>
          <textarea id="recent-artists" placeholder="Artist A|Artist B|Artist C"></textarea>
        </div>
        <div class="inline-options">
          <label><input id="include-video" type="checkbox"> Include video history</label>
          <label><input id="persist-artifacts" type="checkbox" checked> Persist artifacts</label>
          <label><input id="use-feedback-memory" type="checkbox" checked> Seed from feedback memory</label>
        </div>
        <div class="actions">
          <button id="run-session" class="primary" type="button">Generate Session</button>
          <button id="refresh-memory" type="button">Refresh Memory</button>
          <div class="hint">POSTs to <code>/taste-os/session</code>.</div>
        </div>
        <div id="status" class="status">Loading catalog…</div>
      </div>
    </aside>
    <main class="stage">
      <div class="stage-header">
        <div>
          <div class="eyebrow">Session Surface</div>
          <h2>Plan, rationale, and recovery flow</h2>
        </div>
        <div class="hint">Use this page to compare modes without dropping into raw JSON.</div>
      </div>
      <div id="summary-grid" class="summary-grid"></div>
      <div id="history-grid" class="history-grid"></div>
      <div id="result-grid" class="result-grid">
        <div class="empty">Choose a mode and scenario, then generate a session to see the opening choice, candidate stack, baseline plan, guardrails, and adaptive transcript.</div>
      </div>
    </main>
  </div>
  <script>
    const config = {config_json};
    const state = {{
      mode: "focus",
      scenario: "steady",
      catalog: {{ modes: [], scenarios: [] }},
      history: {{ recent_sessions: [], feedback_memory: {{}} }},
      payload: null,
    }};

    const modeGrid = document.getElementById("mode-grid");
    const scenarioGrid = document.getElementById("scenario-grid");
    const resultGrid = document.getElementById("result-grid");
    const summaryGrid = document.getElementById("summary-grid");
    const historyGrid = document.getElementById("history-grid");
    const statusNode = document.getElementById("status");
    const topKInput = document.getElementById("top-k");
    const recentArtistsInput = document.getElementById("recent-artists");
    const includeVideoInput = document.getElementById("include-video");
    const persistArtifactsInput = document.getElementById("persist-artifacts");
    const useFeedbackMemoryInput = document.getElementById("use-feedback-memory");
    const runButton = document.getElementById("run-session");
    const refreshMemoryButton = document.getElementById("refresh-memory");

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function setStatus(message, isError = false) {{
      statusNode.textContent = message;
      statusNode.className = isError ? "status error" : "status";
    }}

    function renderChoiceGrid(items, activeValue, container, key, descriptionKey) {{
      container.innerHTML = items.map((item) => `
        <button class="choice ${{
          item.name === activeValue ? "active" : ""
        }}" data-${{key}}="${{escapeHtml(item.name)}}">
          <strong>${{escapeHtml(item.name)}}</strong>
          <span>${{escapeHtml(item[descriptionKey])}}</span>
        </button>
      `).join("");
    }}

    function renderCatalog() {{
      renderChoiceGrid(state.catalog.modes, state.mode, modeGrid, "mode", "description");
      renderChoiceGrid(state.catalog.scenarios, state.scenario, scenarioGrid, "scenario", "description");
      modeGrid.querySelectorAll("[data-mode]").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.mode = node.dataset.mode;
          renderCatalog();
        }});
      }});
      scenarioGrid.querySelectorAll("[data-scenario]").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.scenario = node.dataset.scenario;
          renderCatalog();
        }});
      }});
    }}

    function summaryCards(payload) {{
      const summary = payload.demo_summary || {{}};
      const request = payload.request || {{}};
      const memory = payload.memory_summary || {{}};
      return [
        {{ label: "Mode", value: request.mode || "n/a" }},
        {{ label: "Scenario", value: request.scenario || "n/a" }},
        {{ label: "Top Artist", value: summary.top_artist || "n/a" }},
        {{ label: "Replans", value: String(summary.adaptive_replans ?? 0) }},
        {{ label: "Safe Route Steps", value: String(summary.adaptive_safe_route_steps ?? 0) }},
        {{ label: "Memory Seeds", value: String((memory.seed_artists || []).length) }},
      ];
    }}

    function renderSummary(payload) {{
      summaryGrid.innerHTML = summaryCards(payload).map((card) => `
        <div class="summary-card">
          <div class="label">${{escapeHtml(card.label)}}</div>
          <div class="value">${{escapeHtml(card.value)}}</div>
        </div>
      `).join("");
    }}

    function renderHistory(history) {{
      const feedback = history.feedback_memory || {{}};
      const recentSessions = Array.isArray(history.recent_sessions) ? history.recent_sessions : [];
      const topAffinities = Array.isArray(feedback.top_affinities) ? feedback.top_affinities : [];
      const avoidArtists = Array.isArray(feedback.avoid_artists) ? feedback.avoid_artists : [];
      const seedArtists = Array.isArray(feedback.seed_artists) ? feedback.seed_artists : [];
      historyGrid.innerHTML = `
        <section class="mini-card">
          <h3>Taste Memory</h3>
          <div class="metric-line">events=${{escapeHtml(feedback.event_count ?? 0)}} | artists=${{escapeHtml(feedback.artist_count ?? 0)}}</div>
          <div class="pill-list">
            ${{seedArtists.map((artist) => `<span class="pill">${{escapeHtml(artist)}}</span>`).join("") || '<span class="metric-line">No memory seeds yet.</span>'}}
          </div>
        </section>
        <section class="mini-card">
          <h3>Affinities</h3>
          <ul>
            ${{
              topAffinities.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">score=${{escapeHtml(row.net_score)}} | likes=${{escapeHtml(row.like_count)}} | repeats=${{escapeHtml(row.repeat_count)}}</div>
                </li>
              `).join("") || "<li>No positive feedback recorded yet.</li>"
            }}
          </ul>
        </section>
        <section class="mini-card">
          <h3>Avoid</h3>
          <ul>
            ${{
              avoidArtists.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">score=${{escapeHtml(row.net_score)}} | dislikes=${{escapeHtml(row.dislike_count)}} | skips=${{escapeHtml(row.skip_count)}}</div>
                </li>
              `).join("") || "<li>No avoid signals recorded yet.</li>"
            }}
          </ul>
        </section>
        <section class="mini-card">
          <h3>Recent Sessions</h3>
          <ol>
            ${{
              recentSessions.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.mode)}} / ${{escapeHtml(row.scenario)}}</strong>
                  <div class="metric-line">top=${{escapeHtml(row.top_artist || "n/a")}} | replans=${{escapeHtml(row.adaptive_replans ?? 0)}}</div>
                </li>
              `).join("") || "<li>No sessions recorded yet.</li>"
            }}
          </ol>
        </section>
      `;
    }}

    function renderResults(payload) {{
      const risk = payload.risk_summary || {{}};
      const fallback = payload.fallback_policy || {{}};
      const adaptive = payload.adaptive_session || {{}};
      const service = payload.service || {{}};
      const memory = payload.memory_summary || {{}};
      const topCandidates = Array.isArray(payload.top_candidates) ? payload.top_candidates : [];
      const journeyPlan = Array.isArray(payload.journey_plan) ? payload.journey_plan : [];
      const whyThisNext = Array.isArray(payload.why_this_next) ? payload.why_this_next : [];
      const transcript = Array.isArray(adaptive.transcript) ? adaptive.transcript : [];
      const effectiveRecentArtists = Array.isArray(memory.effective_recent_artists) ? memory.effective_recent_artists : [];
      const topAffinities = Array.isArray(memory.top_affinities) ? memory.top_affinities : [];
      const avoidArtists = Array.isArray(memory.avoid_artists) ? memory.avoid_artists : [];

      function feedbackButtons(artistName) {{
        return `
          <div class="feedback-buttons">
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="like">Like</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="repeat">Repeat</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="skip">Skip</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="dislike">Dislike</button>
          </div>
        `;
      }}

      const artifactBlock = service.persisted
        ? `<div class="artifact-links">
             <a href="${{escapeHtml(service.artifact_json_url || "#")}}" target="_blank" rel="noreferrer">Artifact JSON</a><div class="metric-line">${{escapeHtml(service.artifact_json || "")}}</div>
             <a href="${{escapeHtml(service.artifact_md_url || "#")}}" target="_blank" rel="noreferrer">Artifact MD</a><div class="metric-line">${{escapeHtml(service.artifact_md || "")}}</div>
           </div>`
        : `<div class="metric-line">Artifact persistence is off for this session.</div>`;

      resultGrid.innerHTML = `
        <section class="result-card">
          <h3>Why This Next</h3>
          <ul>${{whyThisNext.map((line) => `<li>${{escapeHtml(line)}}</li>`).join("") || "<li>No rationale available.</li>"}}</ul>
        </section>
        <section class="result-card">
          <h3>Guardrails</h3>
          <ul>
            <li>Risk state: <strong>${{escapeHtml(risk.risk_state || "n/a")}}</strong></li>
            <li>End risk: <strong>${{escapeHtml(risk.current_end_risk ?? "n/a")}}</strong></li>
            <li>Friction bucket: <strong>${{escapeHtml(risk.friction_bucket || "n/a")}}</strong></li>
            <li>Fallback policy: <strong>${{escapeHtml(fallback.active_policy_name || "n/a")}}</strong></li>
          </ul>
          <div class="metric-line">${{escapeHtml(fallback.reason || "")}}</div>
        </section>
        <section class="result-card wide">
          <h3>Top Candidates</h3>
          <ol class="candidate-list">
            ${{
              topCandidates.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.rank)}}. ${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">
                    model_prob=${{escapeHtml(row.model_probability)}} |
                    surface=${{escapeHtml(row.surface_score)}} |
                    mode=${{escapeHtml(row.mode_score)}} |
                    continuity=${{escapeHtml(row.continuity)}} |
                    novelty=${{escapeHtml(row.novelty)}}
                  </div>
                  <div class="metric-line">
                    memory=${{escapeHtml(row.memory_state || "neutral")}} |
                    memory_score=${{escapeHtml(row.memory_net_score ?? 0)}}
                  </div>
                  ${{feedbackButtons(row.artist_name || "")}}
                </li>
              `).join("") || "<li>No candidates returned.</li>"
            }}
          </ol>
        </section>
        <section class="result-card">
          <h3>Baseline Plan</h3>
          <ol class="plan-list">
            ${{
              journeyPlan.map((row) => `
                <li>
                  <strong>Step ${{escapeHtml(row.step)}} -> ${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">
                    transition=${{escapeHtml(row.transition_probability)}} |
                    mode_score=${{escapeHtml(row.mode_score)}}
                  </div>
                </li>
              `).join("") || "<li>No plan returned.</li>"
            }}
          </ol>
        </section>
        <section class="result-card">
          <h3>Adaptive Session</h3>
          <div class="metric-line">${{escapeHtml(adaptive.description || "")}}</div>
          <ol class="transcript-list">
            ${{
              transcript.map((row) => `
                <li>
                  <strong>Step ${{escapeHtml(row.step)}} -> ${{escapeHtml(row.selected_artist)}}</strong>
                  <div class="metric-line">
                    origin=${{escapeHtml(row.plan_origin)}} |
                    policy=${{escapeHtml(row.policy_name)}} |
                    end_risk=${{escapeHtml(row.end_risk)}}
                  </div>
                  ${{row.why_changed ? `<div class="metric-line">${{escapeHtml(row.why_changed)}}</div>` : ""}}
                  ${{row.event_applied_after_step ? `<div class="metric-line">event=${{escapeHtml(row.event_applied_after_step)}} | ${{escapeHtml(row.event_summary || "")}}</div>` : ""}}
                </li>
              `).join("") || "<li>No transcript available.</li>"
            }}
          </ol>
        </section>
        <section class="result-card wide">
          <h3>Service Output</h3>
          <div class="metric-line">Run dir: ${{escapeHtml(service.run_dir || config.runDir)}}</div>
          <div class="metric-line">Output dir: ${{escapeHtml(service.output_dir || config.outputDir)}}</div>
          <div class="metric-line">Session id: ${{escapeHtml(service.session_id || "n/a")}}</div>
          <div class="metric-line">Created at: ${{escapeHtml(service.created_at || "n/a")}}</div>
          ${{artifactBlock}}
        </section>
        <section class="result-card wide">
          <h3>Memory Context</h3>
          <div class="metric-line">Effective recent artists: ${{escapeHtml(effectiveRecentArtists.join(" | ") || "none")}}</div>
          <div class="pill-list">
            ${{topAffinities.slice(0, 4).map((row) => `<span class="pill">${{escapeHtml(row.artist_name)}} · ${{escapeHtml(row.net_score)}}</span>`).join("")}}
            ${{avoidArtists.slice(0, 3).map((row) => `<span class="pill warn">${{escapeHtml(row.artist_name)}} · ${{escapeHtml(row.net_score)}}</span>`).join("")}}
          </div>
        </section>
      `;

      resultGrid.querySelectorAll("[data-feedback-artist]").forEach((node) => {{
        node.addEventListener("click", async () => {{
          const artistName = node.dataset.feedbackArtist || "";
          const signal = node.dataset.feedbackSignal || "";
          await submitFeedback(service.session_id || "", artistName, signal);
        }});
      }});
    }}

    async function loadCatalog() {{
      const response = await fetch("/taste-os/catalog");
      if (!response.ok) {{
        throw new Error(`Catalog request failed with status ${{response.status}}`);
      }}
      state.catalog = await response.json();
      renderCatalog();
      setStatus("Catalog ready. Generate a session to inspect the planner.");
    }}

    async function loadHistory() {{
      const response = await fetch("/taste-os/history");
      if (!response.ok) {{
        throw new Error(`History request failed with status ${{response.status}}`);
      }}
      state.history = await response.json();
      renderHistory(state.history);
    }}

    async function submitFeedback(sessionId, artistName, signal) {{
      if (!sessionId) {{
        setStatus("Generate a session before recording feedback.", true);
        return;
      }}
      setStatus(`Recording ${{signal}} for ${{artistName}}...`);
      const response = await fetch("/taste-os/feedback", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
        }},
        body: JSON.stringify({{
          session_id: sessionId,
          artist_name: artistName,
          signal,
        }}),
      }});
      const data = await response.json();
      if (!response.ok) {{
        const message = data?.error?.message || `Feedback request failed with status ${{response.status}}`;
        setStatus(message, true);
        return;
      }}
      await loadHistory();
      if (state.payload) {{
        state.payload.memory_summary = data.feedback_memory || state.payload.memory_summary;
        renderSummary(state.payload);
        renderResults(state.payload);
      }}
      setStatus(`Recorded ${{signal}} for ${{artistName}}.`);
    }}

    async function runSession() {{
      runButton.disabled = true;
      setStatus("Generating session...");
      try {{
        const payload = {{
          mode: state.mode,
          scenario: state.scenario,
          top_k: Math.min(config.maxTopK, Math.max(1, Number(topKInput.value || config.defaultTopK))),
          include_video: includeVideoInput.checked,
          persist_artifacts: persistArtifactsInput.checked,
          use_feedback_memory: useFeedbackMemoryInput.checked,
        }};
        const recentArtists = recentArtistsInput.value.trim();
        if (recentArtists) {{
          payload.recent_artists = recentArtists;
        }}
        const response = await fetch("/taste-os/session", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json",
          }},
          body: JSON.stringify(payload),
        }});
        const data = await response.json();
        if (!response.ok) {{
          const message = data?.error?.message || `Session request failed with status ${{response.status}}`;
          throw new Error(message);
        }}
        state.payload = data;
        renderSummary(data);
        renderResults(data);
        await loadHistory();
        setStatus(`Session ready for ${{data.request.mode}} / ${{data.request.scenario}}.`);
      }} catch (error) {{
        setStatus(error.message || "Taste OS session failed.", true);
      }} finally {{
        runButton.disabled = false;
      }}
    }}

    runButton.addEventListener("click", runSession);
    refreshMemoryButton.addEventListener("click", () => {{
      loadHistory().then(() => {{
        setStatus("Memory refreshed.");
      }}).catch((error) => {{
        setStatus(error.message || "Memory refresh failed.", true);
      }});
    }});
    Promise.all([loadCatalog(), loadHistory()]).catch((error) => {{
      setStatus(error.message || "Taste OS studio failed to load.", true);
    }});
  </script>
</body>
</html>"""


class TasteOSService:
    def __init__(
        self,
        *,
        run_dir: Path,
        data_dir: Path,
        output_dir: Path,
        model_name: str,
        include_video: bool,
        max_top_k: int,
        auth_token: str | None,
        logger: logging.Logger,
        digital_twin: ListenerDigitalTwinArtifact | None = None,
        multimodal_space: MultimodalArtistSpace | None = None,
        safe_policy: SafeBanditPolicyArtifact | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_video = include_video
        self.max_top_k = max(1, int(max_top_k))
        self.auth_token = auth_token
        self.logger = logger
        self._context_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._context_cache: dict[bool, _TasteOSContextCacheEntry] = {}
        self.feedback_memory_path = self.output_dir / "feedback_memory.json"
        self.feedback_event_log_path = self.output_dir / "feedback_events.jsonl"
        self.session_history_path = self.output_dir / "session_history.jsonl"

        metadata = json.loads((run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
        self.artist_labels = list(metadata.get("artist_labels", []))
        model_row = resolve_model_row(
            run_dir,
            explicit_model_name=model_name,
            alias_model_name=model_name,
        )
        self.predictor = load_predictor(
            run_dir=run_dir,
            row=model_row,
            artist_labels=self.artist_labels,
        )
        self.model_name = self.predictor.model_name
        self.model_type = self.predictor.model_type
        self.artifact_paths = {
            "multimodal_space": str((run_dir / "analysis" / "multimodal" / "multimodal_artist_space.joblib").resolve()),
            "digital_twin": str((run_dir / "analysis" / "digital_twin" / "listener_digital_twin.joblib").resolve()),
            "safe_policy": str((run_dir / "analysis" / "safe_policy" / "safe_bandit_policy.joblib").resolve()),
        }
        self.multimodal_space = (
            multimodal_space
            if multimodal_space is not None
            else _load_artifact(Path(self.artifact_paths["multimodal_space"]), label="multimodal artist space")
        )
        self.digital_twin = (
            digital_twin
            if digital_twin is not None
            else _load_artifact(Path(self.artifact_paths["digital_twin"]), label="listener digital twin")
        )
        self.safe_policy = (
            safe_policy
            if safe_policy is not None
            else _load_artifact(Path(self.artifact_paths["safe_policy"]), label="safe policy")
        )

        try:
            self._get_prediction_context(include_video=self.include_video)
        except Exception as exc:
            logger.info("Taste OS context warm-up skipped: %s", exc)
        logger.info(
            "Loaded Taste OS service: model=%s type=%s modes=%d scenarios=%d",
            self.model_name,
            self.model_type,
            len(MODE_CONFIGS),
            len(SCENARIOS),
        )

    def _empty_feedback_store(self) -> dict[str, object]:
        return {
            "artists": {},
            "event_count": 0,
            "updated_at": "",
        }

    def _load_feedback_store(self) -> dict[str, object]:
        default_store = self._empty_feedback_store()
        store = _load_json_document(self.feedback_memory_path, default=default_store)
        artists = store.get("artists")
        if not isinstance(artists, dict):
            store["artists"] = {}
        if not isinstance(store.get("event_count"), int):
            store["event_count"] = 0
        if not isinstance(store.get("updated_at"), str):
            store["updated_at"] = ""
        return store

    def _write_feedback_store(self, store: dict[str, object]) -> None:
        self.feedback_memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_memory_path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")

    def _feedback_artist_summary(self, artist_name: str, stats: dict[str, object]) -> dict[str, object]:
        like_count = int(stats.get("like_count", 0))
        repeat_count = int(stats.get("repeat_count", 0))
        skip_count = int(stats.get("skip_count", 0))
        dislike_count = int(stats.get("dislike_count", 0))
        positive_score = float(like_count) + float(repeat_count) * 1.25
        negative_score = float(dislike_count) * 1.3 + float(skip_count) * 0.7
        net_score = round(positive_score - negative_score, 3)
        if net_score >= 1.0:
            memory_state = "favored"
        elif net_score <= -0.75:
            memory_state = "avoid"
        elif like_count or repeat_count or skip_count or dislike_count:
            memory_state = "mixed"
        else:
            memory_state = "neutral"
        return {
            "artist_name": artist_name,
            "like_count": like_count,
            "repeat_count": repeat_count,
            "skip_count": skip_count,
            "dislike_count": dislike_count,
            "net_score": net_score,
            "memory_state": memory_state,
            "last_signal": str(stats.get("last_signal", "")),
            "last_session_id": str(stats.get("last_session_id", "")),
            "updated_at": str(stats.get("updated_at", "")),
        }

    def _feedback_seed_artists(self, *, store: dict[str, object], limit: int) -> list[str]:
        artists = store.get("artists", {})
        if not isinstance(artists, dict):
            return []
        candidates: list[tuple[float, str, str]] = []
        for key, raw_stats in artists.items():
            if not isinstance(raw_stats, dict):
                continue
            artist_name = str(raw_stats.get("artist_name") or key).strip()
            if not artist_name:
                continue
            summary = self._feedback_artist_summary(artist_name, raw_stats)
            if str(summary["memory_state"]) == "avoid":
                continue
            if float(summary["net_score"]) <= 0.0:
                continue
            candidates.append((float(summary["net_score"]), str(summary["updated_at"]), artist_name))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [artist_name for _, _, artist_name in candidates[: max(0, int(limit))]]

    def _feedback_summary(
        self,
        *,
        store: dict[str, object] | None = None,
        limit: int = 5,
        effective_recent_artists: list[str] | None = None,
    ) -> dict[str, object]:
        active_store = store if store is not None else self._load_feedback_store()
        artists = active_store.get("artists", {})
        summaries: list[dict[str, object]] = []
        if isinstance(artists, dict):
            for key, raw_stats in artists.items():
                if not isinstance(raw_stats, dict):
                    continue
                artist_name = str(raw_stats.get("artist_name") or key).strip()
                if not artist_name:
                    continue
                summaries.append(self._feedback_artist_summary(artist_name, raw_stats))
        summaries.sort(key=lambda item: (float(item["net_score"]), str(item["updated_at"])), reverse=True)
        top_affinities = [item for item in summaries if float(item["net_score"]) > 0][:limit]
        avoid_artists = [item for item in sorted(summaries, key=lambda item: (float(item["net_score"]), str(item["updated_at"]))) if float(item["net_score"]) < 0][:limit]
        seed_artists = self._feedback_seed_artists(store=active_store, limit=max(limit, len(effective_recent_artists or [])))
        return {
            "event_count": int(active_store.get("event_count", 0)),
            "artist_count": len(summaries),
            "updated_at": str(active_store.get("updated_at", "")),
            "seed_artists": seed_artists,
            "effective_recent_artists": list(effective_recent_artists or []),
            "top_affinities": top_affinities,
            "avoid_artists": avoid_artists,
            "recent_feedback": _read_jsonl_tail(self.feedback_event_log_path, limit=limit),
        }

    def _annotate_feedback_rows(self, rows: object, *, store: dict[str, object]) -> None:
        if not isinstance(rows, list):
            return
        artists = store.get("artists", {})
        if not isinstance(artists, dict):
            artists = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            artist_name = str(row.get("artist_name", "")).strip()
            if not artist_name:
                continue
            stats = artists.get(_artist_key(artist_name), {})
            if not isinstance(stats, dict):
                stats = {}
            summary = self._feedback_artist_summary(artist_name, stats)
            row["memory_state"] = summary["memory_state"]
            row["memory_net_score"] = summary["net_score"]
            row["memory_like_count"] = summary["like_count"]
            row["memory_dislike_count"] = summary["dislike_count"]

    def _session_history_row(
        self,
        *,
        result: dict[str, object],
        session_id: str,
        created_at: str,
        effective_recent_artists: list[str],
        used_feedback_memory: bool,
    ) -> dict[str, object]:
        request = result.get("request", {}) if isinstance(result, dict) else {}
        summary = result.get("demo_summary", {}) if isinstance(result, dict) else {}
        service = result.get("service", {}) if isinstance(result, dict) else {}
        return {
            "session_id": session_id,
            "created_at": created_at,
            "mode": str(request.get("mode", "")),
            "scenario": str(request.get("scenario", "")),
            "top_artist": str(summary.get("top_artist", "")),
            "adaptive_replans": int(summary.get("adaptive_replans", 0)),
            "safe_route_steps": int(summary.get("adaptive_safe_route_steps", 0)),
            "effective_recent_artists": list(effective_recent_artists),
            "used_feedback_memory": bool(used_feedback_memory),
            "persisted": bool(service.get("persisted", False)),
        }

    def _record_session_history(
        self,
        *,
        result: dict[str, object],
        session_id: str,
        created_at: str,
        effective_recent_artists: list[str],
        used_feedback_memory: bool,
    ) -> None:
        _append_jsonl(
            self.session_history_path,
            self._session_history_row(
                result=result,
                session_id=session_id,
                created_at=created_at,
                effective_recent_artists=effective_recent_artists,
                used_feedback_memory=used_feedback_memory,
            ),
        )

    def history_snapshot(self, *, limit: int = DEFAULT_HISTORY_LIMIT) -> dict[str, object]:
        return {
            "recent_sessions": _read_jsonl_tail(self.session_history_path, limit=limit),
            "feedback_memory": self._feedback_summary(limit=limit),
        }

    def record_feedback(
        self,
        *,
        session_id: str,
        artist_name: str,
        signal: str,
        notes: str | None = None,
    ) -> dict[str, object]:
        timestamp = _utc_now_iso()
        normalized_artist_name = str(artist_name).strip()
        normalized_signal = str(signal).strip().lower()
        with self._memory_lock:
            store = self._load_feedback_store()
            artists = store.setdefault("artists", {})
            if not isinstance(artists, dict):
                artists = {}
                store["artists"] = artists
            key = _artist_key(normalized_artist_name)
            row = artists.get(key)
            if not isinstance(row, dict):
                row = {
                    "artist_name": normalized_artist_name,
                    "like_count": 0,
                    "repeat_count": 0,
                    "skip_count": 0,
                    "dislike_count": 0,
                    "last_signal": "",
                    "last_session_id": "",
                    "updated_at": "",
                }
            row["artist_name"] = normalized_artist_name
            counter_name = f"{normalized_signal}_count"
            row[counter_name] = int(row.get(counter_name, 0)) + 1
            row["last_signal"] = normalized_signal
            row["last_session_id"] = session_id
            row["updated_at"] = timestamp
            artists[key] = row
            store["event_count"] = int(store.get("event_count", 0)) + 1
            store["updated_at"] = timestamp
            self._write_feedback_store(store)
            event = {
                "timestamp": timestamp,
                "session_id": session_id,
                "artist_name": normalized_artist_name,
                "signal": normalized_signal,
                "notes": notes or "",
            }
            _append_jsonl(self.feedback_event_log_path, event)
            summary = self._feedback_summary(store=store)
        return {
            "recorded": True,
            "feedback": event,
            "feedback_memory": summary,
        }

    def artifact_url(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact path is outside the configured output dir: {path}") from exc
        encoded = "/".join(quote(part) for part in relative.parts)
        return f"/taste-os/artifacts/{encoded}"

    def resolve_artifact_path(self, relative_path: str) -> Path:
        candidate = (self.output_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise FileNotFoundError("Artifact path escapes the Taste OS output directory.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Artifact not found: {relative_path}")
        return candidate

    def _prediction_source_signature(self, *, include_video: bool) -> tuple[tuple[str, int, int], ...]:
        return prediction_source_signature(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            include_video=include_video,
        )

    def _get_prediction_context(self, *, include_video: bool) -> PredictionInputContext:
        signature = self._prediction_source_signature(include_video=include_video)
        cached = self._context_cache.get(include_video)
        if cached is not None and cached.signature == signature:
            return cached.context

        with self._context_lock:
            cached = self._context_cache.get(include_video)
            if cached is not None and cached.signature == signature:
                return cached.context

            context = load_prediction_input_context(
                run_dir=self.run_dir,
                data_dir=self.data_dir,
                include_video=include_video,
                logger=self.logger,
            )
            self._context_cache[include_video] = _TasteOSContextCacheEntry(
                signature=signature,
                context=context,
            )
            return context

    def plan_session(
        self,
        *,
        mode: str,
        scenario: str,
        top_k: int,
        recent_artists: list[str] | None,
        include_video: bool,
        persist_artifacts: bool,
        use_feedback_memory: bool,
    ) -> dict[str, object]:
        context = self._get_prediction_context(include_video=include_video)
        feedback_store = self._load_feedback_store()
        requested_recent_artists = list(recent_artists or [])
        effective_recent_artists = list(requested_recent_artists)
        if not effective_recent_artists and use_feedback_memory:
            effective_recent_artists = self._feedback_seed_artists(
                store=feedback_store,
                limit=max(1, int(context.sequence_length)),
            )
        seq_batch, ctx_batch, sequence_names = _prepare_inputs(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            recent_artists=effective_recent_artists or None,
            include_video=include_video,
            logger=self.logger,
            context=context,
        )

        with self._plan_lock:
            payload = build_taste_os_demo_payload(
                predictor=self.predictor,
                artist_labels=list(context.artist_labels),
                sequence_labels=np.asarray(seq_batch[0], dtype="int32"),
                sequence_names=sequence_names,
                context_batch=np.asarray(ctx_batch, dtype="float32"),
                context_raw_batch=context.context_raw,
                context_features=list(context.context_features or []),
                friction_reference=context.friction_reference,
                scaler_mean=context.scaler_mean,
                scaler_scale=context.scaler_scale,
                digital_twin=self.digital_twin,
                multimodal_space=self.multimodal_space,
                safe_policy=self.safe_policy,
                mode_name=mode,
                scenario_name=scenario,
                top_k=top_k,
                artifact_paths=self.artifact_paths,
            )

        result = dict(payload)
        request = result.get("request")
        if isinstance(request, dict):
            request["requested_recent_artists"] = requested_recent_artists
            request["effective_recent_artists"] = effective_recent_artists
            request["used_feedback_memory"] = bool(use_feedback_memory and not requested_recent_artists and effective_recent_artists)

        self._annotate_feedback_rows(result.get("top_candidates"), store=feedback_store)
        self._annotate_feedback_rows(result.get("journey_plan"), store=feedback_store)
        result["memory_summary"] = self._feedback_summary(
            store=feedback_store,
            effective_recent_artists=effective_recent_artists,
        )

        session_id = f"taste-os-{secrets.token_hex(6)}"
        created_at = _utc_now_iso()
        result["service"] = {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "persisted": False,
            "session_id": session_id,
            "created_at": created_at,
        }
        if persist_artifacts:
            json_path, md_path = write_taste_os_demo_artifacts(result, output_dir=self.output_dir)
            result["service"] = {
                "run_dir": str(self.run_dir),
                "output_dir": str(self.output_dir),
                "persisted": True,
                "artifact_json": str(json_path),
                "artifact_md": str(md_path),
                "artifact_json_url": self.artifact_url(json_path),
                "artifact_md_url": self.artifact_url(md_path),
                "session_id": session_id,
                "created_at": created_at,
            }
        self._record_session_history(
            result=result,
            session_id=session_id,
            created_at=created_at,
            effective_recent_artists=effective_recent_artists,
            used_feedback_memory=bool(use_feedback_memory and not requested_recent_artists and effective_recent_artists),
        )
        return result


def _build_handler(service: TasteOSService):
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
                _error_payload(code=code, message=message, details=details),
                extra_headers=extra_headers,
            )

        def log_message(self, fmt: str, *args) -> None:
            service.logger.info("HTTP %s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path in {"", "/", "/taste-os"}:
                self._send_html(200, _taste_os_page_html(service))
                return
            if path.startswith("/taste-os/artifacts/"):
                relative_path = unquote(path.removeprefix("/taste-os/artifacts/"))
                try:
                    artifact_path = service.resolve_artifact_path(relative_path)
                    self._send_bytes(
                        200,
                        artifact_path.read_bytes(),
                        content_type=_guess_content_type(artifact_path),
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
                payload = _read_json_payload(self)
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


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.taste_os_service")

    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    model_name = str(model_row.get("model_name", "")).strip()

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        max_top_k=max(1, int(args.max_top_k)),
        auth_token=(str(args.auth_token).strip() if args.auth_token else None),
        logger=logger,
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), _build_handler(service))
    logger.info("Taste OS service listening on http://%s:%d", args.host, int(args.port))
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
