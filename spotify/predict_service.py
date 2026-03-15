from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_next import PredictionInputContext, _prepare_inputs, load_prediction_input_context
from .serving import load_predictor, resolve_model_row

MAX_REQUEST_BYTES = 1_000_000
DEFAULT_MAX_TOP_K = 20


@dataclass(frozen=True)
class _PredictionContextCacheEntry:
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


class PredictRequestPayload(TypedDict):
    top_k: int
    include_video: bool
    recent_artists: list[str] | None


def _parse_args() -> argparse.Namespace:
    env_max_top_k_raw = os.getenv("SPOTIFY_PREDICT_MAX_TOP_K", str(DEFAULT_MAX_TOP_K)).strip()
    try:
        env_max_top_k = max(1, int(env_max_top_k_raw))
    except ValueError:
        env_max_top_k = DEFAULT_MAX_TOP_K

    parser = argparse.ArgumentParser(
        prog="python -m spotify.predict_service",
        description="Serve Spotify next-artist predictions over HTTP.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--max-top-k", type=int, default=env_max_top_k, help="Maximum top_k value accepted by the API.")
    parser.add_argument(
        "--auth-token",
        type=str,
        default=os.getenv("SPOTIFY_PREDICT_AUTH_TOKEN"),
        help="Optional API token. When set, POST /predict requires Authorization: Bearer <token>.",
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
        from_string = [part.strip() for part in value.split("|") if part.strip()]
        return from_string or None

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
    if isinstance(value, bool):
        raise RequestValidationError(
            status_code=422,
            code="invalid_top_k",
            message="top_k must be an integer.",
        )
    if not isinstance(value, int):
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


def normalize_predict_payload(
    payload: dict[str, object],
    *,
    default_include_video: bool,
    max_top_k: int,
) -> PredictRequestPayload:
    allowed_keys = {"top_k", "include_video", "recent_artists"}
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise RequestValidationError(
            status_code=400,
            code="unknown_fields",
            message="Payload contains unknown fields.",
            details={"fields": unknown_keys},
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

    return {
        "top_k": top_k,
        "include_video": include_video,
        "recent_artists": recent_artists,
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


class PredictionService:
    def __init__(
        self,
        run_dir: Path,
        data_dir: Path,
        model_name: str,
        include_video: bool,
        max_top_k: int,
        auth_token: str | None,
        logger: logging.Logger,
    ):
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.include_video = include_video
        self.max_top_k = max(1, int(max_top_k))
        self.auth_token = auth_token
        self.logger = logger
        self._context_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._context_cache: dict[bool, _PredictionContextCacheEntry] = {}

        metadata_path = run_dir / "feature_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as infile:
            metadata = json.load(infile)
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
        try:
            self._get_prediction_context(include_video=self.include_video)
        except Exception as exc:
            logger.info("Prediction context warm-up skipped: %s", exc)
        logger.info("Loaded serveable predictor: model=%s type=%s", self.model_name, self.model_type)

    def _prediction_source_signature(self, *, include_video: bool) -> tuple[tuple[str, int, int], ...]:
        root = self.data_dir.expanduser().resolve()
        files = sorted(path for path in root.rglob("Streaming_History_Audio_*.json") if path.is_file())
        if include_video:
            files.extend(sorted(path for path in root.rglob("Streaming_History_Video_*.json") if path.is_file()))

        signature: list[tuple[str, int, int]] = []
        for path in files:
            stat = path.stat()
            signature.append(
                (
                    str(path.resolve()),
                    int(stat.st_size),
                    int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                )
            )
        return tuple(signature)

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
            self._context_cache[include_video] = _PredictionContextCacheEntry(
                signature=signature,
                context=context,
            )
            return context

    def predict(self, *, top_k: int, recent_artists: list[str] | None, include_video: bool) -> dict[str, object]:
        context = self._get_prediction_context(include_video=include_video)
        seq_batch, ctx_batch, sequence_names = _prepare_inputs(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            recent_artists=recent_artists,
            include_video=include_video,
            logger=self.logger,
            context=context,
        )

        with self._predict_lock:
            artist_probs = self.predictor.predict_proba(seq_batch, ctx_batch)[0]

        top_k = max(1, int(top_k))
        top_indices = np.argsort(artist_probs)[::-1][:top_k]
        predictions: list[dict[str, object]] = []
        for rank, idx in enumerate(top_indices, start=1):
            label_idx = int(idx)
            artist_name = (
                self.artist_labels[label_idx]
                if 0 <= label_idx < len(self.artist_labels)
                else str(label_idx)
            )
            predictions.append(
                {
                    "rank": rank,
                    "artist_label": label_idx,
                    "artist_name": artist_name,
                    "probability": float(artist_probs[label_idx]),
                }
            )

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "sequence_tail": sequence_names,
            "predictions": predictions,
        }


def _build_handler(service: PredictionService):
    class Handler(BaseHTTPRequestHandler):
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
            if self.path.rstrip("/") == "/health":
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
            self._send_error(404, code="not_found", message="Resource not found.")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/predict":
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
                normalized = normalize_predict_payload(
                    payload,
                    default_include_video=service.include_video,
                    max_top_k=service.max_top_k,
                )
            except RequestValidationError as exc:
                self._send_error(
                    exc.status_code,
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                )
                return

            try:
                result = service.predict(
                    top_k=normalized["top_k"],
                    recent_artists=normalized["recent_artists"],
                    include_video=normalized["include_video"],
                )
                self._send_json(200, result)
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                self._send_error(
                    422,
                    code="prediction_input_error",
                    message=str(exc),
                )
            except Exception:
                service.logger.exception("Unhandled prediction failure")
                self._send_error(
                    500,
                    code="internal_error",
                    message="Prediction failed due to an internal server error.",
                )

    return Handler


def main() -> int:
    load_local_env()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.predict_service")

    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    model_name = str(model_row.get("model_name", "")).strip()

    service = PredictionService(
        run_dir=run_dir,
        data_dir=data_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        max_top_k=max(1, int(args.max_top_k)),
        auth_token=(str(args.auth_token).strip() if args.auth_token else None),
        logger=logger,
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), _build_handler(service))
    logger.info("Prediction service listening on http://%s:%d", args.host, int(args.port))
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
