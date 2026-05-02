from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
import uvicorn

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_service import (
    PredictionService,
    RequestValidationError as PredictRequestValidationError,
    normalize_predict_payload,
)
from .serving import resolve_model_row
from .taste_os_demo import MODE_CONFIGS, SCENARIOS
from .taste_os_http import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MAX_TOP_K as DEFAULT_TASTE_OS_MAX_TOP_K,
    RequestValidationError as TasteOSRequestValidationError,
    normalize_taste_os_feedback_payload,
    normalize_taste_os_payload,
)
from .taste_os_page import render_taste_os_page_html
from .taste_os_service import TasteOSService

DEFAULT_REQUEST_RATE_LIMIT = 240


class _ApiRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(0, int(limit_per_minute))
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        if self.limit_per_minute <= 0:
            return True, 0
        current = float(now if now is not None else time.time())
        window_start = current - 60.0
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] <= window_start:
                bucket.popleft()
            if len(bucket) >= self.limit_per_minute:
                retry_after = max(1, int(60 - max(0.0, current - bucket[0])))
                return False, retry_after
            bucket.append(current)
            return True, 0


class _ApiTelemetry:
    def __init__(self, *, service_name: str) -> None:
        self.service_name = service_name
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._route_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "request_count": 0.0,
                "error_count": 0.0,
                "latency_ms_total": 0.0,
                "latency_ms_max": 0.0,
            }
        )

    def record(self, *, method: str, path: str, status_code: int, latency_ms: float) -> None:
        route_key = f"{method.upper()} {path}"
        with self._lock:
            stats = self._route_stats[route_key]
            stats["request_count"] += 1.0
            stats["latency_ms_total"] += float(latency_ms)
            stats["latency_ms_max"] = max(float(stats["latency_ms_max"]), float(latency_ms))
            if int(status_code) >= 400:
                stats["error_count"] += 1.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            routes = []
            total_requests = 0
            total_errors = 0
            for route_key, stats in sorted(self._route_stats.items()):
                request_count = int(stats["request_count"])
                error_count = int(stats["error_count"])
                total_requests += request_count
                total_errors += error_count
                avg_latency_ms = (float(stats["latency_ms_total"]) / request_count) if request_count else 0.0
                routes.append(
                    {
                        "route": route_key,
                        "request_count": request_count,
                        "error_count": error_count,
                        "avg_latency_ms": round(avg_latency_ms, 3),
                        "max_latency_ms": round(float(stats["latency_ms_max"]), 3),
                    }
                )
        return {
            "service_name": self.service_name,
            "started_at_epoch_s": int(self.started_at),
            "uptime_s": round(max(0.0, time.time() - self.started_at), 3),
            "total_request_count": total_requests,
            "total_error_count": total_errors,
            "routes": routes,
        }


class PredictApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 5
    include_video: bool | None = None
    recent_artists: list[str] | str | None = None
    allow_abstain: bool | None = False
    return_prediction_set: bool | None = False


class TasteOSSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    scenario: str = "steady"
    top_k: int = 5
    include_video: bool | None = None
    recent_artists: list[str] | str | None = None
    persist_artifacts: bool | None = False
    use_feedback_memory: bool | None = False


class TasteOSFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    artist_name: str
    signal: str
    notes: str | None = None


def _request_identity(request: Request) -> str:
    token = request.headers.get("X-API-Key") or request.headers.get("Authorization", "")
    client_host = request.client.host if request.client is not None else "unknown"
    if token:
        return f"token:{token}"
    return f"client:{client_host}"


def _json_error(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: dict[str, object] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }
    headers = {"X-Request-ID": request_id}
    headers.update(extra_headers or {})
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def _install_observability(
    *,
    app: FastAPI,
    logger: logging.Logger,
    telemetry: _ApiTelemetry,
    rate_limiter: _ApiRateLimiter,
) -> None:
    @app.middleware("http")
    async def _request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or secrets.token_hex(12)
        request.state.request_id = request_id
        start = time.perf_counter()
        path = request.url.path

        if path not in {"/health", "/v1/health", "/metrics", "/v1/metrics", "/openapi.json", "/docs"}:
            allowed, retry_after = rate_limiter.allow(_request_identity(request))
            if not allowed:
                response = _json_error(
                    status_code=429,
                    code="rate_limited",
                    message="Rate limit exceeded.",
                    details={"retry_after_seconds": retry_after},
                    request_id=request_id,
                    extra_headers={"Retry-After": str(retry_after)},
                )
                telemetry.record(
                    method=request.method,
                    path=path,
                    status_code=response.status_code,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
                return response

        try:
            response = await call_next(request)
        except HTTPException as exc:
            response = _json_error(
                status_code=int(exc.status_code),
                code="http_error",
                message=str(exc.detail),
                request_id=request_id,
            )
        except Exception:
            logger.exception("Unhandled API failure")
            response = _json_error(
                status_code=500,
                code="internal_error",
                message="Request failed due to an internal server error.",
                request_id=request_id,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        telemetry.record(
            method=request.method,
            path=path,
            status_code=int(response.status_code),
            latency_ms=latency_ms,
        )
        logger.info(
            json.dumps(
                {
                    "event": "api_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": int(response.status_code),
                    "latency_ms": round(latency_ms, 3),
                },
                sort_keys=True,
            )
        )
        return response


def create_prediction_app(
    *,
    service: PredictionService,
    logger: logging.Logger,
    request_rate_limit: int = DEFAULT_REQUEST_RATE_LIMIT,
) -> FastAPI:
    telemetry = _ApiTelemetry(service_name="spotify.predict")
    rate_limiter = _ApiRateLimiter(limit_per_minute=request_rate_limit)
    app = FastAPI(
        title="Spotify Prediction API",
        version="1.0.0",
        description="Production-style ASGI surface for next-artist prediction.",
    )
    _install_observability(app=app, logger=logger, telemetry=telemetry, rate_limiter=rate_limiter)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", "http_error"))
            message = str(detail.get("message", "Request failed."))
            details = detail.get("details")
            details = details if isinstance(details, dict) else {}
        else:
            code = "http_error"
            message = str(detail)
            details = {}
        return _json_error(
            status_code=int(exc.status_code),
            code=code,
            message=message,
            details=details,
            request_id=str(getattr(request.state, "request_id", "")),
        )

    @app.exception_handler(FastAPIRequestValidationError)
    async def _request_validation_handler(request: Request, exc: FastAPIRequestValidationError) -> JSONResponse:
        return _json_error(
            status_code=422,
            code="invalid_request_body",
            message="Request body failed schema validation.",
            details={"errors": exc.errors()},
            request_id=str(getattr(request.state, "request_id", "")),
        )

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, object]:
        return service.health_payload()

    @app.get("/metrics")
    @app.get("/v1/metrics")
    async def metrics() -> dict[str, object]:
        return {
            "health": service.health_payload(),
            "telemetry": telemetry.snapshot(),
            "rate_limit_per_minute": rate_limiter.limit_per_minute,
        }

    @app.post("/predict")
    @app.post("/v1/predict")
    async def predict(payload: PredictApiRequest, request: Request) -> dict[str, object]:
        if not service.auth_token:
            pass
        elif not request.headers:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})
        else:
            from .predict_service import is_authorized_request

            if not is_authorized_request(request.headers, service.auth_token):
                raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})

        try:
            normalized = normalize_predict_payload(
                payload.model_dump(exclude_none=True),
                default_include_video=service.include_video,
                max_top_k=service.max_top_k,
            )
        except PredictRequestValidationError as exc:
            raise HTTPException(
                status_code=int(exc.status_code),
                detail={"code": exc.code, "message": exc.message, "details": exc.details},
            ) from exc

        try:
            return service.predict(
                top_k=normalized["top_k"],
                recent_artists=normalized["recent_artists"],
                include_video=normalized["include_video"],
                allow_abstain=normalized["allow_abstain"],
                return_prediction_set=normalized["return_prediction_set"],
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail={"code": "prediction_input_error", "message": str(exc)}) from exc

    return app


def create_taste_os_app(
    *,
    service: TasteOSService,
    logger: logging.Logger,
    request_rate_limit: int = DEFAULT_REQUEST_RATE_LIMIT,
) -> FastAPI:
    telemetry = _ApiTelemetry(service_name="spotify.taste_os")
    rate_limiter = _ApiRateLimiter(limit_per_minute=request_rate_limit)
    app = FastAPI(
        title="Spotify Taste OS API",
        version="1.0.0",
        description="Production-style ASGI surface for Taste OS session planning.",
    )
    _install_observability(app=app, logger=logger, telemetry=telemetry, rate_limiter=rate_limiter)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", "http_error"))
            message = str(detail.get("message", "Request failed."))
            details = detail.get("details")
            details = details if isinstance(details, dict) else {}
        else:
            code = "http_error"
            message = str(detail)
            details = {}
        return _json_error(
            status_code=int(exc.status_code),
            code=code,
            message=message,
            details=details,
            request_id=str(getattr(request.state, "request_id", "")),
        )

    @app.exception_handler(FastAPIRequestValidationError)
    async def _request_validation_handler(request: Request, exc: FastAPIRequestValidationError) -> JSONResponse:
        return _json_error(
            status_code=422,
            code="invalid_request_body",
            message="Request body failed schema validation.",
            details={"errors": exc.errors()},
            request_id=str(getattr(request.state, "request_id", "")),
        )

    @app.get("/")
    @app.get("/taste-os")
    async def page() -> HTMLResponse:
        return HTMLResponse(render_taste_os_page_html(service))

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, object]:
        return service.health_payload()

    @app.get("/metrics")
    @app.get("/v1/metrics")
    async def metrics() -> dict[str, object]:
        return {
            "health": service.health_payload(),
            "telemetry": telemetry.snapshot(),
            "rate_limit_per_minute": rate_limiter.limit_per_minute,
        }

    @app.get("/taste-os/catalog")
    @app.get("/v1/taste-os/catalog")
    async def catalog() -> dict[str, object]:
        return {
            "modes": [
                {"name": mode.name, "description": mode.description, "planned_horizon": int(mode.horizon)}
                for mode in MODE_CONFIGS.values()
            ],
            "scenarios": [
                {"name": scenario.name, "description": scenario.description, "event_count": len(scenario.events)}
                for scenario in SCENARIOS.values()
            ],
        }

    @app.get("/taste-os/history")
    @app.get("/v1/taste-os/history")
    async def history(limit: int = DEFAULT_HISTORY_LIMIT) -> dict[str, object]:
        return service.history_snapshot(limit=max(1, int(limit)))

    @app.get("/taste-os/artifacts/{relative_path:path}")
    @app.get("/v1/taste-os/artifacts/{relative_path:path}")
    async def artifact(relative_path: str) -> FileResponse:
        try:
            artifact_path = service.resolve_artifact_path(relative_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": str(exc)}) from exc
        return FileResponse(path=artifact_path)

    @app.post("/taste-os/session")
    @app.post("/v1/taste-os/session")
    async def session(payload: TasteOSSessionRequest, request: Request) -> dict[str, object]:
        from .taste_os_http import is_authorized_request

        if service.auth_token and not is_authorized_request(request.headers, service.auth_token):
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})

        try:
            normalized = normalize_taste_os_payload(
                payload.model_dump(exclude_none=True),
                default_include_video=service.include_video,
                max_top_k=service.max_top_k,
            )
        except TasteOSRequestValidationError as exc:
            raise HTTPException(
                status_code=int(exc.status_code),
                detail={"code": exc.code, "message": exc.message, "details": exc.details},
            ) from exc

        try:
            return service.plan_session(
                mode=normalized["mode"],
                scenario=normalized["scenario"],
                top_k=normalized["top_k"],
                recent_artists=normalized["recent_artists"],
                include_video=normalized["include_video"],
                persist_artifacts=normalized["persist_artifacts"],
                use_feedback_memory=normalized["use_feedback_memory"],
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail={"code": "taste_os_input_error", "message": str(exc)}) from exc

    @app.post("/taste-os/feedback")
    @app.post("/v1/taste-os/feedback")
    async def feedback(payload: TasteOSFeedbackRequest, request: Request) -> dict[str, object]:
        from .taste_os_http import is_authorized_request

        if service.auth_token and not is_authorized_request(request.headers, service.auth_token):
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})

        try:
            normalized = normalize_taste_os_feedback_payload(payload.model_dump(exclude_none=True))
        except TasteOSRequestValidationError as exc:
            raise HTTPException(
                status_code=int(exc.status_code),
                detail={"code": exc.code, "message": exc.message, "details": exc.details},
            ) from exc

        return service.record_feedback(
            session_id=normalized["session_id"],
            artist_name=normalized["artist_name"],
            signal=normalized["signal"],
            notes=normalized["notes"],
        )

    return app


def _parse_args() -> argparse.Namespace:
    app_default = str(os.getenv("SPOTIFY_SERVICE_APP", "predict")).strip() or "predict"
    env_max_top_k_raw = os.getenv("SPOTIFY_SERVICE_MAX_TOP_K", str(DEFAULT_TASTE_OS_MAX_TOP_K)).strip()
    try:
        env_max_top_k = max(1, int(env_max_top_k_raw))
    except ValueError:
        env_max_top_k = DEFAULT_TASTE_OS_MAX_TOP_K
    env_rate_limit_raw = os.getenv("SPOTIFY_SERVICE_RATE_LIMIT_PER_MINUTE", str(DEFAULT_REQUEST_RATE_LIMIT)).strip()
    try:
        env_rate_limit = max(0, int(env_rate_limit_raw))
    except ValueError:
        env_rate_limit = DEFAULT_REQUEST_RATE_LIMIT

    parser = argparse.ArgumentParser(
        prog="python -m spotify.service_api",
        description="Run a production-style ASGI API for prediction or Taste OS serving.",
    )
    parser.add_argument("--app", choices=("predict", "taste-os"), default=app_default, help="Service surface to run.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory or champion alias path.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", ""),
        help="Optional raw Streaming_History JSON directory used only to build a missing serving bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("TASTE_OS_OUTPUT_DIR", "outputs/analysis/taste_os_service"),
        help="Taste OS artifact directory.",
    )
    parser.add_argument(
        "--state-db",
        type=str,
        default=os.getenv("TASTE_OS_STATE_DB", ""),
        help="Optional SQLite path for durable Taste OS feedback/session state.",
    )
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="HTTP port.")
    parser.add_argument("--max-top-k", type=int, default=env_max_top_k, help="Maximum top_k allowed by the API.")
    parser.add_argument(
        "--auth-token",
        type=str,
        default=(
            os.getenv("SPOTIFY_SERVICE_AUTH_TOKEN")
            or os.getenv("SPOTIFY_PREDICT_AUTH_TOKEN")
            or os.getenv("SPOTIFY_TASTE_OS_AUTH_TOKEN")
        ),
        help="Optional API token. When set, mutation routes require Authorization: Bearer <token>.",
    )
    parser.add_argument("--include-video", action="store_true", help="Use audio+video context by default.")
    parser.add_argument(
        "--request-rate-limit",
        type=int,
        default=env_rate_limit,
        help="Allowed requests per minute per token/client. Set 0 to disable.",
    )
    parser.add_argument(
        "--require-serving-bundle",
        action="store_true",
        help="Fail fast unless the run already has a materialized prediction serving bundle.",
    )
    return parser.parse_args()


def _build_service_from_args(args: argparse.Namespace, logger: logging.Logger) -> tuple[str, FastAPI]:
    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve() if str(args.data_dir).strip() else None
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    model_name = str(model_row.get("model_name", "")).strip()
    auth_token = str(args.auth_token).strip() if args.auth_token else None

    if args.app == "predict":
        service = PredictionService(
            run_dir=run_dir,
            data_dir=data_dir,
            model_name=model_name,
            include_video=bool(args.include_video),
            max_top_k=max(1, int(args.max_top_k)),
            auth_token=auth_token,
            logger=logger,
            require_serving_bundle=bool(args.require_serving_bundle),
        )
        app = create_prediction_app(
            service=service,
            logger=logger,
            request_rate_limit=max(0, int(args.request_rate_limit)),
        )
        return "predict", app

    output_dir = Path(args.output_dir).expanduser().resolve()
    state_db_path = Path(args.state_db).expanduser().resolve() if str(args.state_db).strip() else None
    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        max_top_k=max(1, int(args.max_top_k)),
        auth_token=auth_token,
        logger=logger,
        state_db_path=state_db_path,
        require_serving_bundle=bool(args.require_serving_bundle),
    )
    app = create_taste_os_app(
        service=service,
        logger=logger,
        request_rate_limit=max(0, int(args.request_rate_limit)),
    )
    return "taste-os", app


def _main_for_app(default_app: str | None = None) -> int:
    load_local_env()
    args = _parse_args()
    if default_app is not None:
        args.app = default_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.service_api")

    app_name, app = _build_service_from_args(args, logger)
    logger.info("ASGI service listening on http://%s:%d (%s)", args.host, int(args.port), app_name)
    uvicorn.run(app, host=str(args.host), port=int(args.port), proxy_headers=True, log_level="info")
    return 0


def main() -> int:
    return _main_for_app()


def main_predict() -> int:
    return _main_for_app("predict")


def main_taste_os() -> int:
    return _main_for_app("taste-os")


if __name__ == "__main__":
    raise SystemExit(main())
