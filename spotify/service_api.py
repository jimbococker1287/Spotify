from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict
import uvicorn

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .predict_next import _prediction_serving_bundle_path
from .predict_service import (
    PredictionService,
    RequestValidationError as PredictRequestValidationError,
    normalize_predict_payload,
)
from .service_auth import ApiAuthSettings, ApiAuthenticator
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
DEFAULT_RATE_LIMIT_BACKEND = "memory"
_OTEL_LOCK = threading.Lock()
_OTEL_PROVIDER_INITIALIZED = False


def _env_flag(name: str, *, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


class _RateLimiter(Protocol):
    limit_per_minute: int
    backend_name: str

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        ...


class _ApiRateLimiter:
    backend_name = "memory"

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


class _RedisApiRateLimiter:
    backend_name = "redis"

    def __init__(self, *, limit_per_minute: int, redis_url: str, logger: logging.Logger) -> None:
        self.limit_per_minute = max(0, int(limit_per_minute))
        self.redis_url = str(redis_url).strip()
        self.logger = logger
        self._fallback = _ApiRateLimiter(limit_per_minute)

        import redis

        self._client = redis.Redis.from_url(self.redis_url)

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        if self.limit_per_minute <= 0:
            return True, 0
        current = float(now if now is not None else time.time())
        current_bucket = int(current // 60.0)
        retry_after = max(1, int(60 - (current % 60.0)))
        bucket_key = f"spotify:service_api:rate_limit:{current_bucket}:{key}"
        try:
            pipeline = self._client.pipeline()
            pipeline.incr(bucket_key)
            pipeline.expire(bucket_key, 120)
            count_raw, _ = pipeline.execute()
            count = int(count_raw)
        except Exception:
            self.logger.exception("Redis rate-limit backend failed; falling back to in-process limiter.")
            return self._fallback.allow(key, now=current)
        if count > self.limit_per_minute:
            return False, retry_after
        return True, 0


class _ApiTelemetry:
    def __init__(self, *, service_name: str) -> None:
        self.service_name = service_name
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._inflight_requests = 0
        self._route_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "request_count": 0.0,
                "error_count": 0.0,
                "latency_ms_total": 0.0,
                "latency_ms_max": 0.0,
            }
        )

    def begin_request(self) -> None:
        with self._lock:
            self._inflight_requests += 1

    def record(self, *, method: str, path: str, status_code: int, latency_ms: float) -> None:
        route_key = f"{method.upper()} {path}"
        with self._lock:
            self._inflight_requests = max(0, self._inflight_requests - 1)
            stats = self._route_stats[route_key]
            stats["request_count"] += 1.0
            stats["latency_ms_total"] += float(latency_ms)
            stats["latency_ms_max"] = max(float(stats["latency_ms_max"]), float(latency_ms))
            if int(status_code) >= 400:
                stats["error_count"] += 1.0

    def snapshot(self) -> dict[str, object]:
        routes, total_requests, total_errors, inflight_requests = self._route_rows()
        return {
            "service_name": self.service_name,
            "started_at_epoch_s": int(self.started_at),
            "uptime_s": round(max(0.0, time.time() - self.started_at), 3),
            "inflight_request_count": inflight_requests,
            "total_request_count": total_requests,
            "total_error_count": total_errors,
            "routes": routes,
        }

    def _route_rows(self) -> tuple[list[dict[str, object]], int, int, int]:
        with self._lock:
            routes = []
            total_requests = 0
            total_errors = 0
            inflight_requests = int(self._inflight_requests)
            for route_key, stats in sorted(self._route_stats.items()):
                request_count = int(stats["request_count"])
                error_count = int(stats["error_count"])
                total_requests += request_count
                total_errors += error_count
                avg_latency_ms = (float(stats["latency_ms_total"]) / request_count) if request_count else 0.0
                method, _, path = route_key.partition(" ")
                routes.append(
                    {
                        "route": route_key,
                        "method": method,
                        "path": path or "/",
                        "request_count": request_count,
                        "error_count": error_count,
                        "avg_latency_ms": round(avg_latency_ms, 3),
                        "latency_ms_total": round(float(stats["latency_ms_total"]), 3),
                        "max_latency_ms": round(float(stats["latency_ms_max"]), 3),
                    }
                )
        return routes, total_requests, total_errors, inflight_requests


class _ApiTracing:
    def __init__(self, *, service_name: str, logger: logging.Logger, enabled: bool) -> None:
        self.enabled = False
        self.tracer = None
        self.propagator = None
        if not enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        except Exception:
            logger.warning("OpenTelemetry SDK is unavailable; request tracing is disabled.")
            return

        global _OTEL_PROVIDER_INITIALIZED
        with _OTEL_LOCK:
            if not _OTEL_PROVIDER_INITIALIZED:
                try:
                    provider = TracerProvider(
                        resource=Resource.create(
                            {
                                "service.name": "spotify.service_api",
                                "service.namespace": "spotify",
                            }
                        )
                    )
                    trace.set_tracer_provider(provider)
                except Exception:
                    logger.exception("Failed to initialize OpenTelemetry tracer provider.")
                    return
                _OTEL_PROVIDER_INITIALIZED = True
        self.tracer = trace.get_tracer(service_name)
        self.propagator = TraceContextTextMapPropagator()
        self.enabled = True

    def context_manager(self, *, method: str, path: str, request: Request):
        if not self.enabled or self.tracer is None or self.propagator is None:
            return nullcontext()
        try:
            from opentelemetry.trace import SpanKind
        except Exception:
            return nullcontext()
        context = self.propagator.extract(dict(request.headers))
        return self.tracer.start_as_current_span(
            f"{method.upper()} {path}",
            context=context,
            kind=SpanKind.SERVER,
        )

    def trace_id(self, span: Any) -> str:
        if not self.enabled or span is None:
            return ""
        try:
            context = span.get_span_context()
        except Exception:
            return ""
        trace_id = int(getattr(context, "trace_id", 0) or 0)
        if trace_id <= 0:
            return ""
        return f"{trace_id:032x}"

    def annotate(self, span: Any, *, method: str, path: str, status_code: int, request_id: str) -> None:
        if not self.enabled or span is None:
            return
        try:
            span.set_attribute("http.request.method", method.upper())
            span.set_attribute("url.path", path)
            span.set_attribute("http.response.status_code", int(status_code))
            span.set_attribute("spotify.request_id", request_id)
            if int(status_code) >= 500:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR))
        except Exception:
            return


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


def _prediction_readiness(service: PredictionService) -> dict[str, object]:
    bundle_path = _prediction_serving_bundle_path(service.run_dir, include_video=service.include_video)
    bundle_ready = bundle_path.exists()
    input_ready = bundle_ready or service.data_dir is not None
    if bool(service.require_serving_bundle):
        input_ready = bundle_ready
    checks = [
        {"name": "run_dir_exists", "ok": service.run_dir.exists(), "detail": str(service.run_dir)},
        {
            "name": "feature_metadata_present",
            "ok": (service.run_dir / "feature_metadata.json").exists(),
            "detail": str(service.run_dir / "feature_metadata.json"),
        },
        {
            "name": "serving_input_source_ready",
            "ok": input_ready,
            "detail": (
                f"bundle={bundle_path}"
                if bundle_ready
                else ("bundle_missing_but_raw_data_configured" if service.data_dir is not None else f"missing_bundle={bundle_path}")
            ),
        },
        {
            "name": "predictor_loaded",
            "ok": getattr(service, "predictor", None) is not None,
            "detail": str(service.model_name),
        },
    ]
    return {
        "status": "ready" if all(bool(check["ok"]) for check in checks) else "not_ready",
        "checks": checks,
    }


def _taste_os_readiness(service: TasteOSService) -> dict[str, object]:
    bundle_path = _prediction_serving_bundle_path(service.run_dir, include_video=service.include_video)
    bundle_ready = bundle_path.exists()
    input_ready = bundle_ready or service.data_dir is not None
    if bool(service.require_serving_bundle):
        input_ready = bundle_ready
    state_health = service.state_store.health_payload()
    checks = [
        {"name": "run_dir_exists", "ok": service.run_dir.exists(), "detail": str(service.run_dir)},
        {"name": "output_dir_exists", "ok": service.output_dir.exists(), "detail": str(service.output_dir)},
        {
            "name": "serving_input_source_ready",
            "ok": input_ready,
            "detail": (
                f"bundle={bundle_path}"
                if bundle_ready
                else ("bundle_missing_but_raw_data_configured" if service.data_dir is not None else f"missing_bundle={bundle_path}")
            ),
        },
        {
            "name": "state_backend_reachable",
            "ok": bool(state_health.get("reachable", False)),
            "detail": str(state_health.get("database_url", "")) or str(state_health.get("db_path", "")),
        },
        {
            "name": "predictor_loaded",
            "ok": getattr(service, "predictor", None) is not None,
            "detail": str(service.model_name),
        },
    ]
    return {
        "status": "ready" if all(bool(check["ok"]) for check in checks) else "not_ready",
        "checks": checks,
    }


def _liveness_payload(*, service_name: str) -> dict[str, object]:
    return {
        "status": "alive",
        "service_name": service_name,
        "timestamp_epoch_s": int(time.time()),
    }


def _prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _float_metric(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int_metric(value: object) -> int:
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


def _prometheus_metrics_payload(
    *,
    telemetry: _ApiTelemetry,
    service_name: str,
    health_payload: dict[str, object],
    readiness_payload: dict[str, object],
    liveness_payload: dict[str, object],
    rate_limiter: _RateLimiter,
) -> str:
    snapshot = telemetry.snapshot()
    route_rows = snapshot.get("routes")
    route_rows = route_rows if isinstance(route_rows, list) else []
    lines = [
        "# HELP spotify_service_up Service liveness state.",
        "# TYPE spotify_service_up gauge",
        f'spotify_service_up{{service="{_prometheus_escape(service_name)}"}} {1 if str(liveness_payload.get("status")) == "alive" else 0}',
        "# HELP spotify_service_ready Service readiness state.",
        "# TYPE spotify_service_ready gauge",
        f'spotify_service_ready{{service="{_prometheus_escape(service_name)}"}} {1 if str(readiness_payload.get("status")) == "ready" else 0}',
        "# HELP spotify_service_rate_limit_per_minute Configured request rate limit.",
        "# TYPE spotify_service_rate_limit_per_minute gauge",
        (
            f'spotify_service_rate_limit_per_minute{{service="{_prometheus_escape(service_name)}",'
            f'backend="{_prometheus_escape(rate_limiter.backend_name)}"}} {int(rate_limiter.limit_per_minute)}'
        ),
        "# HELP spotify_service_uptime_seconds Service uptime in seconds.",
        "# TYPE spotify_service_uptime_seconds gauge",
        f'spotify_service_uptime_seconds{{service="{_prometheus_escape(service_name)}"}} {_float_metric(snapshot.get("uptime_s")):.3f}',
        "# HELP spotify_service_inflight_requests In-flight HTTP requests.",
        "# TYPE spotify_service_inflight_requests gauge",
        f'spotify_service_inflight_requests{{service="{_prometheus_escape(service_name)}"}} {_int_metric(snapshot.get("inflight_request_count"))}',
    ]

    for route_row in route_rows:
        if not isinstance(route_row, dict):
            continue
        method = _prometheus_escape(str(route_row.get("method", "")))
        path = _prometheus_escape(str(route_row.get("path", "")))
        labels = f'service="{_prometheus_escape(service_name)}",method="{method}",path="{path}"'
        lines.extend(
            [
                "# HELP spotify_service_request_total Total HTTP requests by route.",
                "# TYPE spotify_service_request_total counter",
                f"spotify_service_request_total{{{labels}}} {int(route_row.get('request_count', 0) or 0)}",
                "# HELP spotify_service_error_total Total HTTP error responses by route.",
                "# TYPE spotify_service_error_total counter",
                f"spotify_service_error_total{{{labels}}} {int(route_row.get('error_count', 0) or 0)}",
                "# HELP spotify_service_request_latency_ms_total Total request latency in milliseconds.",
                "# TYPE spotify_service_request_latency_ms_total counter",
                f"spotify_service_request_latency_ms_total{{{labels}}} {float(route_row.get('latency_ms_total', 0.0) or 0.0):.3f}",
                "# HELP spotify_service_request_latency_ms_max Maximum request latency in milliseconds.",
                "# TYPE spotify_service_request_latency_ms_max gauge",
                f"spotify_service_request_latency_ms_max{{{labels}}} {float(route_row.get('max_latency_ms', 0.0) or 0.0):.3f}",
            ]
        )

    if "state_reachable" in health_payload:
        lines.extend(
            [
                "# HELP spotify_service_state_backend_reachable State backend health.",
                "# TYPE spotify_service_state_backend_reachable gauge",
                (
                    f'spotify_service_state_backend_reachable{{service="{_prometheus_escape(service_name)}",'
                    f'backend="{_prometheus_escape(str(health_payload.get("state_backend", "")))}"}} '
                    f'{1 if bool(health_payload.get("state_reachable", False)) else 0}'
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _build_rate_limiter(
    *,
    backend_name: str,
    limit_per_minute: int,
    redis_url: str | None,
    logger: logging.Logger,
) -> _RateLimiter:
    normalized_backend = str(backend_name).strip().lower() or DEFAULT_RATE_LIMIT_BACKEND
    if normalized_backend == "redis":
        if str(redis_url or "").strip():
            return _RedisApiRateLimiter(
                limit_per_minute=limit_per_minute,
                redis_url=str(redis_url).strip(),
                logger=logger,
            )
        logger.warning("Redis rate-limit backend requested without a redis URL; falling back to memory.")
    return _ApiRateLimiter(limit_per_minute=limit_per_minute)


def _install_observability(
    *,
    app: FastAPI,
    logger: logging.Logger,
    telemetry: _ApiTelemetry,
    rate_limiter: _RateLimiter,
    tracing: _ApiTracing,
    authenticator: ApiAuthenticator | None = None,
) -> None:
    rate_limit_exempt_paths = {
        "/health",
        "/v1/health",
        "/livez",
        "/v1/livez",
        "/readyz",
        "/v1/readyz",
        "/metrics",
        "/v1/metrics",
        "/metrics/prometheus",
        "/v1/metrics/prometheus",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
    }

    @app.middleware("http")
    async def _request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or secrets.token_hex(12)
        request.state.request_id = request_id
        start = time.perf_counter()
        path = request.url.path
        telemetry.begin_request()

        with tracing.context_manager(method=request.method, path=path, request=request) as span:
            trace_id = tracing.trace_id(span)
            request.state.trace_id = trace_id

            if path not in rate_limit_exempt_paths:
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
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    response.headers["X-Trace-ID"] = trace_id
                    telemetry.record(
                        method=request.method,
                        path=path,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
                    tracing.annotate(
                        span,
                        method=request.method,
                        path=path,
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                    return response

            if authenticator is not None:
                try:
                    authenticator.authenticate_request(request)
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, dict) else {}
                    response = _json_error(
                        status_code=int(exc.status_code),
                        code=str(detail.get("code", "http_error")),
                        message=str(detail.get("message", "Request failed.")),
                        details=detail.get("details") if isinstance(detail.get("details"), dict) else {},
                        request_id=request_id,
                    )
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    if trace_id:
                        response.headers["X-Trace-ID"] = trace_id
                    telemetry.record(
                        method=request.method,
                        path=path,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
                    tracing.annotate(
                        span,
                        method=request.method,
                        path=path,
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                    return response

            try:
                response = await call_next(request)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                response = _json_error(
                    status_code=int(exc.status_code),
                    code=str(detail.get("code", "http_error")),
                    message=str(detail.get("message", str(exc.detail))),
                    details=detail.get("details") if isinstance(detail.get("details"), dict) else {},
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
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id
            telemetry.record(
                method=request.method,
                path=path,
                status_code=int(response.status_code),
                latency_ms=latency_ms,
            )
            tracing.annotate(
                span,
                method=request.method,
                path=path,
                status_code=int(response.status_code),
                request_id=request_id,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "api_request",
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "method": request.method,
                        "path": path,
                        "status_code": int(response.status_code),
                        "latency_ms": round(latency_ms, 3),
                        "auth_type": str(getattr(request.state, "auth_type", "")),
                        "auth_subject": str(getattr(request.state, "auth_subject", "")),
                    },
                    sort_keys=True,
                )
            )
            return response


def _legacy_token_authenticator(*, auth_token: str | None, logger: logging.Logger) -> ApiAuthenticator | None:
    token_value = str(auth_token or "").strip()
    if not token_value:
        return None
    return ApiAuthenticator(
        settings=ApiAuthSettings.from_values(
            mode="token",
            scope="mutations",
            legacy_token=token_value,
            jwt_secret=None,
            jwks_url=None,
            jwt_algorithms=None,
            jwt_issuer=None,
            jwt_audience=None,
            jwt_required_scopes=None,
            jwt_leeway_seconds=0,
        ),
        logger=logger,
    )


def _register_common_service_routes(
    *,
    app: FastAPI,
    service_name: str,
    health_provider: Any,
    readiness_provider: Any,
    telemetry: _ApiTelemetry,
    rate_limiter: _RateLimiter,
    authenticator: ApiAuthenticator | None = None,
) -> None:
    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, object]:
        return health_provider()

    @app.get("/livez")
    @app.get("/v1/livez")
    async def livez() -> dict[str, object]:
        return _liveness_payload(service_name=service_name)

    @app.get("/readyz")
    @app.get("/v1/readyz")
    async def readyz() -> dict[str, object]:
        return readiness_provider()

    @app.get("/metrics")
    @app.get("/v1/metrics")
    async def metrics() -> dict[str, object]:
        return {
            "health": health_provider(),
            "liveness": _liveness_payload(service_name=service_name),
            "readiness": readiness_provider(),
            "telemetry": telemetry.snapshot(),
            "rate_limit_per_minute": rate_limiter.limit_per_minute,
            "rate_limit_backend": rate_limiter.backend_name,
            "auth": authenticator.summary() if authenticator is not None else {"enabled": False, "mode": "off", "scope": "mutations"},
        }

    @app.get("/metrics/prometheus")
    @app.get("/v1/metrics/prometheus")
    async def metrics_prometheus() -> PlainTextResponse:
        health_payload = health_provider()
        readiness_payload = readiness_provider()
        liveness_payload = _liveness_payload(service_name=service_name)
        return PlainTextResponse(
            _prometheus_metrics_payload(
                telemetry=telemetry,
                service_name=service_name,
                health_payload=health_payload,
                readiness_payload=readiness_payload,
                liveness_payload=liveness_payload,
                rate_limiter=rate_limiter,
            ),
            media_type="text/plain; version=0.0.4",
        )


def create_prediction_app(
    *,
    service: PredictionService,
    logger: logging.Logger,
    request_rate_limit: int = DEFAULT_REQUEST_RATE_LIMIT,
    rate_limit_backend: str = DEFAULT_RATE_LIMIT_BACKEND,
    redis_url: str | None = None,
    enable_otel: bool = True,
    authenticator: ApiAuthenticator | None = None,
) -> FastAPI:
    effective_authenticator = authenticator or _legacy_token_authenticator(
        auth_token=getattr(service, "auth_token", None),
        logger=logger,
    )
    telemetry = _ApiTelemetry(service_name="spotify.predict")
    rate_limiter = _build_rate_limiter(
        backend_name=rate_limit_backend,
        limit_per_minute=request_rate_limit,
        redis_url=redis_url,
        logger=logger,
    )
    tracing = _ApiTracing(service_name="spotify.predict", logger=logger, enabled=enable_otel)
    app = FastAPI(
        title="Spotify Prediction API",
        version="1.0.0",
        description="Production-style ASGI surface for next-artist prediction.",
    )
    _install_observability(
        app=app,
        logger=logger,
        telemetry=telemetry,
        rate_limiter=rate_limiter,
        tracing=tracing,
        authenticator=effective_authenticator,
    )
    _register_common_service_routes(
        app=app,
        service_name="spotify.predict",
        health_provider=service.health_payload,
        readiness_provider=lambda: _prediction_readiness(service),
        telemetry=telemetry,
        rate_limiter=rate_limiter,
        authenticator=effective_authenticator,
    )

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

    @app.post("/predict")
    @app.post("/v1/predict")
    async def predict(payload: PredictApiRequest, request: Request) -> dict[str, object]:
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
    rate_limit_backend: str = DEFAULT_RATE_LIMIT_BACKEND,
    redis_url: str | None = None,
    enable_otel: bool = True,
    authenticator: ApiAuthenticator | None = None,
) -> FastAPI:
    effective_authenticator = authenticator or _legacy_token_authenticator(
        auth_token=getattr(service, "auth_token", None),
        logger=logger,
    )
    telemetry = _ApiTelemetry(service_name="spotify.taste_os")
    rate_limiter = _build_rate_limiter(
        backend_name=rate_limit_backend,
        limit_per_minute=request_rate_limit,
        redis_url=redis_url,
        logger=logger,
    )
    tracing = _ApiTracing(service_name="spotify.taste_os", logger=logger, enabled=enable_otel)
    app = FastAPI(
        title="Spotify Taste OS API",
        version="1.0.0",
        description="Production-style ASGI surface for Taste OS session planning.",
    )
    _install_observability(
        app=app,
        logger=logger,
        telemetry=telemetry,
        rate_limiter=rate_limiter,
        tracing=tracing,
        authenticator=effective_authenticator,
    )
    _register_common_service_routes(
        app=app,
        service_name="spotify.taste_os",
        health_provider=service.health_payload,
        readiness_provider=lambda: _taste_os_readiness(service),
        telemetry=telemetry,
        rate_limiter=rate_limiter,
        authenticator=effective_authenticator,
    )

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
    env_rate_limit_backend = str(os.getenv("SPOTIFY_SERVICE_RATE_LIMIT_BACKEND", DEFAULT_RATE_LIMIT_BACKEND)).strip().lower()
    env_otel_enabled = _env_flag("SPOTIFY_SERVICE_OTEL_ENABLED", default=True)

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
    parser.add_argument(
        "--state-db-url",
        type=str,
        default=os.getenv("TASTE_OS_DATABASE_URL", ""),
        help="Optional SQLAlchemy database URL for Taste OS state. Overrides --state-db when set.",
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
    parser.add_argument(
        "--auth-mode",
        choices=("off", "auto", "token", "jwt", "token_or_jwt"),
        default=str(os.getenv("SPOTIFY_SERVICE_AUTH_MODE", "auto")).strip().lower() or "auto",
        help="Authentication mode. `auto` picks token or JWT based on configured secrets.",
    )
    parser.add_argument(
        "--auth-scope",
        choices=("mutations", "all"),
        default=str(os.getenv("SPOTIFY_SERVICE_AUTH_SCOPE", "mutations")).strip().lower() or "mutations",
        help="Whether auth protects only mutations or all non-health application routes.",
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWT_SECRET", ""),
        help="Optional shared secret used for HS* JWT validation.",
    )
    parser.add_argument(
        "--jwks-url",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWKS_URL", ""),
        help="Optional JWKS URL used for RS*/ES* JWT validation.",
    )
    parser.add_argument(
        "--jwt-issuer",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWT_ISSUER", ""),
        help="Optional expected JWT issuer.",
    )
    parser.add_argument(
        "--jwt-audience",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWT_AUDIENCE", ""),
        help="Optional comma-separated JWT audience values.",
    )
    parser.add_argument(
        "--jwt-algorithms",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWT_ALGORITHMS", "RS256"),
        help="Comma-separated allowed JWT algorithms.",
    )
    parser.add_argument(
        "--jwt-required-scopes",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_JWT_REQUIRED_SCOPES", ""),
        help="Optional comma-separated JWT scopes required on protected routes.",
    )
    parser.add_argument(
        "--jwt-leeway-seconds",
        type=int,
        default=max(0, int(str(os.getenv("SPOTIFY_SERVICE_JWT_LEEWAY_SECONDS", "0")).strip() or "0")),
        help="Clock skew leeway for JWT validation.",
    )
    parser.add_argument("--include-video", action="store_true", help="Use audio+video context by default.")
    parser.add_argument(
        "--request-rate-limit",
        type=int,
        default=env_rate_limit,
        help="Allowed requests per minute per token/client. Set 0 to disable.",
    )
    parser.add_argument(
        "--rate-limit-backend",
        choices=("memory", "redis"),
        default=env_rate_limit_backend if env_rate_limit_backend in {"memory", "redis"} else DEFAULT_RATE_LIMIT_BACKEND,
        help="Rate-limit backend. Use redis for multi-instance shared limiting.",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_REDIS_URL", ""),
        help="Optional Redis URL used by the redis rate-limit backend.",
    )
    parser.add_argument(
        "--disable-otel",
        action="store_true",
        default=not env_otel_enabled,
        help="Disable OpenTelemetry request spans and trace headers.",
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
    auth_settings = ApiAuthSettings.from_values(
        mode=str(args.auth_mode),
        scope=str(args.auth_scope),
        legacy_token=auth_token,
        jwt_secret=str(args.jwt_secret),
        jwks_url=str(args.jwks_url),
        jwt_algorithms=str(args.jwt_algorithms),
        jwt_issuer=str(args.jwt_issuer),
        jwt_audience=str(args.jwt_audience),
        jwt_required_scopes=str(args.jwt_required_scopes),
        jwt_leeway_seconds=int(args.jwt_leeway_seconds),
    )
    authenticator = ApiAuthenticator(settings=auth_settings, logger=logger)
    rate_limit_backend = str(args.rate_limit_backend).strip().lower() or DEFAULT_RATE_LIMIT_BACKEND
    redis_url = str(args.redis_url).strip() or None
    enable_otel = not bool(args.disable_otel)

    if args.app == "predict":
        predict_service_instance = PredictionService(
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
            service=predict_service_instance,
            logger=logger,
            request_rate_limit=max(0, int(args.request_rate_limit)),
            rate_limit_backend=rate_limit_backend,
            redis_url=redis_url,
            enable_otel=enable_otel,
            authenticator=authenticator,
        )
        return "predict", app

    output_dir = Path(args.output_dir).expanduser().resolve()
    state_db_path = Path(args.state_db).expanduser().resolve() if str(args.state_db).strip() else None
    state_db_url = str(args.state_db_url).strip() or None
    taste_service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        max_top_k=max(1, int(args.max_top_k)),
        auth_token=auth_token,
        logger=logger,
        state_db_path=state_db_path,
        state_database_url=state_db_url,
        require_serving_bundle=bool(args.require_serving_bundle),
    )
    app = create_taste_os_app(
        service=taste_service,
        logger=logger,
        request_rate_limit=max(0, int(args.request_rate_limit)),
        rate_limit_backend=rate_limit_backend,
        redis_url=redis_url,
        enable_otel=enable_otel,
        authenticator=authenticator,
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
