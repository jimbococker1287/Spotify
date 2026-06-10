from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from .champion_alias import resolve_prediction_run_dir
from .predict_service import PredictionService
from .run_artifacts import write_csv_rows, write_json, write_markdown
from .service_api import (
    build_service_deployment_readiness_provider,
    create_prediction_app,
    create_taste_os_app,
)
from .taste_os_service import TasteOSService


DEFAULT_CHANNEL_RUN_DIR = "outputs/deployments/registry/channels/stable"
CHECK_COLUMNS = [
    "service",
    "check_key",
    "status",
    "severity",
    "message",
    "endpoint",
    "status_code",
    "latency_ms",
    "request_id",
    "trace_id",
]
REQUEST_COLUMNS = [
    "service",
    "method",
    "endpoint",
    "ok",
    "status_code",
    "latency_ms",
    "request_id",
    "trace_id",
    "error",
]
HISTORY_COLUMNS = [
    "generated_at",
    "release_id",
    "run_dir",
    "requested_run_dir",
    "model_name",
    "status",
    "production_ready",
    "check_count",
    "pass_count",
    "warning_count",
    "fail_count",
    "request_count",
    "successful_request_count",
    "max_latency_ms",
    "average_latency_ms",
    "predict_readyz_status",
    "predict_metrics_status",
    "taste_os_readyz_status",
    "taste_os_metrics_status",
    "blocker_count",
]


@dataclass(frozen=True)
class SmokeCheck:
    service: str
    check_key: str
    status: str
    severity: str
    message: str
    endpoint: str = ""
    status_code: int = 0
    latency_ms: float = 0.0
    request_id: str = ""
    trace_id: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "check_key": self.check_key,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "latency_ms": round(float(self.latency_ms), 3),
            "request_id": self.request_id,
            "trace_id": self.trace_id,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    raw = str(value).strip()
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        return Path(unquote(parsed.path)).expanduser().resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _safe_json(response: Any) -> object:
    try:
        return response.json()
    except Exception:
        return {"text": str(getattr(response, "text", ""))[:500]}


def _coerce_int(value: object, default: int = 0) -> int:
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
            return default
    return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "ready"}
    return False


def _request_summary(body: object, *, endpoint: str) -> dict[str, object]:
    if not isinstance(body, dict):
        return {"body_type": type(body).__name__}
    if endpoint.endswith("/readyz"):
        checks = body.get("checks", [])
        checks_list = checks if isinstance(checks, list) else []
        return {
            "status": str(body.get("status", "")),
            "check_count": len(checks_list),
            "failed_checks": [
                str(row.get("name", ""))
                for row in checks_list
                if isinstance(row, dict) and not bool(row.get("ok", False))
            ],
        }
    if endpoint.endswith("/metrics"):
        readiness = body.get("readiness", {})
        health = body.get("health", {})
        telemetry = body.get("telemetry", {})
        return {
            "health_status": str(health.get("status", "")) if isinstance(health, dict) else "",
            "readiness_status": str(readiness.get("status", "")) if isinstance(readiness, dict) else "",
            "telemetry_total_request_count": telemetry.get("total_request_count", 0) if isinstance(telemetry, dict) else 0,
            "rate_limit_backend": str(body.get("rate_limit_backend", "")),
        }
    if endpoint.endswith("/predict"):
        predictions = body.get("predictions", [])
        predictions_list = predictions if isinstance(predictions, list) else []
        top_prediction = predictions_list[0] if predictions_list and isinstance(predictions_list[0], dict) else {}
        return {
            "model_name": str(body.get("model_name", "")),
            "model_type": str(body.get("model_type", "")),
            "decision": str(body.get("decision", "")),
            "prediction_count": len(predictions_list),
            "top_artist": str(top_prediction.get("artist_name", "")) if isinstance(top_prediction, dict) else "",
        }
    if endpoint.endswith("/taste-os/session"):
        service = body.get("service", {})
        candidates = body.get("top_candidates", [])
        journey = body.get("journey_plan", [])
        return {
            "session_id": str(service.get("session_id", "")) if isinstance(service, dict) else "",
            "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
            "journey_step_count": len(journey) if isinstance(journey, list) else 0,
        }
    return {key: body.get(key) for key in sorted(body)[:8]}


def _check(
    *,
    service: str,
    check_key: str,
    condition: bool,
    pass_message: str,
    fail_message: str,
    severity: str = "required",
    request: dict[str, object] | None = None,
) -> SmokeCheck:
    request = request or {}
    return SmokeCheck(
        service=service,
        check_key=check_key,
        status="pass" if condition else ("warn" if severity == "advisory" else "fail"),
        severity=severity,
        message=pass_message if condition else fail_message,
        endpoint=str(request.get("endpoint", "")),
        status_code=_coerce_int(request.get("status_code", 0)),
        latency_ms=_coerce_float(request.get("latency_ms", 0.0)),
        request_id=str(request.get("request_id", "")),
        trace_id=str(request.get("trace_id", "")),
    )


def _headers(auth_token: str | None) -> dict[str, str]:
    token = str(auth_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _client_request(
    client: Any,
    *,
    service: str,
    method: str,
    endpoint: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    try:
        response = client.request(method, endpoint, json=payload, headers=headers or {})
        latency_ms = (time.perf_counter() - start) * 1000.0
        status_code = int(response.status_code)
        body = _safe_json(response)
        return {
            "service": service,
            "method": method,
            "endpoint": endpoint,
            "ok": 200 <= status_code < 300,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "request_id": str(response.headers.get("x-request-id", "")),
            "trace_id": str(response.headers.get("x-trace-id", "")),
            "error": "",
            "response_summary": _request_summary(body, endpoint=endpoint),
            "body": body,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "service": service,
            "method": method,
            "endpoint": endpoint,
            "ok": False,
            "status_code": 0,
            "latency_ms": round(latency_ms, 3),
            "request_id": "",
            "trace_id": "",
            "error": str(exc),
            "response_summary": {},
            "body": {},
        }


def _readiness_checks(service: str, request: dict[str, object]) -> list[SmokeCheck]:
    body = request.get("body", {})
    if not isinstance(body, dict):
        return [
            _check(
                service=service,
                check_key="readyz_body_json",
                condition=False,
                pass_message="Readiness response is JSON.",
                fail_message="Readiness response is not a JSON object.",
                request=request,
            )
        ]
    checks = [
        _check(
            service=service,
            check_key="readyz_status_ready",
            condition=str(body.get("status", "")) == "ready",
            pass_message="Readiness endpoint reports ready.",
            fail_message=f"Readiness endpoint reports {body.get('status', '') or 'unknown'}.",
            request=request,
        )
    ]
    ready_checks = body.get("checks", [])
    if isinstance(ready_checks, list):
        for row in ready_checks:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip() or "unnamed"
            detail = str(row.get("detail", "")).strip()
            checks.append(
                _check(
                    service=service,
                    check_key=f"readyz_{name}",
                    condition=bool(row.get("ok", False)),
                    pass_message=f"{name} is ready.",
                    fail_message=f"{name} is not ready: {detail}",
                    request=request,
                )
            )
    return checks


def _summarize(checks: list[SmokeCheck], requests: list[dict[str, object]]) -> dict[str, object]:
    fail_count = sum(1 for check in checks if check.status == "fail")
    warn_count = sum(1 for check in checks if check.status == "warn")
    pass_count = sum(1 for check in checks if check.status == "pass")
    status = "fail" if fail_count else "attention" if warn_count else "pass"
    latencies = [_coerce_float(row.get("latency_ms", 0.0)) for row in requests]
    return {
        "status": status,
        "production_ready": status == "pass",
        "check_count": len(checks),
        "pass_count": pass_count,
        "warning_count": warn_count,
        "fail_count": fail_count,
        "request_count": len(requests),
        "successful_request_count": sum(1 for row in requests if bool(row.get("ok", False))),
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "average_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
    }


def _blockers(checks: list[SmokeCheck], requests: list[dict[str, object]]) -> list[str]:
    blockers = [check.message for check in checks if check.status == "fail"]
    blockers.extend(
        f"{row.get('service')} {row.get('endpoint')} failed: {row.get('error') or row.get('status_code')}"
        for row in requests
        if not bool(row.get("ok", False))
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        text = str(blocker).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _request_csv_rows(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{column: row.get(column, "") for column in REQUEST_COLUMNS} for row in requests]


def _check_status(checks: list[object], *, service: str, check_key: str) -> str:
    for row in checks:
        if not isinstance(row, dict):
            continue
        if str(row.get("service", "")) == service and str(row.get("check_key", "")) == check_key:
            return str(row.get("status", ""))
    return ""


def _history_row(payload: dict[str, object]) -> dict[str, object]:
    summary = payload.get("summary", {})
    summary_dict = summary if isinstance(summary, dict) else {}
    checks = payload.get("checks", [])
    check_rows = checks if isinstance(checks, list) else []
    blockers = payload.get("blockers", [])
    blocker_rows = blockers if isinstance(blockers, list) else []
    run_dir = str(payload.get("run_dir", ""))
    return {
        "generated_at": str(payload.get("generated_at", "")),
        "release_id": Path(run_dir).name if run_dir else "",
        "run_dir": run_dir,
        "requested_run_dir": str(payload.get("requested_run_dir", "")),
        "model_name": str(payload.get("model_name", "")),
        "status": str(summary_dict.get("status", "")),
        "production_ready": bool(summary_dict.get("production_ready", False)),
        "check_count": _coerce_int(summary_dict.get("check_count", 0)),
        "pass_count": _coerce_int(summary_dict.get("pass_count", 0)),
        "warning_count": _coerce_int(summary_dict.get("warning_count", 0)),
        "fail_count": _coerce_int(summary_dict.get("fail_count", 0)),
        "request_count": _coerce_int(summary_dict.get("request_count", 0)),
        "successful_request_count": _coerce_int(summary_dict.get("successful_request_count", 0)),
        "max_latency_ms": round(_coerce_float(summary_dict.get("max_latency_ms", 0.0)), 3),
        "average_latency_ms": round(_coerce_float(summary_dict.get("average_latency_ms", 0.0)), 3),
        "predict_readyz_status": _check_status(check_rows, service="predict", check_key="readyz_status_ready"),
        "predict_metrics_status": _check_status(check_rows, service="predict", check_key="metrics_readiness_ready"),
        "taste_os_readyz_status": _check_status(check_rows, service="taste-os", check_key="readyz_status_ready"),
        "taste_os_metrics_status": _check_status(check_rows, service="taste-os", check_key="metrics_readiness_ready"),
        "blocker_count": len(blocker_rows),
    }


def _read_history_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as infile:
            return [dict(row) for row in csv.DictReader(infile)]
    except Exception:
        return []


def _trend_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "generated_at": str(row.get("generated_at", "")),
        "release_id": str(row.get("release_id", "")),
        "run_dir": str(row.get("run_dir", "")),
        "model_name": str(row.get("model_name", "")),
        "status": str(row.get("status", "")),
        "production_ready": _coerce_bool(row.get("production_ready", False)),
        "fail_count": _coerce_int(row.get("fail_count", 0)),
        "blocker_count": _coerce_int(row.get("blocker_count", 0)),
        "max_latency_ms": round(_coerce_float(row.get("max_latency_ms", 0.0)), 3),
        "average_latency_ms": round(_coerce_float(row.get("average_latency_ms", 0.0)), 3),
        "predict_readyz_status": str(row.get("predict_readyz_status", "")),
        "taste_os_readyz_status": str(row.get("taste_os_readyz_status", "")),
    }


def _trend_summary(rows: list[dict[str, object]], *, window_size: int = 20) -> dict[str, object]:
    recent = [_trend_row(row) for row in rows[-max(1, int(window_size)) :]]
    latest = recent[-1] if recent else {}
    previous = recent[-2] if len(recent) > 1 else {}
    status_counts: dict[str, int] = {}
    for row in recent:
        status = str(row.get("status", "")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    denominator = max(len(recent), 1)
    latest_max = _coerce_float(latest.get("max_latency_ms", 0.0)) if latest else 0.0
    previous_max = _coerce_float(previous.get("max_latency_ms", 0.0)) if previous else 0.0
    ready_count = sum(1 for row in recent if _coerce_bool(row.get("production_ready", False)))
    both_ready_count = sum(
        1
        for row in recent
        if str(row.get("predict_readyz_status", "")) == "pass"
        and str(row.get("taste_os_readyz_status", "")) == "pass"
    )
    return {
        "history_run_count": len(rows),
        "window_size": len(recent),
        "latest": latest,
        "previous": previous,
        "status_counts": status_counts,
        "production_ready_rate": round(ready_count / denominator, 4),
        "both_services_ready_rate": round(both_ready_count / denominator, 4),
        "average_max_latency_ms": round(
            sum(_coerce_float(row.get("max_latency_ms", 0.0)) for row in recent) / denominator,
            3,
        ),
        "latency_delta_ms": round(latest_max - previous_max, 3) if previous else 0.0,
        "recent_runs": recent,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "/")


def _trend_markdown(trend: dict[str, object], *, history_csv: Path) -> list[str]:
    latest = trend.get("latest", {})
    latest_dict = latest if isinstance(latest, dict) else {}
    recent = trend.get("recent_runs", [])
    recent_rows = recent if isinstance(recent, list) else []
    lines = [
        "# Production Smoke Trend",
        "",
        f"History CSV: `{history_csv}`",
        f"Runs tracked: {trend.get('history_run_count', 0)}",
        f"Window size: {trend.get('window_size', 0)}",
        f"Latest status: `{latest_dict.get('status', '')}`",
        f"Latest max latency ms: {latest_dict.get('max_latency_ms', 0.0)}",
        f"Delta vs previous max latency ms: {trend.get('latency_delta_ms', 0.0)}",
        f"Production-ready rate: {trend.get('production_ready_rate', 0.0)}",
        f"Both-services-ready rate: {trend.get('both_services_ready_rate', 0.0)}",
        "",
        "## Recent Runs",
        "",
        "| Generated At | Release | Status | Ready | Max Latency ms | Failures | Blockers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in recent_rows[-10:]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(column, ""))
                for column in (
                    "generated_at",
                    "release_id",
                    "status",
                    "production_ready",
                    "max_latency_ms",
                    "fail_count",
                    "blocker_count",
                )
            )
            + " |"
        )
    return lines


def _write_history_artifacts(
    *,
    payload: dict[str, object],
    outputs_dir: Path,
    output_root: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    history_csv = outputs_dir / "history" / "production_smoke_history.csv"
    rows = [
        {column: row.get(column, "") for column in HISTORY_COLUMNS}
        for row in _read_history_rows(history_csv)
    ]
    rows.append(_history_row(payload))
    write_csv_rows(history_csv, rows, fieldnames=HISTORY_COLUMNS)

    trend = _trend_summary(rows)
    trend_json = output_root / "production_smoke_trend.json"
    trend_md = output_root / "production_smoke_trend.md"
    write_json(
        trend_json,
        {
            "generated_at": str(payload.get("generated_at", "")),
            "history_csv": str(history_csv),
            **trend,
        },
        sort_keys=True,
    )
    write_markdown(trend_md, _trend_markdown(trend, history_csv=history_csv))
    return {"history_csv": str(history_csv), "trend_json": str(trend_json), "trend_md": str(trend_md)}, trend


def _markdown(payload: dict[str, object]) -> list[str]:
    summary = payload.get("summary", {})
    summary_dict = summary if isinstance(summary, dict) else {}
    trend = payload.get("trend_summary", {})
    trend_dict = trend if isinstance(trend, dict) else {}
    latest = trend_dict.get("latest", {})
    latest_dict = latest if isinstance(latest, dict) else {}
    blockers = payload.get("blockers", [])
    blocker_rows = blockers if isinstance(blockers, list) else []
    requests = payload.get("requests", [])
    request_rows = requests if isinstance(requests, list) else []
    lines = [
        "# Production Smoke",
        "",
        f"Generated at: {payload.get('generated_at', '')}",
        f"Status: {str(summary_dict.get('status', '')).upper()}",
        f"Run: `{payload.get('run_dir', '')}`",
        f"Model: `{payload.get('model_name', '')}`",
        "",
        "## Summary",
        "",
        f"- Checks: {summary_dict.get('check_count', 0)}",
        f"- Passed: {summary_dict.get('pass_count', 0)}",
        f"- Warnings: {summary_dict.get('warning_count', 0)}",
        f"- Failed: {summary_dict.get('fail_count', 0)}",
        f"- Requests: {summary_dict.get('successful_request_count', 0)} / {summary_dict.get('request_count', 0)} succeeded",
        f"- Max latency ms: {summary_dict.get('max_latency_ms', 0.0)}",
        f"- History runs tracked: {trend_dict.get('history_run_count', 0)}",
        f"- Production-ready rate: {trend_dict.get('production_ready_rate', 0.0)}",
        f"- Latest trend max latency ms: {latest_dict.get('max_latency_ms', summary_dict.get('max_latency_ms', 0.0))}",
        f"- Max latency delta vs previous ms: {trend_dict.get('latency_delta_ms', 0.0)}",
        "",
        "## Requests",
        "",
        "| Service | Method | Endpoint | Status | Latency ms | Request ID |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in request_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("service", "")),
                    str(row.get("method", "")),
                    str(row.get("endpoint", "")),
                    str(row.get("status_code", "")),
                    str(row.get("latency_ms", "")),
                    str(row.get("request_id", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Blockers", ""])
    if blocker_rows:
        for blocker in blocker_rows[:12]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- Both ASGI services passed readiness, metrics, and request smoke checks.")
    return lines


def _test_client_class() -> Any:
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover - depends on optional local install state.
        raise RuntimeError("Production smoke requires fastapi.testclient and its httpx runtime.") from exc
    return TestClient


def _service_startup_check(service: str, app: Any | None, exc: Exception | None) -> SmokeCheck:
    return _check(
        service=service,
        check_key="service_startup",
        condition=app is not None and exc is None,
        pass_message=f"{service} ASGI app started.",
        fail_message=f"{service} ASGI app failed to start: {exc}",
    )


def _smoke_prediction_service(
    *,
    requested_run_dir: Path,
    run_dir: Path,
    data_dir: Path | None,
    model_name: str,
    include_video: bool,
    max_top_k: int,
    top_k: int,
    recent_artists: list[str] | None,
    auth_token: str | None,
    require_serving_bundle: bool,
    logger: logging.Logger,
) -> tuple[list[SmokeCheck], list[dict[str, object]]]:
    checks: list[SmokeCheck] = []
    requests: list[dict[str, object]] = []
    app = None
    startup_exc: Exception | None = None
    try:
        service = PredictionService(
            run_dir=run_dir,
            data_dir=data_dir,
            model_name=model_name,
            include_video=include_video,
            max_top_k=max_top_k,
            auth_token=auth_token,
            logger=logger,
            require_serving_bundle=require_serving_bundle,
        )
        deployment_provider = build_service_deployment_readiness_provider(
            requested_run_dir=requested_run_dir,
            service=service,
        )
        app = create_prediction_app(
            service=service,
            logger=logger,
            request_rate_limit=120,
            enable_otel=False,
            deployment_readiness_provider=deployment_provider,
        )
    except Exception as exc:
        startup_exc = exc
    checks.append(_service_startup_check("predict", app, startup_exc))
    if app is None:
        return checks, requests

    try:
        TestClient = _test_client_class()
    except Exception as exc:
        checks.append(
            SmokeCheck(
                service="predict",
                check_key="test_client_available",
                status="fail",
                severity="required",
                message=str(exc),
            )
        )
        return checks, requests

    with TestClient(app) as client:
        ready = _client_request(client, service="predict", method="GET", endpoint="/v1/readyz")
        requests.append(ready)
        checks.append(
            _check(
                service="predict",
                check_key="readyz_http_200",
                condition=bool(ready.get("ok", False)),
                pass_message="Prediction readiness endpoint returned 2xx.",
                fail_message="Prediction readiness endpoint did not return 2xx.",
                request=ready,
            )
        )
        checks.extend(_readiness_checks("predict", ready))

        metrics = _client_request(client, service="predict", method="GET", endpoint="/v1/metrics")
        requests.append(metrics)
        metrics_body = metrics.get("body", {})
        readiness = metrics_body.get("readiness", {}) if isinstance(metrics_body, dict) else {}
        checks.append(
            _check(
                service="predict",
                check_key="metrics_readiness_ready",
                condition=bool(metrics.get("ok", False)) and isinstance(readiness, dict) and readiness.get("status") == "ready",
                pass_message="Prediction metrics endpoint includes ready readiness state.",
                fail_message="Prediction metrics endpoint did not include ready readiness state.",
                request=metrics,
            )
        )

        predict_payload: dict[str, object] = {"top_k": top_k, "include_video": include_video}
        if recent_artists:
            predict_payload["recent_artists"] = recent_artists
        prediction = _client_request(
            client,
            service="predict",
            method="POST",
            endpoint="/v1/predict",
            payload=predict_payload,
            headers=_headers(auth_token),
        )
        requests.append(prediction)
        prediction_body = prediction.get("body", {})
        predictions = prediction_body.get("predictions", []) if isinstance(prediction_body, dict) else []
        checks.append(
            _check(
                service="predict",
                check_key="predict_request_returns_predictions",
                condition=bool(prediction.get("ok", False)) and isinstance(predictions, list) and bool(predictions),
                pass_message="Prediction request returned ranked predictions.",
                fail_message="Prediction request did not return ranked predictions.",
                request=prediction,
            )
        )
    return checks, requests


def _smoke_taste_os_service(
    *,
    requested_run_dir: Path,
    run_dir: Path,
    data_dir: Path | None,
    service_output_dir: Path,
    state_db_path: Path | None,
    state_database_url: str | None,
    model_name: str,
    include_video: bool,
    max_top_k: int,
    top_k: int,
    recent_artists: list[str] | None,
    mode: str,
    scenario: str,
    auth_token: str | None,
    require_serving_bundle: bool,
    logger: logging.Logger,
) -> tuple[list[SmokeCheck], list[dict[str, object]]]:
    checks: list[SmokeCheck] = []
    requests: list[dict[str, object]] = []
    app = None
    startup_exc: Exception | None = None
    try:
        service = TasteOSService(
            run_dir=run_dir,
            data_dir=data_dir,
            output_dir=service_output_dir,
            model_name=model_name,
            include_video=include_video,
            max_top_k=max_top_k,
            auth_token=auth_token,
            logger=logger,
            state_db_path=state_db_path,
            state_database_url=state_database_url,
            require_serving_bundle=require_serving_bundle,
        )
        deployment_provider = build_service_deployment_readiness_provider(
            requested_run_dir=requested_run_dir,
            service=service,
        )
        app = create_taste_os_app(
            service=service,
            logger=logger,
            request_rate_limit=120,
            enable_otel=False,
            deployment_readiness_provider=deployment_provider,
        )
    except Exception as exc:
        startup_exc = exc
    checks.append(_service_startup_check("taste-os", app, startup_exc))
    if app is None:
        return checks, requests

    try:
        TestClient = _test_client_class()
    except Exception as exc:
        checks.append(
            SmokeCheck(
                service="taste-os",
                check_key="test_client_available",
                status="fail",
                severity="required",
                message=str(exc),
            )
        )
        return checks, requests

    with TestClient(app) as client:
        ready = _client_request(client, service="taste-os", method="GET", endpoint="/v1/readyz")
        requests.append(ready)
        checks.append(
            _check(
                service="taste-os",
                check_key="readyz_http_200",
                condition=bool(ready.get("ok", False)),
                pass_message="Taste OS readiness endpoint returned 2xx.",
                fail_message="Taste OS readiness endpoint did not return 2xx.",
                request=ready,
            )
        )
        checks.extend(_readiness_checks("taste-os", ready))

        metrics = _client_request(client, service="taste-os", method="GET", endpoint="/v1/metrics")
        requests.append(metrics)
        metrics_body = metrics.get("body", {})
        readiness = metrics_body.get("readiness", {}) if isinstance(metrics_body, dict) else {}
        checks.append(
            _check(
                service="taste-os",
                check_key="metrics_readiness_ready",
                condition=bool(metrics.get("ok", False)) and isinstance(readiness, dict) and readiness.get("status") == "ready",
                pass_message="Taste OS metrics endpoint includes ready readiness state.",
                fail_message="Taste OS metrics endpoint did not include ready readiness state.",
                request=metrics,
            )
        )

        session_payload: dict[str, object] = {
            "mode": mode,
            "scenario": scenario,
            "top_k": top_k,
            "include_video": include_video,
            "persist_artifacts": False,
            "use_feedback_memory": False,
        }
        if recent_artists:
            session_payload["recent_artists"] = recent_artists
        session = _client_request(
            client,
            service="taste-os",
            method="POST",
            endpoint="/v1/taste-os/session",
            payload=session_payload,
            headers=_headers(auth_token),
        )
        requests.append(session)
        session_body = session.get("body", {})
        service_payload = session_body.get("service", {}) if isinstance(session_body, dict) else {}
        checks.append(
            _check(
                service="taste-os",
                check_key="session_request_returns_session_id",
                condition=bool(session.get("ok", False)) and isinstance(service_payload, dict) and bool(service_payload.get("session_id")),
                pass_message="Taste OS session request returned a session id.",
                fail_message="Taste OS session request did not return a session id.",
                request=session,
            )
        )
    return checks, requests


def build_production_smoke(
    *,
    project_root: Path,
    outputs_dir: Path,
    run_dir: str | Path = DEFAULT_CHANNEL_RUN_DIR,
    output_dir: Path | None = None,
    service_output_dir: Path | None = None,
    data_dir: Path | None = None,
    model_name: str | None = None,
    top_k: int = 3,
    max_top_k: int = 5,
    recent_artists: list[str] | None = None,
    mode: str = "focus",
    scenario: str = "steady",
    include_video: bool = False,
    auth_token: str | None = None,
    state_db_path: Path | None = None,
    state_database_url: str | None = None,
    require_serving_bundle: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    logger = logger or logging.getLogger("spotify.production_smoke")
    project_root = project_root.expanduser().resolve()
    outputs_dir = _resolve_path(outputs_dir, base=project_root)
    output_root = _resolve_path(output_dir, base=project_root) if output_dir is not None else outputs_dir / "analysis" / "production_smoke"
    service_root = _resolve_path(service_output_dir, base=project_root) if service_output_dir is not None else output_root / "taste_os_service"
    data_root = _resolve_path(data_dir, base=project_root) if data_dir is not None else None
    resolved_state_db_path = _resolve_path(state_db_path, base=project_root) if state_db_path is not None else None
    requested_run_dir = _resolve_path(run_dir, base=project_root)
    checks: list[SmokeCheck] = []
    requests: list[dict[str, object]] = []

    resolved_run_dir: Path | None = None
    alias_model_name: str | None = None
    effective_model_name = str(model_name or "").strip()
    try:
        resolved_run_dir, alias_model_name = resolve_prediction_run_dir(str(requested_run_dir), project_root=project_root)
        if not effective_model_name:
            effective_model_name = str(alias_model_name or "").strip()
        checks.append(
            _check(
                service="shared",
                check_key="run_dir_resolves",
                condition=True,
                pass_message="Production smoke run directory resolved.",
                fail_message="Production smoke run directory could not be resolved.",
            )
        )
    except Exception as exc:
        checks.append(
            SmokeCheck(
                service="shared",
                check_key="run_dir_resolves",
                status="fail",
                severity="required",
                message=f"Production smoke run directory could not be resolved: {exc}",
            )
        )

    if resolved_run_dir is not None:
        prediction_checks, prediction_requests = _smoke_prediction_service(
            requested_run_dir=requested_run_dir,
            run_dir=resolved_run_dir,
            data_dir=data_root,
            model_name=effective_model_name,
            include_video=include_video,
            max_top_k=max_top_k,
            top_k=top_k,
            recent_artists=recent_artists,
            auth_token=auth_token,
            require_serving_bundle=require_serving_bundle,
            logger=logger,
        )
        checks.extend(prediction_checks)
        requests.extend(prediction_requests)

        taste_checks, taste_requests = _smoke_taste_os_service(
            requested_run_dir=requested_run_dir,
            run_dir=resolved_run_dir,
            data_dir=data_root,
            service_output_dir=service_root,
            state_db_path=resolved_state_db_path,
            state_database_url=state_database_url,
            model_name=effective_model_name,
            include_video=include_video,
            max_top_k=max_top_k,
            top_k=top_k,
            recent_artists=recent_artists,
            mode=mode,
            scenario=scenario,
            auth_token=auth_token,
            require_serving_bundle=require_serving_bundle,
            logger=logger,
        )
        checks.extend(taste_checks)
        requests.extend(taste_requests)

    summary = _summarize(checks, requests)
    blocker_rows = _blockers(checks, requests)
    generated_at = _utc_now_iso()
    payload: dict[str, object] = {
        "generated_at": generated_at,
        "project_root": str(project_root),
        "outputs_dir": str(outputs_dir),
        "requested_run_dir": str(requested_run_dir),
        "run_dir": str(resolved_run_dir or ""),
        "model_name": effective_model_name,
        "top_k": int(top_k),
        "mode": str(mode),
        "scenario": str(scenario),
        "include_video": bool(include_video),
        "require_serving_bundle": bool(require_serving_bundle),
        "summary": summary,
        "blockers": blocker_rows,
        "checks": [check.as_dict() for check in checks],
        "requests": requests,
    }
    history_paths, trend_summary = _write_history_artifacts(payload=payload, outputs_dir=outputs_dir, output_root=output_root)
    payload["trend_summary"] = trend_summary
    paths = {
        "json": str(write_json(output_root / "production_smoke.json", payload, sort_keys=True)),
        "md": str(write_markdown(output_root / "production_smoke.md", _markdown(payload))),
        "checks_csv": str(write_csv_rows(output_root / "production_smoke_checks.csv", [check.as_dict() for check in checks], fieldnames=CHECK_COLUMNS)),
        "requests_csv": str(write_csv_rows(output_root / "production_smoke_requests.csv", _request_csv_rows(requests), fieldnames=REQUEST_COLUMNS)),
        **history_paths,
    }
    manifest = {
        "generated_at": generated_at,
        "status": summary["status"],
        "production_ready": summary["production_ready"],
        "paths": paths,
        "trend_summary": trend_summary,
    }
    paths["manifest_json"] = str(write_json(output_root / "production_smoke_manifest.json", manifest, sort_keys=True))
    payload["paths"] = paths
    logger.info("Wrote production smoke evidence to %s", output_root)
    return payload


def _parse_recent_artists(value: str | None) -> list[str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split("|") if part.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.production_smoke",
        description="Exercise production-shaped ASGI routes and write release smoke evidence.",
    )
    parser.add_argument("--project-root", type=str, default=".", help="Repository root.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs root for generated smoke artifacts.")
    parser.add_argument("--run-dir", type=str, default=DEFAULT_CHANNEL_RUN_DIR, help="Run directory or registry channel alias path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Production smoke artifact directory.")
    parser.add_argument("--service-output-dir", type=str, default=None, help="Taste OS service artifact/state directory used during smoke.")
    parser.add_argument("--data-dir", type=str, default=os.getenv("DATA_DIR", ""), help="Optional raw history data directory fallback.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional serveable model name override.")
    parser.add_argument("--top-k", type=int, default=3, help="top_k for smoke requests.")
    parser.add_argument("--max-top-k", type=int, default=5, help="Maximum top_k allowed by in-process services.")
    parser.add_argument("--recent-artists", type=str, default="", help="Optional pipe-separated recent artist override.")
    parser.add_argument("--mode", type=str, default="focus", help="Taste OS mode for the session smoke.")
    parser.add_argument("--scenario", type=str, default="steady", help="Taste OS scenario for the session smoke.")
    parser.add_argument("--include-video", action="store_true", help="Use audio+video serving context.")
    parser.add_argument(
        "--auth-token",
        type=str,
        default=os.getenv("SPOTIFY_SERVICE_AUTH_TOKEN", ""),
        help="Optional token used to protect and call mutation routes.",
    )
    parser.add_argument("--state-db", type=str, default="", help="Optional SQLite state path for Taste OS smoke.")
    parser.add_argument("--state-db-url", type=str, default="", help="Optional SQLAlchemy state DB URL for Taste OS smoke.")
    parser.add_argument(
        "--no-require-serving-bundle",
        action="store_true",
        help="Allow raw-data fallback instead of requiring a materialized serving bundle.",
    )
    parser.add_argument("--strict", action="store_true", help="Return nonzero unless all smoke checks pass.")
    parser.add_argument("--stdout-format", choices=("summary", "json"), default="summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    payload = build_production_smoke(
        project_root=Path(args.project_root),
        outputs_dir=Path(args.outputs_dir),
        run_dir=args.run_dir,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        service_output_dir=Path(args.service_output_dir) if args.service_output_dir else None,
        data_dir=Path(args.data_dir) if str(args.data_dir).strip() else None,
        model_name=args.model_name,
        top_k=max(1, int(args.top_k)),
        max_top_k=max(1, int(args.max_top_k)),
        recent_artists=_parse_recent_artists(args.recent_artists),
        mode=str(args.mode),
        scenario=str(args.scenario),
        include_video=bool(args.include_video),
        auth_token=str(args.auth_token).strip() or None,
        state_db_path=Path(args.state_db) if str(args.state_db).strip() else None,
        state_database_url=str(args.state_db_url).strip() or None,
        require_serving_bundle=not bool(args.no_require_serving_bundle),
        logger=logging.getLogger("spotify.production_smoke"),
    )
    summary = payload.get("summary", {})
    summary_dict = summary if isinstance(summary, dict) else {}
    paths = payload.get("paths", {})
    paths_dict = paths if isinstance(paths, dict) else {}
    if args.stdout_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"production_smoke_status={summary_dict.get('status', '')}")
        print(f"production_ready={summary_dict.get('production_ready', False)}")
        print(f"production_smoke_report={paths_dict.get('md', '')}")
        print(f"production_smoke_history={paths_dict.get('history_csv', '')}")
    if args.strict and summary_dict.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
