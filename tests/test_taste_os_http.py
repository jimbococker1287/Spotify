from __future__ import annotations

import http.client
from http.server import ThreadingHTTPServer
import json
import logging
import threading

from spotify.taste_os_http import RequestValidationError, build_taste_os_handler


class _StubTasteOSService:
    auth_token = None
    include_video = False
    max_top_k = 10

    def __init__(self) -> None:
        self.logger = logging.getLogger("spotify.test.taste_os_http")
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())
        self.calls: list[dict[str, object]] = []

    def apply_session_event(self, **kwargs) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        if kwargs["event_id"] == "stale-event":
            raise RequestValidationError(
                status_code=409,
                code="stale_session_version",
                message="Session version 0 is stale; current version is 1.",
                details={"expected_version": 0, "current_version": 1},
            )
        return {
            "service": {
                "session_id": kwargs["session_id"],
                "version": int(kwargs["expected_version"]) + 1,
            },
            "why_this_changed": "The session replanned after your skip.",
        }

    def session_snapshot(self, session_id: str) -> dict[str, object]:
        self.calls.append({"session_id": session_id, "method": "session_snapshot"})
        if session_id == "taste-os-missing":
            raise RequestValidationError(
                status_code=404,
                code="session_not_found",
                message=f"Active Taste OS session not found: {session_id}",
            )
        return {
            "session_id": session_id,
            "version": 2,
            "plan": {
                "service": {
                    "session_id": session_id,
                    "version": 2,
                }
            },
            "adaptation_summary": {
                "event_count": 2,
            },
            "events": [
                {"event_id": "event-1", "resulting_version": 1},
                {"event_id": "event-2", "resulting_version": 2},
            ],
        }


def _json_request(
    port: int,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        request_headers = dict(headers or {})
        body = None
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
            body = json.dumps(payload)
        connection.request(
            method,
            path,
            body=body,
            headers=request_headers,
        )
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body
    finally:
        connection.close()


def _post_json(port: int, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    return _json_request(port, "POST", path, payload)


def test_taste_os_http_session_event_route_and_stale_version_error() -> None:
    service = _StubTasteOSService()
    handler = build_taste_os_handler(service, page_renderer=lambda _: "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        status, body = _post_json(
            port,
            "/taste-os/session/event",
            {
                "session_id": "taste-os-123",
                "event_id": "event-1",
                "event_type": "skip",
                "artist_name": "Artist C",
                "expected_version": 0,
            },
        )
        assert status == 200
        assert body["service"]["session_id"] == "taste-os-123"
        assert body["service"]["version"] == 1
        assert service.calls[0]["event_type"] == "skip"

        stale_status, stale_body = _post_json(
            port,
            "/taste-os/session/event",
            {
                "session_id": "taste-os-123",
                "event_id": "stale-event",
                "event_type": "repeat",
                "artist_name": "Artist B",
                "expected_version": 0,
            },
        )
        assert stale_status == 409
        assert stale_body["error"]["code"] == "stale_session_version"
        assert stale_body["error"]["details"]["current_version"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_taste_os_http_session_snapshot_route_and_not_found_response() -> None:
    service = _StubTasteOSService()
    handler = build_taste_os_handler(service, page_renderer=lambda _: "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        status, body = _json_request(port, "GET", "/taste-os/session/taste-os%2D123")
        assert status == 200
        assert body["session_id"] == "taste-os-123"
        assert body["version"] == 2
        assert body["plan"]["service"]["version"] == 2
        assert [event["event_id"] for event in body["events"]] == ["event-1", "event-2"]

        missing_status, missing_body = _json_request(
            port,
            "GET",
            "/taste-os/session/taste-os-missing",
        )
        assert missing_status == 404
        assert missing_body["error"]["code"] == "session_not_found"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_taste_os_http_keeps_post_auth_behavior_for_session_routes() -> None:
    service = _StubTasteOSService()
    service.auth_token = "token-123"
    handler = build_taste_os_handler(service, page_renderer=lambda _: "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        snapshot_status, _ = _json_request(port, "GET", "/taste-os/session/taste-os-123")
        assert snapshot_status == 200

        unauthorized_status, unauthorized_body = _post_json(
            port,
            "/taste-os/session/event",
            {
                "session_id": "taste-os-123",
                "event_id": "event-1",
                "event_type": "skip",
                "artist_name": "Artist C",
                "expected_version": 0,
            },
        )
        assert unauthorized_status == 401
        assert unauthorized_body["error"]["code"] == "unauthorized"

        authorized_status, _ = _json_request(
            port,
            "POST",
            "/taste-os/session/event",
            {
                "session_id": "taste-os-123",
                "event_id": "event-1",
                "event_type": "skip",
                "artist_name": "Artist C",
                "expected_version": 0,
            },
            headers={"Authorization": "Bearer token-123"},
        )
        assert authorized_status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
