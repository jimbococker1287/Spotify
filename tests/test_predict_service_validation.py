from __future__ import annotations

import pytest

from spotify.predict_service import RequestValidationError, is_authorized_request, normalize_predict_payload


def test_normalize_predict_payload_accepts_supported_fields() -> None:
    payload = {
        "top_k": 5,
        "include_video": True,
        "recent_artists": "Artist A|Artist B|Artist C",
    }

    normalized = normalize_predict_payload(payload, default_include_video=False, max_top_k=10)

    assert normalized["top_k"] == 5
    assert normalized["include_video"] is True
    assert normalized["recent_artists"] == ["Artist A", "Artist B", "Artist C"]


def test_normalize_predict_payload_rejects_top_k_above_limit() -> None:
    with pytest.raises(RequestValidationError) as exc:
        normalize_predict_payload({"top_k": 25}, default_include_video=False, max_top_k=10)

    assert exc.value.code == "invalid_top_k"


def test_normalize_predict_payload_rejects_unknown_field() -> None:
    with pytest.raises(RequestValidationError) as exc:
        normalize_predict_payload({"top_k": 3, "foo": "bar"}, default_include_video=False, max_top_k=10)

    assert exc.value.code == "unknown_fields"


def test_is_authorized_request_supports_bearer_and_api_key() -> None:
    headers_bearer = {"Authorization": "Bearer token-123"}
    headers_api_key = {"X-API-Key": "token-123"}
    headers_invalid = {"Authorization": "Bearer wrong"}

    assert is_authorized_request(headers_bearer, "token-123") is True
    assert is_authorized_request(headers_api_key, "token-123") is True
    assert is_authorized_request(headers_invalid, "token-123") is False
