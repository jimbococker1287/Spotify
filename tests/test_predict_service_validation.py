from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

import spotify.predict_service as predict_service
from spotify.predict_next import PredictionInputContext
from spotify.predict_service import PredictionService, RequestValidationError, is_authorized_request, normalize_predict_payload


def test_normalize_predict_payload_accepts_supported_fields() -> None:
    payload = {
        "top_k": 5,
        "include_video": True,
        "recent_artists": "Artist A|Artist B|Artist C",
        "allow_abstain": True,
        "return_prediction_set": True,
    }

    normalized = normalize_predict_payload(payload, default_include_video=False, max_top_k=10)

    assert normalized["top_k"] == 5
    assert normalized["include_video"] is True
    assert normalized["recent_artists"] == ["Artist A", "Artist B", "Artist C"]
    assert normalized["allow_abstain"] is True
    assert normalized["return_prediction_set"] is True


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


def test_prediction_service_reuses_cached_context_until_source_files_change(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    history_file = data_dir / "Streaming_History_Audio_2024_0.json"
    history_file.write_text("[]", encoding="utf-8")

    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["A", "B", "C"], "sequence_length": 2}),
        encoding="utf-8",
    )

    class _DummyPredictor:
        model_name = "dummy"
        model_type = "classical"

        def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
            return np.array([[0.7, 0.2, 0.1]], dtype="float32")

    load_calls = {"count": 0}

    def _fake_context_loader(run_dir: Path, data_dir: Path, include_video: bool, logger: logging.Logger) -> PredictionInputContext:
        load_calls["count"] += 1
        return PredictionInputContext(
            artist_labels=["A", "B", "C"],
            artist_to_label={"A": 0, "B": 1, "C": 2},
            sequence_length=2,
            latest_sequence_labels=np.array([0, 1], dtype="int32"),
            latest_sequence_names=["A", "B"],
            context_scaled=np.array([[1.0, 2.0]], dtype="float32"),
        )

    monkeypatch.setattr(predict_service, "resolve_model_row", lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy"})
    monkeypatch.setattr(predict_service, "load_predictor", lambda run_dir, row, artist_labels: _DummyPredictor())
    monkeypatch.setattr(predict_service, "load_prediction_input_context", _fake_context_loader)

    logger = logging.getLogger("spotify.test.predict_service")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = PredictionService(
        run_dir=run_dir,
        data_dir=data_dir,
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
    )

    first = service.predict(top_k=2, recent_artists=None, include_video=False)
    second = service.predict(top_k=2, recent_artists=None, include_video=False)

    assert load_calls["count"] == 1
    assert [row["artist_name"] for row in first["predictions"]] == ["A", "B"]
    assert [row["artist_name"] for row in second["predictions"]] == ["A", "B"]

    history_file.write_text("[{}]", encoding="utf-8")
    third = service.predict(top_k=2, recent_artists=None, include_video=False)

    assert load_calls["count"] == 2
    assert [row["artist_name"] for row in third["predictions"]] == ["A", "B"]


def test_prediction_service_can_abstain_with_conformal_summary(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "Streaming_History_Audio_2024_0.json").write_text("[]", encoding="utf-8")

    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["A", "B", "C"], "sequence_length": 2}),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "deep_dummy_conformal_summary.json").write_text(
        json.dumps(
            {
                "tag": "deep_dummy",
                "calibration": {
                    "method": "lac",
                    "alpha": 0.1,
                    "qhat": 0.35,
                    "threshold": 0.65,
                    "sample_count": 10,
                    "empirical_coverage": 0.9,
                    "mean_set_size": 1.4,
                },
            }
        ),
        encoding="utf-8",
    )

    class _DummyPredictor:
        model_name = "dummy"
        model_type = "deep"

        def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
            return np.array([[0.55, 0.30, 0.15]], dtype="float32")

    monkeypatch.setattr(
        predict_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "deep"},
    )
    monkeypatch.setattr(predict_service, "load_predictor", lambda run_dir, row, artist_labels: _DummyPredictor())
    monkeypatch.setattr(
        predict_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger: PredictionInputContext(
            artist_labels=["A", "B", "C"],
            artist_to_label={"A": 0, "B": 1, "C": 2},
            sequence_length=2,
            latest_sequence_labels=np.array([0, 1], dtype="int32"),
            latest_sequence_names=["A", "B"],
            context_scaled=np.array([[1.0, 2.0]], dtype="float32"),
        ),
    )

    logger = logging.getLogger("spotify.test.predict_service.conformal")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = PredictionService(
        run_dir=run_dir,
        data_dir=data_dir,
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
    )

    result = service.predict(
        top_k=2,
        recent_artists=None,
        include_video=False,
        allow_abstain=True,
        return_prediction_set=True,
    )

    assert result["decision"] == "abstain"
    assert result["uncertainty"]["conformal_enabled"] is True
    assert result["uncertainty"]["would_abstain"] is True
    assert result["uncertainty"]["abstained"] is True
    assert result["uncertainty"]["prediction_set_size"] == 0
