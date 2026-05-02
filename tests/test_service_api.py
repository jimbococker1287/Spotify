from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

import spotify.predict_service as predict_service
from spotify.digital_twin import ListenerDigitalTwinArtifact
from spotify.multimodal import MultimodalArtistSpace
from spotify.predict_next import PredictionInputContext
from spotify.predict_service import PredictionService
from spotify.safe_policy import SafeBanditPolicyArtifact
from spotify.service_api import create_prediction_app, create_taste_os_app
import spotify.taste_os_service as taste_os_service
from spotify.taste_os_service import TasteOSService


class _PredictStub:
    model_name = "dummy"
    model_type = "classical"

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        _ = (seq_batch, ctx_batch)
        return np.asarray([[0.6, 0.3, 0.1]], dtype="float32")


class _TasteStubPredictor:
    model_name = "stub_retrieval"
    model_type = "retrieval_reranker"

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        _ = (seq_batch, ctx_batch)
        return np.asarray([[0.10, 0.18, 0.52, 0.20]], dtype="float32")


class _StubEndEstimator:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        _ = features
        return np.asarray([[0.72, 0.28]], dtype="float32")


def _space() -> MultimodalArtistSpace:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype="float32",
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return MultimodalArtistSpace(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        feature_names=["f0", "f1"],
        raw_features=np.zeros((4, 2), dtype="float32"),
        embeddings=embeddings.astype("float32"),
        popularity=np.asarray([0.40, 0.32, 0.18, 0.10], dtype="float32"),
        energy=np.asarray([0.45, 0.55, 0.76, 0.88], dtype="float32"),
        danceability=np.asarray([0.40, 0.50, 0.72, 0.82], dtype="float32"),
        tempo=np.asarray([102.0, 110.0, 128.0, 134.0], dtype="float32"),
    )


def _twin() -> ListenerDigitalTwinArtifact:
    return ListenerDigitalTwinArtifact(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        transition_matrix=np.asarray(
            [
                [0.10, 0.45, 0.35, 0.10],
                [0.08, 0.12, 0.60, 0.20],
                [0.05, 0.15, 0.20, 0.60],
                [0.12, 0.18, 0.46, 0.24],
            ],
            dtype="float32",
        ),
        end_estimator=_StubEndEstimator(),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
        average_track_seconds=180.0,
    )


def _safe_policy() -> SafeBanditPolicyArtifact:
    return SafeBanditPolicyArtifact(
        policy_map={
            "high_friction": {"transition": 0.7, "continuity": 0.4, "novelty": 0.1, "repeat": 0.9},
            "normal_friction": {"transition": 0.9, "continuity": 0.3, "novelty": 0.2, "repeat": 0.7},
        },
        global_policy={"transition": 0.8, "continuity": 0.3, "novelty": 0.2, "repeat": 0.8},
        reward_metric="reward",
    )


def _prediction_context() -> PredictionInputContext:
    return PredictionInputContext(
        artist_labels=["A", "B", "C"],
        artist_to_label={"A": 0, "B": 1, "C": 2},
        sequence_length=2,
        latest_sequence_labels=np.array([0, 1], dtype="int32"),
        latest_sequence_names=["A", "B"],
        context_scaled=np.array([[1.0, 2.0]], dtype="float32"),
    )


def _taste_context() -> PredictionInputContext:
    return PredictionInputContext(
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        artist_to_label={"Artist A": 0, "Artist B": 1, "Artist C": 2, "Artist D": 3},
        sequence_length=2,
        latest_sequence_labels=np.array([0, 1], dtype="int32"),
        latest_sequence_names=["Artist A", "Artist B"],
        context_scaled=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
        context_raw=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
        context_features=["hour", "tech_playback_errors_24h", "offline"],
    )


def test_prediction_api_supports_auth_metrics_and_rate_limits(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["A", "B", "C"], "sequence_length": 2}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        predict_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "classical"},
    )
    monkeypatch.setattr(predict_service, "load_predictor", lambda run_dir, row, artist_labels: _PredictStub())
    monkeypatch.setattr(
        predict_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger, **kwargs: _prediction_context(),
    )

    logger = logging.getLogger("spotify.test.service_api.predict")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = PredictionService(
        run_dir=run_dir,
        data_dir=None,
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token="token-123",
        logger=logger,
        require_serving_bundle=False,
    )
    app = create_prediction_app(service=service, logger=logger, request_rate_limit=1)
    client = TestClient(app)

    unauthorized = client.post("/v1/predict", json={"top_k": 2})
    assert unauthorized.status_code == 401
    assert unauthorized.headers["x-request-id"]

    authorized = client.post(
        "/v1/predict",
        json={"top_k": 2},
        headers={"Authorization": "Bearer token-123"},
    )
    assert authorized.status_code == 200
    assert authorized.headers["x-request-id"]
    assert authorized.json()["predictions"][0]["artist_name"] == "A"

    limited = client.post(
        "/v1/predict",
        json={"top_k": 2},
        headers={"Authorization": "Bearer token-123"},
    )
    assert limited.status_code == 429

    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["telemetry"]["total_request_count"] >= 3


def test_taste_os_api_serves_ui_and_uses_durable_history(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["Artist A", "Artist B", "Artist C", "Artist D"], "sequence_length": 2}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        taste_os_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(taste_os_service, "load_predictor", lambda run_dir, row, artist_labels: _TasteStubPredictor())
    monkeypatch.setattr(
        taste_os_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger, **kwargs: _taste_context(),
    )

    logger = logging.getLogger("spotify.test.service_api.taste_os")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=None,
        output_dir=tmp_path / "outputs" / "analysis" / "taste_os_service",
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
        state_db_path=tmp_path / "outputs" / "analysis" / "taste_os_service" / "taste_os_state.sqlite3",
        require_serving_bundle=False,
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
    )
    app = create_taste_os_app(service=service, logger=logger, request_rate_limit=10)
    client = TestClient(app)

    page = client.get("/taste-os")
    assert page.status_code == 200
    assert "Taste OS Session Studio" in page.text
    assert page.headers["x-request-id"]

    session = client.post(
        "/v1/taste-os/session",
        json={"mode": "focus", "scenario": "skip_recovery", "top_k": 3, "use_feedback_memory": True},
    )
    assert session.status_code == 200
    session_id = session.json()["service"]["session_id"]

    feedback = client.post(
        "/v1/taste-os/feedback",
        json={"session_id": session_id, "artist_name": "Artist C", "signal": "like"},
    )
    assert feedback.status_code == 200
    assert feedback.json()["feedback_memory"]["event_count"] >= 1

    history = client.get("/v1/taste-os/history")
    assert history.status_code == 200
    assert history.json()["recent_sessions"][0]["session_id"] == session_id

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["telemetry"]["total_request_count"] >= 4
