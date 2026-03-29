from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

import spotify.taste_os_service as taste_os_service
from spotify.digital_twin import ListenerDigitalTwinArtifact
from spotify.multimodal import MultimodalArtistSpace
from spotify.predict_next import PredictionInputContext
from spotify.safe_policy import SafeBanditPolicyArtifact
from spotify.taste_os_service import (
    RequestValidationError,
    TasteOSService,
    _taste_os_page_html,
    normalize_taste_os_feedback_payload,
    normalize_taste_os_payload,
)


class _StubPredictor:
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
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


def test_normalize_taste_os_payload_accepts_supported_fields() -> None:
    payload = {
        "mode": "discovery",
        "scenario": "skip_recovery",
        "top_k": 5,
        "include_video": True,
        "recent_artists": "Artist A|Artist B",
        "persist_artifacts": True,
        "use_feedback_memory": False,
    }

    normalized = normalize_taste_os_payload(payload, default_include_video=False, max_top_k=10)

    assert normalized["mode"] == "discovery"
    assert normalized["scenario"] == "skip_recovery"
    assert normalized["top_k"] == 5
    assert normalized["include_video"] is True
    assert normalized["recent_artists"] == ["Artist A", "Artist B"]
    assert normalized["persist_artifacts"] is True
    assert normalized["use_feedback_memory"] is False


def test_normalize_taste_os_payload_rejects_invalid_mode() -> None:
    with pytest.raises(RequestValidationError) as exc:
        normalize_taste_os_payload({"mode": "party"}, default_include_video=False, max_top_k=10)

    assert exc.value.code == "invalid_mode"


def test_normalize_taste_os_feedback_payload_accepts_supported_fields() -> None:
    normalized = normalize_taste_os_feedback_payload(
        {
            "session_id": "taste-os-123",
            "artist_name": "Artist C",
            "signal": "repeat",
            "notes": "good fit",
        }
    )

    assert normalized["session_id"] == "taste-os-123"
    assert normalized["artist_name"] == "Artist C"
    assert normalized["signal"] == "repeat"
    assert normalized["notes"] == "good fit"


def test_normalize_taste_os_feedback_payload_rejects_unknown_signal() -> None:
    with pytest.raises(RequestValidationError) as exc:
        normalize_taste_os_feedback_payload(
            {
                "session_id": "taste-os-123",
                "artist_name": "Artist C",
                "signal": "love",
            }
        )

    assert exc.value.code == "invalid_signal"


def test_taste_os_service_plans_session_and_persists_artifacts(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    history_file = data_dir / "Streaming_History_Audio_2024_0.json"
    history_file.write_text("[]", encoding="utf-8")
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["Artist A", "Artist B", "Artist C", "Artist D"], "sequence_length": 2}),
        encoding="utf-8",
    )

    load_calls = {"count": 0}

    def _fake_context_loader(run_dir: Path, data_dir: Path, include_video: bool, logger: logging.Logger) -> PredictionInputContext:
        _ = (run_dir, data_dir, include_video, logger)
        load_calls["count"] += 1
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

    monkeypatch.setattr(
        taste_os_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(taste_os_service, "load_predictor", lambda run_dir, row, artist_labels: _StubPredictor())
    monkeypatch.setattr(taste_os_service, "load_prediction_input_context", _fake_context_loader)

    logger = logging.getLogger("spotify.test.taste_os_service")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs" / "analysis" / "taste_os_service",
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
    )

    result = service.plan_session(
        mode="focus",
        scenario="skip_recovery",
        top_k=3,
        recent_artists=None,
        include_video=False,
        persist_artifacts=True,
        use_feedback_memory=True,
    )

    assert load_calls["count"] == 1
    assert result["request"]["mode"] == "focus"
    assert result["adaptive_session"]["scenario"] == "skip_recovery"
    assert result["service"]["persisted"] is True
    assert "session_id" in result["service"]
    assert str(result["service"]["artifact_json_url"]).startswith("/taste-os/artifacts/")
    assert str(result["service"]["artifact_md_url"]).startswith("/taste-os/artifacts/")
    assert Path(str(result["service"]["artifact_json"])).exists()
    assert Path(str(result["service"]["artifact_md"])).exists()

    repeat = service.plan_session(
        mode="commute",
        scenario="steady",
        top_k=2,
        recent_artists=["Artist A", "Artist B"],
        include_video=False,
        persist_artifacts=False,
        use_feedback_memory=False,
    )

    assert load_calls["count"] == 1
    assert repeat["request"]["mode"] == "commute"
    assert repeat["service"]["persisted"] is False


def test_taste_os_service_feedback_memory_seeds_future_sessions(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "Streaming_History_Audio_2024_0.json").write_text("[]", encoding="utf-8")
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["Artist A", "Artist B", "Artist C", "Artist D"], "sequence_length": 2}),
        encoding="utf-8",
    )

    captured_recent_artists: list[list[str] | None] = []

    monkeypatch.setattr(
        taste_os_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(taste_os_service, "load_predictor", lambda run_dir, row, artist_labels: _StubPredictor())
    monkeypatch.setattr(
        taste_os_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger: PredictionInputContext(
            artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
            artist_to_label={"Artist A": 0, "Artist B": 1, "Artist C": 2, "Artist D": 3},
            sequence_length=2,
            latest_sequence_labels=np.array([0, 1], dtype="int32"),
            latest_sequence_names=["Artist A", "Artist B"],
            context_scaled=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
            context_raw=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
            context_features=["hour", "tech_playback_errors_24h", "offline"],
        ),
    )

    def _fake_prepare_inputs(
        *,
        run_dir: Path,
        data_dir: Path,
        recent_artists: list[str] | None,
        include_video: bool,
        logger: logging.Logger,
        context: PredictionInputContext,
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        _ = (run_dir, data_dir, include_video, logger, context)
        captured_recent_artists.append(list(recent_artists) if recent_artists is not None else None)
        return [np.array([[0, 1]], dtype="int32")], np.array([[8.0, 0.3, 0.0]], dtype="float32"), ["Artist A", "Artist B"]

    monkeypatch.setattr(taste_os_service, "_prepare_inputs", _fake_prepare_inputs)

    logger = logging.getLogger("spotify.test.taste_os_service.feedback")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs" / "analysis" / "taste_os_service",
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
    )

    feedback = service.record_feedback(
        session_id="taste-os-seed",
        artist_name="Artist D",
        signal="like",
        notes=None,
    )

    assert feedback["recorded"] is True

    result = service.plan_session(
        mode="discovery",
        scenario="steady",
        top_k=3,
        recent_artists=None,
        include_video=False,
        persist_artifacts=False,
        use_feedback_memory=True,
    )

    assert captured_recent_artists[-1] == ["Artist D"]
    assert result["request"]["used_feedback_memory"] is True
    assert result["request"]["effective_recent_artists"] == ["Artist D"]
    assert result["memory_summary"]["seed_artists"] == ["Artist D"]

    history = service.history_snapshot()
    assert history["recent_sessions"][0]["mode"] == "discovery"
    assert history["feedback_memory"]["top_affinities"][0]["artist_name"] == "Artist D"


def test_taste_os_service_page_html_exposes_browser_surface(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "Streaming_History_Audio_2024_0.json").write_text("[]", encoding="utf-8")
    (run_dir / "feature_metadata.json").write_text(
        json.dumps({"artist_labels": ["Artist A", "Artist B", "Artist C", "Artist D"], "sequence_length": 2}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        taste_os_service,
        "resolve_model_row",
        lambda run_dir, explicit_model_name, alias_model_name: {"model_name": "dummy", "model_type": "retrieval_reranker"},
    )
    monkeypatch.setattr(taste_os_service, "load_predictor", lambda run_dir, row, artist_labels: _StubPredictor())
    monkeypatch.setattr(
        taste_os_service,
        "load_prediction_input_context",
        lambda run_dir, data_dir, include_video, logger: PredictionInputContext(
            artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
            artist_to_label={"Artist A": 0, "Artist B": 1, "Artist C": 2, "Artist D": 3},
            sequence_length=2,
            latest_sequence_labels=np.array([0, 1], dtype="int32"),
            latest_sequence_names=["Artist A", "Artist B"],
            context_scaled=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
            context_raw=np.array([[8.0, 0.3, 0.0]], dtype="float32"),
            context_features=["hour", "tech_playback_errors_24h", "offline"],
        ),
    )

    logger = logging.getLogger("spotify.test.taste_os_service.page")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    service = TasteOSService(
        run_dir=run_dir,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs" / "analysis" / "taste_os_service",
        model_name="dummy",
        include_video=False,
        max_top_k=5,
        auth_token=None,
        logger=logger,
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
    )

    html = _taste_os_page_html(service)

    assert "Taste OS Session Studio" in html
    assert "/taste-os/session" in html
    assert "/taste-os/catalog" in html
    assert "/taste-os/history" in html
    assert "/taste-os/feedback" in html
    assert "Generate Session" in html
    assert "Seed from feedback memory" in html
