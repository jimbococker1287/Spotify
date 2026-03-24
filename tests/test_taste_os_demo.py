from __future__ import annotations

import numpy as np

from spotify.digital_twin import ListenerDigitalTwinArtifact
from spotify.multimodal import MultimodalArtistSpace
from spotify.safe_policy import SafeBanditPolicyArtifact
from spotify.taste_os_demo import MODE_CONFIGS, build_taste_os_demo_payload


class _StubPredictor:
    model_name = "stub_retrieval"
    model_type = "retrieval_reranker"
    artist_labels = ["Artist A", "Artist B", "Artist C", "Artist D"]

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


def test_build_taste_os_demo_payload_returns_contract_sections() -> None:
    payload = build_taste_os_demo_payload(
        predictor=_StubPredictor(),
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        sequence_labels=np.asarray([0, 1], dtype="int32"),
        sequence_names=["Artist A", "Artist B"],
        context_batch=np.asarray([[8.0, 0.3, 0.0]], dtype="float32"),
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
        mode_name="focus",
        top_k=3,
        artifact_paths={"multimodal_space": "/tmp/mm.joblib"},
    )

    assert payload["request"]["mode"] == "focus"
    assert payload["current_session"]["model_name"] == "stub_retrieval"
    assert len(payload["top_candidates"]) == 3
    assert len(payload["journey_plan"]) == MODE_CONFIGS["focus"].horizon
    assert payload["why_this_next"]
    assert payload["fallback_policy"]["active_policy_name"] in {"comfort_policy", "safe_global", "safe_bucket_high_friction"}
    assert payload["artifacts_used"]["multimodal_space"] == "/tmp/mm.joblib"


def test_discovery_mode_increases_novelty_priority() -> None:
    focus_payload = build_taste_os_demo_payload(
        predictor=_StubPredictor(),
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        sequence_labels=np.asarray([0, 1], dtype="int32"),
        sequence_names=["Artist A", "Artist B"],
        context_batch=np.asarray([[9.0, 0.0, 0.0]], dtype="float32"),
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
        mode_name="focus",
        top_k=2,
    )
    discovery_payload = build_taste_os_demo_payload(
        predictor=_StubPredictor(),
        artist_labels=["Artist A", "Artist B", "Artist C", "Artist D"],
        sequence_labels=np.asarray([0, 1], dtype="int32"),
        sequence_names=["Artist A", "Artist B"],
        context_batch=np.asarray([[9.0, 0.0, 0.0]], dtype="float32"),
        digital_twin=_twin(),
        multimodal_space=_space(),
        safe_policy=_safe_policy(),
        mode_name="discovery",
        top_k=2,
    )

    focus_top = focus_payload["top_candidates"][0]["artist_name"]
    discovery_top = discovery_payload["top_candidates"][0]["artist_name"]
    focus_novelty = float(focus_payload["top_candidates"][0]["novelty"])
    discovery_novelty = float(discovery_payload["top_candidates"][0]["novelty"])

    assert focus_payload["mode"]["planned_horizon"] == MODE_CONFIGS["focus"].horizon
    assert discovery_payload["mode"]["planned_horizon"] == MODE_CONFIGS["discovery"].horizon
    assert focus_top == "Artist C"
    assert discovery_top in {"Artist C", "Artist D"}
    assert discovery_novelty >= focus_novelty
