from __future__ import annotations

import numpy as np

from spotify.digital_twin import ListenerDigitalTwinArtifact, simulate_rollout, simulate_rollout_batch_summary
from spotify.multimodal import MultimodalArtistSpace


class _StaticEndEstimator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        row_count = len(np.asarray(X))
        return np.column_stack(
            [
                np.ones(row_count, dtype="float32"),
                np.zeros(row_count, dtype="float32"),
            ]
        )


class _LastArtistSensitiveEndEstimator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype="float32")
        last_artist = np.clip(features[:, 0], 0.0, 2.0)
        risk = np.clip(0.05 + (0.25 * last_artist), 0.0, 0.95).astype("float32", copy=False)
        return np.column_stack([1.0 - risk, risk]).astype("float32", copy=False)


def test_simulate_rollout_batch_summary_matches_single_rollouts_without_early_end() -> None:
    twin = ListenerDigitalTwinArtifact(
        artist_labels=["A", "B", "C"],
        transition_matrix=np.array(
            [
                [0.10, 0.80, 0.10],
                [0.20, 0.10, 0.70],
                [0.60, 0.30, 0.10],
            ],
            dtype="float32",
        ),
        end_estimator=_StaticEndEstimator(),
        context_features=["hour", "offline"],
        average_track_seconds=180.0,
    )
    space = MultimodalArtistSpace(
        artist_labels=["A", "B", "C"],
        feature_names=["f0", "f1"],
        raw_features=np.zeros((3, 2), dtype="float32"),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.2, 0.8],
            ],
            dtype="float32",
        ),
        popularity=np.array([0.6, 0.3, 0.1], dtype="float32"),
        energy=np.array([0.4, 0.5, 0.7], dtype="float32"),
        danceability=np.array([0.5, 0.6, 0.8], dtype="float32"),
        tempo=np.array([100.0, 110.0, 130.0], dtype="float32"),
    )
    sequences = np.array([[0, 1], [1, 2], [2, 0]], dtype="int32")
    contexts = np.array([[8.0, 0.0], [9.0, 1.0], [10.0, 0.0]], dtype="float32")
    policy_weights = {"transition": 1.0, "continuity": 0.3, "novelty": 0.2, "repeat": 0.7}
    scenario = {"hour_shift": 2.0, "friction_scale": 1.5}

    batch_summary = simulate_rollout_batch_summary(
        twin=twin,
        multimodal_space=space,
        causal_artifact=None,
        start_sequences=sequences,
        start_contexts=contexts,
        horizon=4,
        policy_weights=policy_weights,
        scenario=scenario,
        rng=np.random.default_rng(11),
    )

    single_rollouts = [
        simulate_rollout(
            twin=twin,
            multimodal_space=space,
            causal_artifact=None,
            start_sequence=sequence,
            start_context=context,
            horizon=4,
            policy_weights=policy_weights,
            scenario=scenario,
            rng=np.random.default_rng(11),
        )
        for sequence, context in zip(sequences, contexts, strict=False)
    ]

    assert np.array_equal(
        np.asarray(batch_summary["session_length"], dtype="int32"),
        np.asarray([row["session_length"] for row in single_rollouts], dtype="int32"),
    )
    assert np.allclose(
        np.asarray(batch_summary["mean_skip_risk"], dtype="float32"),
        np.asarray([row["mean_skip_risk"] for row in single_rollouts], dtype="float32"),
    )
    assert np.allclose(
        np.asarray(batch_summary["mean_end_risk"], dtype="float32"),
        np.asarray([row["mean_end_risk"] for row in single_rollouts], dtype="float32"),
    )
    assert np.array_equal(
        np.asarray(batch_summary["first_choice"], dtype="int32"),
        np.asarray([row["planned_sequence"][0] for row in single_rollouts], dtype="int32"),
    )


def test_simulate_rollout_uses_post_choice_sequence_for_risk_scoring() -> None:
    twin = ListenerDigitalTwinArtifact(
        artist_labels=["A", "B", "C"],
        transition_matrix=np.array(
            [
                [0.20, 0.70, 0.10],
                [0.60, 0.20, 0.20],
                [0.10, 0.10, 0.80],
            ],
            dtype="float32",
        ),
        end_estimator=_LastArtistSensitiveEndEstimator(),
        context_features=["hour", "session_position", "session_repeat_ratio_so_far", "is_artist_repeat_from_prev"],
        average_track_seconds=180.0,
    )
    space = MultimodalArtistSpace(
        artist_labels=["A", "B", "C"],
        feature_names=["f0", "f1"],
        raw_features=np.zeros((3, 2), dtype="float32"),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
            dtype="float32",
        ),
        popularity=np.array([0.8, 0.5, 0.1], dtype="float32"),
        energy=np.array([0.4, 0.5, 0.7], dtype="float32"),
        danceability=np.array([0.5, 0.6, 0.8], dtype="float32"),
        tempo=np.array([100.0, 110.0, 130.0], dtype="float32"),
    )
    sequence = np.array([0, 1], dtype="int32")
    context = np.array([8.0, 1.0, 0.0, 0.0], dtype="float32")

    transition_heavy = simulate_rollout(
        twin=twin,
        multimodal_space=space,
        causal_artifact=None,
        start_sequence=sequence,
        start_context=context,
        horizon=1,
        policy_weights={"transition": 1.2, "continuity": 0.0, "novelty": 0.0, "repeat": 0.2},
        scenario=None,
        rng=np.random.default_rng(5),
    )
    novelty_heavy = simulate_rollout(
        twin=twin,
        multimodal_space=space,
        causal_artifact=None,
        start_sequence=sequence,
        start_context=context,
        horizon=1,
        policy_weights={"transition": 0.0, "continuity": 0.2, "novelty": 1.5, "repeat": 0.0},
        scenario=None,
        rng=np.random.default_rng(5),
    )

    assert transition_heavy["planned_sequence"][0] == 0
    assert novelty_heavy["planned_sequence"][0] == 2
    assert transition_heavy["mean_end_risk"] < novelty_heavy["mean_end_risk"]
