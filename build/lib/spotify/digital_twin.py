from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import build_serving_tabular_features
from .causal_friction import CausalSkipDecompositionArtifact
from .data import PreparedData
from .multimodal import MultimodalArtistSpace
from .run_artifacts import write_csv_rows


@dataclass(frozen=True)
class ListenerDigitalTwinArtifact:
    artist_labels: list[str]
    transition_matrix: np.ndarray
    end_estimator: object
    context_features: list[str]
    average_track_seconds: float


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    return write_csv_rows(path, rows, fieldnames=fieldnames)


def _aligned_frames(data: PreparedData, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = data.df.sort_values("ts").reset_index(drop=True)
    n_train = len(data.X_seq_train)
    n_val = len(data.X_seq_val)
    n_test = len(data.X_seq_test)
    n_total = n_train + n_val + n_test
    aligned = ordered.iloc[sequence_length : sequence_length + n_total].reset_index(drop=True)
    if len(aligned) < n_total:
        aligned = aligned.reindex(range(n_total)).fillna(0.0)
    train = aligned.iloc[:n_train].reset_index(drop=True)
    val = aligned.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = aligned.iloc[n_train + n_val : n_train + n_val + n_test].reset_index(drop=True)
    return train, val, test


def _end_labels(frame) -> np.ndarray:
    if "session_id" not in frame.columns:
        return np.zeros(len(frame), dtype="int32")
    session_ids = frame["session_id"].to_numpy(copy=False)
    shifted = np.roll(session_ids, -1)
    end = (shifted != session_ids).astype("int32")
    if len(end):
        end[-1] = 1
    return end


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(value, dtype="float64")
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -18.0, 18.0)))


def _decision_scores_batch(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype="float32").reshape(-1)
    proba = np.asarray(estimator.predict_proba(X), dtype="float32")
    clipped = np.clip(proba[:, 1], 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype("float32", copy=False)


_SEQUENCE_FEATURE_COUNT = 9
_ROLLOUT_STATIC_CACHE: dict[
    tuple[int, int, int, tuple[str, ...]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = {}


def _rollout_static_inputs(
    twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_key = (
        id(twin.transition_matrix),
        id(multimodal_space.embeddings),
        id(multimodal_space.popularity),
        tuple(str(name) for name in twin.context_features),
    )
    cached = _ROLLOUT_STATIC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    artist_ids = np.arange(len(twin.artist_labels), dtype="int32")
    transition_log = np.log(np.clip(np.asarray(twin.transition_matrix, dtype="float32"), 1e-6, 1.0)).astype("float32", copy=False)
    embeddings = np.asarray(multimodal_space.embeddings, dtype="float32")
    similarity_matrix = (embeddings @ embeddings.T).astype("float32", copy=False)
    novelty = (1.0 - np.asarray(multimodal_space.popularity, dtype="float32")).astype("float32", copy=False)
    friction_indices = np.asarray(
        [idx for idx, name in enumerate(twin.context_features) if str(name).startswith("tech_") or str(name) == "offline"],
        dtype="int64",
    )
    cached = (artist_ids, transition_log, similarity_matrix, novelty, friction_indices)
    _ROLLOUT_STATIC_CACHE[cache_key] = cached
    return cached


def _causal_feature_views(
    context_width: int,
    causal_artifact: CausalSkipDecompositionArtifact,
) -> tuple[np.ndarray, np.ndarray, bool]:
    total_features = _SEQUENCE_FEATURE_COUNT + int(context_width)
    friction_keep = _SEQUENCE_FEATURE_COUNT + np.asarray(causal_artifact.friction_feature_indices, dtype="int64")
    needs_friction_padding = friction_keep.size == 0
    if needs_friction_padding:
        friction_keep = np.asarray([total_features], dtype="int64")
        total_features += 1
    keep_mask = np.ones(total_features, dtype=bool)
    keep_mask[friction_keep] = False
    preference_keep = np.flatnonzero(keep_mask).astype("int64", copy=False)
    return preference_keep, friction_keep, needs_friction_padding


def _context_feature_index_map(context_features: list[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(context_features)}


def _unique_count_per_row(sequences: np.ndarray) -> np.ndarray:
    seq_arr = np.asarray(sequences, dtype="int32")
    if seq_arr.ndim != 2 or seq_arr.shape[1] <= 0:
        return np.zeros(len(seq_arr), dtype="float32")
    sorted_seq = np.sort(seq_arr, axis=1)
    unique_counts = np.ones(len(seq_arr), dtype="float32")
    if seq_arr.shape[1] > 1:
        unique_counts += np.count_nonzero(sorted_seq[:, 1:] != sorted_seq[:, :-1], axis=1).astype("float32")
    return unique_counts


def _recent_unique_ratio(sequences: np.ndarray, width: int) -> np.ndarray:
    seq_arr = np.asarray(sequences, dtype="int32")
    if seq_arr.ndim != 2 or seq_arr.shape[1] <= 0:
        return np.zeros(len(seq_arr), dtype="float32")
    resolved_width = min(max(1, int(width)), seq_arr.shape[1])
    recent = seq_arr[:, -resolved_width:]
    unique_counts = _unique_count_per_row(recent)
    return (unique_counts / float(resolved_width)).astype("float32", copy=False)


def _plays_since_last_occurrence(sequences: np.ndarray, targets: np.ndarray) -> np.ndarray:
    seq_arr = np.asarray(sequences, dtype="int32")
    target_arr = np.asarray(targets, dtype="int32").reshape(-1)
    if seq_arr.ndim != 2 or len(seq_arr) != len(target_arr) or seq_arr.shape[1] <= 0:
        return np.zeros(len(target_arr), dtype="float32")
    reversed_seq = seq_arr[:, ::-1]
    matches = reversed_seq == target_arr[:, None]
    has_match = np.any(matches, axis=1)
    first_match = np.argmax(matches, axis=1).astype("float32") + 1.0
    return np.where(has_match, first_match, float(seq_arr.shape[1] + 1)).astype("float32", copy=False)


def _advance_rollout_context_batch(
    *,
    twin: ListenerDigitalTwinArtifact,
    active_context: np.ndarray,
    previous_sequences: np.ndarray,
    next_sequences: np.ndarray,
    last_artist: np.ndarray,
    chosen: np.ndarray,
    step_index: np.ndarray,
    hour_shift: float,
    friction_scale: float,
    friction_indices: np.ndarray,
) -> np.ndarray:
    next_context = np.asarray(active_context, dtype="float32").copy()
    if next_context.ndim != 2:
        return next_context

    feature_idx = _context_feature_index_map(twin.context_features)
    next_hour = None
    if "hour" in feature_idx:
        hour_idx = feature_idx["hour"]
        next_hour = np.mod(next_context[:, hour_idx] + hour_shift + 1.0, 24.0).astype("float32", copy=False)
        next_context[:, hour_idx] = next_hour
    if next_hour is not None:
        if "hour_sin" in feature_idx:
            next_context[:, feature_idx["hour_sin"]] = np.sin((2.0 * np.pi * next_hour) / 24.0).astype("float32", copy=False)
        if "hour_cos" in feature_idx:
            next_context[:, feature_idx["hour_cos"]] = np.cos((2.0 * np.pi * next_hour) / 24.0).astype("float32", copy=False)

    track_seconds = float(max(1.0, twin.average_track_seconds))
    step_arr = np.asarray(step_index, dtype="float32").reshape(-1)
    if "session_position" in feature_idx:
        current_position = next_context[:, feature_idx["session_position"]]
        next_context[:, feature_idx["session_position"]] = np.maximum(current_position + 1.0, step_arr)
    if "session_elapsed_seconds" in feature_idx:
        next_context[:, feature_idx["session_elapsed_seconds"]] = np.maximum(
            next_context[:, feature_idx["session_elapsed_seconds"]] + track_seconds,
            step_arr * track_seconds,
        )
    if "time_diff" in feature_idx:
        next_context[:, feature_idx["time_diff"]] = track_seconds

    repeat_from_prev = (np.asarray(chosen, dtype="int32") == np.asarray(last_artist, dtype="int32")).astype("float32")
    if "is_artist_repeat_from_prev" in feature_idx:
        next_context[:, feature_idx["is_artist_repeat_from_prev"]] = repeat_from_prev
    if "transition_repeat_count" in feature_idx:
        next_context[:, feature_idx["transition_repeat_count"]] = (
            next_context[:, feature_idx["transition_repeat_count"]] + repeat_from_prev
        )
    if "session_repeat_ratio_so_far" in feature_idx:
        prior_position = np.maximum(1.0, step_arr - 1.0)
        prior_ratio = np.clip(next_context[:, feature_idx["session_repeat_ratio_so_far"]], 0.0, 1.0)
        prior_repeat_count = prior_ratio * prior_position
        next_context[:, feature_idx["session_repeat_ratio_so_far"]] = (
            (prior_repeat_count + repeat_from_prev) / np.maximum(step_arr, 1.0)
        ).astype("float32", copy=False)

    unique_after = _unique_count_per_row(next_sequences)
    if "session_unique_artists_so_far" in feature_idx:
        next_context[:, feature_idx["session_unique_artists_so_far"]] = np.maximum(
            next_context[:, feature_idx["session_unique_artists_so_far"]],
            unique_after,
        )
    if "recent_artist_unique_ratio_5" in feature_idx:
        next_context[:, feature_idx["recent_artist_unique_ratio_5"]] = _recent_unique_ratio(next_sequences, 5)
    if "recent_artist_unique_ratio_20" in feature_idx:
        next_context[:, feature_idx["recent_artist_unique_ratio_20"]] = _recent_unique_ratio(next_sequences, 20)
    if "artist_session_play_count" in feature_idx:
        next_context[:, feature_idx["artist_session_play_count"]] = np.maximum(
            next_context[:, feature_idx["artist_session_play_count"]],
            np.sum(next_sequences == np.asarray(chosen, dtype="int32")[:, None], axis=1).astype("float32", copy=False),
        )
    if "plays_since_last_artist" in feature_idx:
        next_context[:, feature_idx["plays_since_last_artist"]] = _plays_since_last_occurrence(
            previous_sequences,
            np.asarray(chosen, dtype="int32"),
        )
    if "hours_since_last_artist" in feature_idx:
        matched_before = np.any(previous_sequences == np.asarray(chosen, dtype="int32")[:, None], axis=1)
        hours_elapsed = (track_seconds / 3600.0) * np.maximum(step_arr, 1.0)
        next_context[:, feature_idx["hours_since_last_artist"]] = np.where(
            matched_before,
            track_seconds / 3600.0,
            np.maximum(next_context[:, feature_idx["hours_since_last_artist"]], hours_elapsed),
        ).astype("float32", copy=False)
    if "days_since_last" in feature_idx and "hours_since_last_artist" in feature_idx:
        next_context[:, feature_idx["days_since_last"]] = (
            next_context[:, feature_idx["hours_since_last_artist"]] / 24.0
        ).astype("float32", copy=False)

    if friction_indices.size:
        next_context[:, friction_indices] = np.maximum(0.0, next_context[:, friction_indices] * friction_scale)

    return next_context.astype("float32", copy=False)


def fit_listener_digital_twin(
    *,
    data: PreparedData,
    sequence_length: int,
    artist_labels: list[str],
    multimodal_space: MultimodalArtistSpace,
    output_dir: Path,
    logger,
) -> tuple[ListenerDigitalTwinArtifact, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    last_artist = np.asarray(data.X_seq_train[:, -1], dtype="int32")
    targets = np.asarray(data.y_train, dtype="int32")
    transition = np.ones((data.num_artists, data.num_artists), dtype="float32")
    np.add.at(transition, (last_artist, targets), 1.0)
    transition /= transition.sum(axis=1, keepdims=True)

    train_frame, val_frame, test_frame = _aligned_frames(data, sequence_length=sequence_length)
    y_end_train = _end_labels(train_frame)
    end_features_train = build_serving_tabular_features(data.X_seq_train, data.X_ctx_train)
    end_features_val = build_serving_tabular_features(data.X_seq_val, data.X_ctx_val)
    end_features_test = build_serving_tabular_features(data.X_seq_test, data.X_ctx_test)

    end_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
    )
    if np.unique(y_end_train).size < 2:
        y_end_train = np.where(np.arange(len(y_end_train)) == len(y_end_train) - 1, 1, 0)
    end_estimator.fit(end_features_train, y_end_train)

    avg_track_seconds = float(np.nanmedian(_safe_numeric(train_frame, "time_diff"))) if len(train_frame) else 180.0
    if not np.isfinite(avg_track_seconds) or avg_track_seconds <= 0:
        avg_track_seconds = 180.0

    artifact = ListenerDigitalTwinArtifact(
        artist_labels=list(artist_labels),
        transition_matrix=transition.astype("float32", copy=False),
        end_estimator=end_estimator,
        context_features=list(data.context_features),
        average_track_seconds=avg_track_seconds,
    )

    artifact_path = output_dir / "listener_digital_twin.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    def _summary(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        proba = np.asarray(end_estimator.predict_proba(X), dtype="float32")[:, 1]
        if np.unique(y).size < 2:
            auc = float("nan")
        else:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, proba))
        return {"mean_end_probability": float(np.mean(proba)), "auc": auc}

    summary_path = output_dir / "listener_digital_twin_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artist_count": int(len(artist_labels)),
                "average_track_seconds": float(avg_track_seconds),
                "train_end_rate": float(np.mean(y_end_train)) if len(y_end_train) else float("nan"),
                "val": _summary(end_features_val, _end_labels(val_frame)),
                "test": _summary(end_features_test, _end_labels(test_frame)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Built listener digital twin: artists=%d avg_track_s=%.1f",
        len(artist_labels),
        avg_track_seconds,
    )
    return artifact, [artifact_path, summary_path]


def _safe_numeric(frame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype="float32")
    return np.asarray(frame[column], dtype="float32")


def simulate_rollout_batch_summary(
    *,
    twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    causal_artifact: CausalSkipDecompositionArtifact | None,
    start_sequences: np.ndarray,
    start_contexts: np.ndarray,
    horizon: int,
    policy_weights: dict[str, float],
    scenario: dict[str, float] | None = None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    scenario = dict(scenario or {})
    sequences = np.asarray(start_sequences, dtype="int32")
    contexts = np.asarray(start_contexts, dtype="float32")
    if sequences.ndim != 2:
        sequences = sequences.reshape(0, 0)
    if contexts.ndim != 2:
        contexts = contexts.reshape(len(sequences), -1)

    batch_size = int(len(sequences))
    if batch_size == 0:
        empty_float = np.empty(0, dtype="float32")
        empty_int = np.empty(0, dtype="int32")
        return {
            "session_length": empty_int,
            "mean_skip_risk": empty_float,
            "mean_end_risk": empty_float,
            "first_choice": empty_int,
        }

    sequence_state = sequences.copy()
    context_state = contexts.copy()
    session_length = np.zeros(batch_size, dtype="int32")
    skip_sum = np.zeros(batch_size, dtype="float32")
    end_sum = np.zeros(batch_size, dtype="float32")
    first_choice = np.full(batch_size, -1, dtype="int32")
    active_mask = np.ones(batch_size, dtype=bool)

    artist_ids, transition_log, similarity_matrix, novelty, friction_indices = _rollout_static_inputs(
        twin,
        multimodal_space,
    )
    artist_count = int(len(artist_ids))
    novelty_row = novelty.reshape(1, artist_count)

    friction_scale = float(scenario.get("friction_scale", 1.0))
    hour_shift = float(scenario.get("hour_shift", 0.0))
    fatigue_bias = float(scenario.get("fatigue_bias", 0.0))
    repeat_bias = float(scenario.get("repeat_bias", 0.0))
    repeat_weight = float(policy_weights.get("repeat", 0.6)) + repeat_bias
    transition_weight = float(policy_weights.get("transition", 1.0))
    continuity_weight = float(policy_weights.get("continuity", 0.2))
    novelty_weight = float(policy_weights.get("novelty", 0.0))

    preference_keep: np.ndarray | None = None
    friction_keep: np.ndarray | None = None
    needs_friction_padding = False
    if causal_artifact is not None:
        preference_keep, friction_keep, needs_friction_padding = _causal_feature_views(
            context_state.shape[1],
            causal_artifact,
        )

    for _step in range(max(1, int(horizon))):
        active_idx = np.flatnonzero(active_mask)
        if active_idx.size == 0:
            break

        active_seq = sequence_state[active_idx]
        active_ctx = context_state[active_idx]
        last_artist = active_seq[:, -1]
        repeat_penalty = np.zeros((len(active_idx), artist_count), dtype="float32")
        repeat_penalty[np.arange(len(active_idx))[:, None], active_seq] = 1.0
        scores = (
            transition_weight * transition_log[last_artist]
            + continuity_weight * similarity_matrix[last_artist]
            + novelty_weight * novelty_row
            - repeat_weight * repeat_penalty
        )
        chosen = np.argmax(scores, axis=1).astype("int32", copy=False)
        first_choice_mask = session_length[active_idx] == 0
        if np.any(first_choice_mask):
            first_choice[active_idx[first_choice_mask]] = chosen[first_choice_mask]
        session_length[active_idx] += 1

        next_sequence = active_seq.copy()
        next_sequence[:, :-1] = active_seq[:, 1:]
        next_sequence[:, -1] = chosen
        next_context = _advance_rollout_context_batch(
            twin=twin,
            active_context=active_ctx,
            previous_sequences=active_seq,
            next_sequences=next_sequence,
            last_artist=last_artist,
            chosen=chosen,
            step_index=session_length[active_idx],
            hour_shift=hour_shift,
            friction_scale=friction_scale,
            friction_indices=friction_indices,
        )

        serving_features = build_serving_tabular_features(next_sequence, next_context)
        if causal_artifact is not None:
            skip_features = (
                np.pad(serving_features, ((0, 0), (0, 1)), constant_values=0.0)
                if needs_friction_padding
                else serving_features
            )
            preference_scores = _decision_scores_batch(
                causal_artifact.preference_estimator,
                skip_features[:, preference_keep],
            )
            friction_scores = _decision_scores_batch(
                causal_artifact.friction_estimator,
                skip_features[:, friction_keep],
            )
            total_skip = np.asarray(
                causal_artifact.meta_estimator.predict_proba(
                    np.column_stack([preference_scores, friction_scores]).astype("float32", copy=False)
                ),
                dtype="float32",
            )[:, 1]
        else:
            total_skip = np.full(len(active_idx), 0.1, dtype="float32")
        skip_sum[active_idx] += total_skip

        end_risk = np.asarray(twin.end_estimator.predict_proba(serving_features), dtype="float32")[:, 1]
        end_risk = np.clip(end_risk + fatigue_bias, 0.0, 0.99).astype("float32", copy=False)
        end_sum[active_idx] += end_risk

        sequence_state[active_idx] = next_sequence
        context_state[active_idx] = next_context

        ended = rng.random(len(active_idx)) < end_risk
        if np.any(ended):
            active_mask[active_idx[ended]] = False

    mean_skip_risk = np.divide(
        skip_sum,
        np.maximum(session_length, 1),
        out=np.full(batch_size, np.nan, dtype="float32"),
        where=session_length > 0,
    ).astype("float32", copy=False)
    mean_end_risk = np.divide(
        end_sum,
        np.maximum(session_length, 1),
        out=np.full(batch_size, np.nan, dtype="float32"),
        where=session_length > 0,
    ).astype("float32", copy=False)

    return {
        "session_length": session_length,
        "mean_skip_risk": mean_skip_risk,
        "mean_end_risk": mean_end_risk,
        "first_choice": first_choice,
    }


def simulate_rollout(
    *,
    twin: ListenerDigitalTwinArtifact,
    multimodal_space: MultimodalArtistSpace,
    causal_artifact: CausalSkipDecompositionArtifact | None,
    start_sequence: np.ndarray,
    start_context: np.ndarray,
    horizon: int,
    policy_weights: dict[str, float],
    scenario: dict[str, float] | None = None,
    rng: np.random.Generator,
) -> dict[str, object]:
    scenario = dict(scenario or {})
    sequence = np.asarray(start_sequence, dtype="int32").reshape(-1).copy()
    context = np.asarray(start_context, dtype="float32").reshape(-1).copy()
    planned: list[int] = []
    skip_risks: list[float] = []
    end_risks: list[float] = []

    friction_scale = float(scenario.get("friction_scale", 1.0))
    hour_shift = float(scenario.get("hour_shift", 0.0))
    fatigue_bias = float(scenario.get("fatigue_bias", 0.0))
    repeat_bias = float(scenario.get("repeat_bias", 0.0))

    for step in range(max(1, int(horizon))):
        last_artist = int(sequence[-1])
        transition = twin.transition_matrix[last_artist].astype("float32", copy=False)
        artist_ids = np.arange(len(twin.artist_labels), dtype="int32")
        similarity = multimodal_space.embeddings[last_artist] @ multimodal_space.embeddings.T
        novelty = 1.0 - multimodal_space.popularity
        repeat_penalty = np.isin(artist_ids, sequence).astype("float32")
        scores = (
            float(policy_weights.get("transition", 1.0)) * np.log(np.clip(transition, 1e-6, 1.0))
            + float(policy_weights.get("continuity", 0.2)) * similarity
            + float(policy_weights.get("novelty", 0.0)) * novelty
            - (float(policy_weights.get("repeat", 0.6)) + repeat_bias) * repeat_penalty
        )
        choice = int(np.argmax(scores))
        planned.append(choice)

        next_sequence = np.roll(sequence, -1)
        next_sequence[-1] = choice
        next_context = _advance_rollout_context_batch(
            twin=twin,
            active_context=np.asarray([context], dtype="float32"),
            previous_sequences=np.asarray([sequence], dtype="int32"),
            next_sequences=np.asarray([next_sequence], dtype="int32"),
            last_artist=np.asarray([last_artist], dtype="int32"),
            chosen=np.asarray([choice], dtype="int32"),
            step_index=np.asarray([step + 1], dtype="int32"),
            hour_shift=hour_shift,
            friction_scale=friction_scale,
            friction_indices=np.asarray(
                [
                    idx
                    for idx, feature_name in enumerate(twin.context_features)
                    if str(feature_name).startswith("tech_") or str(feature_name) == "offline"
                ],
                dtype="int64",
            ),
        )[0]

        if causal_artifact is not None:
            serving_features = build_serving_tabular_features(
                np.asarray([next_sequence], dtype="int32"),
                np.asarray([next_context], dtype="float32"),
            )
            preference_keep, friction_keep, needs_friction_padding = _causal_feature_views(
                len(next_context),
                causal_artifact,
            )
            skip_features = np.pad(serving_features, ((0, 0), (0, 1)), constant_values=0.0) if needs_friction_padding else serving_features
            pref_logit = _decision_scores_batch(causal_artifact.preference_estimator, skip_features[:, preference_keep])[0]
            friction_logit = _decision_scores_batch(causal_artifact.friction_estimator, skip_features[:, friction_keep])[0]
            total_skip = float(causal_artifact.meta_estimator.predict_proba(np.asarray([[pref_logit, friction_logit]], dtype="float32"))[:, 1][0])
        else:
            total_skip = 0.1
        skip_risks.append(total_skip)

        if causal_artifact is None:
            serving_features = build_serving_tabular_features(
                np.asarray([next_sequence], dtype="int32"),
                np.asarray([next_context], dtype="float32"),
            )
        end_risk = float(np.asarray(twin.end_estimator.predict_proba(serving_features), dtype="float32")[:, 1][0] + fatigue_bias)
        end_risk = float(min(0.99, max(0.0, end_risk)))
        end_risks.append(end_risk)

        sequence = next_sequence
        context = next_context
        if rng.random() < end_risk:
            break

    return {
        "planned_sequence": planned,
        "mean_skip_risk": float(np.mean(skip_risks)) if skip_risks else float("nan"),
        "mean_end_risk": float(np.mean(end_risks)) if end_risks else float("nan"),
        "session_length": int(len(planned)),
    }
