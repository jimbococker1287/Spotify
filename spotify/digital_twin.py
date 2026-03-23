from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
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


@dataclass(frozen=True)
class ListenerDigitalTwinArtifact:
    artist_labels: list[str]
    transition_matrix: np.ndarray
    end_estimator: object
    context_features: list[str]
    average_track_seconds: float


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


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

        next_context = context.copy()
        if len(next_context) >= 1:
            next_context[0] = (next_context[0] + hour_shift + 1.0) % 24.0
        for idx, feature_name in enumerate(twin.context_features):
            if str(feature_name).startswith("tech_") or str(feature_name) == "offline":
                next_context[idx] = float(max(0.0, next_context[idx] * friction_scale))

        if causal_artifact is not None:
            full_features = build_serving_tabular_features(
                np.asarray([sequence], dtype="int32"),
                np.asarray([next_context], dtype="float32"),
            )
            seq_feature_count = full_features.shape[1] - len(causal_artifact.context_features)
            friction_keep = np.asarray([seq_feature_count + idx for idx in causal_artifact.friction_feature_indices], dtype="int64")
            if friction_keep.size == 0:
                friction_keep = np.asarray([full_features.shape[1] - 1], dtype="int64")
            preference_keep = np.asarray(
                [idx for idx in range(full_features.shape[1]) if idx not in set(friction_keep.tolist())],
                dtype="int64",
            )
            pref_logit = np.asarray(causal_artifact.preference_estimator.decision_function(full_features[:, preference_keep]), dtype="float32").reshape(-1)[0]
            friction_logit = np.asarray(causal_artifact.friction_estimator.decision_function(full_features[:, friction_keep]), dtype="float32").reshape(-1)[0]
            total_skip = float(causal_artifact.meta_estimator.predict_proba(np.asarray([[pref_logit, friction_logit]], dtype="float32"))[:, 1][0])
        else:
            total_skip = 0.1
        skip_risks.append(total_skip)

        end_features = build_serving_tabular_features(
            np.asarray([sequence], dtype="int32"),
            np.asarray([next_context], dtype="float32"),
        )
        end_risk = float(np.asarray(twin.end_estimator.predict_proba(end_features), dtype="float32")[:, 1][0] + fatigue_bias)
        end_risk = float(min(0.99, max(0.0, end_risk)))
        end_risks.append(end_risk)

        sequence = np.roll(sequence, -1)
        sequence[-1] = choice
        context = next_context
        if rng.random() < end_risk:
            break

    return {
        "planned_sequence": planned,
        "mean_skip_risk": float(np.mean(skip_risks)) if skip_risks else float("nan"),
        "mean_end_risk": float(np.mean(end_risks)) if end_risks else float("nan"),
        "session_length": int(len(planned)),
    }
