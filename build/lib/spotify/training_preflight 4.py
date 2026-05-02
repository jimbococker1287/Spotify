from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .data import PreparedData
from .run_artifacts import copy_file_if_changed, materialize_cached_file, safe_read_json, write_json


DEEP_TRAINING_CACHE_SCHEMA_VERSION = "deep-training-cache-v1"


@dataclass
class TrainingArtifacts:
    histories: dict[str, object]
    test_metrics: dict[str, dict[str, float]]
    val_metrics: dict[str, dict[str, float]]
    fit_seconds: dict[str, float]
    prediction_bundle_paths: dict[str, str]


@dataclass(frozen=True)
class DeepModelCachePaths:
    cache_key: str
    cache_dir: Path
    result_path: Path
    metadata_path: Path
    checkpoint_path: Path
    prediction_bundle_path: Path


@dataclass
class HistoryArtifact:
    history: dict[str, list[float]]


@dataclass
class CachedDeepModelArtifact:
    history: HistoryArtifact
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    fit_seconds: float
    prediction_bundle_path: str


@dataclass
class DeepTrainingCachePlan:
    enabled: bool
    fingerprint: str
    hit_model_names: tuple[str, ...]
    miss_model_names: tuple[str, ...]
    artifacts: TrainingArtifacts
    cache_contexts: dict[str, tuple[DeepModelCachePaths, dict[str, object]]]


def compute_baselines(data: PreparedData, logger) -> dict[str, float]:
    majority_artist = int(np.bincount(data.y_train.astype(int)).argmax())
    majority_top1 = float(np.mean(data.y_val == majority_artist))

    y_train = np.asarray(data.y_train, dtype="int32")
    y_val = np.asarray(data.y_val, dtype="int32")
    x_seq_train = np.asarray(data.X_seq_train, dtype="int32")
    last_artist_pred = np.asarray(data.X_seq_val[:, -1], dtype="int32")
    last_top1 = float(np.mean(last_artist_pred == data.y_val))

    num_states = int(data.num_artists)
    transitions = np.ones((num_states, num_states), dtype=np.int32)
    if x_seq_train.shape[1] > 1:
        np.add.at(
            transitions,
            (x_seq_train[:, :-1].reshape(-1), x_seq_train[:, 1:].reshape(-1)),
            1,
        )
    np.add.at(transitions, (x_seq_train[:, -1], y_train), 1)

    markov_pred = np.argmax(transitions[last_artist_pred], axis=1)
    markov_top1 = float(np.mean(markov_pred == y_val))

    logger.info("Baseline (majority artist) Top-1 Val Acc: %.4f", majority_top1)
    logger.info("Baseline (last artist) Top-1 Val Acc: %.4f", last_top1)
    logger.info("Baseline (1-step Markov) Top-1 Val Acc: %.4f", markov_top1)

    return {
        "majority_top1": majority_top1,
        "last_artist_top1": last_top1,
        "markov_top1": markov_top1,
    }


def _deep_cache_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_CACHE_DEEP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _normalize_history_payload(history_values: dict[str, object]) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for key, values in history_values.items():
        series = np.asarray(values, dtype="float64").reshape(-1)
        normalized[str(key)] = [float(value) for value in series]
    return normalized


def _normalize_metric_payload(payload: dict[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in payload.items():
        try:
            normalized[str(key)] = float(value)
        except Exception:
            continue
    return normalized


def _deep_training_source_digest() -> str:
    sources = [
        Path(__file__).with_name("training.py").resolve(),
        Path(__file__).with_name("modeling.py").resolve(),
    ]
    hasher = hashlib.sha256()
    for path in sources:
        try:
            hasher.update(path.read_bytes())
        except Exception:
            continue
    return hasher.hexdigest()[:24]


def _build_deep_cache_payload(
    *,
    cache_fingerprint: str,
    model_name: str,
    random_seed: int,
    batch_size: int,
    epochs: int,
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "schema_version": DEEP_TRAINING_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "model_name": model_name,
        "random_seed": int(random_seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "sequence_length": int(sequence_length),
        "num_artists": int(num_artists),
        "num_ctx": int(num_ctx),
        "run_eager": os.getenv("SPOTIFY_RUN_EAGER", "auto").strip().lower(),
        "steps_per_execution": os.getenv("SPOTIFY_STEPS_PER_EXECUTION", "auto").strip().lower(),
        "mixed_precision": os.getenv("SPOTIFY_MIXED_PRECISION", "auto").strip().lower(),
        "distribution_strategy": os.getenv("SPOTIFY_DISTRIBUTION_STRATEGY", "auto").strip().lower(),
        "force_cpu": os.getenv("SPOTIFY_FORCE_CPU", "0").strip().lower(),
        "source_digest": _deep_training_source_digest(),
    }


def _build_deep_cache_key(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _resolve_deep_model_cache_paths(
    *,
    cache_root: Path,
    cache_fingerprint: str,
    model_name: str,
    cache_key: str,
) -> DeepModelCachePaths:
    cache_dir = (cache_root / cache_fingerprint / model_name / cache_key).resolve()
    return DeepModelCachePaths(
        cache_key=cache_key,
        cache_dir=cache_dir,
        result_path=cache_dir / "result.json",
        metadata_path=cache_dir / "cache_meta.json",
        checkpoint_path=cache_dir / f"best_{model_name}.keras",
        prediction_bundle_path=cache_dir / "prediction_bundles" / f"deep_{model_name}.npz",
    )


def _load_cached_deep_artifact(
    *,
    cache_paths: DeepModelCachePaths,
    output_dir: Path,
    model_name: str,
    logger,
) -> CachedDeepModelArtifact | None:
    if not cache_paths.result_path.exists():
        return None
    if not cache_paths.checkpoint_path.exists() or not cache_paths.prediction_bundle_path.exists():
        return None

    payload = safe_read_json(cache_paths.result_path, default=None)
    if not isinstance(payload, dict):
        return None
    result_payload = payload.get("result", payload)
    if not isinstance(result_payload, dict):
        return None

    history_payload = result_payload.get("history")
    val_metrics_payload = result_payload.get("val_metrics")
    test_metrics_payload = result_payload.get("test_metrics")
    if not isinstance(history_payload, dict):
        return None
    if not isinstance(val_metrics_payload, dict) or not isinstance(test_metrics_payload, dict):
        return None

    try:
        checkpoint_dest = output_dir / f"best_{model_name}.keras"
        prediction_dest = output_dir / "prediction_bundles" / f"deep_{model_name}.npz"
        materialize_cached_file(cache_paths.checkpoint_path, checkpoint_dest)
        materialize_cached_file(cache_paths.prediction_bundle_path, prediction_dest)
        fit_seconds = float(result_payload.get("fit_seconds", float("nan")))
    except Exception as exc:
        logger.warning("Deep-training cache load failed for %s (%s). Rebuilding.", model_name, exc)
        return None

    return CachedDeepModelArtifact(
        history=HistoryArtifact(history=_normalize_history_payload(history_payload)),
        val_metrics=_normalize_metric_payload(val_metrics_payload),
        test_metrics=_normalize_metric_payload(test_metrics_payload),
        fit_seconds=fit_seconds,
        prediction_bundle_path=str(prediction_dest),
    )


def _save_deep_artifact_to_cache(
    *,
    cache_paths: DeepModelCachePaths,
    cache_payload: dict[str, object],
    history,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    fit_seconds: float,
    checkpoint_path: Path,
    prediction_bundle_path: Path,
    logger,
) -> None:
    try:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        copy_file_if_changed(checkpoint_path, cache_paths.checkpoint_path)
        copy_file_if_changed(prediction_bundle_path, cache_paths.prediction_bundle_path)
        write_json(
            cache_paths.result_path,
            {
                "cache_schema_version": DEEP_TRAINING_CACHE_SCHEMA_VERSION,
                "result": {
                    "history": _normalize_history_payload(getattr(history, "history", {})),
                    "val_metrics": _normalize_metric_payload(val_metrics),
                    "test_metrics": _normalize_metric_payload(test_metrics),
                    "fit_seconds": float(fit_seconds),
                },
            },
            sort_keys=True,
        )
        write_json(cache_paths.metadata_path, cache_payload, sort_keys=True)
    except Exception as exc:
        logger.warning("Deep-training cache save failed for %s (%s).", checkpoint_path.name, exc)


def _ordered_by_selected(payload: dict[str, object], selected_model_names: list[str]) -> dict[str, object]:
    return {name: payload[name] for name in selected_model_names if name in payload}


def resolve_cached_deep_training_artifacts(
    *,
    data: PreparedData,
    selected_model_names: tuple[str, ...],
    batch_size: int,
    epochs: int,
    output_dir: Path,
    logger,
    random_seed: int = 42,
    cache_root: Path | None = None,
    cache_fingerprint: str = "",
) -> DeepTrainingCachePlan:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_bundles").mkdir(parents=True, exist_ok=True)

    cache_enabled = _deep_cache_enabled_from_env() and cache_root is not None and bool(str(cache_fingerprint).strip())
    selected_names = list(selected_model_names)
    histories: dict[str, object] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    val_metrics: dict[str, dict[str, float]] = {}
    fit_seconds: dict[str, float] = {}
    prediction_bundle_paths: dict[str, str] = {}
    cache_contexts: dict[str, tuple[DeepModelCachePaths, dict[str, object]]] = {}
    hit_model_names: list[str] = []
    miss_model_names: list[str] = []

    for model_name in selected_names:
        if cache_enabled and cache_root is not None:
            cache_payload = _build_deep_cache_payload(
                cache_fingerprint=cache_fingerprint,
                model_name=model_name,
                random_seed=random_seed,
                batch_size=batch_size,
                epochs=epochs,
                sequence_length=int(data.X_seq_train.shape[1]) if getattr(data.X_seq_train, "ndim", 0) >= 2 else 0,
                num_artists=int(data.num_artists),
                num_ctx=int(data.num_ctx),
            )
            cache_key = _build_deep_cache_key(cache_payload)
            cache_paths = _resolve_deep_model_cache_paths(
                cache_root=cache_root,
                cache_fingerprint=cache_fingerprint,
                model_name=model_name,
                cache_key=cache_key,
            )
            cache_contexts[model_name] = (cache_paths, cache_payload)
            cached = _load_cached_deep_artifact(
                cache_paths=cache_paths,
                output_dir=output_dir,
                model_name=model_name,
                logger=logger,
            )
            if cached is not None:
                histories[model_name] = cached.history
                val_metrics[model_name] = cached.val_metrics
                test_metrics[model_name] = cached.test_metrics
                fit_seconds[model_name] = cached.fit_seconds
                prediction_bundle_paths[model_name] = cached.prediction_bundle_path
                hit_model_names.append(model_name)
                continue
        miss_model_names.append(model_name)

    artifacts = TrainingArtifacts(
        histories=_ordered_by_selected(histories, selected_names),
        test_metrics=_ordered_by_selected(test_metrics, selected_names),
        val_metrics=_ordered_by_selected(val_metrics, selected_names),
        fit_seconds=_ordered_by_selected(fit_seconds, selected_names),
        prediction_bundle_paths=_ordered_by_selected(prediction_bundle_paths, selected_names),
    )
    return DeepTrainingCachePlan(
        enabled=bool(cache_enabled),
        fingerprint=(str(cache_fingerprint).strip() if cache_enabled else ""),
        hit_model_names=tuple(hit_model_names),
        miss_model_names=tuple(miss_model_names),
        artifacts=artifacts,
        cache_contexts=cache_contexts,
    )


__all__ = [
    "DEEP_TRAINING_CACHE_SCHEMA_VERSION",
    "CachedDeepModelArtifact",
    "DeepModelCachePaths",
    "DeepTrainingCachePlan",
    "HistoryArtifact",
    "TrainingArtifacts",
    "_build_deep_cache_key",
    "_build_deep_cache_payload",
    "_ordered_by_selected",
    "_resolve_deep_model_cache_paths",
    "_save_deep_artifact_to_cache",
    "compute_baselines",
    "resolve_cached_deep_training_artifacts",
]
