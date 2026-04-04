from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.utils import class_weight

from .data import PreparedData
from .probability_bundles import save_prediction_bundle
from .ranking import ranking_metrics_from_proba
from .run_artifacts import copy_file_if_changed, safe_read_json, write_json


DEEP_TRAINING_CACHE_SCHEMA_VERSION = "deep-training-cache-v1"


@dataclass
class SampleWeights:
    artist_train: np.ndarray
    artist_val: np.ndarray
    artist_test: np.ndarray
    skip_train: np.ndarray
    skip_val: np.ndarray
    skip_test: np.ndarray


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


def _weighted_mean(values: np.ndarray, sample_weights: np.ndarray) -> float:
    values_arr = np.asarray(values, dtype="float64").reshape(-1)
    weights_arr = np.asarray(sample_weights, dtype="float64").reshape(-1)
    if values_arr.size == 0 or values_arr.size != weights_arr.size:
        return float("nan")
    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        return float(np.mean(values_arr))
    return float(np.dot(values_arr, weights_arr) / total_weight)


def _weighted_top1_accuracy_from_proba(
    proba: np.ndarray,
    y_true: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    proba_arr = np.asarray(proba)
    y_arr = np.asarray(y_true).reshape(-1)
    if proba_arr.ndim != 2 or len(proba_arr) != len(y_arr):
        return float("nan")
    pred = np.argmax(proba_arr, axis=1)
    correct = (pred == y_arr).astype("float32")
    return _weighted_mean(correct, sample_weights)


def _weighted_topk_accuracy_from_proba(
    proba: np.ndarray,
    y_true: np.ndarray,
    sample_weights: np.ndarray,
    *,
    k: int,
) -> float:
    proba_arr = np.asarray(proba)
    y_arr = np.asarray(y_true).reshape(-1)
    if proba_arr.ndim != 2 or len(proba_arr) != len(y_arr):
        return float("nan")
    kk = max(1, min(int(k), int(proba_arr.shape[1])))
    topk_idx = np.argpartition(proba_arr, -kk, axis=1)[:, -kk:]
    hits = np.any(topk_idx == y_arr.reshape(-1, 1), axis=1).astype("float32")
    return _weighted_mean(hits, sample_weights)


def compute_baselines(data: PreparedData, logger) -> dict[str, float]:
    majority_artist = int(np.bincount(data.y_train.astype(int)).argmax())
    majority_top1 = float(np.mean(data.y_val == majority_artist))

    y_train = np.asarray(data.y_train, dtype="int32")
    y_val = np.asarray(data.y_val, dtype="int32")
    X_seq_train = np.asarray(data.X_seq_train, dtype="int32")
    last_artist_pred = np.asarray(data.X_seq_val[:, -1], dtype="int32")
    last_top1 = float(np.mean(last_artist_pred == data.y_val))

    num_states = int(data.num_artists)
    transitions = np.ones((num_states, num_states), dtype=np.int32)
    if X_seq_train.shape[1] > 1:
        np.add.at(
            transitions,
            (X_seq_train[:, :-1].reshape(-1), X_seq_train[:, 1:].reshape(-1)),
            1,
        )
    np.add.at(transitions, (X_seq_train[:, -1], y_train), 1)

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


def compute_sample_weights(data: PreparedData) -> SampleWeights:
    artist_y_train = np.asarray(data.y_train, dtype="int64")
    artist_y_val = np.asarray(data.y_val, dtype="int64")
    artist_y_test = np.asarray(data.y_test, dtype="int64")
    skip_y_train = np.asarray(data.y_skip_train, dtype="int64")
    skip_y_val = np.asarray(data.y_skip_val, dtype="int64")
    skip_y_test = np.asarray(data.y_skip_test, dtype="int64")

    artist_classes = np.unique(artist_y_train)
    artist_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=artist_classes,
        y=artist_y_train,
    )

    skip_classes = np.unique(skip_y_train)
    skip_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=skip_classes,
        y=skip_y_train,
    )

    def _lookup_weights(y: np.ndarray, classes: np.ndarray, weights: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(classes, y)
        out = np.ones(y.shape, dtype="float32")
        valid = indices < len(classes)
        if np.any(valid):
            matched = classes[indices[valid]] == y[valid]
            if np.any(matched):
                valid_indices = np.flatnonzero(valid)[matched]
                out[valid_indices] = weights[indices[valid_indices]].astype("float32", copy=False)
        return out

    return SampleWeights(
        artist_train=_lookup_weights(artist_y_train, artist_classes, artist_weights),
        artist_val=_lookup_weights(artist_y_val, artist_classes, artist_weights),
        artist_test=_lookup_weights(artist_y_test, artist_classes, artist_weights),
        skip_train=_lookup_weights(skip_y_train, skip_classes, skip_weights),
        skip_val=_lookup_weights(skip_y_val, skip_classes, skip_weights),
        skip_test=_lookup_weights(skip_y_test, skip_classes, skip_weights),
    )


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
        Path(__file__).resolve(),
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
        copy_file_if_changed(cache_paths.checkpoint_path, checkpoint_dest)
        copy_file_if_changed(cache_paths.prediction_bundle_path, prediction_dest)
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


def train_and_evaluate_models(
    data: PreparedData,
    model_builders,
    batch_size: int,
    epochs: int,
    output_dir: Path,
    strategy,
    logger,
    random_seed: int = 42,
    cache_root: Path | None = None,
    cache_fingerprint: str = "",
    cache_stats_out: dict[str, object] | None = None,
    cache_plan: DeepTrainingCachePlan | None = None,
) -> TrainingArtifacts:
    def _parse_pos_int(raw: str | None, fallback: int) -> int:
        try:
            value = int(str(raw))
            if value > 0:
                return value
        except Exception:
            pass
        return fallback

    def _extract_artist_proba(prediction) -> np.ndarray:
        if isinstance(prediction, dict):
            if "artist_output" in prediction:
                return np.asarray(prediction["artist_output"])
            first_value = next(iter(prediction.values()))
            return np.asarray(first_value)
        if isinstance(prediction, (list, tuple)):
            return np.asarray(prediction[0])
        return np.asarray(prediction)

    output_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[str, object] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    val_metrics: dict[str, dict[str, float]] = {}
    fit_seconds: dict[str, float] = {}
    prediction_bundle_paths: dict[str, str] = {}
    prediction_output_dir = output_dir / "prediction_bundles"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    eager_flag = os.getenv("SPOTIFY_RUN_EAGER", "auto").strip().lower()
    if eager_flag in ("1", "true", "yes", "on"):
        run_eagerly = True
    elif eager_flag in ("0", "false", "no", "off"):
        run_eagerly = False
    else:
        # Graph mode is materially faster; prefer it unless explicitly overridden.
        run_eagerly = False

    log_interval = _parse_pos_int(os.getenv("SPOTIFY_BATCH_LOG_INTERVAL"), 25)
    steps_per_execution_raw = os.getenv("SPOTIFY_STEPS_PER_EXECUTION", "auto").strip().lower()
    if steps_per_execution_raw == "auto":
        steps_per_execution = 1 if run_eagerly else 64
    else:
        steps_per_execution = _parse_pos_int(steps_per_execution_raw, 1 if run_eagerly else 64)

    model_builders = list(model_builders)
    selected_model_names = [name for name, _builder in model_builders]
    resolved_cache_plan = cache_plan or resolve_cached_deep_training_artifacts(
        data=data,
        selected_model_names=tuple(selected_model_names),
        batch_size=batch_size,
        epochs=epochs,
        output_dir=output_dir,
        logger=logger,
        random_seed=random_seed,
        cache_root=cache_root,
        cache_fingerprint=cache_fingerprint,
    )
    histories.update(resolved_cache_plan.artifacts.histories)
    val_metrics.update(resolved_cache_plan.artifacts.val_metrics)
    test_metrics.update(resolved_cache_plan.artifacts.test_metrics)
    fit_seconds.update(resolved_cache_plan.artifacts.fit_seconds)
    prediction_bundle_paths.update(resolved_cache_plan.artifacts.prediction_bundle_paths)
    cache_contexts = dict(resolved_cache_plan.cache_contexts)
    uncached_model_names = set(resolved_cache_plan.miss_model_names)
    uncached_model_builders = [(name, builder) for name, builder in model_builders if name in uncached_model_names]

    if cache_stats_out is not None:
        cache_stats_out.clear()
        cache_stats_out.update(
            {
                "enabled": bool(resolved_cache_plan.enabled),
                "fingerprint": str(resolved_cache_plan.fingerprint),
                "hit_model_names": list(resolved_cache_plan.hit_model_names),
                "miss_model_names": list(resolved_cache_plan.miss_model_names),
            }
        )

    logger.info(
        "Deep-training cache status: enabled=%s fingerprint=%s hits=%d misses=%d",
        resolved_cache_plan.enabled,
        (resolved_cache_plan.fingerprint if resolved_cache_plan.enabled else "disabled"),
        len(resolved_cache_plan.hit_model_names),
        len(uncached_model_builders),
    )

    if not uncached_model_builders:
        return TrainingArtifacts(
            histories=_ordered_by_selected(histories, selected_model_names),
            test_metrics=_ordered_by_selected(test_metrics, selected_model_names),
            val_metrics=_ordered_by_selected(val_metrics, selected_model_names),
            fit_seconds=_ordered_by_selected(fit_seconds, selected_model_names),
            prediction_bundle_paths=_ordered_by_selected(prediction_bundle_paths, selected_model_names),
        )

    import tensorflow as tf
    from tensorflow.keras import callbacks

    class EpochProgressLogger(callbacks.Callback):
        def __init__(self, model_name: str, logger_obj, log_interval: int):
            super().__init__()
            self.model_name = model_name
            self.logger = logger_obj
            self.log_interval = max(1, int(log_interval))
            self._epoch_started = 0.0
            self._steps_per_epoch = None

        def on_train_begin(self, logs=None):
            total_epochs = self.params.get("epochs", "?")
            self._steps_per_epoch = self.params.get("steps", None)
            steps = self._steps_per_epoch if self._steps_per_epoch is not None else "?"
            self.logger.info("[%s] Train begin: epochs=%s steps_per_epoch=%s", self.model_name, total_epochs, steps)

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_started = time.perf_counter()
            total_epochs = self.params.get("epochs", "?")
            self.logger.info("[%s] Epoch %d/%s started", self.model_name, epoch + 1, total_epochs)

        def on_train_batch_end(self, batch, logs=None):
            step = batch + 1
            if step == 1 or step % self.log_interval == 0:
                loss = float((logs or {}).get("loss", float("nan")))
                if self._steps_per_epoch is None:
                    self.logger.info("[%s] Epoch progress: step=%d loss=%.4f", self.model_name, step, loss)
                else:
                    self.logger.info(
                        "[%s] Epoch progress: step=%d/%d loss=%.4f",
                        self.model_name,
                        step,
                        int(self._steps_per_epoch),
                        loss,
                    )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            seconds = time.perf_counter() - self._epoch_started
            loss = logs.get("loss", float("nan"))
            val_loss = logs.get("val_loss", float("nan"))
            val_top1 = logs.get(
                "val_sparse_categorical_accuracy",
                logs.get("val_artist_output_sparse_categorical_accuracy", float("nan")),
            )
            val_top5 = logs.get("val_top_5", logs.get("val_artist_output_top_5", float("nan")))
            self.logger.info(
                "[%s] Epoch %d done in %.1fs | loss=%.4f val_loss=%.4f val_top1=%.4f val_top5=%.4f",
                self.model_name,
                epoch + 1,
                seconds,
                float(loss),
                float(val_loss),
                float(val_top1),
                float(val_top5),
            )

    weights = compute_sample_weights(data)

    def _resolve_dataset_cache_enabled() -> tuple[bool, str]:
        raw = os.getenv("SPOTIFY_TF_DATA_CACHE", "auto").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True, "forced_on"
        if raw in ("0", "false", "no", "off"):
            return False, "forced_off"

        approx_bytes = (
            data.X_seq_train.nbytes
            + data.X_ctx_train.nbytes
            + data.y_train.nbytes
            + data.y_skip_train.nbytes
            + weights.artist_train.nbytes
            + weights.skip_train.nbytes
        )
        try:
            import psutil  # type: ignore

            available_bytes = int(psutil.virtual_memory().available)
            cache_fraction_raw = os.getenv("SPOTIFY_TF_DATA_CACHE_FRACTION", "0.40").strip()
            try:
                cache_fraction = float(cache_fraction_raw)
            except Exception:
                cache_fraction = 0.40
            cache_fraction = min(0.75, max(0.05, cache_fraction))
            threshold_bytes = int(available_bytes * cache_fraction)
            enabled = approx_bytes <= threshold_bytes
            reason = (
                f"auto(approx={approx_bytes // (1024**2)}MiB "
                f"avail={available_bytes // (1024**2)}MiB fraction={cache_fraction:.2f})"
            )
            return enabled, reason
        except Exception:
            return False, "auto(no_psutil)"

    dataset_cache_enabled, dataset_cache_reason = _resolve_dataset_cache_enabled()
    shuffle_buffer = _parse_pos_int(os.getenv("SPOTIFY_SHUFFLE_BUFFER"), min(len(data.y_train), 65536))
    tf_data_threadpool = _parse_pos_int(os.getenv("SPOTIFY_TF_DATA_THREADPOOL"), 0)
    prefetch_raw = os.getenv("SPOTIFY_TF_PREFETCH", "auto").strip().lower()
    if prefetch_raw == "auto":
        prefetch_buffer = tf.data.AUTOTUNE
    else:
        prefetch_buffer = _parse_pos_int(prefetch_raw, 1)

    logger.info(
        "tf.data settings: cache=%s (%s) shuffle_buffer=%d prefetch=%s threadpool=%s",
        dataset_cache_enabled,
        dataset_cache_reason,
        shuffle_buffer,
        ("autotune" if prefetch_buffer == tf.data.AUTOTUNE else str(prefetch_buffer)),
        (str(tf_data_threadpool) if tf_data_threadpool > 0 else "default"),
    )

    def _with_data_options(dataset, training: bool):
        options = tf.data.Options()
        if training:
            options.experimental_deterministic = False
            options.experimental_slack = True
        if tf_data_threadpool > 0:
            options.threading.private_threadpool_size = int(tf_data_threadpool)
        return dataset.with_options(options)

    def _build_weighted_dataset(features, labels, sample_weights, *, training: bool, seed: int):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels, sample_weights))
        if dataset_cache_enabled:
            dataset = dataset.cache()
        if training:
            dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = _with_data_options(dataset, training=training)
        dataset = dataset.prefetch(prefetch_buffer)
        return dataset

    def _build_feature_dataset(features):
        seq_values, ctx_values = features
        dataset = tf.data.Dataset.from_tensor_slices({"seq_input": seq_values, "ctx_input": ctx_values})
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = _with_data_options(dataset, training=False)
        dataset = dataset.prefetch(prefetch_buffer)
        return dataset

    single_data_bundle = None
    multi_data_bundle = None

    for name, builder in uncached_model_builders:
        logger.info("Training %s model", name)
        logger.info("[%s] run_eagerly=%s", name, run_eagerly)
        logger.info("[%s] steps_per_execution=%d", name, steps_per_execution)

        scope = strategy.scope() if strategy is not None else nullcontext()
        with scope:
            model = builder()
            single_head = len(model.outputs) == 1

            if single_head:
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=[
                        "sparse_categorical_accuracy",
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5"),
                    ],
                    run_eagerly=run_eagerly,
                    steps_per_execution=steps_per_execution,
                )
                monitor_metric = "val_sparse_categorical_accuracy"
            else:
                model.compile(
                    optimizer="adam",
                    loss={
                        "artist_output": "sparse_categorical_crossentropy",
                        "skip_output": "binary_crossentropy",
                    },
                    metrics={
                        "artist_output": [
                            "sparse_categorical_accuracy",
                            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5"),
                        ],
                        "skip_output": ["accuracy"],
                    },
                    loss_weights={"artist_output": 1.0, "skip_output": 0.2},
                    run_eagerly=run_eagerly,
                    steps_per_execution=steps_per_execution,
                )
                monitor_metric = "val_artist_output_sparse_categorical_accuracy"

            checkpoint_path = output_dir / f"best_{name}.keras"
            cbs = [
                EpochProgressLogger(name, logger, log_interval=log_interval),
                callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=5,
                    mode="max",
                    restore_best_weights=True,
                ),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    min_delta=0.002,
                    mode="min",
                    restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2,
                    mode="min",
                    verbose=1,
                    min_delta=0.001,
                    cooldown=0,
                    min_lr=1e-6,
                ),
                callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor_metric,
                    save_best_only=True,
                    mode="max",
                ),
            ]

        if single_head:
            if single_data_bundle is None:
                train_dataset = _build_weighted_dataset(
                    features=(data.X_seq_train, data.X_ctx_train),
                    labels=data.y_train,
                    sample_weights=weights.artist_train,
                    training=True,
                    seed=1337,
                )
                val_dataset = _build_weighted_dataset(
                    features=(data.X_seq_val, data.X_ctx_val),
                    labels=data.y_val,
                    sample_weights=weights.artist_val,
                    training=False,
                    seed=0,
                )
                val_predict_dataset = _build_feature_dataset((data.X_seq_val, data.X_ctx_val))
                test_predict_dataset = _build_feature_dataset((data.X_seq_test, data.X_ctx_test))
                single_data_bundle = (
                    train_dataset,
                    val_dataset,
                    val_predict_dataset,
                    test_predict_dataset,
                )
            (
                train_dataset,
                val_dataset,
                val_predict_dataset,
                test_predict_dataset,
            ) = single_data_bundle
        else:
            if multi_data_bundle is None:
                train_dataset = _build_weighted_dataset(
                    features=(data.X_seq_train, data.X_ctx_train),
                    labels=(data.y_train, data.y_skip_train),
                    sample_weights=(weights.artist_train, weights.skip_train),
                    training=True,
                    seed=1337,
                )
                val_dataset = _build_weighted_dataset(
                    features=(data.X_seq_val, data.X_ctx_val),
                    labels=(data.y_val, data.y_skip_val),
                    sample_weights=(weights.artist_val, weights.skip_val),
                    training=False,
                    seed=0,
                )
                val_predict_dataset = _build_feature_dataset((data.X_seq_val, data.X_ctx_val))
                test_predict_dataset = _build_feature_dataset((data.X_seq_test, data.X_ctx_test))
                multi_data_bundle = (
                    train_dataset,
                    val_dataset,
                    val_predict_dataset,
                    test_predict_dataset,
                )
            (
                train_dataset,
                val_dataset,
                val_predict_dataset,
                test_predict_dataset,
            ) = multi_data_bundle

        started = time.perf_counter()
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2,
            callbacks=cbs,
        )
        fit_seconds[name] = float(time.perf_counter() - started)
        histories[name] = history

        val_pred = model.predict(val_predict_dataset, verbose=0)
        test_pred = model.predict(test_predict_dataset, verbose=0)
        val_proba = _extract_artist_proba(val_pred)
        test_proba = _extract_artist_proba(test_pred)
        val_top1 = _weighted_top1_accuracy_from_proba(val_proba, data.y_val, weights.artist_val)
        val_top5 = _weighted_topk_accuracy_from_proba(val_proba, data.y_val, weights.artist_val, k=5)
        top1 = _weighted_top1_accuracy_from_proba(test_proba, data.y_test, weights.artist_test)
        top5 = _weighted_topk_accuracy_from_proba(test_proba, data.y_test, weights.artist_test, k=5)
        bundle_path = save_prediction_bundle(
            prediction_output_dir / f"deep_{name}.npz",
            val_proba=val_proba,
            test_proba=test_proba,
        )
        prediction_bundle_paths[name] = str(bundle_path)
        val_ranking = ranking_metrics_from_proba(
            val_proba,
            data.y_val,
            num_items=data.num_artists,
            k=5,
        )
        test_ranking = ranking_metrics_from_proba(
            test_proba,
            data.y_test,
            num_items=data.num_artists,
            k=5,
        )

        test_metrics[name] = {
            "top1": top1,
            "top5": top5,
            "ndcg_at5": float(test_ranking["ndcg_at_k"]),
            "mrr_at5": float(test_ranking["mrr_at_k"]),
            "coverage_at5": float(test_ranking["coverage_at_k"]),
            "diversity_at5": float(test_ranking["diversity_at_k"]),
        }
        val_metrics[name] = {
            "top1": val_top1,
            "top5": val_top5,
            "ndcg_at5": float(val_ranking["ndcg_at_k"]),
            "mrr_at5": float(val_ranking["mrr_at_k"]),
            "coverage_at5": float(val_ranking["coverage_at_k"]),
            "diversity_at5": float(val_ranking["diversity_at_k"]),
        }
        logger.info("[TEST] %s: Top-1=%.4f | Top-5=%.4f", name, top1, top5)
        if resolved_cache_plan.enabled and name in cache_contexts and checkpoint_path.exists():
            cache_paths, cache_payload = cache_contexts[name]
            _save_deep_artifact_to_cache(
                cache_paths=cache_paths,
                cache_payload=cache_payload,
                history=history,
                val_metrics=val_metrics[name],
                test_metrics=test_metrics[name],
                fit_seconds=fit_seconds[name],
                checkpoint_path=checkpoint_path,
                prediction_bundle_path=bundle_path,
                logger=logger,
            )

    return TrainingArtifacts(
        histories=_ordered_by_selected(histories, selected_model_names),
        test_metrics=_ordered_by_selected(test_metrics, selected_model_names),
        val_metrics=_ordered_by_selected(val_metrics, selected_model_names),
        fit_seconds=_ordered_by_selected(fit_seconds, selected_model_names),
        prediction_bundle_paths=_ordered_by_selected(prediction_bundle_paths, selected_model_names),
    )
