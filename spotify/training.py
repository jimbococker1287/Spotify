from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import numpy as np
from sklearn.utils import class_weight

from .data import PreparedData
from .ranking import ranking_metrics_from_proba


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


def compute_baselines(data: PreparedData, logger) -> dict[str, float]:
    majority_artist = int(np.bincount(data.y_train.astype(int)).argmax())
    majority_top1 = float(np.mean(data.y_val == majority_artist))

    last_artist_pred = data.X_seq_val[:, -1]
    last_top1 = float(np.mean(last_artist_pred == data.y_val))

    num_states = int(data.num_artists)
    transitions = np.ones((num_states, num_states), dtype=np.float64)
    for row in data.X_seq_train:
        for prev, nxt in zip(row[:-1], row[1:]):
            transitions[prev, nxt] += 1.0
    for prev, nxt in zip(data.X_seq_train[:, -1], data.y_train):
        transitions[prev, nxt] += 1.0

    markov_pred = np.argmax(transitions[last_artist_pred], axis=1)
    markov_top1 = float(np.mean(markov_pred == data.y_val))

    logger.info("Baseline (majority artist) Top-1 Val Acc: %.4f", majority_top1)
    logger.info("Baseline (last artist) Top-1 Val Acc: %.4f", last_top1)
    logger.info("Baseline (1-step Markov) Top-1 Val Acc: %.4f", markov_top1)

    return {
        "majority_top1": majority_top1,
        "last_artist_top1": last_top1,
        "markov_top1": markov_top1,
    }


def compute_sample_weights(data: PreparedData) -> SampleWeights:
    artist_classes = np.unique(data.y_train.astype(int))
    artist_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=artist_classes,
        y=data.y_train.astype(int),
    )
    artist_weight_map = dict(zip(artist_classes, artist_weights))

    skip_classes = np.unique(data.y_skip_train.astype(int))
    skip_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=skip_classes,
        y=data.y_skip_train.astype(int),
    )
    skip_weight_map = dict(zip(skip_classes, skip_weights))

    return SampleWeights(
        artist_train=np.array([artist_weight_map.get(int(y), 1.0) for y in data.y_train], dtype="float32"),
        artist_val=np.array([artist_weight_map.get(int(y), 1.0) for y in data.y_val], dtype="float32"),
        artist_test=np.array([artist_weight_map.get(int(y), 1.0) for y in data.y_test], dtype="float32"),
        skip_train=np.array([skip_weight_map.get(int(y), 1.0) for y in data.y_skip_train], dtype="float32"),
        skip_val=np.array([skip_weight_map.get(int(y), 1.0) for y in data.y_skip_val], dtype="float32"),
        skip_test=np.array([skip_weight_map.get(int(y), 1.0) for y in data.y_skip_test], dtype="float32"),
    )


def train_and_evaluate_models(
    data: PreparedData,
    model_builders,
    batch_size: int,
    epochs: int,
    output_dir: Path,
    strategy,
    logger,
) -> TrainingArtifacts:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    def _parse_pos_int(raw: str | None, fallback: int) -> int:
        try:
            value = int(str(raw))
            if value > 0:
                return value
        except Exception:
            pass
        return fallback

    def _parse_bool(raw: str | None, default: bool) -> bool:
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
        return default

    def _resolve_dataset_cache_enabled() -> tuple[bool, str]:
        raw = os.getenv("SPOTIFY_TF_DATA_CACHE", "auto").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True, "forced_on"
        if raw in ("0", "false", "no", "off"):
            return False, "forced_off"

        # Auto mode: only cache if tensors are a modest fraction of available RAM.
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
            threshold_bytes = int(available_bytes * 0.30)
            enabled = approx_bytes <= threshold_bytes
            reason = f"auto(approx={approx_bytes // (1024**2)}MiB avail={available_bytes // (1024**2)}MiB)"
            return enabled, reason
        except Exception:
            # Conservative fallback without psutil.
            return False, "auto(no_psutil)"

    def _extract_artist_proba(prediction) -> np.ndarray:
        if isinstance(prediction, dict):
            if "artist_output" in prediction:
                return np.asarray(prediction["artist_output"])
            first_value = next(iter(prediction.values()))
            return np.asarray(first_value)
        if isinstance(prediction, (list, tuple)):
            return np.asarray(prediction[0])
        return np.asarray(prediction)

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
            val_top1 = logs.get("val_sparse_categorical_accuracy", logs.get("val_artist_output_sparse_categorical_accuracy", float("nan")))
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

    output_dir.mkdir(parents=True, exist_ok=True)

    weights = compute_sample_weights(data)
    histories: dict[str, object] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    val_metrics: dict[str, dict[str, float]] = {}
    fit_seconds: dict[str, float] = {}

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

    for name, builder in model_builders:
        logger.info("Training %s model", name)
        logger.info("[%s] run_eagerly=%s", name, run_eagerly)
        logger.info("[%s] steps_per_execution=%d", name, steps_per_execution)

        with strategy.scope():
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
                test_dataset = _build_weighted_dataset(
                    features=(data.X_seq_test, data.X_ctx_test),
                    labels=data.y_test,
                    sample_weights=weights.artist_test,
                    training=False,
                    seed=0,
                )
                val_predict_dataset = _build_feature_dataset((data.X_seq_val, data.X_ctx_val))
                test_predict_dataset = _build_feature_dataset((data.X_seq_test, data.X_ctx_test))
                single_data_bundle = (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    val_predict_dataset,
                    test_predict_dataset,
                )
            (
                train_dataset,
                val_dataset,
                test_dataset,
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
                test_dataset = _build_weighted_dataset(
                    features=(data.X_seq_test, data.X_ctx_test),
                    labels=(data.y_test, data.y_skip_test),
                    sample_weights=(weights.artist_test, weights.skip_test),
                    training=False,
                    seed=0,
                )
                val_predict_dataset = _build_feature_dataset((data.X_seq_val, data.X_ctx_val))
                test_predict_dataset = _build_feature_dataset((data.X_seq_test, data.X_ctx_test))
                multi_data_bundle = (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    val_predict_dataset,
                    test_predict_dataset,
                )
            (
                train_dataset,
                val_dataset,
                test_dataset,
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

        eval_result = model.evaluate(
            test_dataset,
            verbose=0,
            return_dict=True,
        )
        if single_head:
            top1 = float(eval_result.get("sparse_categorical_accuracy", np.nan))
            top5 = float(eval_result.get("top_5", np.nan))
            val_top1 = float(history.history.get("val_sparse_categorical_accuracy", [np.nan])[-1])
            val_top5 = float(history.history.get("val_top_5", [np.nan])[-1])
        else:
            top1 = float(eval_result.get("artist_output_sparse_categorical_accuracy", np.nan))
            top5 = float(eval_result.get("artist_output_top_5", np.nan))
            val_top1 = float(history.history.get("val_artist_output_sparse_categorical_accuracy", [np.nan])[-1])
            val_top5 = float(history.history.get("val_artist_output_top_5", [np.nan])[-1])
        val_pred = model.predict(val_predict_dataset, verbose=0)
        test_pred = model.predict(test_predict_dataset, verbose=0)
        val_proba = _extract_artist_proba(val_pred)
        test_proba = _extract_artist_proba(test_pred)
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

    return TrainingArtifacts(
        histories=histories,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        fit_seconds=fit_seconds,
    )
