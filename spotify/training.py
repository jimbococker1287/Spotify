from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.utils import class_weight

from .data import PreparedData


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


def _build_datasets(data: PreparedData, weights: SampleWeights, batch_size: int, single_head: bool, tf):
    if single_head:
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((data.X_seq_train, data.X_ctx_train), data.y_train, weights.artist_train)
        ).shuffle(10_000).cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            ((data.X_seq_val, data.X_ctx_val), data.y_val, weights.artist_val)
        ).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices(
            ((data.X_seq_test, data.X_ctx_test), data.y_test, weights.artist_test)
        ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds

    train_sw = (weights.artist_train, weights.skip_train)
    val_sw = (weights.artist_val, weights.skip_val)
    test_sw = (weights.artist_test, weights.skip_test)

    train_ds = tf.data.Dataset.from_tensor_slices(
        ((data.X_seq_train, data.X_ctx_train), (data.y_train, data.y_skip_train), train_sw)
    ).shuffle(10_000).cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ((data.X_seq_val, data.X_ctx_val), (data.y_val, data.y_skip_val), val_sw)
    ).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        ((data.X_seq_test, data.X_ctx_test), (data.y_test, data.y_skip_test), test_sw)
    ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


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

    output_dir.mkdir(parents=True, exist_ok=True)

    weights = compute_sample_weights(data)
    histories: dict[str, object] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    for name, builder in model_builders:
        logger.info("Training %s model", name)

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
                )
                monitor_metric = "val_artist_output_sparse_categorical_accuracy"

            checkpoint_path = output_dir / f"best_{name}.keras"
            cbs = [
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

        train_ds, val_ds, test_ds = _build_datasets(data, weights, batch_size, single_head, tf)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=cbs,
        )
        histories[name] = history

        eval_result = model.evaluate(test_ds, verbose=0, return_dict=True)
        if single_head:
            top1 = float(eval_result.get("sparse_categorical_accuracy", np.nan))
            top5 = float(eval_result.get("top_5", np.nan))
        else:
            top1 = float(eval_result.get("artist_output_sparse_categorical_accuracy", np.nan))
            top5 = float(eval_result.get("artist_output_top_5", np.nan))

        test_metrics[name] = {"top1": top1, "top5": top5}
        logger.info("[TEST] %s: Top-1=%.4f | Top-5=%.4f", name, top1, top5)

    return TrainingArtifacts(histories=histories, test_metrics=test_metrics)
