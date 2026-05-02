from __future__ import annotations

from pathlib import Path

import numpy as np


def align_proba_to_num_classes(
    proba: np.ndarray,
    class_labels: np.ndarray | None,
    num_classes: int,
) -> np.ndarray:
    proba = np.asarray(proba, dtype="float32")
    if proba.ndim != 2:
        return np.empty((0, max(0, int(num_classes))), dtype="float32")

    num_classes = max(0, int(num_classes))
    if num_classes <= 0:
        return np.empty((len(proba), 0), dtype="float32")

    if class_labels is None:
        if proba.shape[1] == num_classes:
            return proba.astype("float32", copy=False)
        width = min(num_classes, int(proba.shape[1]))
        aligned = np.zeros((len(proba), num_classes), dtype="float32")
        aligned[:, :width] = proba[:, :width]
    else:
        labels = np.asarray(class_labels).reshape(-1)
        aligned = np.zeros((len(proba), num_classes), dtype="float32")
        width = min(len(labels), int(proba.shape[1]))
        for src_idx in range(width):
            try:
                label_idx = int(labels[src_idx])
            except Exception:
                continue
            if 0 <= label_idx < num_classes:
                aligned[:, label_idx] = proba[:, src_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    if np.any(valid):
        aligned[valid] = aligned[valid] / row_sums[valid]
    return aligned.astype("float32", copy=False)


def save_prediction_bundle(
    path: Path,
    *,
    val_proba: np.ndarray,
    test_proba: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        val_proba=np.asarray(val_proba, dtype="float32"),
        test_proba=np.asarray(test_proba, dtype="float32"),
    )
    return path


def load_prediction_bundle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["val_proba"], dtype="float32"),
            np.asarray(payload["test_proba"], dtype="float32"),
        )
