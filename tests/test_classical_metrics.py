from __future__ import annotations

import numpy as np

from spotify.benchmarks import evaluate_classical_estimator


class _DummyEstimator:
    def __init__(self) -> None:
        self.classes_ = np.array([10, 20, 30], dtype="int64")
        self._proba = np.array(
            [
                [0.10, 0.85, 0.05],  # true=20
                [0.20, 0.30, 0.50],  # true=30
                [0.70, 0.20, 0.10],  # true=10
            ],
            dtype="float32",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self._proba[: len(X)], axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba[: len(X)]


def test_evaluate_classical_estimator_respects_non_contiguous_class_labels() -> None:
    estimator = _DummyEstimator()
    X = np.zeros((3, 2), dtype="float32")
    y = np.array([20, 30, 10], dtype="int64")

    val_top1, val_top5, test_top1, test_top5, _, _ = evaluate_classical_estimator(estimator, X, y, X, y)

    assert np.isclose(val_top1, 1.0)
    assert np.isclose(test_top1, 1.0)
    assert np.isclose(val_top5, 1.0)
    assert np.isclose(test_top5, 1.0)
