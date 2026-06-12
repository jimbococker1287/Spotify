from __future__ import annotations

import numpy as np

from spotify.benchmarks import evaluate_classical_estimator
from spotify.ranking import ranking_metrics_from_proba


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

    val_top1, val_top5, test_top1, test_top5, val_ranking, test_ranking = evaluate_classical_estimator(
        estimator,
        X,
        y,
        X,
        y,
    )
    expected_ranking = ranking_metrics_from_proba(
        estimator._proba,
        np.array([1, 2, 0], dtype="int64"),
        num_items=3,
        k=5,
    )

    assert np.isclose(val_top1, 1.0)
    assert np.isclose(test_top1, 1.0)
    assert np.isclose(val_top5, 1.0)
    assert np.isclose(test_top5, 1.0)
    for result in (val_ranking, test_ranking):
        assert np.isclose(result["ndcg_at5"], expected_ranking["ndcg_at_k"])
        assert np.isclose(result["mrr_at5"], expected_ranking["mrr_at_k"])
        assert np.isclose(result["coverage_at5"], expected_ranking["coverage_at_k"])
        assert np.isclose(result["diversity_at5"], expected_ranking["diversity_at_k"])


def test_evaluate_classical_estimator_avoids_predict_when_predict_proba_is_available() -> None:
    class _CountingEstimator(_DummyEstimator):
        def __init__(self) -> None:
            super().__init__()
            self.predict_calls = 0
            self.predict_proba_calls = 0

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.predict_calls += 1
            return super().predict(X)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.predict_proba_calls += 1
            return super().predict_proba(X)

    estimator = _CountingEstimator()
    X = np.zeros((3, 2), dtype="float32")
    y = np.array([20, 30, 10], dtype="int64")

    evaluate_classical_estimator(estimator, X, y, X, y)

    assert estimator.predict_calls == 0
    assert estimator.predict_proba_calls == 2
