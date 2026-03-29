from __future__ import annotations

import numpy as np

from spotify.ranking import ranking_metrics_from_proba, topk_indices_1d, topk_indices_2d


def test_ranking_metrics_from_proba_smoke() -> None:
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.4, 0.35, 0.25],
        ],
        dtype="float32",
    )
    y_true = np.array([0, 2, 1], dtype="int32")
    metrics = ranking_metrics_from_proba(proba, y_true, num_items=3, k=2)

    assert np.isclose(metrics["ndcg_at_k"], 0.8769, atol=1e-3)
    assert np.isclose(metrics["mrr_at_k"], 0.8333, atol=1e-3)
    assert np.isclose(metrics["coverage_at_k"], 1.0, atol=1e-6)
    assert np.isclose(metrics["diversity_at_k"], 0.6111, atol=1e-3)


def test_topk_indices_1d_matches_full_sort_for_small_and_large_vectors() -> None:
    small_scores = np.array([0.2, 0.8, 0.5, 0.4], dtype="float32")
    assert topk_indices_1d(small_scores, 3).tolist() == np.argsort(small_scores)[::-1][:3].tolist()

    rng = np.random.default_rng(42)
    large_scores = rng.random(512, dtype="float32")
    assert topk_indices_1d(large_scores, 7).tolist() == np.argsort(large_scores)[::-1][:7].tolist()


def test_topk_indices_2d_matches_full_sort_for_small_and_large_matrices() -> None:
    small_scores = np.array(
        [
            [0.2, 0.8, 0.5, 0.4],
            [0.7, 0.1, 0.6, 0.3],
        ],
        dtype="float32",
    )
    assert topk_indices_2d(small_scores, 3).tolist() == np.argsort(small_scores, axis=1)[:, ::-1][:, :3].tolist()

    rng = np.random.default_rng(7)
    large_scores = rng.random((64, 512), dtype="float32")
    assert topk_indices_2d(large_scores, 5).tolist() == np.argsort(large_scores, axis=1)[:, ::-1][:, :5].tolist()
