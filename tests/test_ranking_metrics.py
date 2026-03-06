from __future__ import annotations

import numpy as np

from spotify.ranking import ranking_metrics_from_proba


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
