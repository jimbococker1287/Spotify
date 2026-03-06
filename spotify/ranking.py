from __future__ import annotations

import numpy as np


def _topk_indices(proba: np.ndarray, k: int) -> np.ndarray:
    if proba.ndim != 2:
        return np.empty((0, 0), dtype="int32")
    kk = max(1, min(int(k), int(proba.shape[1])))
    topk = np.argpartition(proba, -kk, axis=1)[:, -kk:]
    topk_scores = np.take_along_axis(proba, topk, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    return np.take_along_axis(topk, order, axis=1)


def ranking_metrics_from_proba(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    num_items: int,
    k: int = 5,
) -> dict[str, float]:
    if proba.ndim != 2 or len(proba) == 0:
        return {
            "ndcg_at_k": float("nan"),
            "mrr_at_k": float("nan"),
            "coverage_at_k": float("nan"),
            "diversity_at_k": float("nan"),
        }

    y_true = y_true.astype(int).reshape(-1)
    topk = _topk_indices(np.asarray(proba), k)
    if topk.shape[0] != y_true.shape[0]:
        return {
            "ndcg_at_k": float("nan"),
            "mrr_at_k": float("nan"),
            "coverage_at_k": float("nan"),
            "diversity_at_k": float("nan"),
        }

    ndcg_scores = np.zeros(len(y_true), dtype="float64")
    mrr_scores = np.zeros(len(y_true), dtype="float64")
    for idx, true_label in enumerate(y_true):
        matches = np.where(topk[idx] == true_label)[0]
        if matches.size == 0:
            continue
        rank = int(matches[0]) + 1
        ndcg_scores[idx] = 1.0 / np.log2(rank + 1.0)
        mrr_scores[idx] = 1.0 / float(rank)

    coverage = float("nan")
    diversity = float("nan")
    if num_items > 0 and topk.size > 0:
        unique_items = int(np.unique(topk).size)
        coverage = float(unique_items) / float(num_items)

        flat = topk.reshape(-1).astype(int)
        counts = np.bincount(flat, minlength=max(1, int(num_items))).astype("float64")
        total = counts.sum()
        if total > 0:
            p = counts[counts > 0] / total
            diversity = float(1.0 - np.sum(np.square(p)))

    return {
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "mrr_at_k": float(np.mean(mrr_scores)),
        "coverage_at_k": coverage,
        "diversity_at_k": diversity,
    }
