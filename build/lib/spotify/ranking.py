from __future__ import annotations

import numpy as np

_TOPK_1D_ARGPARTITION_MIN_SIZE = 128


def topk_indices_1d(scores: np.ndarray, k: int) -> np.ndarray:
    score_arr = np.asarray(scores)
    if score_arr.ndim != 1 or score_arr.size == 0:
        return np.empty(0, dtype="int32")
    kk = max(1, min(int(k), int(score_arr.shape[0])))
    if kk >= score_arr.shape[0] or score_arr.shape[0] <= _TOPK_1D_ARGPARTITION_MIN_SIZE:
        return np.asarray(np.argsort(score_arr)[::-1][:kk], dtype="int32")
    idx = np.argpartition(score_arr, -kk)[-kk:]
    top_scores = score_arr[idx]
    order = np.argsort(top_scores)[::-1]
    return np.asarray(idx[order], dtype="int32")


def topk_indices_2d(proba: np.ndarray, k: int) -> np.ndarray:
    proba_arr = np.asarray(proba)
    if proba_arr.ndim != 2 or proba_arr.size == 0:
        return np.empty((0, 0), dtype="int32")
    kk = max(1, min(int(k), int(proba_arr.shape[1])))
    if kk >= proba_arr.shape[1]:
        return np.asarray(np.argsort(proba_arr, axis=1)[:, ::-1][:, :kk], dtype="int32")
    topk = np.argpartition(proba_arr, -kk, axis=1)[:, -kk:]
    topk_scores = np.take_along_axis(proba_arr, topk, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    return np.asarray(np.take_along_axis(topk, order, axis=1), dtype="int32")


def _topk_indices(proba: np.ndarray, k: int) -> np.ndarray:
    return topk_indices_2d(proba, k)


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
    topk = topk_indices_2d(np.asarray(proba), k)
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
