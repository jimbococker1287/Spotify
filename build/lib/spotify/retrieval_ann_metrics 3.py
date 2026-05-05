from __future__ import annotations

import time

import numpy as np

from .benchmarks import sample_indices
from .ranking import topk_indices_1d
from .retrieval_common import DEFAULT_RETRIEVAL_ANN_EVAL_ROWS, RandomProjectionANNIndex, _env_int


def _ann_recall_and_latency(
    *,
    ann_index: RandomProjectionANNIndex,
    session_vec: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    random_seed: int,
) -> dict[str, float]:
    session_arr = np.asarray(session_vec, dtype="float32")
    score_arr = np.asarray(scores, dtype="float32")
    if len(session_arr) == 0 or score_arr.ndim != 2:
        return {
            "ann_recall_at_k": float("nan"),
            "exact_p50_ms": float("nan"),
            "exact_p95_ms": float("nan"),
            "ann_p50_ms": float("nan"),
            "ann_p95_ms": float("nan"),
        }
    total_rows = int(len(session_arr))
    ann_eval_rows = _env_int("SPOTIFY_RETRIEVAL_ANN_EVAL_ROWS", DEFAULT_RETRIEVAL_ANN_EVAL_ROWS)
    if ann_eval_rows > 0 and total_rows > ann_eval_rows:
        sampled = np.asarray(
            sample_indices(total_rows, ann_eval_rows, np.random.default_rng(random_seed)),
            dtype="int64",
        )
        session_arr = session_arr[sampled]
        score_arr = score_arr[sampled]
    exact_times: list[float] = []
    ann_times: list[float] = []
    recalls: list[float] = []
    pool = max(int(top_k) * 4, int(top_k))
    for row_idx in range(len(session_arr)):
        started = time.perf_counter()
        exact_idx = topk_indices_1d(score_arr[row_idx], top_k)
        exact_times.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        candidate_ids = ann_index.candidate_ids(session_arr[row_idx : row_idx + 1], candidate_pool=pool)[0]
        ann_scores = score_arr[row_idx, candidate_ids]
        ann_ranked = candidate_ids[topk_indices_1d(ann_scores, top_k)]
        ann_times.append((time.perf_counter() - started) * 1000.0)

        recalls.append(float(len(set(exact_idx.tolist()) & set(ann_ranked.tolist())) / max(1, top_k)))

    return {
        "ann_recall_at_k": float(np.mean(recalls)),
        "exact_p50_ms": float(np.percentile(exact_times, 50)),
        "exact_p95_ms": float(np.percentile(exact_times, 95)),
        "ann_p50_ms": float(np.percentile(ann_times, 50)),
        "ann_p95_ms": float(np.percentile(ann_times, 95)),
        "evaluated_rows": int(len(session_arr)),
        "total_rows": total_rows,
    }


__all__ = ["_ann_recall_and_latency"]
