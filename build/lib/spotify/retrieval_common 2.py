from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os

import numpy as np

from .ranking import ranking_metrics_from_proba

DEFAULT_EMBEDDING_DIM = 32
DEFAULT_PRETRAIN_EPOCHS = 5
DEFAULT_PRETRAIN_MAX_PAIRS = 1_000_000
DEFAULT_RETRIEVAL_EPOCHS = 6
DEFAULT_RETRIEVAL_CANDIDATE_K = 30
DEFAULT_RETRIEVAL_ANN_EVAL_ROWS = 4_096


@dataclass(frozen=True)
class SelfSupervisedPretrainingResult:
    objective_name: str
    embedding_dim: int
    artist_embeddings: np.ndarray
    artist_frequency: np.ndarray
    pair_count: int
    epochs: int
    learning_rate: float
    window_size: int
    negatives: int


@dataclass(frozen=True)
class RandomProjectionANNIndex:
    hyperplanes: np.ndarray
    bucket_codes: np.ndarray
    bucket_lookup: dict[int, np.ndarray]

    def candidate_ids(self, queries: np.ndarray, *, candidate_pool: int) -> list[np.ndarray]:
        query_arr = np.asarray(queries, dtype="float32")
        if query_arr.ndim != 2:
            return []
        raw_codes = _hash_vectors(query_arr, self.hyperplanes)
        item_count = int(len(self.bucket_codes))
        target_pool = max(1, min(int(candidate_pool), item_count))
        out: list[np.ndarray] = []
        for code in raw_codes.tolist():
            candidates = list(np.asarray(self.bucket_lookup.get(int(code), np.array([], dtype="int32")), dtype="int32").tolist())
            if len(candidates) < target_pool:
                for bit_idx in range(int(self.hyperplanes.shape[0])):
                    neighbor_code = int(code) ^ (1 << bit_idx)
                    candidates.extend(
                        np.asarray(self.bucket_lookup.get(neighbor_code, np.array([], dtype="int32")), dtype="int32").tolist()
                    )
                    if len(candidates) >= target_pool:
                        break
            if not candidates:
                candidates = list(range(item_count))
            unique = np.asarray(sorted(set(candidates)), dtype="int32")
            out.append(unique[:target_pool])
        return out


@dataclass
class RetrievalServingArtifact:
    model_name: str
    candidate_k: int
    artist_embeddings: np.ndarray
    sequence_projection: np.ndarray
    context_projection: np.ndarray
    item_bias: np.ndarray
    popularity: np.ndarray
    ann_index: RandomProjectionANNIndex | None = None
    reranker: object | None = None

    def score_items(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_arr = np.asarray(seq_batch, dtype="int32")
        ctx_arr = np.asarray(ctx_batch, dtype="float32")
        session_base = _weighted_session_base(seq_arr, self.artist_embeddings)
        session_vec = (session_base @ self.sequence_projection) + (ctx_arr @ self.context_projection)
        scores = session_vec @ self.artist_embeddings.T
        scores += np.asarray(self.item_bias, dtype="float32").reshape(1, -1)
        return session_vec.astype("float32", copy=False), scores.astype("float32", copy=False)

    def predict_proba(self, seq_batch: np.ndarray, ctx_batch: np.ndarray) -> np.ndarray:
        session_vec, scores = self.score_items(seq_batch, ctx_batch)
        if self.reranker is None:
            return _softmax_rows(scores)
        from .retrieval_stack import _predict_reranked_probabilities

        return _predict_reranked_probabilities(
            seq_batch=np.asarray(seq_batch, dtype="int32"),
            ctx_batch=np.asarray(ctx_batch, dtype="float32"),
            session_vec=session_vec,
            retrieval_scores=scores,
            artifact=self,
        )


@dataclass(frozen=True)
class RetrievalExperimentResult:
    rows: list[dict[str, object]]
    artifact_paths: list[Path]


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return float(value)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype="float32"), -18.0, 18.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype("float32", copy=False)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype="float32")
    if scores.ndim != 2:
        return np.empty((0, 0), dtype="float32")
    exp = scores.copy()
    exp -= exp.max(axis=1, keepdims=True)
    np.exp(exp, out=exp)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    exp /= denom
    return exp.astype("float32", copy=False)


def _resolve_pretraining_objectives() -> tuple[str, ...]:
    raw = str(os.environ.get("SPOTIFY_PRETRAIN_OBJECTIVES", "")).strip()
    if not raw:
        return ("cooccurrence", "masked_tail", "contrastive_session")
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return values or ("cooccurrence",)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim != 2:
        return arr.astype("float32", copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-8] = 1.0
    return (arr / norms).astype("float32", copy=False)


def _sequence_weights(sequence_length: int) -> np.ndarray:
    length = max(1, int(sequence_length))
    weights = np.linspace(1.0, 2.0, num=length, dtype="float32")
    weights /= np.sum(weights)
    return weights


def _weighted_session_base(seq_batch: np.ndarray, artist_embeddings: np.ndarray) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    if seq_arr.ndim != 2:
        return np.empty((0, artist_embeddings.shape[1]), dtype="float32")
    embedding_lookup = np.asarray(artist_embeddings, dtype="float32")[seq_arr]
    weights = _sequence_weights(seq_arr.shape[1]).reshape(1, -1, 1)
    return np.sum(embedding_lookup * weights, axis=1, dtype="float32")


def _topk_indices_and_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    score_arr = np.asarray(scores, dtype="float32")
    if score_arr.ndim != 2:
        return np.empty((0, 0), dtype="int32"), np.empty((0, 0), dtype="float32")
    kk = max(1, min(int(k), int(score_arr.shape[1])))
    idx = np.argpartition(score_arr, -kk, axis=1)[:, -kk:]
    top_scores = np.take_along_axis(score_arr, idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    return (
        np.take_along_axis(idx, order, axis=1).astype("int32", copy=False),
        np.take_along_axis(top_scores, order, axis=1).astype("float32", copy=False),
    )


def _topk_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    candidate_ids, _ = _topk_indices_and_scores(scores, k)
    if len(candidate_ids) == 0:
        return float("nan")
    return float(np.mean(np.any(candidate_ids == np.asarray(y_true).reshape(-1, 1), axis=1)))


def _hash_vectors(vectors: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    projected = np.asarray(vectors, dtype="float32") @ np.asarray(hyperplanes, dtype="float32").T
    bits = projected >= 0.0
    powers = (1 << np.arange(bits.shape[1], dtype="uint64")).reshape(1, -1)
    return np.sum(bits.astype("uint64") * powers, axis=1).astype("uint64")


def _prediction_metrics(proba: np.ndarray, y_true: np.ndarray, *, num_items: int) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    proba_arr = np.asarray(proba, dtype="float32")
    if proba_arr.ndim != 2 or len(proba_arr) != len(y_arr):
        return {
            "top1": float("nan"),
            "top5": float("nan"),
            "ndcg_at5": float("nan"),
            "mrr_at5": float("nan"),
            "coverage_at5": float("nan"),
            "diversity_at5": float("nan"),
        }
    top1 = float(np.mean(np.argmax(proba_arr, axis=1) == y_arr))
    top5 = _topk_accuracy(proba_arr, y_arr, 5)
    ranking = ranking_metrics_from_proba(proba_arr, y_arr, num_items=num_items, k=5)
    return {
        "top1": top1,
        "top5": top5,
        "ndcg_at5": float(ranking["ndcg_at_k"]),
        "mrr_at5": float(ranking["mrr_at_k"]),
        "coverage_at5": float(ranking["coverage_at_k"]),
        "diversity_at5": float(ranking["diversity_at_k"]),
    }


def _positive_label_index(estimator) -> int:
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.size == 0:
        return 1
    matches = np.where(classes == 1)[0]
    if matches.size:
        return int(matches[0])
    return int(min(len(classes) - 1, 1))


def _build_pretraining_pairs(seq_batch: np.ndarray, window_size: int, objective_name: str) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    if seq_arr.ndim != 2 or seq_arr.shape[1] < 2:
        return np.empty((0, 2), dtype="int32")
    pairs: list[tuple[int, int]] = []
    width = max(1, int(window_size))
    for row in seq_arr:
        unique_row = np.unique(row)
        tail = int(row[-1])
        context = row[max(0, len(row) - width - 1) : -1]
        if objective_name == "masked_tail":
            for item in np.unique(context):
                item_id = int(item)
                if item_id != tail:
                    pairs.append((item_id, tail))
        elif objective_name == "contrastive_session":
            for left in unique_row.tolist():
                for right in unique_row.tolist():
                    if int(left) != int(right):
                        pairs.append((int(left), int(right)))
        else:
            for item in np.unique(context):
                item_id = int(item)
                if item_id == tail:
                    continue
                pairs.append((tail, item_id))
                pairs.append((item_id, tail))
    if not pairs:
        return np.empty((0, 2), dtype="int32")
    return np.asarray(pairs, dtype="int32")
