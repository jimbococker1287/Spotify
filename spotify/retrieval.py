from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PreparedData
from .probability_bundles import save_prediction_bundle
from .ranking import ranking_metrics_from_proba

DEFAULT_EMBEDDING_DIM = 32
DEFAULT_PRETRAIN_EPOCHS = 5
DEFAULT_RETRIEVAL_EPOCHS = 6
DEFAULT_RETRIEVAL_CANDIDATE_K = 30


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
    clipped = np.clip(np.asarray(values, dtype="float64"), -18.0, 18.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype="float64")
    if scores.ndim != 2:
        return np.empty((0, 0), dtype="float32")
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (exp / denom).astype("float32")


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


def _build_ann_index(artist_embeddings: np.ndarray, *, random_seed: int) -> RandomProjectionANNIndex:
    rng = np.random.default_rng(random_seed)
    num_bits = _env_int("SPOTIFY_RETRIEVAL_ANN_BITS", 10)
    hyperplanes = rng.normal(size=(num_bits, artist_embeddings.shape[1])).astype("float32")
    bucket_codes = _hash_vectors(np.asarray(artist_embeddings, dtype="float32"), hyperplanes)
    bucket_lookup: dict[int, list[int]] = {}
    for idx, code in enumerate(bucket_codes.tolist()):
        bucket_lookup.setdefault(int(code), []).append(int(idx))
    return RandomProjectionANNIndex(
        hyperplanes=hyperplanes,
        bucket_codes=bucket_codes.astype("uint64"),
        bucket_lookup={int(code): np.asarray(indices, dtype="int32") for code, indices in bucket_lookup.items()},
    )


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


def train_self_supervised_artist_embeddings(
    *,
    data: PreparedData,
    output_dir: Path,
    random_seed: int,
    logger,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    objective_name: str = "cooccurrence",
) -> tuple[SelfSupervisedPretrainingResult, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = _env_int("SPOTIFY_PRETRAIN_EPOCHS", DEFAULT_PRETRAIN_EPOCHS)
    negatives = _env_int("SPOTIFY_PRETRAIN_NEGATIVES", 4)
    batch_size = _env_int("SPOTIFY_PRETRAIN_BATCH_SIZE", 256)
    window_size = _env_int("SPOTIFY_PRETRAIN_WINDOW", 3)
    learning_rate = _env_float("SPOTIFY_PRETRAIN_LR", 0.045)
    l2 = _env_float("SPOTIFY_PRETRAIN_L2", 1e-4)

    rng = np.random.default_rng(random_seed)
    objective = str(objective_name).strip().lower() or "cooccurrence"
    pairs = _build_pretraining_pairs(data.X_seq_train, window_size=window_size, objective_name=objective)
    if len(pairs) == 0:
        embeddings = _normalize_rows(rng.normal(scale=0.05, size=(data.num_artists, embedding_dim)).astype("float32"))
        frequency = np.full(data.num_artists, 1.0 / max(1, data.num_artists), dtype="float32")
        result = SelfSupervisedPretrainingResult(
            objective_name=objective,
            embedding_dim=embedding_dim,
            artist_embeddings=embeddings,
            artist_frequency=frequency,
            pair_count=0,
            epochs=0,
            learning_rate=learning_rate,
            window_size=window_size,
            negatives=negatives,
        )
        artifact_path = output_dir / f"self_supervised_artist_embeddings_{objective}.joblib"
        joblib.dump(result, artifact_path, compress=3)
        return result, artifact_path

    counts = np.bincount(
        np.concatenate([data.X_seq_train.reshape(-1), data.y_train.reshape(-1)]).astype("int32"),
        minlength=data.num_artists,
    ).astype("float64")
    sampling = np.power(counts + 1.0, 0.75)
    sampling /= np.sum(sampling)

    input_embeddings = rng.normal(scale=0.04, size=(data.num_artists, embedding_dim)).astype("float32")
    output_embeddings = rng.normal(scale=0.04, size=(data.num_artists, embedding_dim)).astype("float32")

    logger.info(
        "Training self-supervised session embeddings: objective=%s pairs=%d dim=%d epochs=%d negatives=%d",
        objective,
        len(pairs),
        embedding_dim,
        epochs,
        negatives,
    )

    pair_indices = np.arange(len(pairs), dtype="int64")
    for _epoch in range(epochs):
        rng.shuffle(pair_indices)
        shuffled = pairs[pair_indices]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            centers = batch[:, 0]
            positives = batch[:, 1]
            negative_ids = rng.choice(
                data.num_artists,
                size=(len(batch), negatives),
                replace=True,
                p=sampling,
            ).astype("int32")

            center_vec = input_embeddings[centers]
            pos_vec = output_embeddings[positives]
            neg_vec = output_embeddings[negative_ids]

            pos_logits = np.sum(center_vec * pos_vec, axis=1)
            neg_logits = np.sum(center_vec[:, None, :] * neg_vec, axis=2)
            pos_factor = _sigmoid(pos_logits) - 1.0
            neg_factor = _sigmoid(neg_logits)

            grad_center = (pos_factor[:, None] * pos_vec) + np.sum(neg_factor[:, :, None] * neg_vec, axis=1)
            grad_pos = pos_factor[:, None] * center_vec
            grad_neg = neg_factor[:, :, None] * center_vec[:, None, :]

            for row_idx in range(len(batch)):
                center_id = int(centers[row_idx])
                pos_id = int(positives[row_idx])
                input_embeddings[center_id] -= learning_rate * (
                    grad_center[row_idx] + l2 * input_embeddings[center_id]
                )
                output_embeddings[pos_id] -= learning_rate * (
                    grad_pos[row_idx] + l2 * output_embeddings[pos_id]
                )
                for neg_id, neg_grad in zip(negative_ids[row_idx].tolist(), grad_neg[row_idx], strict=False):
                    output_embeddings[int(neg_id)] -= learning_rate * (
                        neg_grad + l2 * output_embeddings[int(neg_id)]
                    )

    embeddings = _normalize_rows((input_embeddings + output_embeddings) * 0.5)
    frequency = (counts / max(1.0, float(np.sum(counts)))).astype("float32")
    result = SelfSupervisedPretrainingResult(
        objective_name=objective,
        embedding_dim=embedding_dim,
        artist_embeddings=embeddings,
        artist_frequency=frequency,
        pair_count=int(len(pairs)),
        epochs=epochs,
        learning_rate=learning_rate,
        window_size=window_size,
        negatives=negatives,
    )
    artifact_path = output_dir / f"self_supervised_artist_embeddings_{objective}.joblib"
    joblib.dump(result, artifact_path, compress=3)
    return result, artifact_path


def _fit_dual_encoder(
    *,
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    artist_embeddings: np.ndarray,
    popularity: np.ndarray,
    random_seed: int,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(random_seed)
    epochs = _env_int("SPOTIFY_RETRIEVAL_EPOCHS", DEFAULT_RETRIEVAL_EPOCHS)
    batch_size = _env_int("SPOTIFY_RETRIEVAL_BATCH_SIZE", 256)
    learning_rate = _env_float("SPOTIFY_RETRIEVAL_LR", 0.055)
    l2 = _env_float("SPOTIFY_RETRIEVAL_L2", 2e-4)

    embed_dim = int(artist_embeddings.shape[1])
    sequence_projection = np.eye(embed_dim, dtype="float32")
    context_projection = rng.normal(scale=0.02, size=(ctx_train.shape[1], embed_dim)).astype("float32")
    item_bias = np.log(np.clip(np.asarray(popularity, dtype="float32"), 1e-6, 1.0)).astype("float32")

    session_base = _weighted_session_base(seq_train, artist_embeddings)
    indices = np.arange(len(seq_train), dtype="int64")
    logger.info(
        "Training dual-encoder retriever: train_rows=%d dim=%d epochs=%d batch=%d",
        len(seq_train),
        embed_dim,
        epochs,
        batch_size,
    )

    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            base_batch = session_base[batch_idx]
            ctx_batch = ctx_train[batch_idx]
            target = y_train[batch_idx].astype("int64")

            session_vec = (base_batch @ sequence_projection) + (ctx_batch @ context_projection)
            logits = session_vec @ artist_embeddings.T
            logits += item_bias.reshape(1, -1)
            proba = _softmax_rows(logits).astype("float64")
            proba[np.arange(len(target)), target] -= 1.0
            proba /= max(1, len(target))

            grad_session = proba @ artist_embeddings
            grad_sequence = base_batch.T @ grad_session
            grad_context = ctx_batch.T @ grad_session
            grad_bias = proba.sum(axis=0)

            sequence_projection -= learning_rate * (grad_sequence.astype("float32") + l2 * sequence_projection)
            context_projection -= learning_rate * (grad_context.astype("float32") + l2 * context_projection)
            item_bias -= learning_rate * (grad_bias.astype("float32") + l2 * item_bias)

    return sequence_projection, context_projection, item_bias, epochs


def _candidate_feature_matrix(
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
    session_vec: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    artist_embeddings: np.ndarray,
    popularity: np.ndarray,
) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    ctx_arr = np.asarray(ctx_batch, dtype="float32")
    candidates = np.asarray(candidate_ids, dtype="int32")
    scores = np.asarray(candidate_scores, dtype="float32")

    cand_emb = np.asarray(artist_embeddings, dtype="float32")[candidates]
    last_artist = seq_arr[:, -1].astype("int32")
    last_emb = np.asarray(artist_embeddings, dtype="float32")[last_artist]
    matches = seq_arr[:, :, None] == candidates[:, None, :]

    occurrence_fraction = matches.mean(axis=1, dtype="float32")
    in_session = matches.any(axis=1).astype("float32")
    is_last_artist = (candidates == last_artist.reshape(-1, 1)).astype("float32")
    rank_positions = np.broadcast_to(np.arange(candidates.shape[1], dtype="float32"), candidates.shape)
    rank_normalized = rank_positions / float(max(1, candidates.shape[1] - 1))
    top_score = scores[:, :1]
    score_margin = scores - top_score
    popularity_values = np.asarray(popularity, dtype="float32")[candidates]
    similarity_last = np.sum(cand_emb * last_emb[:, None, :], axis=2, dtype="float32")
    similarity_session = np.sum(cand_emb * session_vec[:, None, :], axis=2, dtype="float32")

    scalar_features = np.stack(
        [
            scores,
            score_margin,
            rank_normalized,
            popularity_values,
            in_session,
            occurrence_fraction,
            is_last_artist,
            similarity_last,
            similarity_session,
        ],
        axis=2,
    ).reshape(-1, 9)

    repeated_ctx = np.repeat(ctx_arr.astype("float32", copy=False), candidates.shape[1], axis=0)
    return np.concatenate([scalar_features.astype("float32", copy=False), repeated_ctx], axis=1)


def _inject_true_labels(
    *,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    y_true: np.ndarray,
    full_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids, dtype="int32").copy()
    scores = np.asarray(candidate_scores, dtype="float32").copy()
    y_arr = np.asarray(y_true, dtype="int32").reshape(-1)
    for row_idx in range(len(ids)):
        label = int(y_arr[row_idx])
        if np.any(ids[row_idx] == label):
            continue
        ids[row_idx, -1] = label
        scores[row_idx, -1] = float(full_scores[row_idx, label])
        order = np.argsort(-scores[row_idx])
        ids[row_idx] = ids[row_idx, order]
        scores[row_idx] = scores[row_idx, order]
    return ids, scores


def _fit_reranker(
    *,
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    retrieval_artifact: RetrievalServingArtifact,
    random_seed: int,
) -> object:
    train_cap = _env_int("SPOTIFY_RERANKER_TRAIN_ROWS", 8_000)
    rng = np.random.default_rng(random_seed)
    if len(seq_train) > train_cap:
        selected = np.asarray(rng.choice(len(seq_train), size=train_cap, replace=False), dtype="int64")
    else:
        selected = np.arange(len(seq_train), dtype="int64")

    seq_fit = seq_train[selected]
    ctx_fit = ctx_train[selected]
    y_fit = y_train[selected]
    session_vec, full_scores = retrieval_artifact.score_items(seq_fit, ctx_fit)
    candidate_ids, candidate_scores = _topk_indices_and_scores(full_scores, retrieval_artifact.candidate_k)
    candidate_ids, candidate_scores = _inject_true_labels(
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        y_true=y_fit,
        full_scores=full_scores,
    )
    features = _candidate_feature_matrix(
        seq_batch=seq_fit,
        ctx_batch=ctx_fit,
        session_vec=session_vec,
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        artist_embeddings=retrieval_artifact.artist_embeddings,
        popularity=retrieval_artifact.popularity,
    )
    labels = (candidate_ids == y_fit.reshape(-1, 1)).astype("int8").reshape(-1)

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    estimator.fit(features, labels)
    return estimator


def _predict_reranked_probabilities(
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
    session_vec: np.ndarray,
    retrieval_scores: np.ndarray,
    artifact: RetrievalServingArtifact,
) -> np.ndarray:
    seq_arr = np.asarray(seq_batch, dtype="int32")
    ctx_arr = np.asarray(ctx_batch, dtype="float32")
    session_arr = np.asarray(session_vec, dtype="float32")
    scores = np.asarray(retrieval_scores, dtype="float32")
    if artifact.reranker is None:
        return _softmax_rows(scores)

    candidate_ids, candidate_scores = _topk_indices_and_scores(scores, artifact.candidate_k)
    result = np.zeros_like(scores, dtype="float32")
    positive_index = _positive_label_index(artifact.reranker[-1] if hasattr(artifact.reranker, "__getitem__") else artifact.reranker)

    batch_size = _env_int("SPOTIFY_RERANKER_PREDICT_BATCH", 512)
    for start in range(0, len(seq_arr), batch_size):
        end = min(start + batch_size, len(seq_arr))
        feature_block = _candidate_feature_matrix(
            seq_batch=seq_arr[start:end],
            ctx_batch=ctx_arr[start:end],
            session_vec=session_arr[start:end],
            candidate_ids=candidate_ids[start:end],
            candidate_scores=candidate_scores[start:end],
            artist_embeddings=artifact.artist_embeddings,
            popularity=artifact.popularity,
        )
        rerank_scores = np.asarray(artifact.reranker.predict_proba(feature_block), dtype="float32")[:, positive_index]
        row_count = end - start
        rerank_scores = rerank_scores.reshape(row_count, artifact.candidate_k)
        if rerank_scores.shape != candidate_scores[start:end].shape:
            rerank_scores = candidate_scores[start:end].copy()
        rerank_scores += 1e-3 * _softmax_rows(candidate_scores[start:end])
        rerank_scores = np.clip(rerank_scores, 1e-6, None)
        rerank_scores = rerank_scores / rerank_scores.sum(axis=1, keepdims=True)
        rows = np.arange(row_count, dtype="int64")[:, None]
        result[start:end][rows, candidate_ids[start:end]] = rerank_scores
    return result.astype("float32", copy=False)


def _score_split(
    artifact: RetrievalServingArtifact,
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    session_vec, scores = artifact.score_items(seq_batch, ctx_batch)
    return _softmax_rows(scores), session_vec


def _ann_recall_and_latency(
    *,
    ann_index: RandomProjectionANNIndex,
    session_vec: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    if len(session_vec) == 0 or scores.ndim != 2:
        return {
            "ann_recall_at_k": float("nan"),
            "exact_p50_ms": float("nan"),
            "exact_p95_ms": float("nan"),
            "ann_p50_ms": float("nan"),
            "ann_p95_ms": float("nan"),
        }
    exact_times: list[float] = []
    ann_times: list[float] = []
    recalls: list[float] = []
    pool = max(int(top_k) * 4, int(top_k))
    for row_idx in range(len(session_vec)):
        started = time.perf_counter()
        exact_idx = np.argsort(scores[row_idx])[::-1][:top_k]
        exact_times.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        candidate_ids = ann_index.candidate_ids(session_vec[row_idx : row_idx + 1], candidate_pool=pool)[0]
        ann_scores = scores[row_idx, candidate_ids]
        ann_ranked = candidate_ids[np.argsort(ann_scores)[::-1][:top_k]]
        ann_times.append((time.perf_counter() - started) * 1000.0)

        recalls.append(float(len(set(exact_idx.tolist()) & set(ann_ranked.tolist())) / max(1, top_k)))

    return {
        "ann_recall_at_k": float(np.mean(recalls)),
        "exact_p50_ms": float(np.percentile(exact_times, 50)),
        "exact_p95_ms": float(np.percentile(exact_times, 95)),
        "ann_p50_ms": float(np.percentile(ann_times, 50)),
        "ann_p95_ms": float(np.percentile(ann_times, 95)),
    }


def train_retrieval_stack(
    *,
    data: PreparedData,
    output_dir: Path,
    random_seed: int,
    candidate_k: int,
    enable_self_supervised_pretraining: bool,
    logger,
) -> RetrievalExperimentResult:
    artifact_paths: list[Path] = []
    retrieval_dir = output_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "prediction_bundles"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir = output_dir / "pretraining"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    embedding_dim = _env_int("SPOTIFY_RETRIEVAL_DIM", DEFAULT_EMBEDDING_DIM)
    top_k = max(2, min(int(candidate_k), int(data.num_artists)))

    objective_rows: list[dict[str, object]] = []
    best_selection: tuple[SelfSupervisedPretrainingResult, Path, np.ndarray, np.ndarray, np.ndarray, int] | None = None
    best_val_top1 = float("-inf")

    if enable_self_supervised_pretraining:
        for objective_idx, objective_name in enumerate(_resolve_pretraining_objectives()):
            pretrain_result, pretrain_path = train_self_supervised_artist_embeddings(
                data=data,
                output_dir=pretrain_dir,
                random_seed=random_seed + objective_idx,
                logger=logger,
                embedding_dim=embedding_dim,
                objective_name=objective_name,
            )
            popularity = np.asarray(pretrain_result.artist_frequency, dtype="float32")
            sequence_projection, context_projection, item_bias, retrieval_epochs = _fit_dual_encoder(
                seq_train=data.X_seq_train,
                ctx_train=data.X_ctx_train,
                y_train=data.y_train,
                artist_embeddings=pretrain_result.artist_embeddings,
                popularity=popularity,
                random_seed=random_seed + objective_idx,
                logger=logger,
            )
            candidate_artifact = RetrievalServingArtifact(
                model_name=f"retrieval_dual_encoder_{objective_name}",
                candidate_k=top_k,
                artist_embeddings=np.asarray(pretrain_result.artist_embeddings, dtype="float32"),
                sequence_projection=np.asarray(sequence_projection, dtype="float32"),
                context_projection=np.asarray(context_projection, dtype="float32"),
                item_bias=np.asarray(item_bias, dtype="float32"),
                popularity=popularity.astype("float32"),
                ann_index=None,
                reranker=None,
            )
            candidate_val_proba, _ = _score_split(candidate_artifact, seq_batch=data.X_seq_val, ctx_batch=data.X_ctx_val)
            candidate_val_metrics = _prediction_metrics(candidate_val_proba, data.y_val, num_items=data.num_artists)
            objective_rows.append(
                {
                    "objective_name": objective_name,
                    "embedding_dim": embedding_dim,
                    "pair_count": int(pretrain_result.pair_count),
                    "val_top1": float(candidate_val_metrics["top1"]),
                    "val_top5": float(candidate_val_metrics["top5"]),
                    "artifact_path": str(pretrain_path),
                }
            )
            artifact_paths.append(pretrain_path)
            if float(candidate_val_metrics["top1"]) > best_val_top1:
                best_val_top1 = float(candidate_val_metrics["top1"])
                best_selection = (
                    pretrain_result,
                    pretrain_path,
                    sequence_projection,
                    context_projection,
                    item_bias,
                    retrieval_epochs,
                )
        assert best_selection is not None
        pretrain_result, pretrain_path, sequence_projection, context_projection, item_bias, retrieval_epochs = best_selection
    else:
        rng = np.random.default_rng(random_seed)
        base_embeddings = _normalize_rows(rng.normal(scale=0.05, size=(data.num_artists, embedding_dim)).astype("float32"))
        counts = np.bincount(data.y_train.astype("int32"), minlength=data.num_artists).astype("float32")
        counts += 1.0
        frequency = counts / np.sum(counts)
        pretrain_result = SelfSupervisedPretrainingResult(
            objective_name="supervised_only",
            embedding_dim=embedding_dim,
            artist_embeddings=base_embeddings,
            artist_frequency=frequency.astype("float32"),
            pair_count=0,
            epochs=0,
            learning_rate=0.0,
            window_size=0,
            negatives=0,
        )
        pretrain_path = pretrain_dir / "self_supervised_artist_embeddings_supervised_only.joblib"
        joblib.dump(pretrain_result, pretrain_path, compress=3)
        artifact_paths.append(pretrain_path)
        popularity = np.asarray(pretrain_result.artist_frequency, dtype="float32")
        sequence_projection, context_projection, item_bias, retrieval_epochs = _fit_dual_encoder(
            seq_train=data.X_seq_train,
            ctx_train=data.X_ctx_train,
            y_train=data.y_train,
            artist_embeddings=pretrain_result.artist_embeddings,
            popularity=popularity,
            random_seed=random_seed,
            logger=logger,
        )

    popularity = np.asarray(pretrain_result.artist_frequency, dtype="float32")
    ann_index = _build_ann_index(np.asarray(pretrain_result.artist_embeddings, dtype="float32"), random_seed=random_seed)
    retrieval_artifact = RetrievalServingArtifact(
        model_name="retrieval_dual_encoder",
        candidate_k=top_k,
        artist_embeddings=np.asarray(pretrain_result.artist_embeddings, dtype="float32"),
        sequence_projection=np.asarray(sequence_projection, dtype="float32"),
        context_projection=np.asarray(context_projection, dtype="float32"),
        item_bias=np.asarray(item_bias, dtype="float32"),
        popularity=popularity.astype("float32"),
        ann_index=ann_index,
        reranker=None,
    )

    val_retrieval_proba, _ = _score_split(retrieval_artifact, seq_batch=data.X_seq_val, ctx_batch=data.X_ctx_val)
    test_retrieval_proba, _ = _score_split(retrieval_artifact, seq_batch=data.X_seq_test, ctx_batch=data.X_ctx_test)
    _, val_retrieval_scores = retrieval_artifact.score_items(data.X_seq_val, data.X_ctx_val)
    _, test_retrieval_scores = retrieval_artifact.score_items(data.X_seq_test, data.X_ctx_test)

    retrieval_metrics_val = _prediction_metrics(val_retrieval_proba, data.y_val, num_items=data.num_artists)
    retrieval_metrics_test = _prediction_metrics(test_retrieval_proba, data.y_test, num_items=data.num_artists)
    val_candidate_hit = _topk_accuracy(val_retrieval_scores, data.y_val, top_k)
    test_candidate_hit = _topk_accuracy(test_retrieval_scores, data.y_test, top_k)
    ann_metrics_val = _ann_recall_and_latency(
        ann_index=ann_index,
        session_vec=retrieval_artifact.score_items(data.X_seq_val, data.X_ctx_val)[0],
        scores=val_retrieval_scores,
        top_k=top_k,
    )
    ann_metrics_test = _ann_recall_and_latency(
        ann_index=ann_index,
        session_vec=retrieval_artifact.score_items(data.X_seq_test, data.X_ctx_test)[0],
        scores=test_retrieval_scores,
        top_k=top_k,
    )

    retrieval_bundle_path = save_prediction_bundle(
        prediction_dir / "retrieval_dual_encoder.npz",
        val_proba=val_retrieval_proba,
        test_proba=test_retrieval_proba,
    )
    retrieval_model_path = retrieval_dir / "retrieval_dual_encoder.joblib"
    joblib.dump(retrieval_artifact, retrieval_model_path, compress=3)
    artifact_paths.extend([retrieval_bundle_path, retrieval_model_path])

    logger.info(
        "Retrieval baseline: val_top1=%.4f val_recall@%d=%.4f test_top1=%.4f",
        retrieval_metrics_val["top1"],
        top_k,
        val_candidate_hit,
        retrieval_metrics_test["top1"],
    )

    reranker_estimator = _fit_reranker(
        seq_train=data.X_seq_train,
        ctx_train=data.X_ctx_train,
        y_train=data.y_train,
        retrieval_artifact=retrieval_artifact,
        random_seed=random_seed,
    )
    reranker_path = retrieval_dir / "retrieval_reranker_estimator.joblib"
    joblib.dump(reranker_estimator, reranker_path, compress=3)
    artifact_paths.append(reranker_path)

    reranker_artifact = RetrievalServingArtifact(
        model_name="retrieval_reranker",
        candidate_k=top_k,
        artist_embeddings=retrieval_artifact.artist_embeddings,
        sequence_projection=retrieval_artifact.sequence_projection,
        context_projection=retrieval_artifact.context_projection,
        item_bias=retrieval_artifact.item_bias,
        popularity=retrieval_artifact.popularity,
        reranker=reranker_estimator,
    )
    val_rerank_proba = reranker_artifact.predict_proba(data.X_seq_val, data.X_ctx_val)
    test_rerank_proba = reranker_artifact.predict_proba(data.X_seq_test, data.X_ctx_test)
    rerank_metrics_val = _prediction_metrics(val_rerank_proba, data.y_val, num_items=data.num_artists)
    rerank_metrics_test = _prediction_metrics(test_rerank_proba, data.y_test, num_items=data.num_artists)

    reranker_bundle_path = save_prediction_bundle(
        prediction_dir / "retrieval_reranker.npz",
        val_proba=val_rerank_proba,
        test_proba=test_rerank_proba,
    )
    reranker_model_path = retrieval_dir / "retrieval_reranker.joblib"
    joblib.dump(reranker_artifact, reranker_model_path, compress=3)
    artifact_paths.extend([reranker_bundle_path, reranker_model_path])

    summary_path = retrieval_dir / "retrieval_summary.json"
    summary_payload = {
        "candidate_k": top_k,
        "embedding_dim": embedding_dim,
        "enable_self_supervised_pretraining": bool(enable_self_supervised_pretraining),
        "selected_pretraining_objective": str(pretrain_result.objective_name),
        "pretraining_pairs": int(pretrain_result.pair_count),
        "retrieval_epochs": int(retrieval_epochs),
        "val_candidate_hit_rate": float(val_candidate_hit),
        "test_candidate_hit_rate": float(test_candidate_hit),
        "pretraining_objectives": objective_rows,
        "ann_validation": ann_metrics_val,
        "ann_test": ann_metrics_test,
        "retrieval": {
            "val": retrieval_metrics_val,
            "test": retrieval_metrics_test,
        },
        "reranker": {
            "val": rerank_metrics_val,
            "test": rerank_metrics_test,
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    artifact_paths.append(summary_path)

    fit_seconds = float(time.perf_counter() - started)
    rows = [
        {
            "model_name": "retrieval_dual_encoder",
            "model_type": "retrieval",
            "model_family": "dual_encoder",
            "val_top1": float(retrieval_metrics_val["top1"]),
            "val_top5": float(retrieval_metrics_val["top5"]),
            "val_ndcg_at5": float(retrieval_metrics_val["ndcg_at5"]),
            "val_mrr_at5": float(retrieval_metrics_val["mrr_at5"]),
            "val_coverage_at5": float(retrieval_metrics_val["coverage_at5"]),
            "val_diversity_at5": float(retrieval_metrics_val["diversity_at5"]),
            "test_top1": float(retrieval_metrics_test["top1"]),
            "test_top5": float(retrieval_metrics_test["top5"]),
            "test_ndcg_at5": float(retrieval_metrics_test["ndcg_at5"]),
            "test_mrr_at5": float(retrieval_metrics_test["mrr_at5"]),
            "test_coverage_at5": float(retrieval_metrics_test["coverage_at5"]),
            "test_diversity_at5": float(retrieval_metrics_test["diversity_at5"]),
            "fit_seconds": fit_seconds,
            "epochs": retrieval_epochs,
            "prediction_bundle_path": str(retrieval_bundle_path),
            "retrieval_artifact_path": str(retrieval_model_path),
            "pretraining_artifact_path": str(pretrain_path),
            "pretraining_objective": str(pretrain_result.objective_name),
            f"val_recall_at{top_k}": float(val_candidate_hit),
            f"test_recall_at{top_k}": float(test_candidate_hit),
            f"val_ann_recall_at{top_k}": float(ann_metrics_val["ann_recall_at_k"]),
            f"test_ann_recall_at{top_k}": float(ann_metrics_test["ann_recall_at_k"]),
        },
        {
            "model_name": "retrieval_reranker",
            "model_type": "retrieval_reranker",
            "model_family": "candidate_reranker",
            "val_top1": float(rerank_metrics_val["top1"]),
            "val_top5": float(rerank_metrics_val["top5"]),
            "val_ndcg_at5": float(rerank_metrics_val["ndcg_at5"]),
            "val_mrr_at5": float(rerank_metrics_val["mrr_at5"]),
            "val_coverage_at5": float(rerank_metrics_val["coverage_at5"]),
            "val_diversity_at5": float(rerank_metrics_val["diversity_at5"]),
            "test_top1": float(rerank_metrics_test["top1"]),
            "test_top5": float(rerank_metrics_test["top5"]),
            "test_ndcg_at5": float(rerank_metrics_test["ndcg_at5"]),
            "test_mrr_at5": float(rerank_metrics_test["mrr_at5"]),
            "test_coverage_at5": float(rerank_metrics_test["coverage_at5"]),
            "test_diversity_at5": float(rerank_metrics_test["diversity_at5"]),
            "fit_seconds": fit_seconds,
            "epochs": retrieval_epochs,
            "prediction_bundle_path": str(reranker_bundle_path),
            "retrieval_artifact_path": str(reranker_model_path),
            "estimator_artifact_path": str(reranker_path),
            "pretraining_artifact_path": str(pretrain_path),
            "pretraining_objective": str(pretrain_result.objective_name),
            f"val_recall_at{top_k}": float(val_candidate_hit),
            f"test_recall_at{top_k}": float(test_candidate_hit),
            f"val_ann_recall_at{top_k}": float(ann_metrics_val["ann_recall_at_k"]),
            f"test_ann_recall_at{top_k}": float(ann_metrics_test["ann_recall_at_k"]),
        },
    ]
    return RetrievalExperimentResult(rows=rows, artifact_paths=artifact_paths)
