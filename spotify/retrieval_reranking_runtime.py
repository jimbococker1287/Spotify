from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .retrieval_common import (
    RetrievalServingArtifact,
    _env_int,
    _positive_label_index,
    _softmax_rows,
    _topk_indices_and_scores,
)
from .retrieval_reranking_features import (
    _apply_repeat_mitigation,
    _candidate_feature_matrix,
    _inject_true_labels,
    _reranker_sample_weights,
)


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
    sample_weight = _reranker_sample_weights(
        seq_batch=seq_fit,
        candidate_ids=candidate_ids,
        y_true=y_fit,
    )
    estimator.fit(features, labels, logisticregression__sample_weight=sample_weight)
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
        rerank_scores = _apply_repeat_mitigation(
            seq_batch=seq_arr[start:end],
            candidate_ids=candidate_ids[start:end],
            candidate_scores=candidate_scores[start:end],
            rerank_scores=rerank_scores,
        )
        rows = np.arange(row_count, dtype="int64")[:, None]
        result[start:end][rows, candidate_ids[start:end]] = rerank_scores
    return result.astype("float32", copy=False)


__all__ = [
    "_fit_reranker",
    "_predict_reranked_probabilities",
]
