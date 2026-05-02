from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval_common import RetrievalServingArtifact, _prediction_metrics, _softmax_rows, _topk_accuracy
from .retrieval_reranking import _fit_reranker, _predict_reranked_probabilities
from .retrieval_training import _ann_recall_and_latency, _build_ann_index


def _build_transition_prior_matrix(data) -> np.ndarray:
    num_artists = max(1, int(getattr(data, "num_artists", 0) or 0))
    seq_train = np.asarray(getattr(data, "X_seq_train", np.empty((0, 0), dtype="int32")), dtype="int32")
    y_train = np.asarray(getattr(data, "y_train", np.empty((0,), dtype="int32")), dtype="int32").reshape(-1)
    if seq_train.ndim != 2 or len(seq_train) == 0 or len(seq_train) != len(y_train):
        return np.full((num_artists, num_artists), 1.0 / float(num_artists), dtype="float32")

    last_artist = seq_train[:, -1].astype("int32", copy=False)
    valid = (
        (last_artist >= 0)
        & (last_artist < num_artists)
        & (y_train >= 0)
        & (y_train < num_artists)
    )
    counts = np.zeros((num_artists, num_artists), dtype="float32")
    if np.any(valid):
        np.add.at(counts, (last_artist[valid], y_train[valid]), 1.0)
    global_counts = np.bincount(np.clip(y_train[valid], 0, num_artists - 1), minlength=num_artists).astype("float32")
    counts += 0.35 * global_counts.reshape(1, -1)
    counts += 0.25
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return (counts / row_sums).astype("float32", copy=False)


@dataclass
class RetrievalBaselineEvaluation:
    retrieval_artifact: RetrievalServingArtifact
    val_session_vec: np.ndarray
    test_session_vec: np.ndarray
    val_retrieval_scores: np.ndarray
    test_retrieval_scores: np.ndarray
    retrieval_metrics_val: dict[str, float]
    retrieval_metrics_test: dict[str, float]
    val_candidate_hit: float
    test_candidate_hit: float
    ann_metrics_val: dict[str, float]
    ann_metrics_test: dict[str, float]


@dataclass
class RetrievalRerankerEvaluation:
    reranker_artifact: RetrievalServingArtifact
    reranker_estimator: object
    reranker_model_name: str
    rerank_metrics_val: dict[str, float]
    rerank_metrics_test: dict[str, float]
    val_rerank_proba: np.ndarray
    test_rerank_proba: np.ndarray


def evaluate_retrieval_baseline(
    *,
    artist_embeddings: np.ndarray,
    context_projection: np.ndarray,
    data,
    item_bias: np.ndarray,
    popularity: np.ndarray,
    random_seed: int,
    sequence_projection: np.ndarray,
    top_k: int,
) -> RetrievalBaselineEvaluation:
    ann_index = _build_ann_index(np.asarray(artist_embeddings, dtype="float32"), random_seed=random_seed)
    transition_prior = _build_transition_prior_matrix(data)
    retrieval_artifact = RetrievalServingArtifact(
        model_name="retrieval_dual_encoder",
        candidate_k=top_k,
        artist_embeddings=np.asarray(artist_embeddings, dtype="float32"),
        sequence_projection=np.asarray(sequence_projection, dtype="float32"),
        context_projection=np.asarray(context_projection, dtype="float32"),
        item_bias=np.asarray(item_bias, dtype="float32"),
        popularity=np.asarray(popularity, dtype="float32"),
        context_feature_names=tuple(getattr(data, "context_features", ()) or ()),
        transition_prior=transition_prior,
        ann_index=ann_index,
        reranker=None,
    )

    val_session_vec, val_retrieval_scores = retrieval_artifact.score_items(data.X_seq_val, data.X_ctx_val)
    test_session_vec, test_retrieval_scores = retrieval_artifact.score_items(data.X_seq_test, data.X_ctx_test)
    val_retrieval_proba = _softmax_rows(val_retrieval_scores)
    test_retrieval_proba = _softmax_rows(test_retrieval_scores)

    retrieval_metrics_val = _prediction_metrics(val_retrieval_proba, data.y_val, num_items=data.num_artists)
    retrieval_metrics_test = _prediction_metrics(test_retrieval_proba, data.y_test, num_items=data.num_artists)
    val_candidate_hit = _topk_accuracy(val_retrieval_scores, data.y_val, top_k)
    test_candidate_hit = _topk_accuracy(test_retrieval_scores, data.y_test, top_k)
    ann_metrics_val = _ann_recall_and_latency(
        ann_index=ann_index,
        session_vec=val_session_vec,
        scores=val_retrieval_scores,
        top_k=top_k,
        random_seed=random_seed,
    )
    ann_metrics_test = _ann_recall_and_latency(
        ann_index=ann_index,
        session_vec=test_session_vec,
        scores=test_retrieval_scores,
        top_k=top_k,
        random_seed=random_seed + 1,
    )

    return RetrievalBaselineEvaluation(
        retrieval_artifact=retrieval_artifact,
        val_session_vec=val_session_vec,
        test_session_vec=test_session_vec,
        val_retrieval_scores=val_retrieval_scores,
        test_retrieval_scores=test_retrieval_scores,
        retrieval_metrics_val=retrieval_metrics_val,
        retrieval_metrics_test=retrieval_metrics_test,
        val_candidate_hit=val_candidate_hit,
        test_candidate_hit=test_candidate_hit,
        ann_metrics_val=ann_metrics_val,
        ann_metrics_test=ann_metrics_test,
    )


def train_and_evaluate_reranker(
    *,
    baseline: RetrievalBaselineEvaluation,
    data,
    random_seed: int,
) -> RetrievalRerankerEvaluation:
    reranker_estimator = _fit_reranker(
        seq_train=data.X_seq_train,
        ctx_train=data.X_ctx_train,
        y_train=data.y_train,
        seq_val=data.X_seq_val,
        ctx_val=data.X_ctx_val,
        y_val=data.y_val,
        retrieval_artifact=baseline.retrieval_artifact,
        random_seed=random_seed,
        context_feature_names=tuple(getattr(data, "context_features", ()) or ()),
    )

    reranker_artifact = RetrievalServingArtifact(
        model_name="retrieval_reranker",
        candidate_k=baseline.retrieval_artifact.candidate_k,
        artist_embeddings=baseline.retrieval_artifact.artist_embeddings,
        sequence_projection=baseline.retrieval_artifact.sequence_projection,
        context_projection=baseline.retrieval_artifact.context_projection,
        item_bias=baseline.retrieval_artifact.item_bias,
        popularity=baseline.retrieval_artifact.popularity,
        context_feature_names=tuple(getattr(baseline.retrieval_artifact, "context_feature_names", ()) or ()),
        transition_prior=np.asarray(getattr(baseline.retrieval_artifact, "transition_prior", None), dtype="float32")
        if getattr(baseline.retrieval_artifact, "transition_prior", None) is not None
        else None,
        reranker=reranker_estimator,
    )
    val_rerank_proba = _predict_reranked_probabilities(
        seq_batch=data.X_seq_val,
        ctx_batch=data.X_ctx_val,
        session_vec=baseline.val_session_vec,
        retrieval_scores=baseline.val_retrieval_scores,
        artifact=reranker_artifact,
    )
    test_rerank_proba = _predict_reranked_probabilities(
        seq_batch=data.X_seq_test,
        ctx_batch=data.X_ctx_test,
        session_vec=baseline.test_session_vec,
        retrieval_scores=baseline.test_retrieval_scores,
        artifact=reranker_artifact,
    )
    rerank_metrics_val = _prediction_metrics(val_rerank_proba, data.y_val, num_items=data.num_artists)
    rerank_metrics_test = _prediction_metrics(test_rerank_proba, data.y_test, num_items=data.num_artists)
    return RetrievalRerankerEvaluation(
        reranker_artifact=reranker_artifact,
        reranker_estimator=reranker_estimator,
        reranker_model_name=str(getattr(reranker_estimator, "_spotify_reranker_model_name", "")),
        rerank_metrics_val=rerank_metrics_val,
        rerank_metrics_test=rerank_metrics_test,
        val_rerank_proba=val_rerank_proba,
        test_rerank_proba=test_rerank_proba,
    )


__all__ = [
    "RetrievalBaselineEvaluation",
    "RetrievalRerankerEvaluation",
    "evaluate_retrieval_baseline",
    "train_and_evaluate_reranker",
]
