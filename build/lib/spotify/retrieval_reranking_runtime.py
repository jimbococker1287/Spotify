from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _reranker_model_names() -> tuple[str, ...]:
    raw = os.getenv("SPOTIFY_RERANKER_MODELS", "logreg,hist_gbm").strip()
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    supported = tuple(name for name in values if name in {"logreg", "hist_gbm"})
    return supported or ("logreg",)


def _build_reranker_estimator(*, model_name: str, random_seed: int):
    if model_name == "hist_gbm":
        return HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=180,
            learning_rate=0.06,
            min_samples_leaf=24,
            random_state=random_seed,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )


def _fit_estimator(estimator, *, features: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray):
    if hasattr(estimator, "steps"):
        estimator.fit(features, labels, logisticregression__sample_weight=sample_weight)
    else:
        estimator.fit(features, labels, sample_weight=sample_weight)
    return estimator


def _validation_top1(
    *,
    estimator,
    retrieval_artifact: RetrievalServingArtifact,
    seq_val: np.ndarray,
    ctx_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    seq_arr = np.asarray(seq_val, dtype="int32")
    ctx_arr = np.asarray(ctx_val, dtype="float32")
    y_arr = np.asarray(y_val, dtype="int32").reshape(-1)
    if len(seq_arr) == 0 or len(seq_arr) != len(y_arr):
        return float("-inf")
    session_vec, retrieval_scores = retrieval_artifact.score_items(seq_arr, ctx_arr)
    candidate_artifact = RetrievalServingArtifact(
        model_name=retrieval_artifact.model_name,
        candidate_k=retrieval_artifact.candidate_k,
        artist_embeddings=retrieval_artifact.artist_embeddings,
        sequence_projection=retrieval_artifact.sequence_projection,
        context_projection=retrieval_artifact.context_projection,
        item_bias=retrieval_artifact.item_bias,
        popularity=retrieval_artifact.popularity,
        ann_index=retrieval_artifact.ann_index,
        reranker=estimator,
    )
    reranked_proba = _predict_reranked_probabilities(
        seq_batch=seq_arr,
        ctx_batch=ctx_arr,
        session_vec=session_vec,
        retrieval_scores=retrieval_scores,
        artifact=candidate_artifact,
    )
    return float(np.mean(np.argmax(reranked_proba, axis=1) == y_arr))


def _fit_reranker(
    *,
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray | None,
    ctx_val: np.ndarray | None,
    y_val: np.ndarray | None,
    retrieval_artifact: RetrievalServingArtifact,
    random_seed: int,
) -> object:
    train_cap = _env_int("SPOTIFY_RERANKER_TRAIN_ROWS", 12_000)
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

    best_estimator = None
    best_score = float("-inf")
    best_name = ""
    for model_idx, model_name in enumerate(_reranker_model_names()):
        estimator = _build_reranker_estimator(model_name=model_name, random_seed=random_seed + model_idx)
        estimator = _fit_estimator(
            estimator,
            features=features,
            labels=labels,
            sample_weight=sample_weight,
        )
        score = _validation_top1(
            estimator=estimator,
            retrieval_artifact=retrieval_artifact,
            seq_val=(np.asarray(seq_val, dtype="int32") if seq_val is not None else np.empty((0, 0), dtype="int32")),
            ctx_val=(np.asarray(ctx_val, dtype="float32") if ctx_val is not None else np.empty((0, 0), dtype="float32")),
            y_val=(np.asarray(y_val, dtype="int32") if y_val is not None else np.empty((0,), dtype="int32")),
        )
        if score > best_score:
            best_score = float(score)
            best_estimator = estimator
            best_name = model_name

    if best_estimator is None:
        best_estimator = _fit_estimator(
            _build_reranker_estimator(model_name="logreg", random_seed=random_seed),
            features=features,
            labels=labels,
            sample_weight=sample_weight,
        )
        best_name = "logreg"
        best_score = float("-inf")

    setattr(best_estimator, "_spotify_reranker_model_name", best_name)
    setattr(best_estimator, "_spotify_validation_top1", float(best_score))
    return best_estimator


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
