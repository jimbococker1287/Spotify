from __future__ import annotations

import numpy as np

from .retrieval_common import (
    DEFAULT_RETRIEVAL_EPOCHS,
    RandomProjectionANNIndex,
    RetrievalServingArtifact,
    _env_float,
    _env_int,
    _hash_vectors,
    _softmax_rows,
    _weighted_session_base,
)


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


def _fit_dual_encoder(
    *,
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    artist_embeddings: np.ndarray,
    popularity: np.ndarray,
    random_seed: int,
    logger,
    epochs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(random_seed)
    resolved_epochs = max(1, int(epochs)) if epochs is not None else _env_int("SPOTIFY_RETRIEVAL_EPOCHS", DEFAULT_RETRIEVAL_EPOCHS)
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
        resolved_epochs,
        batch_size,
    )

    for _epoch in range(resolved_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            base_batch = session_base[batch_idx]
            ctx_batch = ctx_train[batch_idx]
            target = y_train[batch_idx].astype("int64")

            session_vec = (base_batch @ sequence_projection) + (ctx_batch @ context_projection)
            logits = session_vec @ artist_embeddings.T
            logits += item_bias.reshape(1, -1)
            proba = _softmax_rows(logits)
            proba[np.arange(len(target)), target] -= 1.0
            proba /= float(max(1, len(target)))

            grad_session = proba @ artist_embeddings
            grad_sequence = base_batch.T @ grad_session
            grad_context = ctx_batch.T @ grad_session
            grad_bias = proba.sum(axis=0)

            sequence_projection -= learning_rate * (grad_sequence.astype("float32") + l2 * sequence_projection)
            context_projection -= learning_rate * (grad_context.astype("float32") + l2 * context_projection)
            item_bias -= learning_rate * (grad_bias.astype("float32") + l2 * item_bias)

    return sequence_projection, context_projection, item_bias, resolved_epochs


def _score_split(
    artifact: RetrievalServingArtifact,
    *,
    seq_batch: np.ndarray,
    ctx_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    session_vec, scores = artifact.score_items(seq_batch, ctx_batch)
    return _softmax_rows(scores), session_vec


__all__ = [
    "_build_ann_index",
    "_fit_dual_encoder",
    "_score_split",
]
