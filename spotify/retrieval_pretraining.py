from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .benchmarks import sample_indices
from .data import PreparedData
from .retrieval_common import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_PRETRAIN_MAX_PAIRS,
    SelfSupervisedPretrainingResult,
    _build_pretraining_pairs,
    _env_float,
    _env_int,
    _normalize_rows,
    _resolve_pretraining_objectives,
    _sigmoid,
)


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
    pair_count_before_sampling = int(len(pairs))
    max_pairs = _env_int("SPOTIFY_PRETRAIN_MAX_PAIRS", DEFAULT_PRETRAIN_MAX_PAIRS)
    if max_pairs > 0 and len(pairs) > max_pairs:
        selected = sample_indices(len(pairs), max_pairs, rng)
        pairs = pairs[selected]
        logger.info(
            "Sampling self-supervised pretraining pairs for %s: using %d/%d pairs.",
            objective,
            len(pairs),
            pair_count_before_sampling,
        )
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
            center_updates = (learning_rate * grad_center).astype("float32", copy=False)
            positive_updates = (learning_rate * grad_pos).astype("float32", copy=False)
            negative_ids_flat = negative_ids.reshape(-1)
            negative_updates = (learning_rate * grad_neg.reshape(-1, embedding_dim)).astype("float32", copy=False)

            if l2 > 0.0:
                center_counts = np.bincount(centers, minlength=data.num_artists).astype("float32")
                output_counts = np.bincount(positives, minlength=data.num_artists).astype("float32")
                output_counts += np.bincount(negative_ids_flat, minlength=data.num_artists).astype("float32")
                input_embeddings *= (1.0 - (learning_rate * l2 * center_counts).reshape(-1, 1)).astype("float32", copy=False)
                output_embeddings *= (1.0 - (learning_rate * l2 * output_counts).reshape(-1, 1)).astype("float32", copy=False)

            np.add.at(input_embeddings, centers, -center_updates)
            np.add.at(output_embeddings, positives, -positive_updates)
            np.add.at(output_embeddings, negative_ids_flat, -negative_updates)

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


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_PRETRAIN_EPOCHS",
    "DEFAULT_PRETRAIN_MAX_PAIRS",
    "SelfSupervisedPretrainingResult",
    "train_self_supervised_artist_embeddings",
    "_resolve_pretraining_objectives",
]
