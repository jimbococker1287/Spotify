from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

from .data import PreparedData
from .retrieval_common import (
    RetrievalServingArtifact,
    SelfSupervisedPretrainingResult,
    _normalize_rows,
    _prediction_metrics,
    _resolve_pretraining_objectives,
)
from .retrieval_pretraining import train_self_supervised_artist_embeddings
from .retrieval_training import _fit_dual_encoder, _score_split


def _pretraining_blend_topk() -> int:
    raw = os.getenv("SPOTIFY_PRETRAIN_BLEND_TOPK", "2").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 2


def _select_pretraining_seed(
    *,
    data: PreparedData,
    pretrain_dir: Path,
    random_seed: int,
    logger,
    embedding_dim: int,
    top_k: int,
    artifact_paths: list[Path],
) -> tuple[SelfSupervisedPretrainingResult, Path, np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, object]]]:
    objective_rows: list[dict[str, object]] = []
    objective_candidates: list[dict[str, object]] = []
    best_selection: tuple[SelfSupervisedPretrainingResult, Path, np.ndarray, np.ndarray, np.ndarray, int] | None = None
    best_val_top1 = float("-inf")

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
        objective_candidates.append(
            {
                "objective_name": objective_name,
                "pretrain_result": pretrain_result,
                "pretrain_path": pretrain_path,
                "val_metrics": candidate_val_metrics,
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

    blend_topk = min(_pretraining_blend_topk(), len(objective_candidates))
    if blend_topk >= 2:
        top_candidates = sorted(
            objective_candidates,
            key=lambda item: float(item["val_metrics"]["top1"]),
            reverse=True,
        )[:blend_topk]
        blend_names = [str(item["objective_name"]) for item in top_candidates]
        raw_weights = np.asarray(
            [
                max(1e-6, float(item["val_metrics"]["top1"]) + (0.25 * float(item["val_metrics"]["top5"])))
                for item in top_candidates
            ],
            dtype="float32",
        )
        raw_weights /= np.sum(raw_weights)
        blended_embeddings = _normalize_rows(
            np.sum(
                [
                    float(weight) * np.asarray(item["pretrain_result"].artist_embeddings, dtype="float32")
                    for weight, item in zip(raw_weights.tolist(), top_candidates)
                ],
                axis=0,
                dtype="float32",
            )
        )
        blended_frequency = np.sum(
            [
                float(weight) * np.asarray(item["pretrain_result"].artist_frequency, dtype="float32")
                for weight, item in zip(raw_weights.tolist(), top_candidates)
            ],
            axis=0,
            dtype="float32",
        )
        blended_frequency /= max(1e-6, float(np.sum(blended_frequency)))
        blended_pair_count = int(sum(int(item["pretrain_result"].pair_count) for item in top_candidates))
        blended_objective = "blend_" + "_".join(blend_names)
        learning_rate = float(
            np.mean([float(item["pretrain_result"].learning_rate) for item in top_candidates])
        )
        window_size = int(
            round(np.mean([float(item["pretrain_result"].window_size) for item in top_candidates]))
        )
        negatives = int(
            round(np.mean([float(item["pretrain_result"].negatives) for item in top_candidates]))
        )
        blended_result = SelfSupervisedPretrainingResult(
            objective_name=blended_objective,
            embedding_dim=embedding_dim,
            artist_embeddings=blended_embeddings.astype("float32", copy=False),
            artist_frequency=blended_frequency.astype("float32", copy=False),
            pair_count=blended_pair_count,
            epochs=max(int(item["pretrain_result"].epochs) for item in top_candidates),
            learning_rate=learning_rate,
            window_size=max(1, window_size),
            negatives=max(1, negatives),
        )
        blended_path = pretrain_dir / f"self_supervised_artist_embeddings_{blended_objective}.joblib"
        joblib.dump(blended_result, blended_path, compress=3)
        artifact_paths.append(blended_path)

        sequence_projection, context_projection, item_bias, retrieval_epochs = _fit_dual_encoder(
            seq_train=data.X_seq_train,
            ctx_train=data.X_ctx_train,
            y_train=data.y_train,
            artist_embeddings=blended_result.artist_embeddings,
            popularity=np.asarray(blended_result.artist_frequency, dtype="float32"),
            random_seed=random_seed + 1000,
            logger=logger,
        )
        blended_artifact = RetrievalServingArtifact(
            model_name=f"retrieval_dual_encoder_{blended_objective}",
            candidate_k=top_k,
            artist_embeddings=np.asarray(blended_result.artist_embeddings, dtype="float32"),
            sequence_projection=np.asarray(sequence_projection, dtype="float32"),
            context_projection=np.asarray(context_projection, dtype="float32"),
            item_bias=np.asarray(item_bias, dtype="float32"),
            popularity=np.asarray(blended_result.artist_frequency, dtype="float32"),
            ann_index=None,
            reranker=None,
        )
        blended_val_proba, _ = _score_split(blended_artifact, seq_batch=data.X_seq_val, ctx_batch=data.X_ctx_val)
        blended_val_metrics = _prediction_metrics(blended_val_proba, data.y_val, num_items=data.num_artists)
        objective_rows.append(
            {
                "objective_name": blended_objective,
                "embedding_dim": embedding_dim,
                "pair_count": blended_pair_count,
                "val_top1": float(blended_val_metrics["top1"]),
                "val_top5": float(blended_val_metrics["top5"]),
                "artifact_path": str(blended_path),
                "blend_topk": int(blend_topk),
                "blend_weights": {
                    str(name): float(weight)
                    for name, weight in zip(blend_names, raw_weights.tolist())
                },
            }
        )
        if float(blended_val_metrics["top1"]) > best_val_top1:
            best_val_top1 = float(blended_val_metrics["top1"])
            best_selection = (
                blended_result,
                blended_path,
                sequence_projection,
                context_projection,
                item_bias,
                retrieval_epochs,
            )

    assert best_selection is not None
    pretrain_result, pretrain_path, sequence_projection, context_projection, item_bias, retrieval_epochs = best_selection
    return (
        pretrain_result,
        pretrain_path,
        sequence_projection,
        context_projection,
        item_bias,
        retrieval_epochs,
        objective_rows,
    )


def _build_supervised_only_seed(
    *,
    data: PreparedData,
    pretrain_dir: Path,
    random_seed: int,
    logger,
    embedding_dim: int,
    artifact_paths: list[Path],
) -> tuple[SelfSupervisedPretrainingResult, Path, np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, object]]]:
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
    return (
        pretrain_result,
        pretrain_path,
        sequence_projection,
        context_projection,
        item_bias,
        retrieval_epochs,
        [],
    )


__all__ = [
    "_build_supervised_only_seed",
    "_select_pretraining_seed",
]
