from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .data import PreparedData
from .retrieval_common import (
    DEFAULT_EMBEDDING_DIM,
    RetrievalExperimentResult,
    _env_int,
)
from .retrieval_runtime_eval import evaluate_retrieval_baseline, train_and_evaluate_reranker
from .retrieval_runtime_persistence import persist_retrieval_outputs
from .retrieval_seed_selection import train_pretraining_seed


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

    (
        pretrain_result,
        pretrain_path,
        sequence_projection,
        context_projection,
        item_bias,
        retrieval_epochs,
        objective_rows,
    ) = train_pretraining_seed(
        data=data,
        pretrain_dir=pretrain_dir,
        random_seed=random_seed,
        logger=logger,
        embedding_dim=embedding_dim,
        top_k=top_k,
        enable_self_supervised_pretraining=enable_self_supervised_pretraining,
        artifact_paths=artifact_paths,
    )

    popularity = np.asarray(pretrain_result.artist_frequency, dtype="float32")
    baseline = evaluate_retrieval_baseline(
        artist_embeddings=np.asarray(pretrain_result.artist_embeddings, dtype="float32"),
        context_projection=np.asarray(context_projection, dtype="float32"),
        data=data,
        item_bias=np.asarray(item_bias, dtype="float32"),
        popularity=popularity.astype("float32"),
        random_seed=random_seed,
        sequence_projection=np.asarray(sequence_projection, dtype="float32"),
        top_k=top_k,
    )

    reranker = train_and_evaluate_reranker(
        baseline=baseline,
        data=data,
        random_seed=random_seed,
    )

    fit_seconds = float(time.perf_counter() - started)
    _, rows = persist_retrieval_outputs(
        artifact_paths=artifact_paths,
        baseline=baseline,
        enable_self_supervised_pretraining=enable_self_supervised_pretraining,
        top_k=top_k,
        logger=logger,
        objective_rows=objective_rows,
        output_dir=output_dir,
        pretrain_path=pretrain_path,
        pretrain_result=pretrain_result,
        reranker=reranker,
        fit_seconds=fit_seconds,
        retrieval_epochs=retrieval_epochs,
    )
    return RetrievalExperimentResult(rows=rows, artifact_paths=artifact_paths)


__all__ = [
    "train_retrieval_stack",
]
