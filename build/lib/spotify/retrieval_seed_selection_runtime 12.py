from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import PreparedData
from .retrieval_common import SelfSupervisedPretrainingResult
from .retrieval_seed_candidates import _build_supervised_only_seed, _select_pretraining_seed


def train_pretraining_seed(
    *,
    data: PreparedData,
    pretrain_dir: Path,
    random_seed: int,
    logger,
    embedding_dim: int,
    top_k: int,
    enable_self_supervised_pretraining: bool,
    artifact_paths: list[Path],
) -> tuple[SelfSupervisedPretrainingResult, Path, np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, object]]]:
    if enable_self_supervised_pretraining:
        return _select_pretraining_seed(
            data=data,
            pretrain_dir=pretrain_dir,
            random_seed=random_seed,
            logger=logger,
            embedding_dim=embedding_dim,
            top_k=top_k,
            artifact_paths=artifact_paths,
        )
    return _build_supervised_only_seed(
        data=data,
        pretrain_dir=pretrain_dir,
        random_seed=random_seed,
        logger=logger,
        embedding_dim=embedding_dim,
        artifact_paths=artifact_paths,
    )


__all__ = ["train_pretraining_seed"]
