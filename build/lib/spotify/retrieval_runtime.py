from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

from .data import PreparedData
from .retrieval_common import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_PRETRAIN_EPOCHS,
    DEFAULT_PRETRAIN_MAX_PAIRS,
    DEFAULT_RETRIEVAL_ANN_EVAL_ROWS,
    DEFAULT_RETRIEVAL_EPOCHS,
    RetrievalExperimentResult,
    _env_float,
    _env_int,
    _resolve_pretraining_objectives,
)
from .retrieval_runtime_eval import evaluate_retrieval_baseline, train_and_evaluate_reranker
from .retrieval_runtime_persistence import persist_retrieval_outputs
from .retrieval_seed_selection import train_pretraining_seed
from .run_artifacts import copy_file_if_changed, safe_read_json, write_json


RETRIEVAL_CACHE_SCHEMA_VERSION = "retrieval-stack-cache-v1"


@dataclass(frozen=True)
class RetrievalStackCachePaths:
    cache_key: str
    cache_dir: Path
    result_path: Path
    metadata_path: Path
    artifact_dir: Path


def _retrieval_cache_enabled_from_env() -> bool:
    raw = os.getenv("SPOTIFY_CACHE_RETRIEVAL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _retrieval_source_digest() -> str:
    source_names = (
        "retrieval_runtime.py",
        "retrieval_common.py",
        "retrieval_artifact_payloads.py",
        "retrieval_runtime_eval.py",
        "retrieval_runtime_persistence.py",
        "retrieval_seed_selection.py",
        "retrieval_seed_candidates.py",
        "retrieval_pretraining.py",
        "retrieval_training.py",
        "retrieval_dual_encoder.py",
        "retrieval_ann_metrics.py",
        "retrieval_reranking_runtime.py",
        "retrieval_reranking_features.py",
        "retrieval_stack.py",
    )
    root_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in source_names:
        path = root_dir / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:24]
def _relative_output_artifact_path(*, source_path: Path, output_dir: Path) -> str:
    try:
        return source_path.relative_to(output_dir).as_posix()
    except ValueError:
        pass
    try:
        return source_path.resolve().relative_to(output_dir.resolve()).as_posix()
    except Exception:
        return source_path.name


def _serialize_cached_retrieval_row(row: dict[str, object], *, output_dir: Path) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text.endswith("_path") and str(value).strip():
            path_value = Path(str(value).strip()).expanduser()
            serialized[key_text] = _relative_output_artifact_path(source_path=path_value, output_dir=output_dir)
        else:
            serialized[key_text] = value
    return serialized


def _restore_cached_retrieval_row(row: dict[str, object], *, output_dir: Path) -> dict[str, object]:
    restored: dict[str, object] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text.endswith("_path") and str(value).strip():
            path_value = Path(str(value).strip())
            if path_value.is_absolute():
                restored[key_text] = str(path_value)
            else:
                restored[key_text] = str((output_dir / path_value).resolve())
        else:
            restored[key_text] = value
    return restored


def _retrieval_runtime_config(*, num_artists: int, candidate_k: int, enable_self_supervised_pretraining: bool) -> dict[str, object]:
    resolved_candidate_k = max(2, min(int(candidate_k), int(num_artists)))
    return {
        "embedding_dim": _env_int("SPOTIFY_RETRIEVAL_DIM", DEFAULT_EMBEDDING_DIM),
        "candidate_k": int(resolved_candidate_k),
        "enable_self_supervised_pretraining": bool(enable_self_supervised_pretraining),
        "retrieval_epochs": _env_int("SPOTIFY_RETRIEVAL_EPOCHS", DEFAULT_RETRIEVAL_EPOCHS),
        "retrieval_ann_bits": _env_int("SPOTIFY_RETRIEVAL_ANN_BITS", 10),
        "retrieval_batch_size": _env_int("SPOTIFY_RETRIEVAL_BATCH_SIZE", 256),
        "retrieval_learning_rate": _env_float("SPOTIFY_RETRIEVAL_LR", 0.055),
        "retrieval_l2": _env_float("SPOTIFY_RETRIEVAL_L2", 2e-4),
        "ann_eval_rows": _env_int("SPOTIFY_RETRIEVAL_ANN_EVAL_ROWS", DEFAULT_RETRIEVAL_ANN_EVAL_ROWS),
        "pretrain_objectives": list(_resolve_pretraining_objectives()),
        "pretrain_epochs": _env_int("SPOTIFY_PRETRAIN_EPOCHS", DEFAULT_PRETRAIN_EPOCHS),
        "pretrain_negatives": _env_int("SPOTIFY_PRETRAIN_NEGATIVES", 4),
        "pretrain_batch_size": _env_int("SPOTIFY_PRETRAIN_BATCH_SIZE", 256),
        "pretrain_window": _env_int("SPOTIFY_PRETRAIN_WINDOW", 3),
        "pretrain_learning_rate": _env_float("SPOTIFY_PRETRAIN_LR", 0.045),
        "pretrain_l2": _env_float("SPOTIFY_PRETRAIN_L2", 1e-4),
        "pretrain_max_pairs": _env_int("SPOTIFY_PRETRAIN_MAX_PAIRS", DEFAULT_PRETRAIN_MAX_PAIRS),
        "pretrain_blend_topk": _env_int("SPOTIFY_PRETRAIN_BLEND_TOPK", 2),
        "reranker_train_rows": _env_int("SPOTIFY_RERANKER_TRAIN_ROWS", 12_000),
        "reranker_predict_batch": _env_int("SPOTIFY_RERANKER_PREDICT_BATCH", 512),
        "reranker_repeat_penalty_bps": _env_int("SPOTIFY_RERANKER_REPEAT_PENALTY_BPS", 220),
        "reranker_immediate_repeat_penalty_bps": _env_int("SPOTIFY_RERANKER_IMMEDIATE_REPEAT_PENALTY_BPS", 360),
        "reranker_novelty_boost_bps": _env_int("SPOTIFY_RERANKER_NOVELTY_BOOST_BPS", 120),
    }


def _build_retrieval_cache_payload(
    *,
    cache_fingerprint: str,
    data: PreparedData,
    random_seed: int,
    candidate_k: int,
    enable_self_supervised_pretraining: bool,
) -> dict[str, object]:
    return {
        "cache_schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
        "prepared_fingerprint": str(cache_fingerprint).strip(),
        "random_seed": int(random_seed),
        "sequence_length": int(data.X_seq_train.shape[1]) if data.X_seq_train.ndim >= 2 else 0,
        "num_artists": int(data.num_artists),
        "num_ctx": int(data.num_ctx),
        "train_rows": int(len(data.y_train)),
        "val_rows": int(len(data.y_val)),
        "test_rows": int(len(data.y_test)),
        "runtime": _retrieval_runtime_config(
            num_artists=int(data.num_artists),
            candidate_k=int(candidate_k),
            enable_self_supervised_pretraining=enable_self_supervised_pretraining,
        ),
        "source_digest": _retrieval_source_digest(),
    }


def _build_retrieval_cache_key(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _resolve_retrieval_cache_paths(
    *,
    cache_root: Path,
    cache_fingerprint: str,
    cache_key: str,
) -> RetrievalStackCachePaths:
    cache_dir = (cache_root / cache_fingerprint / cache_key).resolve()
    return RetrievalStackCachePaths(
        cache_key=cache_key,
        cache_dir=cache_dir,
        result_path=cache_dir / "result.json",
        metadata_path=cache_dir / "cache_meta.json",
        artifact_dir=cache_dir / "artifacts",
    )


def _read_cached_retrieval_manifest(
    *,
    cache_paths: RetrievalStackCachePaths,
) -> tuple[list[dict[str, object]], list[str]] | None:
    payload = safe_read_json(cache_paths.result_path, default=None)
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_schema_version") != RETRIEVAL_CACHE_SCHEMA_VERSION:
        return None
    rows_payload = payload.get("rows")
    artifact_names = payload.get("artifact_names")
    if not isinstance(rows_payload, list) or not isinstance(artifact_names, list):
        return None
    normalized_rows = [dict(row) for row in rows_payload if isinstance(row, dict)]
    normalized_names: list[str] = []
    for name in artifact_names:
        name_text = str(name).strip().replace("\\", "/")
        if not name_text:
            continue
        if not (cache_paths.artifact_dir / name_text).exists():
            return None
        normalized_names.append(name_text)
    return normalized_rows, normalized_names


def _load_cached_retrieval_result(
    *,
    cache_paths: RetrievalStackCachePaths,
    output_dir: Path,
    logger,
) -> RetrievalExperimentResult | None:
    try:
        cached_manifest = _read_cached_retrieval_manifest(cache_paths=cache_paths)
        if cached_manifest is None:
            return None
        rows_payload, artifact_names = cached_manifest
        restored_artifact_paths: list[Path] = []
        for rel_path in artifact_names:
            destination_path = output_dir / rel_path
            copy_file_if_changed(cache_paths.artifact_dir / rel_path, destination_path)
            restored_artifact_paths.append(destination_path)
        return RetrievalExperimentResult(
            rows=[_restore_cached_retrieval_row(dict(row), output_dir=output_dir) for row in rows_payload],
            artifact_paths=restored_artifact_paths,
        )
    except Exception as exc:
        logger.warning("Retrieval cache load failed (%s). Rebuilding.", exc)
        return None


def _save_retrieval_result_to_cache(
    *,
    cache_paths: RetrievalStackCachePaths,
    cache_payload: dict[str, object],
    output_dir: Path,
    result: RetrievalExperimentResult,
) -> None:
    try:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_paths.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_names: list[str] = []
        seen: set[str] = set()
        for source_path in result.artifact_paths:
            if not source_path.exists():
                continue
            rel_path = _relative_output_artifact_path(source_path=source_path, output_dir=output_dir)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            artifact_names.append(rel_path)
            copy_file_if_changed(source_path, cache_paths.artifact_dir / rel_path)
        write_json(
            cache_paths.result_path,
            {
                "cache_schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
                "rows": [_serialize_cached_retrieval_row(dict(row), output_dir=output_dir) for row in result.rows],
                "artifact_names": artifact_names,
            },
            sort_keys=True,
        )
        write_json(cache_paths.metadata_path, cache_payload, sort_keys=True)
    except Exception:
        return None


def train_retrieval_stack(
    *,
    data: PreparedData,
    output_dir: Path,
    random_seed: int,
    candidate_k: int,
    enable_self_supervised_pretraining: bool,
    logger,
    cache_root: Path | None = None,
    cache_fingerprint: str = "",
    cache_stats_out: dict[str, object] | None = None,
) -> RetrievalExperimentResult:
    artifact_paths: list[Path] = []
    retrieval_dir = output_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "prediction_bundles"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir = output_dir / "pretraining"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    embedding_dim = _env_int("SPOTIFY_RETRIEVAL_DIM", DEFAULT_EMBEDDING_DIM)
    top_k = max(2, min(int(candidate_k), int(data.num_artists)))
    cache_enabled = _retrieval_cache_enabled_from_env() and cache_root is not None and bool(str(cache_fingerprint).strip())
    cache_paths = None
    cached_result = None
    if cache_enabled and cache_root is not None:
        cache_payload = _build_retrieval_cache_payload(
            cache_fingerprint=str(cache_fingerprint).strip(),
            data=data,
            random_seed=random_seed,
            candidate_k=top_k,
            enable_self_supervised_pretraining=enable_self_supervised_pretraining,
        )
        cache_key = _build_retrieval_cache_key(cache_payload)
        cache_paths = _resolve_retrieval_cache_paths(
            cache_root=cache_root,
            cache_fingerprint=str(cache_fingerprint).strip(),
            cache_key=cache_key,
        )
        cached_result = _load_cached_retrieval_result(
            cache_paths=cache_paths,
            output_dir=output_dir,
            logger=logger,
        )
    else:
        cache_payload = {}

    if cache_stats_out is not None:
        cache_stats_out.clear()
        cache_stats_out.update(
            {
                "enabled": bool(cache_enabled),
                "fingerprint": (str(cache_fingerprint).strip() if cache_enabled else ""),
                "cache_key": (cache_paths.cache_key if cache_paths is not None else ""),
                "hit": bool(cached_result is not None),
                "candidate_k": int(top_k),
            }
        )

    logger.info(
        "Retrieval cache status: enabled=%s fingerprint=%s hit=%s",
        cache_enabled,
        (cache_fingerprint if cache_enabled else "disabled"),
        bool(cached_result is not None),
    )
    if cached_result is not None:
        return cached_result

    started = time.perf_counter()

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
    result = RetrievalExperimentResult(rows=rows, artifact_paths=artifact_paths)
    if cache_enabled and cache_paths is not None:
        _save_retrieval_result_to_cache(
            cache_paths=cache_paths,
            cache_payload=cache_payload,
            output_dir=output_dir,
            result=result,
        )
    return result


__all__ = [
    "train_retrieval_stack",
]
