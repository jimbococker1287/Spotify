from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import logging
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import load_streaming_history
from .ranking import topk_indices_2d
from .run_artifacts import write_csv_rows, write_json, write_markdown
from .track_level_data import (
    TrackLevelExample,
    build_track_level_dataset,
    split_track_level_examples,
)
from .track_retrieval import EASERetriever, SessionCooccurrenceRetriever


PAD_ITEM_ID = 0
OOV_ITEM_ID = 1
CONTEXT_FEATURE_NAMES = (
    "log_target_gap",
    "log_history_length",
    "history_unique_ratio",
    "history_repeat_ratio",
    "log_mean_history_gap",
    "log_max_history_gap",
    "log_session_position",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)


@dataclass(frozen=True)
class TrackTrainingConfig:
    raw_data_dir: Path
    output_dir: Path
    include_video: bool = True
    max_history: int = 256
    sequence_length: int = 64
    evaluation_k: int = 100
    retrieval_evaluation_limit: int = 5_000
    cooccurrence_max_items: int = 1_500
    cooccurrence_shrinkage: float = 10.0
    ease_max_items: int = 400
    ease_l2: float = 100.0
    neural_models: tuple[str, ...] = ("meantime", "mmoe", "ple")
    neural_max_items: int = 2_000
    max_train_examples: int = 6_000
    max_validation_examples: int = 1_500
    max_test_examples: int = 1_500
    epochs: int = 1
    batch_size: int = 128
    random_seed: int = 42


@dataclass(frozen=True)
class TrackVocabulary:
    items: tuple[str, ...]
    item_to_id: Mapping[str, int]

    @property
    def vocabulary_size(self) -> int:
        return len(self.items) + 2

    def encode(self, item: str) -> int:
        return int(self.item_to_id.get(item, OOV_ITEM_ID))

    def to_dict(self) -> dict[str, object]:
        return {
            "padding_item_id": PAD_ITEM_ID,
            "oov_item_id": OOV_ITEM_ID,
            "vocabulary_size": self.vocabulary_size,
            "trained_item_count": len(self.items),
            "items": list(self.items),
        }


@dataclass(frozen=True)
class ContextScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype("float32")


@dataclass(frozen=True)
class EncodedTrackSplit:
    sequence_ids: np.ndarray
    time_gaps: np.ndarray
    context: np.ndarray
    target_ids: np.ndarray
    target_in_vocabulary: np.ndarray
    targets: dict[str, np.ndarray]
    sample_weights: dict[str, np.ndarray]
    example_ids: np.ndarray

    def __len__(self) -> int:
        return int(len(self.target_ids))


@dataclass(frozen=True)
class TrackModelData:
    vocabulary: TrackVocabulary
    context_scaler: ContextScaler
    dwell_log_scale: float
    train: EncodedTrackSplit
    validation: EncodedTrackSplit
    test: EncodedTrackSplit


def reconstruct_session_interactions(
    examples: Sequence[TrackLevelExample],
) -> pd.DataFrame:
    """Reconstruct one ordered row per session event without history duplication."""
    sessions: dict[int, list[TrackLevelExample]] = {}
    for example in examples:
        sessions.setdefault(int(example.session_id), []).append(example)

    rows: list[dict[str, object]] = []
    for session_id, session_examples in sorted(sessions.items()):
        ordered = sorted(
            session_examples,
            key=lambda value: (value.session_position, value.example_id),
        )
        if not ordered:
            continue
        first = ordered[0]
        tracks = [*first.history_track_uris, *(example.target_track_uri for example in ordered)]
        for position, track_id in enumerate(tracks):
            rows.append(
                {
                    "session_id": session_id,
                    "position": position,
                    "track_id": track_id,
                }
            )
    return pd.DataFrame(rows, columns=["session_id", "position", "track_id"])


def fit_track_vocabulary(
    train_examples: Sequence[TrackLevelExample],
    *,
    max_items: int,
) -> TrackVocabulary:
    if max_items < 2:
        raise ValueError("max_items must be at least 2")
    interactions = reconstruct_session_interactions(train_examples)
    if interactions.empty:
        raise ValueError("Cannot fit a track vocabulary without training interactions.")
    counts = interactions["track_id"].value_counts()
    ranked = sorted(
        ((str(item), int(count)) for item, count in counts.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    items = tuple(item for item, _count in ranked[: int(max_items)])
    return TrackVocabulary(
        items=items,
        item_to_id={item: index + 2 for index, item in enumerate(items)},
    )


def _bounded_examples(
    examples: Sequence[TrackLevelExample],
    limit: int,
) -> tuple[TrackLevelExample, ...]:
    values = tuple(examples)
    if limit <= 0 or len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=int(limit), dtype="int64")
    return tuple(values[int(index)] for index in indices)


def _raw_context(examples: Sequence[TrackLevelExample]) -> np.ndarray:
    rows: list[list[float]] = []
    for example in examples:
        history = example.history_track_uris
        gaps = np.asarray(example.history_time_gaps_seconds, dtype="float64")
        history_count = len(history)
        unique_count = len(set(history))
        unique_ratio = unique_count / history_count if history_count else 0.0
        repeat_ratio = 1.0 - unique_ratio if history_count else 0.0
        positive_gaps = gaps[gaps > 0.0]
        mean_gap = float(np.mean(positive_gaps)) if positive_gaps.size else 0.0
        max_gap = float(np.max(positive_gaps)) if positive_gaps.size else 0.0
        timestamp = example.target_timestamp
        hour = timestamp.hour + (timestamp.minute / 60.0)
        day = float(timestamp.dayofweek)
        rows.append(
            [
                math.log1p(max(0.0, example.target_time_gap_seconds)),
                math.log1p(history_count),
                unique_ratio,
                repeat_ratio,
                math.log1p(mean_gap),
                math.log1p(max_gap),
                math.log1p(max(0, example.session_position)),
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                math.sin(2.0 * math.pi * day / 7.0),
                math.cos(2.0 * math.pi * day / 7.0),
            ]
        )
    return np.asarray(rows, dtype="float32")


def _fit_context_scaler(train_examples: Sequence[TrackLevelExample]) -> ContextScaler:
    values = _raw_context(train_examples)
    mean = np.mean(values, axis=0, dtype="float64").astype("float32")
    scale = np.std(values, axis=0, dtype="float64").astype("float32")
    scale = np.where(scale < 1e-6, 1.0, scale).astype("float32")
    return ContextScaler(mean=mean, scale=scale)


def _fit_dwell_scale(train_examples: Sequence[TrackLevelExample]) -> float:
    values = [
        math.log1p(int(example.labels.listen_duration_ms))
        for example in train_examples
        if example.labels.listen_duration_ms is not None
        and example.labels.listen_duration_ms >= 0
    ]
    if not values:
        return 1.0
    return max(1.0, float(np.percentile(np.asarray(values, dtype="float64"), 95.0)))


def encode_track_examples(
    examples: Sequence[TrackLevelExample],
    *,
    vocabulary: TrackVocabulary,
    sequence_length: int,
    context_scaler: ContextScaler,
    dwell_log_scale: float,
) -> EncodedTrackSplit:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    values = tuple(examples)
    row_count = len(values)
    sequences = np.zeros((row_count, sequence_length), dtype="int32")
    time_gaps = np.zeros((row_count, sequence_length), dtype="float32")
    target_ids = np.zeros(row_count, dtype="int32")
    target_in_vocabulary = np.zeros(row_count, dtype=bool)
    example_ids = np.zeros(row_count, dtype="int64")

    skip = np.zeros((row_count, 1), dtype="float32")
    dwell = np.zeros((row_count, 1), dtype="float32")
    session_end = np.zeros((row_count, 1), dtype="float32")
    explicit_positive = np.zeros((row_count, 1), dtype="float32")
    repeat = np.zeros((row_count, 1), dtype="float32")

    next_item_weight = np.zeros(row_count, dtype="float32")
    skip_weight = np.zeros(row_count, dtype="float32")
    dwell_weight = np.zeros(row_count, dtype="float32")
    session_end_weight = np.ones(row_count, dtype="float32")
    explicit_positive_weight = np.zeros(row_count, dtype="float32")
    repeat_weight = np.ones(row_count, dtype="float32")

    for row_index, example in enumerate(values):
        history = example.history_track_uris[-sequence_length:]
        history_gaps = example.history_time_gaps_seconds[-sequence_length:]
        start = sequence_length - len(history)
        sequences[row_index, start:] = [
            vocabulary.encode(track_id) for track_id in history
        ]
        if history_gaps:
            time_gaps[row_index, start:] = np.asarray(history_gaps, dtype="float32")

        target_id = vocabulary.encode(example.target_track_uri)
        target_ids[row_index] = target_id
        target_in_vocabulary[row_index] = target_id != OOV_ITEM_ID
        next_item_weight[row_index] = float(target_id != OOV_ITEM_ID)
        example_ids[row_index] = example.example_id

        if example.labels.skipped is not None:
            skip[row_index, 0] = float(example.labels.skipped)
            skip_weight[row_index] = 1.0
        if example.labels.listen_duration_ms is not None:
            dwell[row_index, 0] = float(
                np.clip(
                    math.log1p(max(0, example.labels.listen_duration_ms))
                    / dwell_log_scale,
                    0.0,
                    1.0,
                )
            )
            dwell_weight[row_index] = 1.0
        session_end[row_index, 0] = float(example.labels.session_end)
        repeat[row_index, 0] = float(example.labels.repeat)

    context = context_scaler.transform(_raw_context(values))
    return EncodedTrackSplit(
        sequence_ids=sequences,
        time_gaps=time_gaps,
        context=context,
        target_ids=target_ids,
        target_in_vocabulary=target_in_vocabulary,
        targets={
            "next_item_output": target_ids,
            "skip_output": skip,
            "dwell_output": dwell,
            "session_end_output": session_end,
            "explicit_positive_output": explicit_positive,
            "repeat_output": repeat,
        },
        sample_weights={
            "next_item_output": next_item_weight,
            "skip_output": skip_weight,
            "dwell_output": dwell_weight,
            "session_end_output": session_end_weight,
            "explicit_positive_output": explicit_positive_weight,
            "repeat_output": repeat_weight,
        },
        example_ids=example_ids,
    )


def prepare_track_model_data(
    train_examples: Sequence[TrackLevelExample],
    validation_examples: Sequence[TrackLevelExample],
    test_examples: Sequence[TrackLevelExample],
    *,
    max_items: int,
    sequence_length: int,
    max_train_examples: int,
    max_validation_examples: int,
    max_test_examples: int,
) -> TrackModelData:
    vocabulary = fit_track_vocabulary(train_examples, max_items=max_items)
    context_scaler = _fit_context_scaler(train_examples)
    dwell_log_scale = _fit_dwell_scale(train_examples)
    bounded_train = _bounded_examples(train_examples, max_train_examples)
    bounded_validation = _bounded_examples(validation_examples, max_validation_examples)
    bounded_test = _bounded_examples(test_examples, max_test_examples)
    encode = lambda rows: encode_track_examples(  # noqa: E731
        rows,
        vocabulary=vocabulary,
        sequence_length=sequence_length,
        context_scaler=context_scaler,
        dwell_log_scale=dwell_log_scale,
    )
    return TrackModelData(
        vocabulary=vocabulary,
        context_scaler=context_scaler,
        dwell_log_scale=dwell_log_scale,
        train=encode(bounded_train),
        validation=encode(bounded_validation),
        test=encode(bounded_test),
    )


def _top_catalog_interactions(
    interactions: pd.DataFrame,
    max_items: int,
) -> pd.DataFrame:
    counts = interactions["track_id"].value_counts()
    ranked = sorted(
        ((str(item), int(count)) for item, count in counts.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    keep = {item for item, _count in ranked[: int(max_items)]}
    return interactions.loc[interactions["track_id"].isin(keep)].copy()


def _stream_retrieval_metrics(
    retriever,
    examples: Sequence[TrackLevelExample],
    *,
    k: int,
    limit: int,
) -> dict[str, object]:
    selected = _bounded_examples(examples, limit)
    catalog = set(retriever.catalog)
    hits = 0
    in_catalog_hits = 0
    in_catalog_targets = 0
    ndcg_total = 0.0
    mrr_total = 0.0
    unique_candidates: set[object] = set()
    candidate_counts: list[int] = []
    for example in selected:
        candidates = retriever.recommend(
            example.history_track_uris,
            k=k,
            exclude_seen=False,
        )
        candidate_ids = [candidate.item_id for candidate in candidates]
        candidate_counts.append(len(candidate_ids))
        unique_candidates.update(candidate_ids)
        target_in_catalog = example.target_track_uri in catalog
        in_catalog_targets += int(target_in_catalog)
        try:
            rank = candidate_ids.index(example.target_track_uri) + 1
        except ValueError:
            continue
        hits += 1
        in_catalog_hits += int(target_in_catalog)
        ndcg_total += 1.0 / math.log2(rank + 1.0)
        mrr_total += 1.0 / float(rank)

    query_count = len(selected)
    return {
        "evaluated_examples": query_count,
        "recall_at_k": hits / query_count if query_count else float("nan"),
        "in_catalog_recall_at_k": (
            in_catalog_hits / in_catalog_targets if in_catalog_targets else float("nan")
        ),
        "target_catalog_coverage": (
            in_catalog_targets / query_count if query_count else float("nan")
        ),
        "ndcg_at_k": ndcg_total / query_count if query_count else float("nan"),
        "mrr_at_k": mrr_total / query_count if query_count else float("nan"),
        "candidate_catalog_coverage": (
            len(unique_candidates) / len(catalog) if catalog else float("nan")
        ),
        "mean_candidate_count": (
            float(np.mean(candidate_counts)) if candidate_counts else 0.0
        ),
        "k": int(k),
        "exclude_seen": False,
    }


def run_bounded_retrieval_benchmarks(
    train_examples: Sequence[TrackLevelExample],
    validation_examples: Sequence[TrackLevelExample],
    *,
    k: int,
    evaluation_limit: int,
    cooccurrence_max_items: int,
    cooccurrence_shrinkage: float,
    ease_max_items: int,
    ease_l2: float,
) -> list[dict[str, object]]:
    interactions = reconstruct_session_interactions(train_examples)
    if interactions.empty:
        return []
    definitions = (
        (
            "session_cooccurrence",
            SessionCooccurrenceRetriever,
            cooccurrence_max_items,
            {"shrinkage": cooccurrence_shrinkage},
        ),
        (
            "ease",
            EASERetriever,
            ease_max_items,
            {"l2": ease_l2},
        ),
    )
    results: list[dict[str, object]] = []
    for model_name, factory, max_items, fit_kwargs in definitions:
        bounded = _top_catalog_interactions(interactions, max_items)
        started = time.perf_counter()
        retriever = factory().fit(bounded, **fit_kwargs)
        fit_seconds = time.perf_counter() - started
        metrics = _stream_retrieval_metrics(
            retriever,
            validation_examples,
            k=k,
            limit=evaluation_limit,
        )
        item_count = len(retriever.catalog)
        results.append(
            {
                "model_name": model_name,
                "status": "complete",
                "fit_seconds": fit_seconds,
                "catalog_items": item_count,
                "training_sessions": int(bounded["session_id"].nunique()),
                "training_interactions": int(len(bounded)),
                "weight_matrix_mb": (
                    float(retriever.item_weights_.nbytes) / (1024.0 * 1024.0)
                ),
                **metrics,
            }
        )
        del retriever
        gc.collect()
    return results


def _batch_indices(
    row_count: int,
    *,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    order = np.arange(row_count, dtype="int64")
    if rng is not None:
        order = rng.permutation(order)
    for start in range(0, row_count, batch_size):
        yield order[start : start + batch_size]


def _ranking_metrics_from_topk(
    topk: np.ndarray,
    target_ids: np.ndarray,
    target_in_vocabulary: np.ndarray,
    *,
    trained_item_count: int,
) -> dict[str, float]:
    matches = topk == target_ids[:, None]
    hit_mask = np.any(matches, axis=1)
    ranks = np.argmax(matches, axis=1) + 1
    ndcg = np.zeros(len(ranks), dtype="float64")
    mrr = np.zeros(len(ranks), dtype="float64")
    ndcg[hit_mask] = 1.0 / np.log2(ranks[hit_mask] + 1.0)
    mrr[hit_mask] = 1.0 / ranks[hit_mask]
    in_vocab = np.asarray(target_in_vocabulary, dtype=bool)
    predicted_real_items = topk[topk >= 2]
    return {
        "recall_at_k": float(np.mean(hit_mask)) if len(hit_mask) else float("nan"),
        "in_vocab_recall_at_k": (
            float(np.mean(hit_mask[in_vocab])) if np.any(in_vocab) else float("nan")
        ),
        "target_vocabulary_coverage": (
            float(np.mean(in_vocab)) if len(in_vocab) else float("nan")
        ),
        "ndcg_at_k": float(np.mean(ndcg)) if len(ndcg) else float("nan"),
        "mrr_at_k": float(np.mean(mrr)) if len(mrr) else float("nan"),
        "prediction_catalog_coverage": (
            float(len(np.unique(predicted_real_items))) / float(trained_item_count)
            if trained_item_count and predicted_real_items.size
            else 0.0
        ),
        "k": int(topk.shape[1]) if topk.ndim == 2 else 0,
    }


def _predict_topk(
    model,
    split: EncodedTrackSplit,
    *,
    model_name: str,
    batch_size: int,
    k: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(split) == 0:
        return np.empty((0, 0), dtype="int32"), {}

    topk_rows: list[np.ndarray] = []
    auxiliary: dict[str, list[np.ndarray]] = {}
    for selector in _batch_indices(len(split), batch_size=batch_size):
        if model_name == "meantime":
            prediction = model(
                [
                    split.sequence_ids[selector],
                    split.time_gaps[selector],
                    split.context[selector],
                ],
                training=False,
            )
            next_item = np.asarray(prediction)
        else:
            raw = model(
                [split.sequence_ids[selector], split.context[selector]],
                training=False,
            )
            outputs = raw if isinstance(raw, (list, tuple)) else [raw]
            mapped = {
                name: np.asarray(value)
                for name, value in zip(model.output_names, outputs)
            }
            next_item = mapped.pop("next_item_output")
            for name, values in mapped.items():
                auxiliary.setdefault(name, []).append(values)
        scores = np.asarray(next_item, dtype="float32").copy()
        scores[:, :2] = -np.inf
        topk_rows.append(topk_indices_2d(scores, k))
    return (
        np.concatenate(topk_rows, axis=0),
        {
            name: np.concatenate(parts, axis=0)
            for name, parts in auxiliary.items()
        },
    )


def _binary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(weight).reshape(-1) > 0.0
    if not np.any(mask):
        return {"accuracy": float("nan"), "auc": float("nan"), "labeled_rows": 0}
    y_true = np.asarray(truth).reshape(-1)[mask]
    y_score = np.asarray(prediction).reshape(-1)[mask]
    accuracy = float(np.mean((y_score >= 0.5) == (y_true >= 0.5)))
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_score))
    return {"accuracy": accuracy, "auc": auc, "labeled_rows": int(np.sum(mask))}


def _auxiliary_metrics(
    split: EncodedTrackSplit,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in ("skip_output", "session_end_output", "repeat_output"):
        if name in predictions:
            metrics[name] = _binary_metrics(
                split.targets[name],
                predictions[name],
                split.sample_weights[name],
            )
    if "dwell_output" in predictions:
        mask = split.sample_weights["dwell_output"].reshape(-1) > 0.0
        metrics["dwell_output"] = {
            "mae": (
                float(
                    np.mean(
                        np.abs(
                            split.targets["dwell_output"].reshape(-1)[mask]
                            - predictions["dwell_output"].reshape(-1)[mask]
                        )
                    )
                )
                if np.any(mask)
                else float("nan")
            ),
            "labeled_rows": int(np.sum(mask)),
        }
    return metrics


def _train_meantime(
    data: TrackModelData,
    *,
    config: TrackTrainingConfig,
    checkpoint_dir: Path,
) -> dict[str, object]:
    import tensorflow as tf

    from .meantime_model import build_meantime_model

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_meantime_model(
        sequence_length=config.sequence_length,
        vocabulary_size=data.vocabulary.vocabulary_size,
        num_ctx=len(CONTEXT_FEATURE_NAMES),
        params={
            "embedding_dim": 32,
            "num_heads": 2,
            "feed_forward_dim": 64,
            "dropout_rate": 0.1,
            "num_blocks": 1,
            "num_time_buckets": 32,
            "context_dim": 32,
        },
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        run_eagerly=True,
    )
    started = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    final_loss = float("nan")
    for _epoch in range(config.epochs):
        for selector in _batch_indices(
            len(data.train),
            batch_size=config.batch_size,
            rng=rng,
        ):
            final_loss = float(
                model.train_on_batch(
                    [
                        data.train.sequence_ids[selector],
                        data.train.time_gaps[selector],
                        data.train.context[selector],
                    ],
                    data.train.target_ids[selector],
                    sample_weight=data.train.sample_weights["next_item_output"][selector],
                )
            )
    fit_seconds = time.perf_counter() - started
    checkpoint = checkpoint_dir / "meantime_track.keras"
    model.save(checkpoint)
    val_topk, _ = _predict_topk(
        model,
        data.validation,
        model_name="meantime",
        batch_size=config.batch_size,
        k=config.evaluation_k,
    )
    test_topk, _ = _predict_topk(
        model,
        data.test,
        model_name="meantime",
        batch_size=config.batch_size,
        k=config.evaluation_k,
    )
    return {
        "model_name": "meantime",
        "status": "complete",
        "fit_seconds": fit_seconds,
        "epochs": config.epochs,
        "final_train_loss": final_loss,
        "checkpoint": str(checkpoint),
        "validation": _ranking_metrics_from_topk(
            val_topk,
            data.validation.target_ids,
            data.validation.target_in_vocabulary,
            trained_item_count=len(data.vocabulary.items),
        ),
        "test": _ranking_metrics_from_topk(
            test_topk,
            data.test.target_ids,
            data.test.target_in_vocabulary,
            trained_item_count=len(data.vocabulary.items),
        ),
    }


def _train_multitask(
    data: TrackModelData,
    *,
    architecture: str,
    config: TrackTrainingConfig,
    checkpoint_dir: Path,
) -> dict[str, object]:
    import tensorflow as tf

    from .multitask_model import build_multitask_recommender

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_multitask_recommender(
        sequence_length=config.sequence_length,
        num_items=data.vocabulary.vocabulary_size,
        num_ctx=len(CONTEXT_FEATURE_NAMES),
        params={
            "architecture": architecture,
            "sequence_encoder": "average",
            "embedding_dim": 32,
            "sequence_dim": 32,
            "context_dim": 24,
            "fusion_dim": 64,
            "num_experts": 3,
            "task_experts": 2,
            "expert_units": 48,
            "tower_units": 24,
            "dropout_rate": 0.1,
            "learning_rate": 1e-3,
        },
    )
    started = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    final_loss = float("nan")
    for _epoch in range(config.epochs):
        for selector in _batch_indices(
            len(data.train),
            batch_size=config.batch_size,
            rng=rng,
        ):
            ordered_targets = [
                data.train.targets[name][selector] for name in model.output_names
            ]
            ordered_weights = [
                data.train.sample_weights[name][selector] for name in model.output_names
            ]
            result = model.train_on_batch(
                [
                    data.train.sequence_ids[selector],
                    data.train.context[selector],
                ],
                ordered_targets,
                sample_weight=ordered_weights,
                return_dict=True,
            )
            final_loss = float(result["loss"])
    fit_seconds = time.perf_counter() - started
    checkpoint = checkpoint_dir / f"{architecture}_track_multitask.keras"
    model.save(checkpoint)
    val_topk, val_aux = _predict_topk(
        model,
        data.validation,
        model_name=architecture,
        batch_size=config.batch_size,
        k=config.evaluation_k,
    )
    test_topk, test_aux = _predict_topk(
        model,
        data.test,
        model_name=architecture,
        batch_size=config.batch_size,
        k=config.evaluation_k,
    )
    return {
        "model_name": architecture,
        "status": "complete",
        "fit_seconds": fit_seconds,
        "epochs": config.epochs,
        "final_train_loss": final_loss,
        "checkpoint": str(checkpoint),
        "validation": {
            **_ranking_metrics_from_topk(
                val_topk,
                data.validation.target_ids,
                data.validation.target_in_vocabulary,
                trained_item_count=len(data.vocabulary.items),
            ),
            "auxiliary": _auxiliary_metrics(data.validation, val_aux),
        },
        "test": {
            **_ranking_metrics_from_topk(
                test_topk,
                data.test.target_ids,
                data.test.target_in_vocabulary,
                trained_item_count=len(data.vocabulary.items),
            ),
            "auxiliary": _auxiliary_metrics(data.test, test_aux),
        },
    }


def _training_markdown(manifest: dict[str, object]) -> list[str]:
    retrieval = manifest.get("retrieval_results", [])
    neural = manifest.get("neural_results", [])
    lines = [
        "# Track Expansion Training",
        "",
        f"- Status: `{manifest.get('status', 'unknown')}`",
        f"- Generated: `{manifest.get('generated_at', '')}`",
        "",
        "## Retrieval",
        "",
    ]
    if isinstance(retrieval, list) and retrieval:
        for row in retrieval:
            lines.append(
                f"- `{row['model_name']}`: Recall@{row['k']} "
                f"`{float(row['recall_at_k']):.6f}`, in-catalog recall "
                f"`{float(row['in_catalog_recall_at_k']):.6f}`, target coverage "
                f"`{float(row['target_catalog_coverage']):.6f}`."
            )
    else:
        lines.append("- No retrieval benchmark completed.")
    lines.extend(["", "## Neural Models", ""])
    if isinstance(neural, list) and neural:
        for row in neural:
            validation = row.get("validation", {})
            lines.append(
                f"- `{row['model_name']}`: validation Recall@"
                f"{validation.get('k', '?')} `{float(validation.get('recall_at_k', float('nan'))):.6f}`, "
                f"test recall `{float(row.get('test', {}).get('recall_at_k', float('nan'))):.6f}`."
            )
    else:
        lines.append("- Neural training was not requested.")
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "Run `make train-recommender-expansion` to repeat the bounded pass.",
            "Increase limits through `EXTRA_ARGS` only after reviewing memory and runtime from this manifest.",
        ]
    )
    return lines


def run_track_expansion_training(
    *,
    config: TrackTrainingConfig,
    logger: logging.Logger,
) -> list[Path]:
    raw = load_streaming_history(
        config.raw_data_dir,
        include_video=config.include_video,
        logger=logger,
    )
    dataset = build_track_level_dataset(
        raw,
        max_history=max(config.max_history, config.sequence_length),
    )
    splits = split_track_level_examples(dataset)
    root = config.output_dir / "analysis" / "recommender_expansion" / "training"
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running bounded track retrieval benchmarks.")
    retrieval_results = run_bounded_retrieval_benchmarks(
        splits.train,
        splits.validation,
        k=config.evaluation_k,
        evaluation_limit=config.retrieval_evaluation_limit,
        cooccurrence_max_items=config.cooccurrence_max_items,
        cooccurrence_shrinkage=config.cooccurrence_shrinkage,
        ease_max_items=config.ease_max_items,
        ease_l2=config.ease_l2,
    )

    neural_results: list[dict[str, object]] = []
    model_data: TrackModelData | None = None
    requested_models = tuple(
        model.strip().lower() for model in config.neural_models if model.strip()
    )
    unknown = sorted(set(requested_models) - {"meantime", "mmoe", "ple"})
    if unknown:
        raise ValueError(f"Unknown track neural models: {', '.join(unknown)}")
    if requested_models:
        logger.info("Preparing leakage-safe track tensors for %s.", ", ".join(requested_models))
        model_data = prepare_track_model_data(
            splits.train,
            splits.validation,
            splits.test,
            max_items=config.neural_max_items,
            sequence_length=config.sequence_length,
            max_train_examples=config.max_train_examples,
            max_validation_examples=config.max_validation_examples,
            max_test_examples=config.max_test_examples,
        )
        for model_name in requested_models:
            logger.info("Training bounded track model %s.", model_name)
            if model_name == "meantime":
                result = _train_meantime(
                    model_data,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                )
            else:
                result = _train_multitask(
                    model_data,
                    architecture=model_name,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                )
            neural_results.append(result)

    tensor_summary: dict[str, object] = {}
    vocabulary_payload: dict[str, object] = {}
    if model_data is not None:
        vocabulary_payload = model_data.vocabulary.to_dict()
        tensor_summary = {
            "sequence_length": config.sequence_length,
            "context_features": list(CONTEXT_FEATURE_NAMES),
            "dwell_log_scale": model_data.dwell_log_scale,
            "train_rows": len(model_data.train),
            "validation_rows": len(model_data.validation),
            "test_rows": len(model_data.test),
            "train_target_vocabulary_coverage": float(
                np.mean(model_data.train.target_in_vocabulary)
            ),
            "validation_target_vocabulary_coverage": float(
                np.mean(model_data.validation.target_in_vocabulary)
            ),
            "test_target_vocabulary_coverage": float(
                np.mean(model_data.test.target_in_vocabulary)
            ),
            "labeled_rows": {
                name: int(np.sum(weights > 0.0))
                for name, weights in model_data.train.sample_weights.items()
            },
        }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": {
            **asdict(config),
            "raw_data_dir": str(config.raw_data_dir.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "dataset": {
            "examples": len(dataset.examples),
            "unique_tracks": dataset.unique_track_count,
            "sessions": dataset.session_count,
            "train_examples": len(splits.train),
            "validation_examples": len(splits.validation),
            "test_examples": len(splits.test),
        },
        "retrieval_results": retrieval_results,
        "neural_results": neural_results,
        "tensor_summary": tensor_summary,
    }
    paths = [
        write_json(root / "training_manifest.json", manifest),
        write_json(root / "retrieval_benchmarks.json", retrieval_results),
        write_csv_rows(
            root / "retrieval_benchmarks.csv",
            retrieval_results,
            fieldnames=list(retrieval_results[0]) if retrieval_results else ["model_name"],
        ),
        write_json(root / "neural_results.json", neural_results),
        write_json(root / "track_vocabulary.json", vocabulary_payload),
        write_json(root / "tensor_summary.json", tensor_summary),
        write_markdown(root / "CONTINUE_TRAINING.md", _training_markdown(manifest)),
    ]

    from .recommender_expansion_lab import (
        ExpansionRunConfig,
        build_recommender_expansion_lab,
    )

    paths.extend(
        build_recommender_expansion_lab(
            config=ExpansionRunConfig(
                raw_data_dir=config.raw_data_dir,
                output_dir=config.output_dir,
                max_history=config.max_history,
                evaluation_k=config.evaluation_k,
                evaluation_limit=max(
                    config.retrieval_evaluation_limit,
                    config.max_validation_examples,
                ),
                include_video=config.include_video,
            ),
            logger=logger,
        )
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train bounded track-level retrieval and neural expansion models."
    )
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-history", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--evaluation-k", type=int, default=100)
    parser.add_argument("--retrieval-evaluation-limit", type=int, default=5_000)
    parser.add_argument("--cooccurrence-max-items", type=int, default=1_500)
    parser.add_argument("--cooccurrence-shrinkage", type=float, default=10.0)
    parser.add_argument("--ease-max-items", type=int, default=400)
    parser.add_argument("--ease-l2", type=float, default=100.0)
    parser.add_argument("--models", default="meantime,mmoe,ple")
    parser.add_argument("--neural-max-items", type=int, default=2_000)
    parser.add_argument("--max-train-examples", type=int, default=6_000)
    parser.add_argument("--max-validation-examples", type=int, default=1_500)
    parser.add_argument("--max-test-examples", type=int, default=1_500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("spotify.track_expansion_training")
    paths = run_track_expansion_training(
        config=TrackTrainingConfig(
            raw_data_dir=Path(args.raw_data_dir),
            output_dir=Path(args.output_dir),
            include_video=not args.no_video,
            max_history=args.max_history,
            sequence_length=args.sequence_length,
            evaluation_k=args.evaluation_k,
            retrieval_evaluation_limit=args.retrieval_evaluation_limit,
            cooccurrence_max_items=args.cooccurrence_max_items,
            cooccurrence_shrinkage=args.cooccurrence_shrinkage,
            ease_max_items=args.ease_max_items,
            ease_l2=args.ease_l2,
            neural_models=tuple(
                value.strip() for value in str(args.models).split(",") if value.strip()
            ),
            neural_max_items=args.neural_max_items,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            max_test_examples=args.max_test_examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
        ),
        logger=logger,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
