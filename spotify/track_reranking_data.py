from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .run_artifacts import write_json
from .track_level_data import TrackLevelExample, TrackLevelTemporalSplits
from .track_retrieval import (
    PopularityRetriever,
    ScoredCandidate,
    SessionCooccurrenceRetriever,
)


PAD_ITEM_ID = 0
OOV_ITEM_ID = 1

CONTEXT_FEATURE_NAMES = (
    "log_history_length",
    "history_unique_ratio",
    "history_repeat_ratio",
    "log_session_position",
    "log_mean_history_gap",
    "log_max_history_gap",
    "log_last_history_gap",
    "known_history_ratio",
)

CANDIDATE_FEATURE_NAMES = (
    "cooccurrence_score",
    "cooccurrence_reciprocal_rank",
    "log_popularity_count",
    "popularity_reciprocal_rank",
    "retriever_agreement",
    "history_frequency_ratio",
    "history_reciprocal_recency",
    "same_as_last_track",
    "session_support_ratio",
)


@dataclass(frozen=True)
class TrackRerankingConfig:
    """Bounded, temporal candidate-generation settings."""

    max_items: int = 1_500
    candidate_count: int = 50
    retrieval_pool_size: int = 200
    retriever_fit_fraction: float = 0.75
    cooccurrence_shrinkage: float = 10.0
    max_train_queries: int = 20_000
    max_validation_queries: int = 5_000
    max_test_queries: int = 5_000
    random_seed: int = 42

    def validate(self) -> None:
        if self.max_items < 2:
            raise ValueError("max_items must be at least 2")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least 2")
        if self.retrieval_pool_size < self.candidate_count:
            raise ValueError("retrieval_pool_size must be at least candidate_count")
        if not 0.0 < self.retriever_fit_fraction < 1.0:
            raise ValueError("retriever_fit_fraction must be between 0 and 1")
        if self.cooccurrence_shrinkage < 0.0:
            raise ValueError("cooccurrence_shrinkage cannot be negative")
        for name in (
            "max_train_queries",
            "max_validation_queries",
            "max_test_queries",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_items": self.max_items,
            "candidate_count": self.candidate_count,
            "retrieval_pool_size": self.retrieval_pool_size,
            "retriever_fit_fraction": self.retriever_fit_fraction,
            "cooccurrence_shrinkage": self.cooccurrence_shrinkage,
            "max_train_queries": self.max_train_queries,
            "max_validation_queries": self.max_validation_queries,
            "max_test_queries": self.max_test_queries,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class RerankingTrackVocabulary:
    """Track IDs and counts learned only from the retriever-fit prefix."""

    items: tuple[str, ...]
    counts: tuple[int, ...]
    item_to_id: Mapping[str, int]

    @property
    def vocabulary_size(self) -> int:
        return len(self.items) + 2

    def contains(self, item: str) -> bool:
        return item in self.item_to_id

    def encode(self, item: str) -> int:
        return self.item_to_id.get(item, OOV_ITEM_ID)

    def to_dict(self) -> dict[str, object]:
        return {
            "padding_item_id": PAD_ITEM_ID,
            "oov_item_id": OOV_ITEM_ID,
            "vocabulary_size": self.vocabulary_size,
            "trained_item_count": len(self.items),
            "items": list(self.items),
            "counts": list(self.counts),
        }


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype="float32")
        scale = np.asarray(self.scale, dtype="float32")
        return ((np.asarray(values, dtype="float32") - mean) / scale).astype("float32")

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "scale": list(self.scale)}


@dataclass(frozen=True)
class TrackRerankingSplit:
    """Flat candidate rows plus offsets that recover each ranking query."""

    query_ids: tuple[str, ...]
    example_ids: np.ndarray
    group_ids: np.ndarray
    group_offsets: np.ndarray
    candidate_track_uris: tuple[str, ...]
    candidate_item_ids: np.ndarray
    context_features: np.ndarray
    candidate_features: np.ndarray
    labels: np.ndarray
    source_example_count: int
    skipped_oov_target_count: int

    def __len__(self) -> int:
        return int(len(self.labels))

    @property
    def query_count(self) -> int:
        return len(self.query_ids)

    def group_slice(self, group_index: int) -> slice:
        start = int(self.group_offsets[group_index])
        stop = int(self.group_offsets[group_index + 1])
        return slice(start, stop)

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "query_ids": np.asarray(self.query_ids, dtype=np.str_),
            "example_ids": self.example_ids,
            "group_ids": self.group_ids,
            "group_offsets": self.group_offsets,
            "candidate_track_uris": np.asarray(self.candidate_track_uris, dtype=np.str_),
            "candidate_item_ids": self.candidate_item_ids,
            "context_features": self.context_features,
            "candidate_features": self.candidate_features,
            "labels": self.labels,
        }

    def to_manifest_dict(self) -> dict[str, int | float]:
        positive_count = int(np.sum(self.labels))
        return {
            "source_examples": self.source_example_count,
            "queries": self.query_count,
            "candidate_rows": len(self),
            "positive_rows": positive_count,
            "skipped_oov_targets": self.skipped_oov_target_count,
            "mean_candidates_per_query": len(self) / self.query_count if self.query_count else 0.0,
        }


@dataclass(frozen=True)
class TrackRerankingData:
    config: TrackRerankingConfig
    vocabulary: RerankingTrackVocabulary
    context_standardizer: FeatureStandardizer
    candidate_standardizer: FeatureStandardizer
    retriever_fit_example_count: int
    reranker_train_source_example_count: int
    train: TrackRerankingSplit
    validation: TrackRerankingSplit
    test: TrackRerankingSplit

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "status": "complete",
            "config": self.config.to_dict(),
            "dataset": {
                "retriever_fit_examples": self.retriever_fit_example_count,
                "reranker_train_source_examples": self.reranker_train_source_example_count,
                "train": self.train.to_manifest_dict(),
                "validation": self.validation.to_manifest_dict(),
                "test": self.test.to_manifest_dict(),
            },
            "features": {
                "context": list(CONTEXT_FEATURE_NAMES),
                "candidate": list(CANDIDATE_FEATURE_NAMES),
                "context_count": len(CONTEXT_FEATURE_NAMES),
                "candidate_count": len(CANDIDATE_FEATURE_NAMES),
            },
            "vocabulary": self.vocabulary.to_dict(),
            "standardizers": {
                "context": self.context_standardizer.to_dict(),
                "candidate": self.candidate_standardizer.to_dict(),
            },
            "leakage_controls": {
                "retriever_fit_scope": "earlier_training_sessions_only",
                "reranker_train_scope": "later_training_sessions_only",
                "validation_and_test_fit_access": False,
                "target_dependent_features": False,
                "unknown_targets_skipped": True,
            },
        }


@dataclass(frozen=True)
class _RawRerankingSplit:
    query_ids: tuple[str, ...]
    example_ids: np.ndarray
    group_ids: np.ndarray
    group_offsets: np.ndarray
    candidate_track_uris: tuple[str, ...]
    candidate_item_ids: np.ndarray
    context_features: np.ndarray
    candidate_features: np.ndarray
    labels: np.ndarray
    source_example_count: int
    skipped_oov_target_count: int


@dataclass(frozen=True)
class _FittedCandidateSources:
    vocabulary: RerankingTrackVocabulary
    cooccurrence: SessionCooccurrenceRetriever
    popularity: PopularityRetriever
    popularity_counts: Mapping[str, float]
    popularity_ranks: Mapping[str, int]
    session_support: Mapping[str, float]
    training_session_count: int


def _ordered_examples(examples: Sequence[TrackLevelExample]) -> tuple[TrackLevelExample, ...]:
    return tuple(
        sorted(
            examples,
            key=lambda example: (
                example.target_timestamp.value,
                example.session_id,
                example.session_position,
                example.example_id,
            ),
        )
    )


def temporal_bounded_sample(
    examples: Sequence[TrackLevelExample],
    *,
    limit: int,
) -> tuple[TrackLevelExample, ...]:
    """Deterministically span the full temporal range while preserving order."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = _ordered_examples(examples)
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=limit, dtype="int64")
    return tuple(ordered[int(index)] for index in indices)


def _partition_training_sessions(
    examples: Sequence[TrackLevelExample],
    *,
    fit_fraction: float,
) -> tuple[tuple[TrackLevelExample, ...], tuple[TrackLevelExample, ...]]:
    ordered = _ordered_examples(examples)
    sessions: list[list[TrackLevelExample]] = []
    for example in ordered:
        if not sessions or sessions[-1][0].session_id != example.session_id:
            sessions.append([example])
        else:
            sessions[-1].append(example)
    if len(sessions) < 2:
        raise ValueError("Leakage-safe reranking data requires at least two training sessions")

    cumulative = np.cumsum([len(session) for session in sessions])
    target = len(ordered) * fit_fraction
    boundary = min(
        range(1, len(sessions)),
        key=lambda index: (abs(float(cumulative[index - 1]) - target), index),
    )
    fit_examples = tuple(example for session in sessions[:boundary] for example in session)
    query_examples = tuple(example for session in sessions[boundary:] for example in session)
    return fit_examples, query_examples


def _reconstruct_interactions(examples: Sequence[TrackLevelExample]) -> pd.DataFrame:
    sessions: dict[int, list[TrackLevelExample]] = {}
    for example in examples:
        sessions.setdefault(int(example.session_id), []).append(example)

    rows: list[dict[str, object]] = []
    for session_id, values in sorted(sessions.items()):
        ordered = sorted(values, key=lambda example: (example.session_position, example.example_id))
        first = ordered[0]
        tracks = [*first.history_track_uris, *(example.target_track_uri for example in ordered)]
        for position, track_id in enumerate(tracks):
            rows.append({"session_id": session_id, "position": position, "track_id": track_id})
    return pd.DataFrame(rows, columns=["session_id", "position", "track_id"])


def fit_reranking_vocabulary(
    train_examples: Sequence[TrackLevelExample],
    *,
    max_items: int,
) -> RerankingTrackVocabulary:
    """Fit a deterministic frequency vocabulary from training examples only."""
    if max_items < 2:
        raise ValueError("max_items must be at least 2")
    interactions = _reconstruct_interactions(train_examples)
    if interactions.empty:
        raise ValueError("Cannot fit a reranking vocabulary without interactions")
    counts = interactions["track_id"].value_counts()
    ranked = sorted(
        ((str(item), int(count)) for item, count in counts.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )[:max_items]
    return RerankingTrackVocabulary(
        items=tuple(item for item, _count in ranked),
        counts=tuple(count for _item, count in ranked),
        item_to_id={item: index + 2 for index, (item, _count) in enumerate(ranked)},
    )


def _fit_candidate_sources(
    fit_examples: Sequence[TrackLevelExample],
    *,
    config: TrackRerankingConfig,
) -> _FittedCandidateSources:
    vocabulary = fit_reranking_vocabulary(fit_examples, max_items=config.max_items)
    interactions = _reconstruct_interactions(fit_examples)
    interactions = interactions.loc[interactions["track_id"].isin(vocabulary.items)].copy()
    popularity = PopularityRetriever().fit(interactions)
    cooccurrence = SessionCooccurrenceRetriever().fit(
        interactions,
        shrinkage=config.cooccurrence_shrinkage,
    )
    counts = interactions["track_id"].value_counts()
    popularity_counts = {str(item): float(count) for item, count in counts.items()}
    ranked_popularity = sorted(
        vocabulary.items,
        key=lambda item: (-popularity_counts.get(item, 0.0), item),
    )
    popularity_ranks = {item: rank for rank, item in enumerate(ranked_popularity, start=1)}
    support = interactions.groupby("track_id")["session_id"].nunique()
    return _FittedCandidateSources(
        vocabulary=vocabulary,
        cooccurrence=cooccurrence,
        popularity=popularity,
        popularity_counts=popularity_counts,
        popularity_ranks=popularity_ranks,
        session_support={str(item): float(value) for item, value in support.items()},
        training_session_count=int(interactions["session_id"].nunique()),
    )


def _context_features(
    example: TrackLevelExample,
    *,
    vocabulary: RerankingTrackVocabulary,
) -> np.ndarray:
    history = example.history_track_uris
    history_count = len(history)
    unique_ratio = len(set(history)) / history_count if history_count else 0.0
    gaps = np.asarray(example.history_time_gaps_seconds, dtype="float64")
    positive_gaps = gaps[gaps > 0.0]
    known_ratio = (
        sum(vocabulary.contains(item) for item in history) / history_count
        if history_count
        else 0.0
    )
    return np.asarray(
        [
            math.log1p(history_count),
            unique_ratio,
            1.0 - unique_ratio if history_count else 0.0,
            math.log1p(max(0, example.session_position)),
            math.log1p(float(np.mean(positive_gaps))) if positive_gaps.size else 0.0,
            math.log1p(float(np.max(positive_gaps))) if positive_gaps.size else 0.0,
            math.log1p(max(0.0, float(gaps[-1]))) if gaps.size else 0.0,
            known_ratio,
        ],
        dtype="float32",
    )


def _scored_maps(
    candidates: Sequence[ScoredCandidate],
) -> tuple[dict[str, float], dict[str, int]]:
    scores = {str(candidate.item_id): float(candidate.score) for candidate in candidates}
    ranks = {str(candidate.item_id): rank for rank, candidate in enumerate(candidates, start=1)}
    return scores, ranks


def _candidate_features(
    candidate: str,
    *,
    example: TrackLevelExample,
    sources: _FittedCandidateSources,
    cooccurrence_scores: Mapping[str, float],
    cooccurrence_ranks: Mapping[str, int],
    popularity_pool_ranks: Mapping[str, int],
    history_counts: Mapping[str, int],
    history_recency: Mapping[str, int],
) -> np.ndarray:
    history = example.history_track_uris
    history_count = len(history)
    history_frequency = history_counts.get(candidate, 0)
    reverse_position = history_recency.get(candidate)
    cooccurrence_rank = cooccurrence_ranks.get(candidate)
    popularity_rank = sources.popularity_ranks[candidate]
    return np.asarray(
        [
            cooccurrence_scores.get(candidate, 0.0),
            1.0 / cooccurrence_rank if cooccurrence_rank is not None else 0.0,
            math.log1p(sources.popularity_counts.get(candidate, 0.0)),
            1.0 / popularity_rank,
            float(
                cooccurrence_rank is not None
                and candidate in popularity_pool_ranks
            ),
            history_frequency / history_count if history_count else 0.0,
            1.0 / reverse_position if reverse_position is not None else 0.0,
            float(bool(history) and history[-1] == candidate),
            (
                sources.session_support.get(candidate, 0.0) / sources.training_session_count
                if sources.training_session_count
                else 0.0
            ),
        ],
        dtype="float32",
    )


def _stable_permutation(length: int, *, query_id: str, random_seed: int) -> np.ndarray:
    digest = hashlib.blake2b(
        f"{random_seed}:{query_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(seed).permutation(length)


def _candidate_group(
    example: TrackLevelExample,
    *,
    query_id: str,
    sources: _FittedCandidateSources,
    config: TrackRerankingConfig,
) -> tuple[list[str], np.ndarray]:
    pool_size = min(config.retrieval_pool_size, len(sources.vocabulary.items))
    cooccurrence = sources.cooccurrence.recommend(
        example.history_track_uris,
        k=pool_size,
        exclude_seen=False,
    )
    popularity = sources.popularity.recommend(
        example.history_track_uris,
        k=pool_size,
        exclude_seen=False,
    )
    cooccurrence_scores, cooccurrence_ranks = _scored_maps(cooccurrence)
    popularity_scores, popularity_pool_ranks = _scored_maps(popularity)

    pool = set(cooccurrence_scores) | set(popularity_scores)
    ranked_pool = sorted(
        pool,
        key=lambda item: (
            -(
                (1.0 / cooccurrence_ranks[item] if item in cooccurrence_ranks else 0.0)
                + (1.0 / popularity_pool_ranks[item] if item in popularity_pool_ranks else 0.0)
            ),
            -cooccurrence_scores.get(item, 0.0),
            -popularity_scores.get(item, 0.0),
            item,
        ),
    )
    target = example.target_track_uri
    negatives = [item for item in ranked_pool if item != target]
    if len(negatives) < config.candidate_count - 1:
        fallback = sorted(
            sources.vocabulary.items,
            key=lambda item: (sources.popularity_ranks[item], item),
        )
        used = set(negatives) | {target}
        negatives.extend(item for item in fallback if item not in used)

    candidates = sorted([target, *negatives[: config.candidate_count - 1]])
    permutation = _stable_permutation(
        len(candidates),
        query_id=query_id,
        random_seed=config.random_seed,
    )
    candidates = [candidates[int(index)] for index in permutation]
    history_counts: dict[str, int] = {}
    history_recency: dict[str, int] = {}
    for reverse_position, item in enumerate(reversed(example.history_track_uris), start=1):
        history_counts[item] = history_counts.get(item, 0) + 1
        history_recency.setdefault(item, reverse_position)
    features = np.vstack(
        [
            _candidate_features(
                candidate,
                example=example,
                sources=sources,
                cooccurrence_scores=cooccurrence_scores,
                cooccurrence_ranks=cooccurrence_ranks,
                popularity_pool_ranks=popularity_pool_ranks,
                history_counts=history_counts,
                history_recency=history_recency,
            )
            for candidate in candidates
        ]
    )
    return candidates, features


def _build_raw_split(
    examples: Sequence[TrackLevelExample],
    *,
    split_name: str,
    limit: int,
    sources: _FittedCandidateSources,
    config: TrackRerankingConfig,
) -> _RawRerankingSplit:
    eligible = tuple(
        example
        for example in examples
        if sources.vocabulary.contains(example.target_track_uri)
    )
    selected = temporal_bounded_sample(eligible, limit=limit) if eligible else ()
    query_ids: list[str] = []
    example_ids: list[int] = []
    group_ids: list[int] = []
    group_offsets = [0]
    candidate_uris: list[str] = []
    candidate_ids: list[int] = []
    context_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    labels: list[float] = []

    for group_id, example in enumerate(selected):
        query_id = f"{split_name}:{example.example_id}"
        candidates, features = _candidate_group(
            example,
            query_id=query_id,
            sources=sources,
            config=config,
        )
        context = _context_features(example, vocabulary=sources.vocabulary)
        query_ids.append(query_id)
        example_ids.append(example.example_id)
        group_ids.extend([group_id] * len(candidates))
        candidate_uris.extend(candidates)
        candidate_ids.extend(sources.vocabulary.encode(candidate) for candidate in candidates)
        context_rows.extend([context] * len(candidates))
        candidate_rows.extend(features)
        labels.extend(float(candidate == example.target_track_uri) for candidate in candidates)
        group_offsets.append(len(candidate_uris))

    context_array = (
        np.vstack(context_rows).astype("float32")
        if context_rows
        else np.empty((0, len(CONTEXT_FEATURE_NAMES)), dtype="float32")
    )
    candidate_array = (
        np.vstack(candidate_rows).astype("float32")
        if candidate_rows
        else np.empty((0, len(CANDIDATE_FEATURE_NAMES)), dtype="float32")
    )
    return _RawRerankingSplit(
        query_ids=tuple(query_ids),
        example_ids=np.asarray(example_ids, dtype="int64"),
        group_ids=np.asarray(group_ids, dtype="int64"),
        group_offsets=np.asarray(group_offsets, dtype="int64"),
        candidate_track_uris=tuple(candidate_uris),
        candidate_item_ids=np.asarray(candidate_ids, dtype="int32"),
        context_features=context_array,
        candidate_features=candidate_array,
        labels=np.asarray(labels, dtype="float32").reshape(-1, 1),
        source_example_count=len(examples),
        skipped_oov_target_count=len(examples) - len(eligible),
    )


def _fit_standardizer(values: np.ndarray, *, feature_count: int) -> FeatureStandardizer:
    if values.shape != (len(values), feature_count) or not len(values):
        raise ValueError("Cannot fit a feature standardizer without correctly shaped rows")
    mean = np.mean(values, axis=0, dtype="float64")
    scale = np.std(values, axis=0, dtype="float64")
    scale = np.where(scale < 1e-6, 1.0, scale)
    return FeatureStandardizer(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
    )


def _standardize_split(
    raw: _RawRerankingSplit,
    *,
    context_standardizer: FeatureStandardizer,
    candidate_standardizer: FeatureStandardizer,
) -> TrackRerankingSplit:
    return TrackRerankingSplit(
        query_ids=raw.query_ids,
        example_ids=raw.example_ids,
        group_ids=raw.group_ids,
        group_offsets=raw.group_offsets,
        candidate_track_uris=raw.candidate_track_uris,
        candidate_item_ids=raw.candidate_item_ids,
        context_features=context_standardizer.transform(raw.context_features),
        candidate_features=candidate_standardizer.transform(raw.candidate_features),
        labels=raw.labels,
        source_example_count=raw.source_example_count,
        skipped_oov_target_count=raw.skipped_oov_target_count,
    )


def build_track_reranking_data(
    splits: TrackLevelTemporalSplits,
    *,
    config: TrackRerankingConfig | None = None,
) -> TrackRerankingData:
    """Build pointwise DCN rows from temporal candidate groups without target leakage."""
    resolved = config or TrackRerankingConfig()
    resolved.validate()
    fit_examples, reranker_train_examples = _partition_training_sessions(
        splits.train,
        fit_fraction=resolved.retriever_fit_fraction,
    )
    sources = _fit_candidate_sources(fit_examples, config=resolved)
    raw_train = _build_raw_split(
        reranker_train_examples,
        split_name="train",
        limit=resolved.max_train_queries,
        sources=sources,
        config=resolved,
    )
    if not len(raw_train.labels):
        raise ValueError("No in-vocabulary reranker training queries remain")
    raw_validation = _build_raw_split(
        splits.validation,
        split_name="validation",
        limit=resolved.max_validation_queries,
        sources=sources,
        config=resolved,
    )
    raw_test = _build_raw_split(
        splits.test,
        split_name="test",
        limit=resolved.max_test_queries,
        sources=sources,
        config=resolved,
    )
    context_standardizer = _fit_standardizer(
        raw_train.context_features,
        feature_count=len(CONTEXT_FEATURE_NAMES),
    )
    candidate_standardizer = _fit_standardizer(
        raw_train.candidate_features,
        feature_count=len(CANDIDATE_FEATURE_NAMES),
    )
    standardize = lambda raw: _standardize_split(  # noqa: E731
        raw,
        context_standardizer=context_standardizer,
        candidate_standardizer=candidate_standardizer,
    )
    return TrackRerankingData(
        config=resolved,
        vocabulary=sources.vocabulary,
        context_standardizer=context_standardizer,
        candidate_standardizer=candidate_standardizer,
        retriever_fit_example_count=len(fit_examples),
        reranker_train_source_example_count=len(reranker_train_examples),
        train=standardize(raw_train),
        validation=standardize(raw_validation),
        test=standardize(raw_test),
    )


def save_track_reranking_data(
    data: TrackRerankingData,
    output_dir: Path,
) -> Path:
    """Write compressed tensors and a training-manifest-style JSON artifact."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        split = getattr(data, split_name)
        path = root / f"{split_name}_reranking.npz"
        np.savez_compressed(path, **split.to_npz_payload())
        artifact_paths[split_name] = str(path)

    manifest = data.to_manifest_dict()
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["artifacts"] = artifact_paths
    return write_json(root / "reranking_data_manifest.json", manifest)


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "FeatureStandardizer",
    "OOV_ITEM_ID",
    "PAD_ITEM_ID",
    "RerankingTrackVocabulary",
    "TrackRerankingConfig",
    "TrackRerankingData",
    "TrackRerankingSplit",
    "build_track_reranking_data",
    "fit_reranking_vocabulary",
    "save_track_reranking_data",
    "temporal_bounded_sample",
]
