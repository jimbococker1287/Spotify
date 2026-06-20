from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, overload

import numpy as np

from .track_dcn_training import DCNCandidateSplit, DCNTemporalDataset

TemporalFallback = Literal["auto", "row_order", "query_order"]


@dataclass(frozen=True)
class DCNTemporalReweightingConfig:
    """Bounded recency weighting config for DCN drift experiments."""

    min_weight: float = 0.5
    max_weight: float = 2.0
    half_life: float | None = None
    strength: float = 1.0
    fallback: TemporalFallback = "auto"

    def validate(self) -> None:
        min_weight = _finite_float(self.min_weight, name="min_weight")
        max_weight = _finite_float(self.max_weight, name="max_weight")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be positive and finite")
        if max_weight <= 0.0:
            raise ValueError("max_weight must be positive and finite")
        if min_weight > max_weight:
            raise ValueError("min_weight must be less than or equal to max_weight")
        if min_weight > 1.0 or max_weight < 1.0:
            raise ValueError("min_weight and max_weight must bound a mean-normalized weight of 1.0")
        if self.half_life is not None:
            half_life = _finite_float(self.half_life, name="half_life")
            if half_life <= 0.0:
                raise ValueError("half_life must be positive and finite")
        strength = _finite_float(self.strength, name="strength")
        if strength <= 0.0:
            raise ValueError("strength must be positive and finite")
        if self.fallback not in {"auto", "row_order", "query_order"}:
            raise ValueError("fallback must be 'auto', 'row_order', or 'query_order'")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "half_life": None if self.half_life is None else float(self.half_life),
            "strength": float(self.strength),
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class DCNRecencyWeightResult:
    """Computed recency weights plus JSON-friendly provenance."""

    weights: np.ndarray
    method: Literal["event_times", "fallback"]
    fallback: Literal["row_order", "query_order"] | None
    row_count: int
    source_min: float | None
    source_max: float | None
    config: DCNTemporalReweightingConfig

    @property
    def weight_min(self) -> float | None:
        if not len(self.weights):
            return None
        return float(np.min(self.weights))

    @property
    def weight_max(self) -> float | None:
        if not len(self.weights):
            return None
        return float(np.max(self.weights))

    @property
    def weight_mean(self) -> float | None:
        if not len(self.weights):
            return None
        return float(np.mean(self.weights))

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "fallback": self.fallback,
            "row_count": int(self.row_count),
            "source_min": self.source_min,
            "source_max": self.source_max,
            "weight_min": self.weight_min,
            "weight_mean": self.weight_mean,
            "weight_max": self.weight_max,
            "config": self.config.to_dict(),
            "weights": [float(weight) for weight in np.asarray(self.weights).reshape(-1)],
        }


def compute_normalized_recency_weights(
    event_times: Sequence[object] | np.ndarray | None,
    *,
    row_count: int | None = None,
    query_ids: Sequence[object] | np.ndarray | None = None,
    config: DCNTemporalReweightingConfig | None = None,
) -> DCNRecencyWeightResult:
    """Compute bounded mean-1 recency weights from timestamps or stable order."""

    cfg = config or DCNTemporalReweightingConfig()
    cfg.validate()
    if row_count is None:
        if event_times is not None:
            row_count = int(np.asarray(event_times).reshape(-1).shape[0])
        elif query_ids is not None:
            row_count = int(np.asarray(query_ids).reshape(-1).shape[0])
        else:
            raise ValueError("row_count is required when event_times and query_ids are missing")
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if row_count == 0:
        return DCNRecencyWeightResult(
            weights=np.empty(0, dtype="float32"),
            method="event_times" if event_times is not None else "fallback",
            fallback=None if event_times is not None else _resolve_fallback(cfg.fallback, query_ids),
            row_count=0,
            source_min=None,
            source_max=None,
            config=cfg,
        )

    if event_times is not None:
        source = _coerce_event_times(event_times, expected_count=row_count)
        scores = _event_time_scores(source, cfg)
        method: Literal["event_times", "fallback"] = "event_times"
        fallback: Literal["row_order", "query_order"] | None = None
    else:
        fallback = _resolve_fallback(cfg.fallback, query_ids)
        source = _fallback_source(row_count, fallback=fallback, query_ids=query_ids)
        scores = _monotonic_scores(source, strength=cfg.strength)
        method = "fallback"

    weights = _bounded_mean_one_weights(scores, cfg).astype("float32")
    return DCNRecencyWeightResult(
        weights=weights,
        method=method,
        fallback=fallback,
        row_count=row_count,
        source_min=float(np.min(source)),
        source_max=float(np.max(source)),
        config=cfg,
    )


@overload
def apply_temporal_reweighting(
    data: DCNCandidateSplit,
    config: DCNTemporalReweightingConfig | None = None,
) -> DCNCandidateSplit:
    ...


@overload
def apply_temporal_reweighting(
    data: DCNTemporalDataset,
    config: DCNTemporalReweightingConfig | None = None,
) -> DCNTemporalDataset:
    ...


def apply_temporal_reweighting(
    data: DCNCandidateSplit | DCNTemporalDataset,
    config: DCNTemporalReweightingConfig | None = None,
) -> DCNCandidateSplit | DCNTemporalDataset:
    """Return a copied split/dataset with recency weights multiplied into train weights."""

    if isinstance(data, DCNTemporalDataset):
        train = _reweighted_split(data.train, config=config, name="train")
        return DCNTemporalDataset(
            train=train,
            validation=_copy_split(data.validation),
            test=_copy_split(data.test),
        )
    return _reweighted_split(data, config=config, name="train")


def _reweighted_split(
    split: DCNCandidateSplit,
    *,
    config: DCNTemporalReweightingConfig | None,
    name: str,
) -> DCNCandidateSplit:
    split.validate(name=name)
    recency = compute_normalized_recency_weights(
        split.event_times,
        row_count=len(split),
        query_ids=split.query_ids,
        config=config,
    ).weights
    existing = (
        np.ones(len(split), dtype="float32")
        if split.sample_weights is None
        else np.asarray(split.sample_weights, dtype="float32").reshape(-1)
    )
    copied = _copy_split(split)
    return DCNCandidateSplit(
        context_features=copied.context_features,
        item_features=copied.item_features,
        labels=copied.labels,
        query_ids=copied.query_ids,
        candidate_ids=copied.candidate_ids,
        event_times=copied.event_times,
        sample_weights=(existing * recency).astype("float32"),
    )


def _copy_split(split: DCNCandidateSplit) -> DCNCandidateSplit:
    def optional(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.asarray(values).copy()

    return DCNCandidateSplit(
        context_features=np.asarray(split.context_features).copy(),
        item_features=np.asarray(split.item_features).copy(),
        labels=np.asarray(split.labels).reshape(-1).copy(),
        query_ids=np.asarray(split.query_ids).reshape(-1).copy(),
        candidate_ids=optional(split.candidate_ids),
        event_times=optional(split.event_times),
        sample_weights=optional(split.sample_weights),
    )


def _coerce_event_times(
    event_times: Sequence[object] | np.ndarray,
    *,
    expected_count: int,
) -> np.ndarray:
    values = np.asarray(event_times).reshape(-1)
    if len(values) != expected_count:
        raise ValueError(f"event_times has {len(values)} rows; expected {expected_count}")
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype("float64", copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("event_times must contain finite values")
        return numeric
    try:
        timestamps = values.astype("datetime64[ns]").astype("int64")
    except (TypeError, ValueError) as exc:
        raise ValueError("event_times must contain numeric or datetime-like values") from exc
    if (timestamps == np.iinfo("int64").min).any():
        raise ValueError("event_times must not contain missing timestamps")
    return timestamps.astype("float64")


def _event_time_scores(
    source: np.ndarray,
    config: DCNTemporalReweightingConfig,
) -> np.ndarray:
    if config.half_life is None:
        return _monotonic_scores(source, strength=config.strength)
    newest = float(np.max(source))
    age = np.maximum(0.0, newest - source.astype("float64", copy=False))
    decayed = np.power(0.5, age / float(config.half_life))
    return _min_max_scores(decayed)


def _monotonic_scores(source: np.ndarray, *, strength: float) -> np.ndarray:
    normalized = _min_max_scores(source.astype("float64", copy=False))
    return np.power(normalized, strength)


def _min_max_scores(source: np.ndarray) -> np.ndarray:
    minimum = float(np.min(source))
    maximum = float(np.max(source))
    if minimum == maximum:
        return np.ones(len(source), dtype="float64")
    return (source - minimum) / (maximum - minimum)


def _bounded_mean_one_weights(
    scores: np.ndarray,
    config: DCNTemporalReweightingConfig,
) -> np.ndarray:
    if len(scores) == 0:
        return np.empty(0, dtype="float64")
    centered = scores.astype("float64", copy=False) - float(np.mean(scores))
    lower_extent = abs(float(np.min(centered)))
    upper_extent = float(np.max(centered))
    if lower_extent == 0.0 and upper_extent == 0.0:
        return np.ones(len(scores), dtype="float64")
    lower_scale = (
        np.inf
        if lower_extent == 0.0
        else (1.0 - float(config.min_weight)) / lower_extent
    )
    upper_scale = (
        np.inf
        if upper_extent == 0.0
        else (float(config.max_weight) - 1.0) / upper_extent
    )
    scale = min(lower_scale, upper_scale)
    weights = 1.0 + (centered * scale)
    return np.clip(weights, config.min_weight, config.max_weight)


def _resolve_fallback(
    fallback: TemporalFallback,
    query_ids: Sequence[object] | np.ndarray | None,
) -> Literal["row_order", "query_order"]:
    if fallback == "auto":
        return "query_order" if query_ids is not None else "row_order"
    return fallback


def _fallback_source(
    row_count: int,
    *,
    fallback: Literal["row_order", "query_order"],
    query_ids: Sequence[object] | np.ndarray | None,
) -> np.ndarray:
    if fallback == "row_order":
        return np.arange(row_count, dtype="float64")
    if query_ids is None:
        raise ValueError("query_ids are required for query_order fallback")
    ids = np.asarray(query_ids).reshape(-1)
    if len(ids) != row_count:
        raise ValueError(f"query_ids has {len(ids)} rows; expected {row_count}")
    query_ordinals: dict[object, int] = {}
    source = np.empty(row_count, dtype="float64")
    for row_index, query_id in enumerate(ids.tolist()):
        query_ordinals.setdefault(query_id, len(query_ordinals))
        source[row_index] = float(query_ordinals[query_id])
    return source


def _finite_float(value: float | None, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


__all__ = [
    "DCNRecencyWeightResult",
    "DCNTemporalReweightingConfig",
    "apply_temporal_reweighting",
    "compute_normalized_recency_weights",
]
