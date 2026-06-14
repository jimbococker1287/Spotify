from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np


AugmentationName = Literal["crop", "subsequence", "mask"]
AttributeTables = np.ndarray | Mapping[str, np.ndarray]


@dataclass(frozen=True)
class MaskedItemBatch:
    """A compact masked-item batch with one input row per source sequence."""

    input_sequences: np.ndarray
    target_items: np.ndarray
    source_rows: np.ndarray
    target_positions: np.ndarray
    prediction_mask: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class SequenceView:
    """An augmented sequence view and its relationship to the source positions."""

    sequences: np.ndarray
    valid_mask: np.ndarray
    source_positions: np.ndarray


@dataclass(frozen=True)
class PositivePairBatch:
    """Two sequence views known to represent the same next-item intent."""

    left: SequenceView
    right: SequenceView
    left_rows: np.ndarray
    right_rows: np.ndarray
    pair_kind: str


@dataclass(frozen=True)
class AttributePredictionBatch:
    """Selected item positions and metadata targets for auxiliary prediction."""

    source_rows: np.ndarray
    positions: np.ndarray
    item_ids: np.ndarray
    targets: dict[str, np.ndarray]


def _validate_sequences(sequences: object) -> np.ndarray:
    values = np.asarray(sequences)
    if values.ndim != 2:
        raise ValueError("sequences must be a rank-2 array")
    if values.shape[1] == 0:
        raise ValueError("sequences must contain at least one position")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("sequences must contain integer item IDs")
    return values.astype("int32", copy=False)


def _valid_mask(sequences: np.ndarray, padding_id: int | None) -> np.ndarray:
    if padding_id is None:
        return np.ones(sequences.shape, dtype=bool)
    return sequences != int(padding_id)


def _protected_mask(valid_mask: np.ndarray, protect_last_n: int) -> np.ndarray:
    if protect_last_n < 0:
        raise ValueError("protect_last_n must be non-negative")
    protected = np.zeros(valid_mask.shape, dtype=bool)
    if protect_last_n == 0:
        return protected
    for row_index, row_mask in enumerate(valid_mask):
        positions = np.flatnonzero(row_mask)
        protected[row_index, positions[-protect_last_n:]] = True
    return protected


def _sample_positions(
    eligible: np.ndarray,
    probability: float,
    *,
    rng: np.random.Generator,
    ensure_one_per_sequence: bool,
) -> np.ndarray:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in the interval [0, 1]")
    selected = (rng.random(eligible.shape) < probability) & eligible
    if ensure_one_per_sequence:
        for row_index, row_eligible in enumerate(eligible):
            candidates = np.flatnonzero(row_eligible)
            if candidates.size and not selected[row_index].any():
                selected[row_index, int(rng.choice(candidates))] = True
    return selected


def build_masked_item_batch(
    sequences: object,
    *,
    mask_token_id: int,
    mask_probability: float = 0.2,
    padding_id: int | None = None,
    protect_last_n: int = 0,
    ensure_one_per_sequence: bool = True,
    seed: int = 0,
) -> MaskedItemBatch:
    """Mask valid positions without expanding source rows.

    Targets are flattened for easy gathering from an encoder's token states.
    ``source_rows`` and ``target_positions`` identify the corresponding states.
    Protected recent items remain visible, which lets callers retain a stable
    next-item intent anchor while masking older context.
    """

    values = _validate_sequences(sequences)
    valid = _valid_mask(values, padding_id)
    protected = _protected_mask(valid, protect_last_n)
    selected = _sample_positions(
        valid & ~protected,
        mask_probability,
        rng=np.random.default_rng(seed),
        ensure_one_per_sequence=ensure_one_per_sequence,
    )
    source_rows, target_positions = np.nonzero(selected)
    targets = values[source_rows, target_positions].astype("int32", copy=True)
    masked = values.copy()
    masked[selected] = int(mask_token_id)
    return MaskedItemBatch(
        input_sequences=masked,
        target_items=targets,
        source_rows=source_rows.astype("int32", copy=False),
        target_positions=target_positions.astype("int32", copy=False),
        prediction_mask=selected,
        valid_mask=valid,
    )


def _pack_view(
    source: np.ndarray,
    kept_positions: list[np.ndarray],
    *,
    padding_id: int,
) -> SequenceView:
    packed = np.full(source.shape, int(padding_id), dtype="int32")
    source_positions = np.full(source.shape, -1, dtype="int32")
    valid = np.zeros(source.shape, dtype=bool)
    for row_index, positions in enumerate(kept_positions):
        if positions.size == 0:
            continue
        output_positions = np.arange(source.shape[1] - positions.size, source.shape[1])
        packed[row_index, output_positions] = source[row_index, positions]
        source_positions[row_index, output_positions] = positions
        valid[row_index, output_positions] = True
    return SequenceView(packed, valid, source_positions)


def crop_sequences(
    sequences: object,
    *,
    crop_ratio: float = 0.7,
    padding_id: int = -1,
    protect_last_n: int = 1,
    min_items: int = 1,
    seed: int = 0,
) -> SequenceView:
    """Create fixed-width contiguous crops.

    With a protected suffix, each crop is anchored at the latest event so the
    recent intent remains unchanged. Without protection, the crop start is
    sampled deterministically from ``seed``.
    """

    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in the interval (0, 1]")
    if min_items <= 0:
        raise ValueError("min_items must be positive")
    values = _validate_sequences(sequences)
    valid = _valid_mask(values, padding_id)
    if protect_last_n < 0:
        raise ValueError("protect_last_n must be non-negative")
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for row_mask in valid:
        positions = np.flatnonzero(row_mask)
        if positions.size == 0:
            kept.append(positions)
            continue
        keep_count = min(
            positions.size,
            max(min_items, protect_last_n, int(np.ceil(positions.size * crop_ratio))),
        )
        if protect_last_n:
            start = positions.size - keep_count
        else:
            start = int(rng.integers(0, positions.size - keep_count + 1))
        kept.append(positions[start : start + keep_count])
    return _pack_view(values, kept, padding_id=padding_id)


def subsequence_sequences(
    sequences: object,
    *,
    drop_probability: float = 0.2,
    padding_id: int = -1,
    protect_last_n: int = 1,
    min_items: int = 1,
    seed: int = 0,
) -> SequenceView:
    """Drop non-protected items while retaining order and a recent suffix."""

    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must be in the interval [0, 1)")
    if min_items <= 0:
        raise ValueError("min_items must be positive")
    values = _validate_sequences(sequences)
    valid = _valid_mask(values, padding_id)
    protected = _protected_mask(valid, protect_last_n)
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for row_index, row_mask in enumerate(valid):
        positions = np.flatnonzero(row_mask)
        if positions.size == 0:
            kept.append(positions)
            continue
        row_keep = protected[row_index].copy()
        candidates = np.flatnonzero(row_mask & ~protected[row_index])
        row_keep[candidates] = rng.random(candidates.size) >= drop_probability
        required = min(min_items, positions.size)
        missing = required - int(np.count_nonzero(row_keep))
        if missing > 0:
            available = np.flatnonzero(row_mask & ~row_keep)
            chosen = rng.choice(available, size=missing, replace=False)
            row_keep[chosen] = True
        kept.append(np.flatnonzero(row_keep))
    return _pack_view(values, kept, padding_id=padding_id)


def mask_sequences(
    sequences: object,
    *,
    mask_token_id: int,
    mask_probability: float = 0.2,
    padding_id: int = -1,
    protect_last_n: int = 1,
    ensure_one_per_sequence: bool = True,
    seed: int = 0,
) -> SequenceView:
    """Mask older context while keeping protected recent items untouched."""

    batch = build_masked_item_batch(
        sequences,
        mask_token_id=mask_token_id,
        mask_probability=mask_probability,
        padding_id=padding_id,
        protect_last_n=protect_last_n,
        ensure_one_per_sequence=ensure_one_per_sequence,
        seed=seed,
    )
    source_positions = np.broadcast_to(
        np.arange(batch.input_sequences.shape[1], dtype="int32"),
        batch.input_sequences.shape,
    ).copy()
    source_positions[~batch.valid_mask] = -1
    return SequenceView(batch.input_sequences, batch.valid_mask, source_positions)


def augment_sequences(
    sequences: object,
    augmentation: AugmentationName,
    *,
    mask_token_id: int | None = None,
    crop_ratio: float = 0.7,
    drop_probability: float = 0.2,
    mask_probability: float = 0.2,
    padding_id: int = -1,
    protect_last_n: int = 1,
    min_items: int = 1,
    seed: int = 0,
) -> SequenceView:
    """Dispatch a named augmentation with consistent intent constraints."""

    if augmentation == "crop":
        return crop_sequences(
            sequences,
            crop_ratio=crop_ratio,
            padding_id=padding_id,
            protect_last_n=protect_last_n,
            min_items=min_items,
            seed=seed,
        )
    if augmentation == "subsequence":
        return subsequence_sequences(
            sequences,
            drop_probability=drop_probability,
            padding_id=padding_id,
            protect_last_n=protect_last_n,
            min_items=min_items,
            seed=seed,
        )
    if augmentation == "mask":
        if mask_token_id is None:
            raise ValueError("mask_token_id is required for mask augmentation")
        return mask_sequences(
            sequences,
            mask_token_id=mask_token_id,
            mask_probability=mask_probability,
            padding_id=padding_id,
            protect_last_n=protect_last_n,
            seed=seed,
        )
    raise ValueError(f"unknown augmentation: {augmentation!r}")


def build_augmented_positive_pairs(
    sequences: object,
    *,
    left_augmentation: AugmentationName = "crop",
    right_augmentation: AugmentationName = "subsequence",
    mask_token_id: int | None = None,
    crop_ratio: float = 0.7,
    drop_probability: float = 0.2,
    mask_probability: float = 0.2,
    padding_id: int = -1,
    protect_last_n: int = 1,
    min_items: int = 1,
    seed: int = 0,
) -> PositivePairBatch:
    """Build two deterministic views of every source sequence."""

    values = _validate_sequences(sequences)
    common = {
        "mask_token_id": mask_token_id,
        "crop_ratio": crop_ratio,
        "drop_probability": drop_probability,
        "mask_probability": mask_probability,
        "padding_id": padding_id,
        "protect_last_n": protect_last_n,
        "min_items": min_items,
    }
    left = augment_sequences(values, left_augmentation, seed=seed, **common)
    right = augment_sequences(values, right_augmentation, seed=seed + 1, **common)
    rows = np.arange(values.shape[0], dtype="int32")
    return PositivePairBatch(left, right, rows, rows.copy(), "augmented")


def _subset_view(sequences: np.ndarray, rows: np.ndarray, padding_id: int | None) -> SequenceView:
    subset = sequences[rows].copy()
    valid = _valid_mask(subset, padding_id)
    positions = np.broadcast_to(np.arange(subset.shape[1], dtype="int32"), subset.shape).copy()
    positions[~valid] = -1
    return SequenceView(subset, valid, positions)


def build_same_target_positive_pairs(
    sequences: object,
    targets: object,
    *,
    padding_id: int | None = None,
    seed: int = 0,
) -> PositivePairBatch:
    """Pair sequences sharing a supervised next-item target, as in DuoRec.

    Singleton target groups are omitted instead of being paired with themselves.
    Each eligible source row contributes exactly one pair.
    """

    values = _validate_sequences(sequences)
    target_values = np.asarray(targets)
    if target_values.ndim != 1 or target_values.shape[0] != values.shape[0]:
        raise ValueError("targets must be a rank-1 array matching the sequence batch")
    rng = np.random.default_rng(seed)
    left_rows: list[int] = []
    right_rows: list[int] = []
    for target in np.unique(target_values):
        group = np.flatnonzero(target_values == target)
        if group.size < 2:
            continue
        for source_row in group:
            candidates = group[group != source_row]
            left_rows.append(int(source_row))
            right_rows.append(int(rng.choice(candidates)))
    left_indices = np.asarray(left_rows, dtype="int32")
    right_indices = np.asarray(right_rows, dtype="int32")
    return PositivePairBatch(
        _subset_view(values, left_indices, padding_id),
        _subset_view(values, right_indices, padding_id),
        left_indices,
        right_indices,
        "same_target",
    )


def numpy_info_nce_loss(
    left_embeddings: object,
    right_embeddings: object,
    *,
    temperature: float = 0.2,
    symmetric: bool = True,
    group_ids: object | None = None,
) -> float:
    """Compute cosine InfoNCE with in-batch negatives.

    ``group_ids`` can suppress false negatives that share a semantic target.
    The diagonal positive is retained even when all examples share a group.
    """

    left = np.asarray(left_embeddings, dtype="float64")
    right = np.asarray(right_embeddings, dtype="float64")
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left_embeddings and right_embeddings must have the same rank-2 shape")
    if left.shape[0] == 0:
        raise ValueError("embedding batches must not be empty")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    left = left / np.maximum(left_norms, 1e-12)
    right = right / np.maximum(right_norms, 1e-12)
    logits = (left @ right.T) / float(temperature)

    if group_ids is not None:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or groups.shape[0] != left.shape[0]:
            raise ValueError("group_ids must be rank-1 and match the embedding batch")
        false_negatives = groups[:, None] == groups[None, :]
        np.fill_diagonal(false_negatives, False)
        logits = np.where(false_negatives, -np.inf, logits)

    def directional_loss(scores: np.ndarray) -> float:
        row_max = np.max(scores, axis=1, keepdims=True)
        log_normalizer = row_max[:, 0] + np.log(np.exp(scores - row_max).sum(axis=1))
        return float(np.mean(log_normalizer - np.diag(scores)))

    loss = directional_loss(logits)
    if symmetric:
        loss = 0.5 * (loss + directional_loss(logits.T))
    return float(loss)


def info_nce_loss(
    left_embeddings: object,
    right_embeddings: object,
    *,
    temperature: float = 0.2,
    symmetric: bool = True,
    group_ids: object | None = None,
) -> float:
    """Alias for the NumPy implementation used by lightweight trainers."""

    return numpy_info_nce_loss(
        left_embeddings,
        right_embeddings,
        temperature=temperature,
        symmetric=symmetric,
        group_ids=group_ids,
    )


def _attribute_tables(item_attributes: AttributeTables) -> dict[str, np.ndarray]:
    if isinstance(item_attributes, Mapping):
        tables = {str(name): np.asarray(values) for name, values in item_attributes.items()}
        if not tables:
            raise ValueError("item_attributes must not be empty")
    else:
        tables = {"attributes": np.asarray(item_attributes)}
    for name, table in tables.items():
        if table.ndim == 0:
            raise ValueError(f"attribute table {name!r} must have an item axis")
    return tables


def build_attribute_prediction_batch(
    sequences: object,
    item_attributes: AttributeTables,
    *,
    selection_probability: float = 0.2,
    padding_id: int | None = None,
    protect_last_n: int = 0,
    ensure_one_per_sequence: bool = True,
    seed: int = 0,
) -> AttributePredictionBatch:
    """Sample item positions and look up one or more auxiliary target tables."""

    values = _validate_sequences(sequences)
    valid = _valid_mask(values, padding_id)
    protected = _protected_mask(valid, protect_last_n)
    selected = _sample_positions(
        valid & ~protected,
        selection_probability,
        rng=np.random.default_rng(seed),
        ensure_one_per_sequence=ensure_one_per_sequence,
    )
    source_rows, positions = np.nonzero(selected)
    item_ids = values[source_rows, positions].astype("int32", copy=False)
    if np.any(item_ids < 0):
        raise ValueError("selected item IDs must be non-negative for attribute lookup")
    tables = _attribute_tables(item_attributes)
    targets: dict[str, np.ndarray] = {}
    for name, table in tables.items():
        if item_ids.size and int(item_ids.max()) >= table.shape[0]:
            raise ValueError(f"attribute table {name!r} does not cover every selected item ID")
        targets[name] = table[item_ids].copy()
    return AttributePredictionBatch(
        source_rows.astype("int32", copy=False),
        positions.astype("int32", copy=False),
        item_ids,
        targets,
    )


def build_metadata_prediction_batch(
    sequences: object,
    item_metadata: AttributeTables,
    **kwargs: object,
) -> AttributePredictionBatch:
    """Metadata-oriented alias for ``build_attribute_prediction_batch``."""

    return build_attribute_prediction_batch(sequences, item_metadata, **kwargs)


def build_sequence_attribute_targets(
    sequences: object,
    item_attributes: AttributeTables,
    *,
    padding_id: int | None = None,
    aggregation: Literal["mean", "max", "last"] = "max",
) -> dict[str, np.ndarray]:
    """Aggregate item metadata into sequence-level auxiliary targets."""

    if aggregation not in {"mean", "max", "last"}:
        raise ValueError("aggregation must be one of: mean, max, last")
    values = _validate_sequences(sequences)
    valid = _valid_mask(values, padding_id)
    tables = _attribute_tables(item_attributes)
    outputs: dict[str, np.ndarray] = {}
    for name, table in tables.items():
        valid_items = values[valid]
        if valid_items.size and (np.any(valid_items < 0) or int(valid_items.max()) >= table.shape[0]):
            raise ValueError(f"attribute table {name!r} does not cover every valid item ID")
        output_shape = (values.shape[0],) + table.shape[1:]
        output_dtype = np.result_type(table.dtype, np.float32) if aggregation == "mean" else table.dtype
        output = np.zeros(output_shape, dtype=output_dtype)
        for row_index, row_mask in enumerate(valid):
            item_ids = values[row_index, row_mask]
            if item_ids.size == 0:
                continue
            row_attributes = table[item_ids]
            if aggregation == "last":
                output[row_index] = row_attributes[-1]
            elif aggregation == "mean":
                output[row_index] = np.mean(row_attributes, axis=0)
            else:
                output[row_index] = np.max(row_attributes, axis=0)
        outputs[name] = output
    return outputs


__all__ = [
    "AttributePredictionBatch",
    "MaskedItemBatch",
    "PositivePairBatch",
    "SequenceView",
    "augment_sequences",
    "build_attribute_prediction_batch",
    "build_augmented_positive_pairs",
    "build_masked_item_batch",
    "build_metadata_prediction_batch",
    "build_same_target_positive_pairs",
    "build_sequence_attribute_targets",
    "crop_sequences",
    "info_nce_loss",
    "mask_sequences",
    "numpy_info_nce_loss",
    "subsequence_sequences",
]
