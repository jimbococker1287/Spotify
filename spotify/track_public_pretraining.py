from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .public_training_data import (
    DatasetSourceManifest,
    PublicTrainingDataError,
    file_sha256,
    load_source_manifest,
    validate_canonical_interactions,
    validate_source_manifest,
)
from .sequence_pretraining import (
    MaskedItemBatch,
    PositivePairBatch,
    build_augmented_positive_pairs,
    build_masked_item_batch,
)


PADDING_ITEM_ID = 0
MASK_ITEM_ID = 1
OOV_ITEM_ID = 2
_ITEM_ID_OFFSET = 3
_TRAINING_SPLITS = ("", "train", "training", "pretrain", "pretraining", "public_train")

RecordLoader = Callable[[DatasetSourceManifest], pd.DataFrame | Iterable[pd.DataFrame | Mapping[str, object]]]
PretrainingTrainer = Callable[["PublicPretrainingTrainingRequest"], "PublicPretrainingCheckpoint"]


class PublicPretrainingError(ValueError):
    """Raised when approved public records cannot satisfy the pretraining contract."""


@dataclass(frozen=True)
class PublicPretrainingConfig:
    max_source_records: int = 250_000
    max_sequences: int = 20_000
    max_sequence_length: int = 128
    min_sequence_items: int = 2
    max_vocabulary_items: int = 100_000
    max_masked_sequences: int = 4_096
    max_masked_predictions: int = 32_768
    max_contrastive_pairs: int = 4_096
    mask_probability: float = 0.2
    crop_ratio: float = 0.7
    drop_probability: float = 0.2
    protect_last_n: int = 1
    allowed_training_splits: tuple[str, ...] = _TRAINING_SPLITS
    seed: int = 42

    def __post_init__(self) -> None:
        positive_fields = (
            "max_source_records",
            "max_sequences",
            "max_sequence_length",
            "min_sequence_items",
            "max_vocabulary_items",
            "max_masked_sequences",
            "max_masked_predictions",
            "max_contrastive_pairs",
        )
        for name in positive_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_sequence_items > self.max_sequence_length:
            raise ValueError("min_sequence_items cannot exceed max_sequence_length")
        if not 0.0 <= self.mask_probability <= 1.0:
            raise ValueError("mask_probability must be in the interval [0, 1]")
        if not 0.0 < self.crop_ratio <= 1.0:
            raise ValueError("crop_ratio must be in the interval (0, 1]")
        if not 0.0 <= self.drop_probability < 1.0:
            raise ValueError("drop_probability must be in the interval [0, 1)")
        if self.protect_last_n < 0:
            raise ValueError("protect_last_n must be non-negative")
        if not self.allowed_training_splits:
            raise ValueError("allowed_training_splits must not be empty")


@dataclass(frozen=True)
class PublicPretrainingExamples:
    sequences: np.ndarray
    session_keys: tuple[str, ...]
    item_to_index: Mapping[str, int]
    vocabulary_digest: str
    source_record_count: int
    eligible_record_count: int
    excluded_split_rows: int
    dropped_unkeyed_rows: int
    dropped_short_sessions: int
    truncated_item_count: int

    def summary(self) -> dict[str, object]:
        return {
            "source_record_count": int(self.source_record_count),
            "eligible_record_count": int(self.eligible_record_count),
            "sequence_count": int(self.sequences.shape[0]),
            "sequence_length": int(self.sequences.shape[1]),
            "vocabulary_size": int(len(self.item_to_index)),
            "vocabulary_digest": self.vocabulary_digest,
            "excluded_split_rows": int(self.excluded_split_rows),
            "dropped_unkeyed_rows": int(self.dropped_unkeyed_rows),
            "dropped_short_sessions": int(self.dropped_short_sessions),
            "truncated_item_count": int(self.truncated_item_count),
            "private_validation_or_test_included": False,
        }


@dataclass(frozen=True)
class PublicPretrainingBatches:
    masked_items: MaskedItemBatch
    masked_example_rows: np.ndarray
    contrastive_pairs: PositivePairBatch
    contrastive_example_rows: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "masked_sequence_count": int(self.masked_items.input_sequences.shape[0]),
            "masked_prediction_count": int(self.masked_items.target_items.shape[0]),
            "contrastive_pair_count": int(self.contrastive_pairs.left_rows.shape[0]),
            "bounds_applied": True,
        }


@dataclass(frozen=True)
class PublicPretrainingTrainingRequest:
    dataset_id: str
    manifest_digest: str
    source_file_digests: tuple[tuple[str, str], ...]
    examples: PublicPretrainingExamples
    batches: PublicPretrainingBatches
    config: PublicPretrainingConfig
    reserved_item_ids: Mapping[str, int] = field(
        default_factory=lambda: {
            "padding": PADDING_ITEM_ID,
            "mask": MASK_ITEM_ID,
            "oov": OOV_ITEM_ID,
        }
    )


@dataclass(frozen=True)
class PublicPretrainingCheckpoint:
    checkpoint_path: str | Path
    encoder_family: str
    embedding_dim: int
    transferable_components: tuple[str, ...]
    target_model_families: tuple[str, ...]
    metrics: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointTransferContract:
    schema_version: int
    checkpoint_path: str
    checkpoint_sha256: str
    encoder_family: str
    embedding_dim: int
    transferable_components: tuple[str, ...]
    target_model_families: tuple[str, ...]
    dataset_id: str
    manifest_digest: str
    source_file_digests: tuple[tuple[str, str], ...]
    vocabulary_digest: str
    vocabulary_size: int
    sequence_length: int
    reserved_item_ids: Mapping[str, int]
    private_data_included: bool = False

    def to_dict(self) -> dict[str, object]:
        return _json_friendly(asdict(self))


@dataclass(frozen=True)
class GovernanceDecision:
    status: Literal["blocked", "approved"]
    dataset_id: str
    manifest_digest: str
    training_use_approved: bool
    checksums_verified: bool
    source_files: tuple[Mapping[str, object], ...]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return _json_friendly(asdict(self))


@dataclass(frozen=True)
class PublicPretrainingResult:
    status: Literal["blocked", "ready"]
    stage: Literal["governance", "data", "batches", "training", "complete"]
    governance: GovernanceDecision
    data: Mapping[str, object] = field(default_factory=dict)
    batches: Mapping[str, object] = field(default_factory=dict)
    training: Mapping[str, object] = field(default_factory=dict)
    transfer_contract: CheckpointTransferContract | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = {
            "status": self.status,
            "stage": self.stage,
            "governance": self.governance.to_dict(),
            "data": self.data,
            "batches": self.batches,
            "training": self.training,
            "transfer_contract": (
                self.transfer_contract.to_dict() if self.transfer_contract is not None else None
            ),
            "reason": self.reason,
        }
        return _json_friendly(payload)


def manifest_digest(manifest: DatasetSourceManifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def map_public_records_to_examples(
    records: pd.DataFrame,
    *,
    manifest: DatasetSourceManifest,
    config: PublicPretrainingConfig | None = None,
) -> PublicPretrainingExamples:
    """Validate and map approved public interactions into bounded session tensors."""

    validate_source_manifest(manifest, verify_checksums=True)
    return _map_validated_public_records(records, manifest=manifest, config=config or PublicPretrainingConfig())


def _map_validated_public_records(
    records: pd.DataFrame,
    *,
    manifest: DatasetSourceManifest,
    config: PublicPretrainingConfig,
) -> PublicPretrainingExamples:
    frame = records.copy()
    validate_canonical_interactions(frame)
    source_ids = set(frame["source_dataset"].dropna().astype("string"))
    if source_ids != {manifest.dataset_id}:
        raise PublicPretrainingError(
            "Every public pretraining row must use the approved manifest dataset_id."
        )

    split_values = frame["split"].map(_normalized_text).str.casefold()
    allowed_splits = {value.strip().casefold() for value in config.allowed_training_splits}
    eligible_mask = split_values.isin(allowed_splits)
    excluded_split_rows = int((~eligible_mask).sum())
    eligible = frame.loc[eligible_mask].copy()
    eligible["_source_order"] = np.arange(len(eligible), dtype="int64")
    eligible["_event_time"] = pd.to_datetime(eligible["timestamp"], errors="coerce", utc=True)
    eligible["_session_key"] = eligible.apply(_session_key, axis=1)
    dropped_unkeyed_rows = int(eligible["_session_key"].isna().sum())
    eligible = eligible.loc[eligible["_session_key"].notna()].copy()

    sessions: list[tuple[str, list[str]]] = []
    dropped_short_sessions = 0
    for session_key, group in eligible.groupby("_session_key", sort=True):
        ordered = group.sort_values(["_event_time", "_source_order"], kind="stable", na_position="last")
        items = [str(value) for value in ordered["item_id"]]
        if len(items) < config.min_sequence_items:
            dropped_short_sessions += 1
            continue
        sessions.append((str(session_key), items))
    if not sessions:
        raise PublicPretrainingError("No eligible public sessions contain enough training items.")

    selected_rows = _bounded_indices(len(sessions), config.max_sequences, seed=config.seed)
    selected_sessions = [sessions[int(index)] for index in selected_rows]
    item_counts = Counter(item for _, items in selected_sessions for item in items)
    ranked_items = sorted(item_counts, key=lambda item: (-item_counts[item], item))
    retained_items = ranked_items[: config.max_vocabulary_items]
    item_to_index = {item: index + _ITEM_ID_OFFSET for index, item in enumerate(retained_items)}
    truncated_item_count = len(ranked_items) - len(retained_items)

    sequences = np.full(
        (len(selected_sessions), config.max_sequence_length),
        PADDING_ITEM_ID,
        dtype="int32",
    )
    session_keys: list[str] = []
    for row_index, (session_key, items) in enumerate(selected_sessions):
        encoded = [item_to_index.get(item, OOV_ITEM_ID) for item in items[-config.max_sequence_length :]]
        sequences[row_index, -len(encoded) :] = encoded
        session_keys.append(session_key)
    vocabulary_payload = json.dumps(item_to_index, sort_keys=True, separators=(",", ":"))
    vocabulary_digest = hashlib.sha256(vocabulary_payload.encode("utf-8")).hexdigest()
    return PublicPretrainingExamples(
        sequences=sequences,
        session_keys=tuple(session_keys),
        item_to_index=item_to_index,
        vocabulary_digest=vocabulary_digest,
        source_record_count=int(len(frame)),
        eligible_record_count=int(len(eligible)),
        excluded_split_rows=excluded_split_rows,
        dropped_unkeyed_rows=dropped_unkeyed_rows,
        dropped_short_sessions=dropped_short_sessions,
        truncated_item_count=truncated_item_count,
    )


def build_public_pretraining_batches(
    examples: PublicPretrainingExamples,
    *,
    config: PublicPretrainingConfig | None = None,
) -> PublicPretrainingBatches:
    """Build deterministic masked-item and augmented-session batches within explicit bounds."""

    resolved = config or PublicPretrainingConfig()
    sequence_count = int(examples.sequences.shape[0])
    if sequence_count == 0:
        raise PublicPretrainingError("At least one public sequence is required to build batches.")

    masked_rows = _bounded_indices(sequence_count, resolved.max_masked_sequences, seed=resolved.seed + 11)
    masked = build_masked_item_batch(
        examples.sequences[masked_rows],
        mask_token_id=MASK_ITEM_ID,
        mask_probability=resolved.mask_probability,
        padding_id=PADDING_ITEM_ID,
        protect_last_n=resolved.protect_last_n,
        seed=resolved.seed + 12,
    )
    masked = _cap_masked_predictions(
        masked,
        max_predictions=resolved.max_masked_predictions,
        seed=resolved.seed + 13,
    )

    contrastive_rows = _bounded_indices(
        sequence_count,
        resolved.max_contrastive_pairs,
        seed=resolved.seed + 21,
    )
    contrastive = build_augmented_positive_pairs(
        examples.sequences[contrastive_rows],
        left_augmentation="crop",
        right_augmentation="subsequence",
        crop_ratio=resolved.crop_ratio,
        drop_probability=resolved.drop_probability,
        padding_id=PADDING_ITEM_ID,
        protect_last_n=resolved.protect_last_n,
        seed=resolved.seed + 22,
    )
    return PublicPretrainingBatches(
        masked_items=masked,
        masked_example_rows=masked_rows.astype("int32", copy=False),
        contrastive_pairs=contrastive,
        contrastive_example_rows=contrastive_rows.astype("int32", copy=False),
    )


def run_public_pretraining(
    manifest: DatasetSourceManifest | str | Path,
    *,
    record_loader: RecordLoader,
    config: PublicPretrainingConfig | None = None,
    trainer: PretrainingTrainer | None = None,
) -> PublicPretrainingResult:
    """Gate, load, batch, and optionally train from an approved public source.

    Governance validation always completes before ``record_loader`` or ``trainer``
    is invoked. This ordering is the primary boundary preventing unlicensed data
    from entering training.
    """

    resolved_config = config or PublicPretrainingConfig()
    try:
        source_manifest = (
            manifest if isinstance(manifest, DatasetSourceManifest) else load_source_manifest(manifest)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        governance = GovernanceDecision("blocked", "", "", False, False, (), str(exc))
        return PublicPretrainingResult("blocked", "governance", governance, reason=str(exc))

    digest = manifest_digest(source_manifest)
    try:
        validate_source_manifest(source_manifest, verify_checksums=True)
    except (PublicTrainingDataError, OSError, ValueError) as exc:
        governance = _governance_decision(source_manifest, digest, status="blocked", reason=str(exc))
        return PublicPretrainingResult("blocked", "governance", governance, reason=str(exc))
    governance = _governance_decision(source_manifest, digest, status="approved")

    try:
        records = _load_bounded_records(record_loader(source_manifest), resolved_config.max_source_records)
        examples = _map_validated_public_records(
            records,
            manifest=source_manifest,
            config=resolved_config,
        )
    except Exception as exc:  # The loader is an explicit dependency boundary.
        return PublicPretrainingResult(
            "blocked",
            "data",
            governance,
            reason=f"{type(exc).__name__}: {exc}",
        )

    try:
        batches = build_public_pretraining_batches(examples, config=resolved_config)
    except (PublicPretrainingError, ValueError) as exc:
        return PublicPretrainingResult(
            "blocked",
            "batches",
            governance,
            data=examples.summary(),
            reason=str(exc),
        )

    training: dict[str, object] = {"status": "ready_for_trainer"}
    transfer_contract: CheckpointTransferContract | None = None
    if trainer is not None:
        request = PublicPretrainingTrainingRequest(
            dataset_id=source_manifest.dataset_id,
            manifest_digest=digest,
            source_file_digests=_source_file_digests(source_manifest),
            examples=examples,
            batches=batches,
            config=resolved_config,
        )
        try:
            checkpoint = trainer(request)
            transfer_contract = _checkpoint_contract(checkpoint, request)
            training = {
                "status": "trained",
                "metrics": _json_friendly(checkpoint.metrics),
                "metadata": _json_friendly(checkpoint.metadata),
            }
        except Exception as exc:  # Trainer failures must remain visible in the result artifact.
            return PublicPretrainingResult(
                "blocked",
                "training",
                governance,
                data=examples.summary(),
                batches=batches.summary(),
                training={"status": "failed"},
                reason=f"{type(exc).__name__}: {exc}",
            )

    return PublicPretrainingResult(
        "ready",
        "complete",
        governance,
        data=examples.summary(),
        batches=batches.summary(),
        training=training,
        transfer_contract=transfer_contract,
    )


def _load_bounded_records(
    loaded: pd.DataFrame | Iterable[pd.DataFrame | Mapping[str, object]],
    limit: int,
) -> pd.DataFrame:
    chunks: Iterable[pd.DataFrame | Mapping[str, object]]
    chunks = (loaded,) if isinstance(loaded, pd.DataFrame) else loaded
    collected: list[pd.DataFrame] = []
    remaining = limit
    for chunk in chunks:
        frame = chunk if isinstance(chunk, pd.DataFrame) else pd.DataFrame([dict(chunk)])
        validate_canonical_interactions(frame)
        if frame.empty:
            continue
        selected = frame.iloc[:remaining].copy()
        collected.append(selected)
        remaining -= len(selected)
        if remaining == 0:
            break
    if not collected:
        raise PublicPretrainingError("The approved public record loader returned no interactions.")
    return pd.concat(collected, ignore_index=True)


def _cap_masked_predictions(
    batch: MaskedItemBatch,
    *,
    max_predictions: int,
    seed: int,
) -> MaskedItemBatch:
    if len(batch.target_items) <= max_predictions:
        return batch
    retained = _bounded_indices(len(batch.target_items), max_predictions, seed=seed)
    keep_mask = np.zeros(len(batch.target_items), dtype=bool)
    keep_mask[retained] = True
    input_sequences = batch.input_sequences.copy()
    prediction_mask = np.zeros_like(batch.prediction_mask)
    prediction_mask[batch.source_rows[retained], batch.target_positions[retained]] = True
    dropped = np.flatnonzero(~keep_mask)
    input_sequences[batch.source_rows[dropped], batch.target_positions[dropped]] = batch.target_items[dropped]
    return MaskedItemBatch(
        input_sequences=input_sequences,
        target_items=batch.target_items[retained],
        source_rows=batch.source_rows[retained],
        target_positions=batch.target_positions[retained],
        prediction_mask=prediction_mask,
        valid_mask=batch.valid_mask,
    )


def _checkpoint_contract(
    checkpoint: PublicPretrainingCheckpoint,
    request: PublicPretrainingTrainingRequest,
) -> CheckpointTransferContract:
    if not checkpoint.encoder_family.strip():
        raise PublicPretrainingError("Checkpoint encoder_family must not be empty.")
    if checkpoint.embedding_dim <= 0:
        raise PublicPretrainingError("Checkpoint embedding_dim must be positive.")
    if not checkpoint.transferable_components or not checkpoint.target_model_families:
        raise PublicPretrainingError(
            "Checkpoint transfer components and target model families must be declared."
        )
    checkpoint_path = Path(checkpoint.checkpoint_path).expanduser().resolve()
    checkpoint_digest = file_sha256(checkpoint_path)
    return CheckpointTransferContract(
        schema_version=1,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_digest,
        encoder_family=checkpoint.encoder_family,
        embedding_dim=int(checkpoint.embedding_dim),
        transferable_components=tuple(checkpoint.transferable_components),
        target_model_families=tuple(checkpoint.target_model_families),
        dataset_id=request.dataset_id,
        manifest_digest=request.manifest_digest,
        source_file_digests=request.source_file_digests,
        vocabulary_digest=request.examples.vocabulary_digest,
        vocabulary_size=len(request.examples.item_to_index),
        sequence_length=int(request.examples.sequences.shape[1]),
        reserved_item_ids=dict(request.reserved_item_ids),
    )


def _governance_decision(
    manifest: DatasetSourceManifest,
    digest: str,
    *,
    status: Literal["blocked", "approved"],
    reason: str = "",
) -> GovernanceDecision:
    files = tuple(
        {
            "path": record.path,
            "sha256": record.sha256,
            "size_bytes": int(record.size_bytes),
            "source_url": record.source_url,
            "acquired_at": record.acquired_at,
        }
        for record in manifest.files
    )
    return GovernanceDecision(
        status=status,
        dataset_id=manifest.dataset_id,
        manifest_digest=digest,
        training_use_approved=bool(manifest.training_use_approved),
        checksums_verified=status == "approved",
        source_files=files,
        reason=reason,
    )


def _source_file_digests(manifest: DatasetSourceManifest) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(Path(record.path).expanduser().resolve()), record.sha256) for record in manifest.files))


def _bounded_indices(size: int, limit: int, *, seed: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype="int64")
    selected = np.random.default_rng(seed).choice(size, size=limit, replace=False)
    return np.sort(selected.astype("int64", copy=False))


def _normalized_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _session_key(row: pd.Series) -> str | None:
    user_id = _normalized_text(row["user_id"])
    session_id = _normalized_text(row["session_id"])
    if not user_id and not session_id:
        return None
    return f"{row['source_dataset']}|user={user_id}|session={session_id}"


def _json_friendly(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_friendly(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "CheckpointTransferContract",
    "GovernanceDecision",
    "MASK_ITEM_ID",
    "OOV_ITEM_ID",
    "PADDING_ITEM_ID",
    "PublicPretrainingBatches",
    "PublicPretrainingCheckpoint",
    "PublicPretrainingConfig",
    "PublicPretrainingError",
    "PublicPretrainingExamples",
    "PublicPretrainingResult",
    "PublicPretrainingTrainingRequest",
    "build_public_pretraining_batches",
    "manifest_digest",
    "map_public_records_to_examples",
    "run_public_pretraining",
]
