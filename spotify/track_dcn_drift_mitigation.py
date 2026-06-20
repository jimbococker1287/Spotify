from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .track_dcn_training import DCNCandidateSplit, DCNTemporalDataset


DEFAULT_CONTEXT_FEATURE_PREFIX = "context_feature"
DEFAULT_ITEM_FEATURE_PREFIX = "item_feature"
MITIGATION_METHOD = "low_drift_feature_mask"
PROMOTION_DECISION = "not_evaluated"


@dataclass(frozen=True)
class TrackDCNDriftMitigationConfig:
    """Controls how grouped drift evidence is collapsed into DCN feature masks."""

    minimum_context_features: int = 1
    minimum_item_features: int = 1
    context_feature_prefix: str = DEFAULT_CONTEXT_FEATURE_PREFIX
    item_feature_prefix: str = DEFAULT_ITEM_FEATURE_PREFIX
    comparison_policy: str = "all_comparison_splits"

    def validate(self) -> None:
        if (
            isinstance(self.minimum_context_features, bool)
            or not isinstance(self.minimum_context_features, int)
            or self.minimum_context_features < 1
        ):
            raise ValueError("minimum_context_features must be a positive integer")
        if (
            isinstance(self.minimum_item_features, bool)
            or not isinstance(self.minimum_item_features, int)
            or self.minimum_item_features < 1
        ):
            raise ValueError("minimum_item_features must be a positive integer")
        if not self.context_feature_prefix:
            raise ValueError("context_feature_prefix must be non-empty")
        if not self.item_feature_prefix:
            raise ValueError("item_feature_prefix must be non-empty")
        if self.comparison_policy != "all_comparison_splits":
            raise ValueError("comparison_policy must be 'all_comparison_splits'")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_context_features": int(self.minimum_context_features),
            "minimum_item_features": int(self.minimum_item_features),
            "context_feature_prefix": self.context_feature_prefix,
            "item_feature_prefix": self.item_feature_prefix,
            "comparison_policy": self.comparison_policy,
        }


@dataclass(frozen=True)
class TrackDCNFeatureDrop:
    """One feature removed by the mitigation mask and the evidence that caused it."""

    feature_group: str
    feature_name: str
    feature_index: int
    failed_comparison_splits: tuple[str, ...]
    source_rows: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "feature_name": self.feature_name,
            "feature_index": int(self.feature_index),
            "failed_comparison_splits": list(self.failed_comparison_splits),
            "source_rows": [_json_ready(row) for row in self.source_rows],
        }


@dataclass(frozen=True)
class TrackDCNFeatureGroupMask:
    """A low-drift mask for either the context or item DCN tensor."""

    feature_group: str
    feature_names: tuple[str, ...]
    low_drift_mask: tuple[bool, ...]
    retained_feature_indices: tuple[int, ...]
    retained_features: tuple[str, ...]
    dropped_features: tuple[TrackDCNFeatureDrop, ...]
    source_comparison_splits: tuple[str, ...]

    @property
    def original_feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def retained_feature_count(self) -> int:
        return len(self.retained_feature_indices)

    @property
    def dropped_feature_count(self) -> int:
        return len(self.dropped_features)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "original_feature_count": self.original_feature_count,
            "retained_feature_count": self.retained_feature_count,
            "dropped_feature_count": self.dropped_feature_count,
            "feature_names": list(self.feature_names),
            "low_drift_mask": [bool(value) for value in self.low_drift_mask],
            "retained_feature_indices": [int(index) for index in self.retained_feature_indices],
            "retained_features": list(self.retained_features),
            "dropped_features": [drop.to_dict() for drop in self.dropped_features],
            "source_comparison_splits": list(self.source_comparison_splits),
        }


@dataclass(frozen=True)
class TrackDCNDriftMitigationPlan:
    """JSON-friendly artifact describing masks to try in a later DCN runner pass."""

    status: str
    method: str
    promotion_decision: str
    source_report_status: str | None
    reference_split: str | None
    comparison_splits: tuple[str, ...]
    source_drift_config: Mapping[str, object]
    config: TrackDCNDriftMitigationConfig
    context: TrackDCNFeatureGroupMask
    item: TrackDCNFeatureGroupMask
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dropped_feature_count(self) -> int:
        return self.context.dropped_feature_count + self.item.dropped_feature_count

    def to_dict(self) -> dict[str, object]:
        dropped = [
            f"{drop.feature_group}.{drop.feature_name}"
            for drop in (*self.context.dropped_features, *self.item.dropped_features)
        ]
        return {
            "status": self.status,
            "method": self.method,
            "promotion_decision": self.promotion_decision,
            "source_report_status": self.source_report_status,
            "reference_split": self.reference_split,
            "comparison_splits": list(self.comparison_splits),
            "source_drift_config": _json_ready(dict(self.source_drift_config)),
            "config": self.config.to_dict(),
            "dropped_feature_count": self.dropped_feature_count,
            "dropped_features": dropped,
            "context": self.context.to_dict(),
            "item": self.item.to_dict(),
            "notes": list(self.notes),
        }


def build_track_dcn_drift_mitigation_plan(
    report: object,
    config: TrackDCNDriftMitigationConfig | None = None,
) -> TrackDCNDriftMitigationPlan:
    """Build context/item low-drift masks from drift evidence.

    The mask keeps a feature only when every observed comparison split marks
    that feature as low drift. The returned plan is an artifact for a later
    train/evaluate integration; it does not assert that model promotion passes.
    """

    resolved_config = config or TrackDCNDriftMitigationConfig()
    resolved_config.validate()
    payload = _report_payload(report)
    groups = _report_groups(payload)
    drift_config = _mapping(payload.get("config"))

    context = _build_group_mask(
        groups,
        feature_group="context",
        feature_prefix=resolved_config.context_feature_prefix,
        minimum_features=resolved_config.minimum_context_features,
        drift_config=drift_config,
    )
    item = _build_group_mask(
        groups,
        feature_group="item",
        feature_prefix=resolved_config.item_feature_prefix,
        minimum_features=resolved_config.minimum_item_features,
        drift_config=drift_config,
    )
    status = "noop" if context.dropped_feature_count == 0 and item.dropped_feature_count == 0 else "ready"
    return TrackDCNDriftMitigationPlan(
        status=status,
        method=MITIGATION_METHOD,
        promotion_decision=PROMOTION_DECISION,
        source_report_status=_optional_str(payload.get("status")),
        reference_split=_optional_str(payload.get("reference_split")),
        comparison_splits=_string_tuple(payload.get("comparison_splits")),
        source_drift_config=drift_config,
        config=resolved_config,
        context=context,
        item=item,
        notes=(
            "Feature masks are a mitigation plan only; retrain and re-evaluate the DCN reranker before promotion.",
        ),
    )


def apply_track_dcn_drift_mitigation(
    dataset: object,
    mitigation: object,
    config: TrackDCNDriftMitigationConfig | None = None,
) -> DCNTemporalDataset:
    """Return a DCNTemporalDataset with context/item tensors masked by a plan."""

    plan = (
        mitigation
        if isinstance(mitigation, TrackDCNDriftMitigationPlan)
        else build_track_dcn_drift_mitigation_plan(mitigation, config=config)
    )
    context_mask = np.asarray(plan.context.low_drift_mask, dtype=bool)
    item_mask = np.asarray(plan.item.low_drift_mask, dtype=bool)
    masked = DCNTemporalDataset(
        train=_apply_split_masks(
            _dataset_split(dataset, "train"),
            context_mask=context_mask,
            item_mask=item_mask,
            split_name="train",
        ),
        validation=_apply_split_masks(
            _dataset_split(dataset, "validation"),
            context_mask=context_mask,
            item_mask=item_mask,
            split_name="validation",
        ),
        test=_apply_split_masks(
            _dataset_split(dataset, "test"),
            context_mask=context_mask,
            item_mask=item_mask,
            split_name="test",
        ),
    )
    masked.validate()
    return masked


def save_track_dcn_drift_mitigation_plan(
    plan: TrackDCNDriftMitigationPlan,
    output_path: str | Path,
) -> Path:
    """Write a deterministic JSON mitigation-plan artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_group_mask(
    groups: Sequence[Mapping[str, object]],
    *,
    feature_group: str,
    feature_prefix: str,
    minimum_features: int,
    drift_config: Mapping[str, object],
) -> TrackDCNFeatureGroupMask:
    matching = [group for group in groups if _optional_str(group.get("feature_group")) == feature_group]
    if not matching:
        raise ValueError(f"drift report does not contain a {feature_group} feature group")

    first_mask = _group_low_drift_mask(matching[0], feature_group=feature_group)
    feature_names = _group_feature_names(matching[0], width=len(first_mask), feature_prefix=feature_prefix)
    combined = np.ones(len(first_mask), dtype=bool)
    source_comparison_splits: list[str] = []
    masks_by_split: list[tuple[str, np.ndarray, Mapping[int, Mapping[str, object]]]] = []
    for group in matching:
        comparison_split = _optional_str(group.get("comparison_split")) or "unknown"
        mask = _group_low_drift_mask(group, feature_group=feature_group)
        if len(mask) != len(feature_names):
            raise ValueError(
                f"{comparison_split}.{feature_group} low_drift_mask has {len(mask)} entries; "
                f"expected {len(feature_names)}"
            )
        group_names = _group_feature_names(group, width=len(mask), feature_prefix=feature_prefix)
        if group_names != feature_names:
            raise ValueError(f"{comparison_split}.{feature_group} feature names do not match the reference group")
        if comparison_split not in source_comparison_splits:
            source_comparison_splits.append(comparison_split)
        rows_by_index = _feature_rows_by_index(group, width=len(mask))
        masks_by_split.append((comparison_split, mask, rows_by_index))
        combined &= mask

    retained_indices = tuple(int(index) for index in np.flatnonzero(combined))
    if len(retained_indices) < minimum_features:
        raise ValueError(
            f"{feature_group} low-drift mask retains {len(retained_indices)} features; "
            f"expected at least {minimum_features}"
        )
    retained_features = tuple(feature_names[index] for index in retained_indices)
    dropped: list[TrackDCNFeatureDrop] = []
    for index in np.flatnonzero(~combined).tolist():
        failed_splits: list[str] = []
        source_rows: list[dict[str, object]] = []
        for comparison_split, mask, rows_by_index in masks_by_split:
            if bool(mask[index]):
                continue
            failed_splits.append(comparison_split)
            row = rows_by_index.get(index, {})
            source_rows.append(
                _feature_source_row(
                    row,
                    comparison_split=comparison_split,
                    feature_group=feature_group,
                    feature_name=feature_names[index],
                    feature_index=index,
                    drift_config=drift_config,
                )
            )
        dropped.append(
            TrackDCNFeatureDrop(
                feature_group=feature_group,
                feature_name=feature_names[index],
                feature_index=index,
                failed_comparison_splits=tuple(failed_splits),
                source_rows=tuple(source_rows),
            )
        )

    return TrackDCNFeatureGroupMask(
        feature_group=feature_group,
        feature_names=feature_names,
        low_drift_mask=tuple(bool(value) for value in combined.tolist()),
        retained_feature_indices=retained_indices,
        retained_features=retained_features,
        dropped_features=tuple(dropped),
        source_comparison_splits=tuple(source_comparison_splits),
    )


def _apply_split_masks(
    split: object,
    *,
    context_mask: np.ndarray,
    item_mask: np.ndarray,
    split_name: str,
) -> DCNCandidateSplit:
    context = _feature_matrix(split, "context_features", split_name=split_name)
    item = _feature_matrix(split, "item_features", split_name=split_name)
    if context.shape[1] != len(context_mask):
        raise ValueError(
            f"{split_name}.context_features has {context.shape[1]} columns; "
            f"expected {len(context_mask)} from drift mitigation plan"
        )
    if item.shape[1] != len(item_mask):
        raise ValueError(
            f"{split_name}.item_features has {item.shape[1]} columns; "
            f"expected {len(item_mask)} from drift mitigation plan"
        )
    return DCNCandidateSplit(
        context_features=context[:, context_mask].copy(),
        item_features=item[:, item_mask].copy(),
        labels=_required_vector(split, "labels", split_name=split_name),
        query_ids=_required_vector(split, "query_ids", split_name=split_name),
        candidate_ids=_optional_vector(split, "candidate_ids"),
        event_times=_optional_vector(split, "event_times"),
        sample_weights=_optional_vector(split, "sample_weights"),
    )


def _report_payload(report: object) -> Mapping[str, object]:
    if isinstance(report, Mapping):
        return report
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("report must be a mapping or expose to_dict() returning a mapping")


def _report_groups(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    groups = payload.get("groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ValueError("drift report must contain a groups sequence")
    resolved: list[Mapping[str, object]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"drift report group {index} must be a mapping")
        resolved.append(group)
    return tuple(resolved)


def _group_low_drift_mask(group: Mapping[str, object], *, feature_group: str) -> np.ndarray:
    raw = group.get("low_drift_mask")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{feature_group} group is missing low_drift_mask")
    values = [_bool_value(value, name=f"{feature_group}.low_drift_mask") for value in raw]
    if not values:
        raise ValueError(f"{feature_group} low_drift_mask must contain at least one feature")
    return np.asarray(values, dtype=bool)


def _group_feature_names(
    group: Mapping[str, object],
    *,
    width: int,
    feature_prefix: str,
) -> tuple[str, ...]:
    raw_names = group.get("feature_names")
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        names = tuple(str(name) for name in raw_names)
        if len(names) == width:
            return names

    features = group.get("features")
    names_by_index: list[str | None] = [None] * width
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        for position, row in enumerate(features):
            if not isinstance(row, Mapping):
                continue
            feature_index = _optional_int(row.get("feature_index"))
            if feature_index is None:
                feature_index = position
            if 0 <= feature_index < width and names_by_index[feature_index] is None:
                names_by_index[feature_index] = str(row.get("feature_name") or f"{feature_prefix}_{feature_index}")
    return tuple(name if name is not None else f"{feature_prefix}_{index}" for index, name in enumerate(names_by_index))


def _feature_rows_by_index(
    group: Mapping[str, object],
    *,
    width: int,
) -> dict[int, Mapping[str, object]]:
    features = group.get("features")
    rows: dict[int, Mapping[str, object]] = {}
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return rows
    for position, row in enumerate(features):
        if not isinstance(row, Mapping):
            continue
        feature_index = _optional_int(row.get("feature_index"))
        if feature_index is None:
            feature_index = position
        if 0 <= feature_index < width:
            rows[feature_index] = row
    return rows


def _feature_source_row(
    row: Mapping[str, object],
    *,
    comparison_split: str,
    feature_group: str,
    feature_name: str,
    feature_index: int,
    drift_config: Mapping[str, object],
) -> dict[str, object]:
    source: dict[str, object] = {
        "comparison_split": comparison_split,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "feature_index": int(feature_index),
        "low_drift": False,
    }
    for key in (
        "abs_standardized_mean_shift",
        "standardized_mean_shift",
        "js_distance",
        "drift_score",
        "reference_mean",
        "comparison_mean",
    ):
        if key in row:
            source[key] = _json_ready(row[key])
    if drift_config:
        source["thresholds"] = {
            key: _json_ready(drift_config[key])
            for key in ("max_abs_standardized_mean_shift", "max_js_distance")
            if key in drift_config
        }
    return source


def _dataset_split(dataset: object, split_name: str) -> object:
    if not hasattr(dataset, split_name):
        raise ValueError(f"dataset is missing required split {split_name}")
    return getattr(dataset, split_name)


def _feature_matrix(split: object, attribute: str, *, split_name: str) -> np.ndarray:
    if not hasattr(split, attribute):
        raise ValueError(f"{split_name} is missing required attribute {attribute}")
    matrix = np.asarray(getattr(split, attribute))
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{split_name}.{attribute} must be rank-2 with at least one column")
    return matrix


def _required_vector(split: object, attribute: str, *, split_name: str) -> np.ndarray:
    if not hasattr(split, attribute):
        raise ValueError(f"{split_name} is missing required attribute {attribute}")
    return np.asarray(getattr(split, attribute)).reshape(-1).copy()


def _optional_vector(split: object, attribute: str) -> np.ndarray | None:
    values = getattr(split, attribute, None)
    if values is None:
        return None
    return np.asarray(values).reshape(-1).copy()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return ()


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bool_value(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} entries must be booleans")


def _json_ready(value: Any) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "MITIGATION_METHOD",
    "PROMOTION_DECISION",
    "TrackDCNDriftMitigationConfig",
    "TrackDCNDriftMitigationPlan",
    "TrackDCNFeatureDrop",
    "TrackDCNFeatureGroupMask",
    "apply_track_dcn_drift_mitigation",
    "build_track_dcn_drift_mitigation_plan",
    "save_track_dcn_drift_mitigation_plan",
]
