from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_CONTEXT_FEATURE_PREFIX = "context_feature"
DEFAULT_ITEM_FEATURE_PREFIX = "item_feature"


@dataclass(frozen=True)
class TrackDriftEvidenceConfig:
    """Thresholds and histogram settings for reranker tensor drift evidence."""

    max_abs_standardized_mean_shift: float = 0.20
    max_js_distance: float = 0.20
    histogram_bins: int = 20
    top_feature_count: int = 10

    def validate(self) -> None:
        if (
            isinstance(self.max_abs_standardized_mean_shift, bool)
            or not isinstance(self.max_abs_standardized_mean_shift, (int, float))
            or not math.isfinite(self.max_abs_standardized_mean_shift)
            or self.max_abs_standardized_mean_shift < 0.0
        ):
            raise ValueError("max_abs_standardized_mean_shift must be a finite non-negative number")
        if (
            isinstance(self.max_js_distance, bool)
            or not isinstance(self.max_js_distance, (int, float))
            or not math.isfinite(self.max_js_distance)
            or self.max_js_distance < 0.0
        ):
            raise ValueError("max_js_distance must be a finite non-negative number")
        if (
            isinstance(self.histogram_bins, bool)
            or not isinstance(self.histogram_bins, int)
            or self.histogram_bins < 2
        ):
            raise ValueError("histogram_bins must be an integer >= 2")
        if (
            isinstance(self.top_feature_count, bool)
            or not isinstance(self.top_feature_count, int)
            or self.top_feature_count < 1
        ):
            raise ValueError("top_feature_count must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_abs_standardized_mean_shift": float(self.max_abs_standardized_mean_shift),
            "max_js_distance": float(self.max_js_distance),
            "histogram_bins": int(self.histogram_bins),
            "top_feature_count": int(self.top_feature_count),
        }


@dataclass(frozen=True)
class FeatureDriftReport:
    """Per-feature drift metrics between one reference and comparison tensor."""

    feature_group: str
    feature_name: str
    feature_index: int
    reference_rows: int
    comparison_rows: int
    reference_mean: float | None
    comparison_mean: float | None
    standardized_mean_shift: float | None
    abs_standardized_mean_shift: float | None
    js_distance: float | None
    low_drift: bool

    def drift_score(self) -> float | None:
        values = [
            value
            for value in (self.abs_standardized_mean_shift, self.js_distance)
            if value is not None
        ]
        return max(values) if values else None

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "feature_name": self.feature_name,
            "feature_index": int(self.feature_index),
            "reference_rows": int(self.reference_rows),
            "comparison_rows": int(self.comparison_rows),
            "reference_mean": _json_float(self.reference_mean),
            "comparison_mean": _json_float(self.comparison_mean),
            "standardized_mean_shift": _json_float(self.standardized_mean_shift),
            "abs_standardized_mean_shift": _json_float(self.abs_standardized_mean_shift),
            "js_distance": _json_float(self.js_distance),
            "drift_score": _json_float(self.drift_score()),
            "low_drift": bool(self.low_drift),
        }


@dataclass(frozen=True)
class FeatureGroupDriftReport:
    """Feature-level evidence for either context or item tensors."""

    feature_group: str
    reference_split: str
    comparison_split: str
    rows: tuple[FeatureDriftReport, ...]
    low_drift_mask: tuple[bool, ...]
    failing_features: tuple[str, ...]

    @property
    def status(self) -> str:
        return "pass" if not self.failing_features else "fail"

    def max_abs_standardized_mean_shift(self) -> float | None:
        values = [
            row.abs_standardized_mean_shift
            for row in self.rows
            if row.abs_standardized_mean_shift is not None
        ]
        return max(values) if values else None

    def max_js_distance(self) -> float | None:
        values = [row.js_distance for row in self.rows if row.js_distance is not None]
        return max(values) if values else None

    def max_drift_score(self) -> float | None:
        values = [row.drift_score() for row in self.rows if row.drift_score() is not None]
        return max(values) if values else None

    def top_features(self, limit: int = 10) -> tuple[FeatureDriftReport, ...]:
        ordered = sorted(
            self.rows,
            key=lambda row: (
                -1.0 if row.drift_score() is None else row.drift_score(),
                row.feature_name,
            ),
            reverse=True,
        )
        return tuple(ordered[:limit])

    def to_dict(self, *, top_feature_count: int = 10) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "reference_split": self.reference_split,
            "comparison_split": self.comparison_split,
            "status": self.status,
            "feature_count": len(self.rows),
            "low_drift_feature_count": int(sum(self.low_drift_mask)),
            "low_drift_mask": [bool(value) for value in self.low_drift_mask],
            "failing_features": list(self.failing_features),
            "max_abs_standardized_mean_shift": _json_float(
                self.max_abs_standardized_mean_shift()
            ),
            "max_js_distance": _json_float(self.max_js_distance()),
            "max_drift_score": _json_float(self.max_drift_score()),
            "top_features": [
                row.to_dict() for row in self.top_features(limit=top_feature_count)
            ],
            "features": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class TrackDriftEvidenceReport:
    """JSON-friendly drift summary for DCN reranking context and item tensors."""

    status: str
    reference_split: str
    comparison_splits: tuple[str, ...]
    config: TrackDriftEvidenceConfig
    groups: tuple[FeatureGroupDriftReport, ...]
    gate_blockers: tuple[dict[str, object], ...] = field(default_factory=tuple)

    @property
    def within_threshold(self) -> bool:
        return self.status == "pass"

    def max_drift_score(self) -> float | None:
        values = [group.max_drift_score() for group in self.groups if group.max_drift_score() is not None]
        return max(values) if values else None

    def failing_features(self) -> tuple[str, ...]:
        names: list[str] = []
        for group in self.groups:
            names.extend(
                f"{group.comparison_split}.{group.feature_group}.{name}"
                for name in group.failing_features
            )
        return tuple(names)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "within_threshold": self.within_threshold,
            "reference_split": self.reference_split,
            "comparison_splits": list(self.comparison_splits),
            "config": self.config.to_dict(),
            "max_drift_score": _json_float(self.max_drift_score()),
            "failing_features": list(self.failing_features()),
            "gate_blockers": list(self.gate_blockers),
            "groups": [
                group.to_dict(top_feature_count=self.config.top_feature_count)
                for group in self.groups
            ],
        }


def standardized_mean_shift(
    reference_values: Sequence[float] | np.ndarray,
    comparison_values: Sequence[float] | np.ndarray,
) -> float | None:
    """Return signed pooled-standard-deviation mean shift for one feature."""

    reference = _finite_vector(reference_values)
    comparison = _finite_vector(comparison_values)
    if reference.size == 0 or comparison.size == 0:
        return None
    reference_mean = float(np.mean(reference))
    comparison_mean = float(np.mean(comparison))
    pooled_scale = _pooled_standard_deviation(reference, comparison)
    if pooled_scale <= 0.0:
        pooled_scale = 1.0
    return float((comparison_mean - reference_mean) / pooled_scale)


def jensen_shannon_distance(
    reference_values: Sequence[float] | np.ndarray,
    comparison_values: Sequence[float] | np.ndarray,
    *,
    bins: int = 20,
) -> float | None:
    """Return numeric Jensen-Shannon distribution distance in base 2."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    reference = _finite_vector(reference_values)
    comparison = _finite_vector(comparison_values)
    if reference.size == 0 or comparison.size == 0:
        return None
    lower = float(min(np.min(reference), np.min(comparison)))
    upper = float(max(np.max(reference), np.max(comparison)))
    if math.isclose(lower, upper):
        return 0.0
    edges = np.linspace(lower, upper, bins + 1, dtype="float64")
    reference_counts = np.histogram(reference, bins=edges)[0].astype("float64")
    comparison_counts = np.histogram(comparison, bins=edges)[0].astype("float64")
    return _js_distance_from_counts(reference_counts, comparison_counts)


def select_low_drift_feature_mask(
    rows: Sequence[FeatureDriftReport | Mapping[str, object]],
    *,
    max_abs_standardized_mean_shift: float = 0.20,
    max_js_distance: float = 0.20,
) -> np.ndarray:
    """Return a boolean feature mask where both supported drift metrics pass."""

    if max_abs_standardized_mean_shift < 0.0 or max_js_distance < 0.0:
        raise ValueError("low-drift thresholds must be non-negative")
    mask: list[bool] = []
    for row in rows:
        if isinstance(row, FeatureDriftReport):
            abs_shift = row.abs_standardized_mean_shift
            js_distance = row.js_distance
        else:
            abs_shift = _optional_float(row.get("abs_standardized_mean_shift"))
            js_distance = _optional_float(row.get("js_distance"))
        mask.append(
            abs_shift is not None
            and js_distance is not None
            and abs_shift <= max_abs_standardized_mean_shift
            and js_distance <= max_js_distance
        )
    return np.asarray(mask, dtype=bool)


def compute_feature_drift_report(
    *,
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    feature_names: Sequence[str],
    feature_group: str,
    reference_split: str,
    comparison_split: str,
    config: TrackDriftEvidenceConfig | None = None,
) -> FeatureGroupDriftReport:
    """Build per-feature drift rows for one tensor group."""

    resolved_config = config or TrackDriftEvidenceConfig()
    resolved_config.validate()
    reference = _as_feature_matrix(reference_values, name=f"{reference_split}.{feature_group}")
    comparison = _as_feature_matrix(comparison_values, name=f"{comparison_split}.{feature_group}")
    if reference.shape[1] != comparison.shape[1]:
        raise ValueError(
            f"{feature_group} feature width mismatch: reference has {reference.shape[1]} "
            f"columns, comparison has {comparison.shape[1]}"
        )
    names = _resolve_feature_names(feature_names, reference.shape[1], prefix=feature_group)
    rows = tuple(
        _feature_row(
            reference_column=reference[:, idx],
            comparison_column=comparison[:, idx],
            feature_group=feature_group,
            feature_name=names[idx],
            feature_index=idx,
            config=resolved_config,
        )
        for idx in range(reference.shape[1])
    )
    mask = select_low_drift_feature_mask(
        rows,
        max_abs_standardized_mean_shift=resolved_config.max_abs_standardized_mean_shift,
        max_js_distance=resolved_config.max_js_distance,
    )
    failing = tuple(row.feature_name for row, low_drift in zip(rows, mask) if not low_drift)
    return FeatureGroupDriftReport(
        feature_group=feature_group,
        reference_split=reference_split,
        comparison_split=comparison_split,
        rows=rows,
        low_drift_mask=tuple(bool(value) for value in mask.tolist()),
        failing_features=failing,
    )


def build_track_drift_evidence(
    *,
    reference_split: object,
    comparison_splits: Mapping[str, object],
    reference_name: str = "train",
    context_feature_names: Sequence[str] | None = None,
    item_feature_names: Sequence[str] | None = None,
    config: TrackDriftEvidenceConfig | None = None,
) -> TrackDriftEvidenceReport:
    """Build grouped context/item drift evidence for DCN candidate splits."""

    if not comparison_splits:
        raise ValueError("comparison_splits must contain at least one split")
    resolved_config = config or TrackDriftEvidenceConfig()
    resolved_config.validate()
    reference_context = _split_matrix(reference_split, "context_features")
    reference_items = _split_matrix(reference_split, "item_features")
    context_names = _resolve_feature_names(
        context_feature_names,
        reference_context.shape[1],
        prefix=DEFAULT_CONTEXT_FEATURE_PREFIX,
    )
    item_names = _resolve_feature_names(
        item_feature_names,
        reference_items.shape[1],
        prefix=DEFAULT_ITEM_FEATURE_PREFIX,
    )

    groups: list[FeatureGroupDriftReport] = []
    blockers: list[dict[str, object]] = []
    for comparison_name, comparison_split in comparison_splits.items():
        comparison_context = _split_matrix(comparison_split, "context_features")
        comparison_items = _split_matrix(comparison_split, "item_features")
        for group in (
            compute_feature_drift_report(
                reference_values=reference_context,
                comparison_values=comparison_context,
                feature_names=context_names,
                feature_group="context",
                reference_split=reference_name,
                comparison_split=str(comparison_name),
                config=resolved_config,
            ),
            compute_feature_drift_report(
                reference_values=reference_items,
                comparison_values=comparison_items,
                feature_names=item_names,
                feature_group="item",
                reference_split=reference_name,
                comparison_split=str(comparison_name),
                config=resolved_config,
            ),
        ):
            groups.append(group)
            blockers.extend(_gate_blockers_for_group(group, config=resolved_config))
    status = "pass" if not blockers else "fail"
    return TrackDriftEvidenceReport(
        status=status,
        reference_split=reference_name,
        comparison_splits=tuple(str(name) for name in comparison_splits),
        config=resolved_config,
        groups=tuple(groups),
        gate_blockers=tuple(blockers),
    )


def save_track_drift_evidence(
    report: TrackDriftEvidenceReport,
    output_path: str | Path,
) -> Path:
    """Write a drift evidence report as deterministic pretty JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _feature_row(
    *,
    reference_column: np.ndarray,
    comparison_column: np.ndarray,
    feature_group: str,
    feature_name: str,
    feature_index: int,
    config: TrackDriftEvidenceConfig,
) -> FeatureDriftReport:
    reference = _finite_vector(reference_column)
    comparison = _finite_vector(comparison_column)
    shift = standardized_mean_shift(reference, comparison)
    distance = jensen_shannon_distance(reference, comparison, bins=config.histogram_bins)
    abs_shift = None if shift is None else abs(shift)
    low_drift = (
        abs_shift is not None
        and distance is not None
        and abs_shift <= config.max_abs_standardized_mean_shift
        and distance <= config.max_js_distance
    )
    return FeatureDriftReport(
        feature_group=feature_group,
        feature_name=feature_name,
        feature_index=feature_index,
        reference_rows=int(reference.size),
        comparison_rows=int(comparison.size),
        reference_mean=_mean_or_none(reference),
        comparison_mean=_mean_or_none(comparison),
        standardized_mean_shift=shift,
        abs_standardized_mean_shift=abs_shift,
        js_distance=distance,
        low_drift=low_drift,
    )


def _gate_blockers_for_group(
    group: FeatureGroupDriftReport,
    *,
    config: TrackDriftEvidenceConfig,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for row in group.rows:
        reasons: list[str] = []
        if row.abs_standardized_mean_shift is None or row.js_distance is None:
            reasons.append("missing_supported_metric")
        elif row.abs_standardized_mean_shift > config.max_abs_standardized_mean_shift:
            reasons.append("standardized_mean_shift")
        if row.js_distance is not None and row.js_distance > config.max_js_distance:
            reasons.append("js_distance")
        if reasons:
            blockers.append(
                {
                    "comparison_split": group.comparison_split,
                    "feature_group": group.feature_group,
                    "feature_name": row.feature_name,
                    "feature_index": int(row.feature_index),
                    "reasons": reasons,
                    "abs_standardized_mean_shift": _json_float(
                        row.abs_standardized_mean_shift
                    ),
                    "js_distance": _json_float(row.js_distance),
                    "drift_score": _json_float(row.drift_score()),
                    "thresholds": {
                        "max_abs_standardized_mean_shift": float(
                            config.max_abs_standardized_mean_shift
                        ),
                        "max_js_distance": float(config.max_js_distance),
                    },
                }
            )
    return blockers


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a rank-2 array with at least one column")
    return matrix


def _finite_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype="float64").reshape(-1)
    return vector[np.isfinite(vector)]


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _js_distance_from_counts(reference_counts: np.ndarray, comparison_counts: np.ndarray) -> float:
    reference_total = float(np.sum(reference_counts))
    comparison_total = float(np.sum(comparison_counts))
    if reference_total <= 0.0 or comparison_total <= 0.0:
        return 0.0
    reference = reference_counts / reference_total
    comparison = comparison_counts / comparison_total
    midpoint = 0.5 * (reference + comparison)
    divergence = 0.5 * _kl_divergence_base2(reference, midpoint)
    divergence += 0.5 * _kl_divergence_base2(comparison, midpoint)
    return float(math.sqrt(max(0.0, divergence)))


def _kl_divergence_base2(source: np.ndarray, target: np.ndarray) -> float:
    mask = source > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.sum(source[mask] * np.log2(source[mask] / target[mask])))


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pooled_standard_deviation(reference: np.ndarray, comparison: np.ndarray) -> float:
    reference_var = float(np.var(reference)) if reference.size else 0.0
    comparison_var = float(np.var(comparison)) if comparison.size else 0.0
    return math.sqrt(max(0.0, 0.5 * (reference_var + comparison_var)))


def _resolve_feature_names(
    names: Sequence[str] | None,
    width: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}_{idx}" for idx in range(width))
    resolved = tuple(str(name) for name in names)
    if len(resolved) != width:
        raise ValueError(f"{prefix} feature names has {len(resolved)} entries; expected {width}")
    return resolved


def _split_matrix(split: object, attribute: str) -> np.ndarray:
    if not hasattr(split, attribute):
        raise ValueError(f"split is missing required attribute {attribute}")
    return _as_feature_matrix(getattr(split, attribute), name=attribute)


__all__ = [
    "FeatureDriftReport",
    "FeatureGroupDriftReport",
    "TrackDriftEvidenceConfig",
    "TrackDriftEvidenceReport",
    "build_track_drift_evidence",
    "compute_feature_drift_report",
    "jensen_shannon_distance",
    "save_track_drift_evidence",
    "select_low_drift_feature_mask",
    "standardized_mean_shift",
]
