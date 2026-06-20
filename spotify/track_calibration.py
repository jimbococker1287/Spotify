from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np

ScoreInputType = Literal["auto", "probability", "logit"]

_EPS = 1e-12
_VALID_INPUT_TYPES = {"auto", "probability", "logit"}


@dataclass(frozen=True)
class CalibrationConfig:
    input_type: ScoreInputType = "auto"
    n_bins: int = 15
    positive_label: int = 1
    min_temperature: float = 0.05
    max_temperature: float = 10.0
    temperature_candidates: int = 81

    def normalized(self) -> CalibrationConfig:
        return CalibrationConfig(
            input_type=_validate_input_type(self.input_type),
            n_bins=max(1, int(self.n_bins)),
            positive_label=max(0, int(self.positive_label)),
            min_temperature=max(_EPS, float(self.min_temperature)),
            max_temperature=max(max(_EPS, float(self.min_temperature)), float(self.max_temperature)),
            temperature_candidates=max(3, int(self.temperature_candidates)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float
    input_type: ScoreInputType
    positive_label: int
    n_bins: int
    validation_count: int
    validation_log_loss_before: float
    validation_log_loss_after: float
    validation_ece_before: float
    validation_ece_after: float
    status: str = "complete"
    reason: str | None = None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return scores_to_probabilities(
            scores,
            input_type=self.input_type,
            temperature=self.temperature,
            positive_label=self.positive_label,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return _json_ready(payload)


@dataclass(frozen=True)
class CalibrationSummary:
    method: str
    status: str
    temperature: float
    input_type: ScoreInputType
    positive_label: int
    n_bins: int
    validation_count: int
    test_count: int
    validation_log_loss_before: float
    validation_log_loss_after: float
    validation_ece_before: float
    validation_ece_after: float
    test_log_loss_before: float
    test_log_loss_after: float
    test_ece_before: float
    test_ece_after: float
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return _json_ready(asdict(self))


def expected_calibration_error(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    input_type: ScoreInputType = "probability",
    n_bins: int = 15,
    positive_label: int = 1,
    sample_weight: np.ndarray | None = None,
) -> float:
    probabilities, labels, weights = _valid_binary_inputs(
        scores,
        y_true,
        input_type=input_type,
        positive_label=positive_label,
        sample_weight=sample_weight,
    )
    if labels.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, max(1, int(n_bins)) + 1, dtype="float64")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan")

    ece = 0.0
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == len(edges) - 2:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not np.any(mask):
            continue
        bin_weight = weights[mask]
        bin_total = float(np.sum(bin_weight))
        if bin_total <= 0.0:
            continue
        mean_probability = float(np.average(probabilities[mask], weights=bin_weight))
        observed_rate = float(np.average(labels[mask], weights=bin_weight))
        ece += (bin_total / total_weight) * abs(mean_probability - observed_rate)
    return float(ece)


def binary_log_loss(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    input_type: ScoreInputType = "probability",
    temperature: float = 1.0,
    positive_label: int = 1,
    sample_weight: np.ndarray | None = None,
) -> float:
    probabilities, labels, weights = _valid_binary_inputs(
        scores,
        y_true,
        input_type=input_type,
        temperature=temperature,
        positive_label=positive_label,
        sample_weight=sample_weight,
    )
    if labels.size == 0:
        return float("nan")
    clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.average(losses, weights=weights))


def apply_temperature_scaling(
    scores: np.ndarray,
    temperature: float,
    *,
    input_type: ScoreInputType = "auto",
) -> np.ndarray:
    score_arr = np.asarray(scores, dtype="float64")
    if score_arr.size == 0:
        return score_arr.astype("float32", copy=False)

    resolved_type = _resolve_input_type(score_arr, input_type)
    temp = _valid_temperature(temperature)
    if score_arr.ndim <= 1 or (score_arr.ndim == 2 and score_arr.shape[1] == 1):
        logits = _binary_logits(score_arr, resolved_type)
        return _sigmoid(logits / temp).astype("float32", copy=False)

    logits = _multiclass_logits(score_arr, resolved_type)
    logits = logits / temp
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    denominators = np.sum(exp_logits, axis=1, keepdims=True)
    denominators[denominators <= 0.0] = 1.0
    return (exp_logits / denominators).astype("float32", copy=False)


def scores_to_probabilities(
    scores: np.ndarray,
    *,
    input_type: ScoreInputType = "auto",
    temperature: float = 1.0,
    positive_label: int = 1,
) -> np.ndarray:
    probabilities = apply_temperature_scaling(scores, temperature, input_type=input_type)
    proba_arr = np.asarray(probabilities, dtype="float64")
    if proba_arr.ndim <= 1:
        return proba_arr.reshape(-1).astype("float32", copy=False)
    if proba_arr.ndim == 2 and proba_arr.shape[1] == 1:
        return proba_arr[:, 0].astype("float32", copy=False)
    column = min(max(0, int(positive_label)), int(proba_arr.shape[1]) - 1)
    return proba_arr[:, column].astype("float32", copy=False)


def fit_temperature_scaler(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: CalibrationConfig | None = None,
    sample_weight: np.ndarray | None = None,
) -> TemperatureScaler:
    cfg = (config or CalibrationConfig()).normalized()
    score_arr = np.asarray(validation_scores, dtype="float64")
    resolved_type = _resolve_input_type(score_arr, cfg.input_type)
    _, labels, _ = _valid_binary_inputs(
        score_arr,
        validation_labels,
        input_type=resolved_type,
        positive_label=cfg.positive_label,
        sample_weight=sample_weight,
    )
    if labels.size == 0:
        return TemperatureScaler(
            temperature=1.0,
            input_type=resolved_type,
            positive_label=cfg.positive_label,
            n_bins=cfg.n_bins,
            validation_count=0,
            validation_log_loss_before=float("nan"),
            validation_log_loss_after=float("nan"),
            validation_ece_before=float("nan"),
            validation_ece_after=float("nan"),
            status="unavailable",
            reason="no valid binary validation rows",
        )

    before_loss = binary_log_loss(
        score_arr,
        validation_labels,
        input_type=resolved_type,
        positive_label=cfg.positive_label,
        sample_weight=sample_weight,
    )
    before_ece = expected_calibration_error(
        score_arr,
        validation_labels,
        input_type=resolved_type,
        n_bins=cfg.n_bins,
        positive_label=cfg.positive_label,
        sample_weight=sample_weight,
    )

    best_temperature = 1.0
    best_loss = before_loss
    for candidate in _temperature_grid(cfg):
        loss = binary_log_loss(
            score_arr,
            validation_labels,
            input_type=resolved_type,
            temperature=float(candidate),
            positive_label=cfg.positive_label,
            sample_weight=sample_weight,
        )
        if math.isfinite(loss) and (not math.isfinite(best_loss) or loss < best_loss - 1e-15):
            best_loss = float(loss)
            best_temperature = float(candidate)

    after_loss = binary_log_loss(
        score_arr,
        validation_labels,
        input_type=resolved_type,
        temperature=best_temperature,
        positive_label=cfg.positive_label,
        sample_weight=sample_weight,
    )
    calibrated_probabilities = scores_to_probabilities(
        score_arr,
        input_type=resolved_type,
        temperature=best_temperature,
        positive_label=cfg.positive_label,
    )
    after_ece = expected_calibration_error(
        calibrated_probabilities,
        validation_labels,
        input_type="probability",
        n_bins=cfg.n_bins,
        positive_label=cfg.positive_label,
        sample_weight=sample_weight,
    )

    return TemperatureScaler(
        temperature=float(best_temperature),
        input_type=resolved_type,
        positive_label=cfg.positive_label,
        n_bins=cfg.n_bins,
        validation_count=int(labels.size),
        validation_log_loss_before=float(before_loss),
        validation_log_loss_after=float(after_loss),
        validation_ece_before=float(before_ece),
        validation_ece_after=float(after_ece),
    )


def calibrate_validation_test(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    config: CalibrationConfig | None = None,
    validation_weight: np.ndarray | None = None,
    test_weight: np.ndarray | None = None,
) -> CalibrationSummary:
    cfg = (config or CalibrationConfig()).normalized()
    scaler = fit_temperature_scaler(
        validation_scores,
        validation_labels,
        config=cfg,
        sample_weight=validation_weight,
    )
    test_probabilities, test_y, _ = _valid_binary_inputs(
        test_scores,
        test_labels,
        input_type=scaler.input_type,
        positive_label=scaler.positive_label,
        sample_weight=test_weight,
    )
    del test_probabilities
    test_loss_before = binary_log_loss(
        test_scores,
        test_labels,
        input_type=scaler.input_type,
        positive_label=scaler.positive_label,
        sample_weight=test_weight,
    )
    test_loss_after = binary_log_loss(
        test_scores,
        test_labels,
        input_type=scaler.input_type,
        temperature=scaler.temperature,
        positive_label=scaler.positive_label,
        sample_weight=test_weight,
    )
    test_ece_before = expected_calibration_error(
        test_scores,
        test_labels,
        input_type=scaler.input_type,
        n_bins=scaler.n_bins,
        positive_label=scaler.positive_label,
        sample_weight=test_weight,
    )
    test_ece_after = expected_calibration_error(
        scaler.transform(test_scores),
        test_labels,
        input_type="probability",
        n_bins=scaler.n_bins,
        positive_label=scaler.positive_label,
        sample_weight=test_weight,
    )
    return CalibrationSummary(
        method="temperature_scaling",
        status=scaler.status,
        temperature=float(scaler.temperature),
        input_type=scaler.input_type,
        positive_label=scaler.positive_label,
        n_bins=scaler.n_bins,
        validation_count=scaler.validation_count,
        test_count=int(test_y.size),
        validation_log_loss_before=scaler.validation_log_loss_before,
        validation_log_loss_after=scaler.validation_log_loss_after,
        validation_ece_before=scaler.validation_ece_before,
        validation_ece_after=scaler.validation_ece_after,
        test_log_loss_before=float(test_loss_before),
        test_log_loss_after=float(test_loss_after),
        test_ece_before=float(test_ece_before),
        test_ece_after=float(test_ece_after),
        reason=scaler.reason,
    )


def _validate_input_type(input_type: str) -> ScoreInputType:
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {sorted(_VALID_INPUT_TYPES)}, got {input_type!r}")
    return input_type  # type: ignore[return-value]


def _resolve_input_type(scores: np.ndarray, input_type: ScoreInputType) -> ScoreInputType:
    input_type = _validate_input_type(input_type)
    if input_type != "auto":
        return input_type
    score_arr = np.asarray(scores, dtype="float64")
    finite = score_arr[np.isfinite(score_arr)]
    if finite.size > 0 and float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return "probability"
    return "logit"


def _valid_temperature(temperature: float) -> float:
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be a positive finite value")
    return max(_EPS, temp)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype="float64")
    positive = values_arr >= 0.0
    result = np.empty_like(values_arr, dtype="float64")
    result[positive] = 1.0 / (1.0 + np.exp(-values_arr[positive]))
    exp_values = np.exp(values_arr[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _binary_logits(scores: np.ndarray, input_type: ScoreInputType) -> np.ndarray:
    score_arr = np.asarray(scores, dtype="float64")
    if input_type == "probability":
        clipped = np.clip(score_arr, _EPS, 1.0 - _EPS)
        return np.log(clipped / (1.0 - clipped))
    return score_arr


def _multiclass_logits(scores: np.ndarray, input_type: ScoreInputType) -> np.ndarray:
    score_arr = np.asarray(scores, dtype="float64")
    if input_type == "probability":
        clipped = np.clip(score_arr, _EPS, 1.0)
        row_sums = np.sum(clipped, axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return np.log(clipped / row_sums)
    return score_arr


def _valid_binary_inputs(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    input_type: ScoreInputType,
    temperature: float = 1.0,
    positive_label: int = 1,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = scores_to_probabilities(
        scores,
        input_type=input_type,
        temperature=temperature,
        positive_label=positive_label,
    ).astype("float64", copy=False)
    labels = np.asarray(y_true, dtype="float64").reshape(-1)
    if probabilities.shape[0] != labels.shape[0]:
        return (
            np.empty(0, dtype="float64"),
            np.empty(0, dtype="float64"),
            np.empty(0, dtype="float64"),
        )

    if sample_weight is None:
        weights = np.ones(labels.shape[0], dtype="float64")
    else:
        weights = np.asarray(sample_weight, dtype="float64").reshape(-1)
        if weights.shape[0] != labels.shape[0]:
            return (
                np.empty(0, dtype="float64"),
                np.empty(0, dtype="float64"),
                np.empty(0, dtype="float64"),
            )

    mask = (
        np.isfinite(probabilities)
        & np.isfinite(labels)
        & np.isfinite(weights)
        & (weights >= 0.0)
        & ((labels == 0.0) | (labels == 1.0))
    )
    if not np.any(mask):
        return (
            np.empty(0, dtype="float64"),
            np.empty(0, dtype="float64"),
            np.empty(0, dtype="float64"),
        )
    return (
        np.clip(probabilities[mask], _EPS, 1.0 - _EPS),
        labels[mask].astype("float64", copy=False),
        weights[mask].astype("float64", copy=False),
    )


def _temperature_grid(config: CalibrationConfig) -> np.ndarray:
    values = np.geomspace(config.min_temperature, config.max_temperature, config.temperature_candidates)
    values = np.concatenate([values, np.array([1.0], dtype="float64")])
    values = values[np.isfinite(values) & (values > 0.0)]
    return np.asarray(sorted({round(float(value), 8) for value in values.tolist()}), dtype="float64")


def _json_float(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _json_float(value)


__all__ = [
    "CalibrationConfig",
    "CalibrationSummary",
    "ScoreInputType",
    "TemperatureScaler",
    "apply_temperature_scaling",
    "binary_log_loss",
    "calibrate_validation_test",
    "expected_calibration_error",
    "fit_temperature_scaler",
    "scores_to_probabilities",
]
