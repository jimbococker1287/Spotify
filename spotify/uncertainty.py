from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SplitConformalCalibration:
    method: str
    alpha: float
    qhat: float
    threshold: float
    sample_count: int
    empirical_coverage: float
    mean_set_size: float
    operating_threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def abstention_threshold(self) -> float:
        value = self.operating_threshold
        if value is None:
            return float(self.threshold)
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return float(self.threshold)
        if math.isnan(threshold):
            return float(self.threshold)
        return threshold


def _as_2d_float_array(proba: np.ndarray) -> np.ndarray:
    proba_arr = np.asarray(proba, dtype="float64")
    if proba_arr.ndim == 1:
        proba_arr = proba_arr.reshape(1, -1)
    if proba_arr.ndim != 2:
        return np.empty((0, 0), dtype="float64")
    return proba_arr


def _valid_label_mask(y_true: np.ndarray, width: int) -> np.ndarray:
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if width <= 0 or y_arr.size == 0:
        return np.zeros(y_arr.shape[0], dtype=bool)
    return (y_arr >= 0) & (y_arr < width)


def apply_temperature_scaling(proba: np.ndarray, temperature: float) -> np.ndarray:
    proba_arr = _as_2d_float_array(proba)
    if proba_arr.shape[0] == 0:
        return proba_arr.astype("float32", copy=False)
    temp = max(float(temperature), 1e-3)
    clipped = np.clip(proba_arr, 1e-9, 1.0)
    logits = np.log(clipped) / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (exp / denom).astype("float32")


def _negative_log_likelihood(proba: np.ndarray, y_true: np.ndarray) -> float:
    proba_arr = _as_2d_float_array(proba)
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if proba_arr.shape[0] == 0 or proba_arr.shape[0] != y_arr.shape[0]:
        return float("inf")
    valid = _valid_label_mask(y_arr, proba_arr.shape[1])
    if not np.any(valid):
        return float("inf")
    valid_idx = np.flatnonzero(valid)
    y_valid = y_arr[valid]
    clipped = np.clip(proba_arr[valid_idx, y_valid], 1e-9, 1.0)
    return float(-np.mean(np.log(clipped)))


def fit_temperature_scaling(proba: np.ndarray, y_true: np.ndarray) -> float:
    proba_arr = _as_2d_float_array(proba)
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if proba_arr.shape[0] == 0 or proba_arr.shape[0] != y_arr.shape[0]:
        return 1.0
    valid = _valid_label_mask(y_arr, proba_arr.shape[1])
    if not np.any(valid):
        return 1.0

    temperatures = np.concatenate(
        [
            np.array([0.65, 0.8, 1.0, 1.25, 1.6], dtype="float64"),
            np.geomspace(0.5, 3.0, num=15, dtype="float64"),
        ]
    )
    best_temp = 1.0
    best_nll = float("inf")
    for temp in sorted({round(float(item), 6) for item in temperatures.tolist()}):
        calibrated = apply_temperature_scaling(proba_arr, temp)
        nll = _negative_log_likelihood(calibrated, y_arr)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return best_temp


def fit_split_conformal_classifier(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    alpha: float = 0.10,
) -> SplitConformalCalibration | None:
    proba_arr = _as_2d_float_array(proba)
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if proba_arr.shape[0] == 0 or proba_arr.shape[0] != y_arr.shape[0]:
        return None

    alpha_value = min(0.99, max(1e-6, float(alpha)))
    valid = _valid_label_mask(y_arr, proba_arr.shape[1])
    if not np.any(valid):
        return None

    valid_idx = np.flatnonzero(valid)
    y_valid = y_arr[valid]
    true_scores = 1.0 - np.clip(proba_arr[valid_idx, y_valid], 0.0, 1.0)
    n = int(true_scores.size)
    rank = int(np.ceil((n + 1) * (1.0 - alpha_value)))
    rank = min(max(rank, 1), n)
    qhat = float(np.partition(true_scores, rank - 1)[rank - 1])
    threshold = float(min(1.0, max(0.0, 1.0 - qhat)))

    prediction_mask = conformal_prediction_mask(proba_arr, threshold=threshold)
    empirical_coverage = float(np.mean(prediction_mask[valid_idx, y_valid]))
    mean_set_size = float(np.mean(np.sum(prediction_mask[valid], axis=1)))

    return SplitConformalCalibration(
        method="lac",
        alpha=alpha_value,
        qhat=qhat,
        threshold=threshold,
        sample_count=n,
        empirical_coverage=empirical_coverage,
        mean_set_size=mean_set_size,
    )


def fit_operating_abstention_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    base_threshold: float,
    target_selective_risk: float = 0.50,
    min_accepted_rate: float = 0.10,
    min_risk_drop: float = 0.02,
) -> float:
    proba_arr = _as_2d_float_array(proba)
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if proba_arr.shape[0] == 0 or proba_arr.shape[0] != y_arr.shape[0]:
        return float(base_threshold)

    valid = _valid_label_mask(y_arr, proba_arr.shape[1])
    if not np.any(valid):
        return float(base_threshold)

    y_valid = y_arr[valid]
    max_confidence = np.max(proba_arr[valid], axis=1).astype("float64", copy=False)
    predicted = np.argmax(proba_arr[valid], axis=1).astype("int64", copy=False)
    correct = (predicted == y_valid).astype("float64", copy=False)
    if max_confidence.size == 0:
        return float(base_threshold)

    order = np.argsort(max_confidence)[::-1]
    sorted_conf = max_confidence[order]
    sorted_correct = correct[order]
    counts = np.arange(1, len(sorted_conf) + 1, dtype="float64")
    accepted_rate = counts / float(len(sorted_conf))
    selective_accuracy = np.cumsum(sorted_correct, dtype="float64") / counts
    selective_risk = 1.0 - selective_accuracy
    full_selective_risk = float(selective_risk[-1]) if selective_risk.size else float("nan")

    eligible = np.flatnonzero(accepted_rate >= max(1e-6, float(min_accepted_rate)))
    if eligible.size == 0:
        return float(base_threshold)

    target = min(0.99, max(0.0, float(target_selective_risk)))
    satisfying = eligible[selective_risk[eligible] <= target]
    if satisfying.size:
        best_idx = int(satisfying[-1])
    else:
        improvements = full_selective_risk - selective_risk[eligible]
        utility = improvements - (0.20 * (1.0 - accepted_rate[eligible]))
        best_offset = int(np.argmax(utility))
        best_idx = int(eligible[best_offset])
        if not np.isfinite(improvements[best_offset]) or improvements[best_offset] < float(min_risk_drop):
            return float(base_threshold)

    return float(max(float(base_threshold), float(sorted_conf[best_idx])))


def conformal_prediction_mask(
    proba: np.ndarray,
    *,
    calibration: SplitConformalCalibration | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    proba_arr = _as_2d_float_array(proba)
    if proba_arr.shape[0] == 0:
        return np.zeros(proba_arr.shape, dtype=bool)

    if calibration is not None:
        threshold_value = float(calibration.threshold)
    elif threshold is not None:
        threshold_value = float(threshold)
    else:
        raise ValueError("Either calibration or threshold must be provided.")

    threshold_value = min(1.0, max(0.0, threshold_value))
    return proba_arr >= (threshold_value - 1e-12)


def conformal_prediction_sets(
    proba: np.ndarray,
    *,
    calibration: SplitConformalCalibration | None = None,
    threshold: float | None = None,
) -> list[np.ndarray]:
    proba_arr = _as_2d_float_array(proba)
    mask = conformal_prediction_mask(proba_arr, calibration=calibration, threshold=threshold)
    out: list[np.ndarray] = []
    for row_idx in range(mask.shape[0]):
        indices = np.flatnonzero(mask[row_idx])
        if indices.size == 2:
            left, right = int(indices[0]), int(indices[1])
            if proba_arr[row_idx, right] >= proba_arr[row_idx, left]:
                indices = indices[::-1]
        elif indices.size > 2:
            order = np.argsort(proba_arr[row_idx, indices])[::-1]
            indices = indices[order]
        out.append(indices.astype("int64", copy=False))
    return out


def summarize_prediction_sets(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    calibration: SplitConformalCalibration,
) -> dict[str, float | int]:
    proba_arr = _as_2d_float_array(proba)
    y_arr = np.asarray(y_true, dtype="int64").reshape(-1)
    if proba_arr.shape[0] == 0 or proba_arr.shape[0] != y_arr.shape[0]:
        return {
            "coverage": float("nan"),
            "mean_set_size": float("nan"),
            "median_set_size": float("nan"),
            "max_set_size": 0,
            "abstention_rate": float("nan"),
            "top1_confidence_mean": float("nan"),
        }

    mask = conformal_prediction_mask(proba_arr, calibration=calibration)
    set_sizes = np.sum(mask, axis=1).astype("int64")
    max_confidence = np.max(proba_arr, axis=1) if proba_arr.shape[1] > 0 else np.zeros(len(proba_arr), dtype="float64")
    abstention_threshold = float(calibration.abstention_threshold)
    abstained = max_confidence < (abstention_threshold - 1e-12)

    valid = _valid_label_mask(y_arr, proba_arr.shape[1])
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        coverage = float(np.mean(mask[valid_idx, y_arr[valid]]))
    else:
        coverage = float("nan")

    predicted = np.argmax(proba_arr, axis=1) if proba_arr.shape[1] > 0 else np.zeros(len(proba_arr), dtype="int64")
    accepted = ~abstained
    if np.any(accepted):
        selective_accuracy = float(np.mean(predicted[accepted] == y_arr[accepted]))
        selective_risk = float(1.0 - selective_accuracy)
        accepted_rate = float(np.mean(accepted))
    else:
        selective_accuracy = float("nan")
        selective_risk = float("nan")
        accepted_rate = 0.0

    return {
        "coverage": coverage,
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes.size else float("nan"),
        "median_set_size": float(np.median(set_sizes)) if set_sizes.size else float("nan"),
        "max_set_size": int(np.max(set_sizes)) if set_sizes.size else 0,
        "abstention_rate": float(np.mean(abstained)) if abstained.size else float("nan"),
        "accepted_rate": accepted_rate,
        "selective_accuracy": selective_accuracy,
        "selective_risk": selective_risk,
        "top1_confidence_mean": float(np.mean(max_confidence)) if max_confidence.size else float("nan"),
        "abstention_threshold": abstention_threshold,
    }


def calibration_from_payload(payload: dict[str, object] | None) -> SplitConformalCalibration | None:
    if not isinstance(payload, dict):
        return None

    try:
        method = str(payload.get("method", "")).strip()
        alpha = float(payload.get("alpha", float("nan")))
        qhat = float(payload.get("qhat", float("nan")))
        threshold = float(payload.get("threshold", float("nan")))
        sample_count = int(payload.get("sample_count", 0))
        empirical_coverage = float(payload.get("empirical_coverage", float("nan")))
        mean_set_size = float(payload.get("mean_set_size", float("nan")))
        operating_threshold_raw = payload.get("operating_threshold")
        operating_threshold = (
            None
            if operating_threshold_raw is None
            else float(operating_threshold_raw)
        )
    except (TypeError, ValueError):
        return None

    if not method:
        return None

    return SplitConformalCalibration(
        method=method,
        alpha=alpha,
        qhat=qhat,
        threshold=threshold,
        sample_count=sample_count,
        empirical_coverage=empirical_coverage,
        mean_set_size=mean_set_size,
        operating_threshold=operating_threshold,
    )
