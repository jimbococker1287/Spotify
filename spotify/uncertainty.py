from __future__ import annotations

from dataclasses import asdict, dataclass

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
    abstained = max_confidence < (float(calibration.threshold) - 1e-12)

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
    )
