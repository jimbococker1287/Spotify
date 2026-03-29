from __future__ import annotations

import numpy as np

from spotify.uncertainty import (
    calibration_from_payload,
    conformal_prediction_sets,
    fit_split_conformal_classifier,
    summarize_prediction_sets,
)


def test_fit_split_conformal_classifier_returns_threshold_and_summary() -> None:
    proba = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.70, 0.20],
            [0.20, 0.25, 0.55],
            [0.62, 0.20, 0.18],
        ],
        dtype="float32",
    )
    y_true = np.array([0, 1, 2, 0], dtype="int32")

    calibration = fit_split_conformal_classifier(proba, y_true, alpha=0.10)

    assert calibration is not None
    assert calibration.method == "lac"
    assert 0.0 <= calibration.threshold <= 1.0
    assert calibration.sample_count == 4
    assert calibration.empirical_coverage >= 0.75


def test_conformal_prediction_sets_and_summary_use_calibrated_threshold() -> None:
    proba = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.58, 0.57, 0.05],
        ],
        dtype="float32",
    )
    y_true = np.array([0, 1], dtype="int32")
    calibration = fit_split_conformal_classifier(
        np.array(
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.70, 0.20],
                [0.20, 0.25, 0.55],
                [0.62, 0.20, 0.18],
            ],
            dtype="float32",
        ),
        np.array([0, 1, 2, 0], dtype="int32"),
        alpha=0.10,
    )

    assert calibration is not None
    prediction_sets = conformal_prediction_sets(proba, calibration=calibration)
    summary = summarize_prediction_sets(proba, y_true, calibration=calibration)

    assert prediction_sets[0].tolist() == [0]
    assert prediction_sets[1].tolist() == [0, 1]
    assert summary["coverage"] >= 0.5
    assert summary["mean_set_size"] >= 1.0


def test_conformal_prediction_sets_keep_descending_probability_order_for_small_sets() -> None:
    proba = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.40, 0.60, 0.10],
            [0.55, 0.20, 0.55],
        ],
        dtype="float32",
    )

    prediction_sets = conformal_prediction_sets(proba, threshold=0.35)

    assert prediction_sets[0].tolist() == [0]
    assert prediction_sets[1].tolist() == [1, 0]
    assert prediction_sets[2].tolist() == [2, 0]


def test_calibration_from_payload_rehydrates_dataclass() -> None:
    payload = {
        "method": "lac",
        "alpha": 0.1,
        "qhat": 0.4,
        "threshold": 0.6,
        "sample_count": 12,
        "empirical_coverage": 0.91,
        "mean_set_size": 1.6,
    }

    calibration = calibration_from_payload(payload)

    assert calibration is not None
    assert calibration.threshold == 0.6
    assert calibration.sample_count == 12
