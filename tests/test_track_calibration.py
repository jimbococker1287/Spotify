from __future__ import annotations

import json

import numpy as np
import pytest

from spotify.track_calibration import (
    CalibrationConfig,
    apply_temperature_scaling,
    binary_log_loss,
    calibrate_validation_test,
    expected_calibration_error,
    fit_temperature_scaler,
    scores_to_probabilities,
)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype("float64"), 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def test_expected_calibration_error_uses_binary_positive_rate_bins() -> None:
    probabilities = np.array([0.10, 0.20, 0.80, 0.90], dtype="float32")
    labels = np.array([0, 0, 1, 1], dtype="int32")

    ece = expected_calibration_error(probabilities, labels, n_bins=2)

    assert ece == pytest.approx(0.15)


def test_expected_calibration_error_accepts_logits_equivalent_to_probabilities() -> None:
    probabilities = np.array([0.10, 0.20, 0.80, 0.90], dtype="float32")
    labels = np.array([0, 0, 1, 1], dtype="int32")

    ece_from_probabilities = expected_calibration_error(probabilities, labels, n_bins=2)
    ece_from_logits = expected_calibration_error(_logit(probabilities), labels, input_type="logit", n_bins=2)

    assert ece_from_logits == pytest.approx(ece_from_probabilities)


def test_apply_temperature_scaling_softens_overconfident_logits() -> None:
    logits = np.array([-4.0, 0.0, 4.0], dtype="float32")

    softened = apply_temperature_scaling(logits, 2.0, input_type="logit")

    assert softened[0] > apply_temperature_scaling(logits, 1.0, input_type="logit")[0]
    assert softened[1] == pytest.approx(0.5)
    assert softened[2] < apply_temperature_scaling(logits, 1.0, input_type="logit")[2]


def test_scores_to_probabilities_extracts_positive_column_from_two_class_scores() -> None:
    class_probabilities = np.array(
        [
            [0.80, 0.20],
            [0.25, 0.75],
        ],
        dtype="float32",
    )

    positive = scores_to_probabilities(class_probabilities, input_type="probability", positive_label=1)

    assert positive.tolist() == pytest.approx([0.20, 0.75])


def test_fit_temperature_scaler_uses_validation_labels_and_reduces_log_loss() -> None:
    validation_logits = np.array([4.0, 4.0, 4.0, -4.0, -4.0, -4.0], dtype="float32")
    validation_labels = np.array([1, 1, 0, 0, 0, 1], dtype="int32")

    scaler = fit_temperature_scaler(
        validation_logits,
        validation_labels,
        config=CalibrationConfig(input_type="logit", temperature_candidates=61),
    )

    assert scaler.status == "complete"
    assert scaler.temperature > 1.0
    assert scaler.validation_log_loss_after < scaler.validation_log_loss_before
    assert scaler.transform(validation_logits).shape == validation_logits.shape


def test_calibrate_validation_test_fits_only_validation_and_applies_to_test() -> None:
    validation_logits = np.array([4.0, 4.0, 4.0, -4.0, -4.0, -4.0], dtype="float32")
    validation_labels = np.array([1, 1, 0, 0, 0, 1], dtype="int32")
    test_logits = np.array([3.0, -3.0, 3.0, -3.0], dtype="float32")
    test_labels = np.array([1, 0, 0, 1], dtype="int32")

    summary = calibrate_validation_test(
        validation_logits,
        validation_labels,
        test_logits,
        test_labels,
        config=CalibrationConfig(input_type="logit", n_bins=4, temperature_candidates=61),
    )
    payload = summary.to_dict()

    assert summary.validation_count == 6
    assert summary.test_count == 4
    assert summary.temperature > 1.0
    assert summary.test_log_loss_after < summary.test_log_loss_before
    assert payload["method"] == "temperature_scaling"
    json.dumps(payload, sort_keys=True)


def test_invalid_validation_contract_returns_json_friendly_unavailable_summary() -> None:
    summary = calibrate_validation_test(
        np.array([], dtype="float32"),
        np.array([], dtype="int32"),
        np.array([0.2, 0.8], dtype="float32"),
        np.array([0, 1], dtype="int32"),
    )
    payload = summary.to_dict()

    assert summary.status == "unavailable"
    assert summary.temperature == 1.0
    assert payload["validation_log_loss_before"] is None
    assert payload["reason"] == "no valid binary validation rows"


def test_sample_weight_changes_log_loss_without_breaking_ece() -> None:
    probabilities = np.array([0.90, 0.90, 0.10], dtype="float32")
    labels = np.array([1, 0, 0], dtype="int32")
    weights = np.array([1.0, 10.0, 1.0], dtype="float32")

    weighted = binary_log_loss(probabilities, labels, sample_weight=weights)
    unweighted = binary_log_loss(probabilities, labels)
    ece = expected_calibration_error(probabilities, labels, sample_weight=weights)

    assert weighted > unweighted
    assert 0.0 <= ece <= 1.0
