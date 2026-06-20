from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from spotify.track_drift_evidence import (
    FeatureDriftReport,
    TrackDriftEvidenceConfig,
    build_track_drift_evidence,
    compute_feature_drift_report,
    jensen_shannon_distance,
    save_track_drift_evidence,
    select_low_drift_feature_mask,
    standardized_mean_shift,
)


def _split(
    *,
    context_features: list[list[float]],
    item_features: list[list[float]],
) -> SimpleNamespace:
    return SimpleNamespace(
        context_features=np.asarray(context_features, dtype="float32"),
        item_features=np.asarray(item_features, dtype="float32"),
    )


def test_standardized_shift_and_js_distance_are_deterministic() -> None:
    reference = np.asarray([0.0, 0.0, 1.0, 1.0], dtype="float32")
    same = np.asarray([0.0, 0.0, 1.0, 1.0], dtype="float32")
    shifted = np.asarray([4.0, 4.0, 5.0, 5.0], dtype="float32")
    separated = np.asarray([2.0, 2.0, 2.0, 2.0], dtype="float32")

    assert standardized_mean_shift(reference, same) == pytest.approx(0.0)
    assert jensen_shannon_distance(reference, same, bins=4) == pytest.approx(0.0)
    assert standardized_mean_shift(reference, shifted) == pytest.approx(8.0)
    assert jensen_shannon_distance(reference, separated, bins=3) == pytest.approx(1.0)


def test_feature_report_names_low_and_high_drift_features() -> None:
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
        ],
        dtype="float32",
    )
    comparison = np.asarray(
        [
            [0.0, 4.0, 1.0],
            [0.0, 5.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 5.0, 1.0],
        ],
        dtype="float32",
    )

    report = compute_feature_drift_report(
        reference_values=reference,
        comparison_values=comparison,
        feature_names=("stable", "mean_shift", "shape_shift"),
        feature_group="context",
        reference_split="train",
        comparison_split="validation",
        config=TrackDriftEvidenceConfig(
            max_abs_standardized_mean_shift=0.5,
            max_js_distance=0.2,
            histogram_bins=4,
            top_feature_count=2,
        ),
    )

    assert report.status == "fail"
    assert report.low_drift_mask == (True, False, False)
    assert report.failing_features == ("mean_shift", "shape_shift")
    assert report.rows[0].low_drift is True
    assert report.rows[1].standardized_mean_shift == pytest.approx(8.0)
    assert report.rows[2].js_distance == pytest.approx(1.0)
    assert [row.feature_name for row in report.top_features(limit=2)] == [
        "mean_shift",
        "shape_shift",
    ]
    payload = report.to_dict(top_feature_count=2)
    assert payload["max_abs_standardized_mean_shift"] == pytest.approx(8.0)
    assert payload["top_features"][0]["feature_name"] == "mean_shift"
    json.dumps(payload, sort_keys=True)


def test_low_drift_mask_accepts_dataclasses_and_mapping_rows() -> None:
    rows = [
        FeatureDriftReport(
            feature_group="context",
            feature_name="stable",
            feature_index=0,
            reference_rows=2,
            comparison_rows=2,
            reference_mean=0.0,
            comparison_mean=0.0,
            standardized_mean_shift=0.0,
            abs_standardized_mean_shift=0.0,
            js_distance=0.0,
            low_drift=True,
        ),
        {
            "abs_standardized_mean_shift": 0.1,
            "js_distance": 0.5,
        },
        {
            "abs_standardized_mean_shift": 0.5,
            "js_distance": 0.1,
        },
    ]

    mask = select_low_drift_feature_mask(
        rows,
        max_abs_standardized_mean_shift=0.2,
        max_js_distance=0.2,
    )

    assert mask.tolist() == [True, False, False]


def test_track_drift_evidence_groups_context_and_item_blockers(tmp_path) -> None:
    train = _split(
        context_features=[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        item_features=[
            [0.0, 2.0],
            [0.0, 3.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
    )
    validation = _split(
        context_features=[
            [0.0, 10.0],
            [0.0, 11.0],
            [1.0, 10.0],
            [1.0, 11.0],
        ],
        item_features=[
            [3.0, 2.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [4.0, 3.0],
        ],
    )

    report = build_track_drift_evidence(
        reference_split=train,
        comparison_splits={"validation": validation},
        context_feature_names=("hour", "session_position"),
        item_feature_names=("artist_affinity", "candidate_popularity"),
        config=TrackDriftEvidenceConfig(
            max_abs_standardized_mean_shift=0.5,
            max_js_distance=0.5,
            histogram_bins=4,
            top_feature_count=1,
        ),
    )

    assert report.status == "fail"
    assert report.within_threshold is False
    assert report.failing_features() == (
        "validation.context.session_position",
        "validation.item.artist_affinity",
    )
    blocker_names = {
        (row["feature_group"], row["feature_name"]) for row in report.gate_blockers
    }
    assert blocker_names == {
        ("context", "session_position"),
        ("item", "artist_affinity"),
    }
    payload = report.to_dict()
    assert payload["max_drift_score"] == pytest.approx(20.0)
    assert payload["groups"][0]["low_drift_mask"] == [True, False]
    assert payload["groups"][1]["low_drift_mask"] == [False, True]

    path = save_track_drift_evidence(report, tmp_path / "drift_evidence.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["failing_features"] == list(report.failing_features())
    assert saved["gate_blockers"][0]["reasons"]


def test_validation_rejects_mismatched_feature_widths() -> None:
    with pytest.raises(ValueError, match="feature width mismatch"):
        compute_feature_drift_report(
            reference_values=np.ones((2, 2)),
            comparison_values=np.ones((2, 3)),
            feature_names=("a", "b"),
            feature_group="item",
            reference_split="train",
            comparison_split="test",
        )
