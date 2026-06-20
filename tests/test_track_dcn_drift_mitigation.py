from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from spotify.track_dcn_drift_mitigation import (
    PROMOTION_DECISION,
    TrackDCNDriftMitigationConfig,
    apply_track_dcn_drift_mitigation,
    build_track_dcn_drift_mitigation_plan,
    save_track_dcn_drift_mitigation_plan,
)
from spotify.track_dcn_training import DCNCandidateSplit, DCNTemporalDataset
from spotify.track_drift_evidence import TrackDriftEvidenceConfig, build_track_drift_evidence


def _report_dict() -> dict[str, object]:
    return {
        "status": "fail",
        "reference_split": "train",
        "comparison_splits": ["validation", "test"],
        "config": {
            "max_abs_standardized_mean_shift": 0.2,
            "max_js_distance": 0.2,
        },
        "groups": [
            {
                "feature_group": "context",
                "comparison_split": "validation",
                "low_drift_mask": [True, False, True],
                "failing_features": ["session_position"],
                "features": [
                    _feature("context", "hour", 0, low_drift=True),
                    _feature("context", "session_position", 1, low_drift=False, shift=1.4),
                    _feature("context", "history_depth", 2, low_drift=True),
                ],
            },
            {
                "feature_group": "context",
                "comparison_split": "test",
                "low_drift_mask": [True, True, True],
                "failing_features": [],
                "features": [
                    _feature("context", "hour", 0, low_drift=True),
                    _feature("context", "session_position", 1, low_drift=True),
                    _feature("context", "history_depth", 2, low_drift=True),
                ],
            },
            {
                "feature_group": "item",
                "comparison_split": "validation",
                "low_drift_mask": [False, True, True],
                "failing_features": ["artist_affinity"],
                "features": [
                    _feature("item", "artist_affinity", 0, low_drift=False, shift=2.0),
                    _feature("item", "candidate_recency", 1, low_drift=True),
                    _feature("item", "popularity", 2, low_drift=True),
                ],
            },
            {
                "feature_group": "item",
                "comparison_split": "test",
                "low_drift_mask": [True, True, False],
                "failing_features": ["popularity"],
                "features": [
                    _feature("item", "artist_affinity", 0, low_drift=True),
                    _feature("item", "candidate_recency", 1, low_drift=True),
                    _feature("item", "popularity", 2, low_drift=False, js=0.7),
                ],
            },
        ],
    }


def _feature(
    feature_group: str,
    feature_name: str,
    feature_index: int,
    *,
    low_drift: bool,
    shift: float = 0.0,
    js: float = 0.0,
) -> dict[str, object]:
    return {
        "feature_group": feature_group,
        "feature_name": feature_name,
        "feature_index": feature_index,
        "reference_rows": 8,
        "comparison_rows": 8,
        "reference_mean": 0.0,
        "comparison_mean": shift,
        "standardized_mean_shift": shift,
        "abs_standardized_mean_shift": abs(shift),
        "js_distance": js,
        "drift_score": max(abs(shift), js),
        "low_drift": low_drift,
    }


def _split(prefix: str, *, time_start: int) -> DCNCandidateSplit:
    context = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype="float32",
    )
    items = np.asarray(
        [
            [5.0, 50.0, 500.0],
            [6.0, 60.0, 600.0],
            [7.0, 70.0, 700.0],
            [8.0, 80.0, 800.0],
        ],
        dtype="float32",
    )
    return DCNCandidateSplit(
        context_features=context,
        item_features=items,
        labels=np.asarray([1, 0, 1, 0], dtype="float32"),
        query_ids=np.asarray([f"{prefix}-q1", f"{prefix}-q1", f"{prefix}-q2", f"{prefix}-q2"], dtype=object),
        candidate_ids=np.asarray([f"{prefix}-a", f"{prefix}-b", f"{prefix}-c", f"{prefix}-d"], dtype=object),
        event_times=np.arange(time_start, time_start + 4, dtype="int64"),
        sample_weights=np.asarray([1.0, 0.5, 2.0, 1.5], dtype="float32"),
    )


def test_plan_from_report_dict_is_json_ready_and_summarizes_dropped_features(tmp_path) -> None:
    plan = build_track_dcn_drift_mitigation_plan(_report_dict())

    assert plan.status == "ready"
    assert plan.promotion_decision == PROMOTION_DECISION
    assert plan.context.low_drift_mask == (True, False, True)
    assert plan.item.low_drift_mask == (False, True, False)
    assert [drop.feature_name for drop in plan.context.dropped_features] == ["session_position"]
    assert [drop.feature_name for drop in plan.item.dropped_features] == ["artist_affinity", "popularity"]

    payload = plan.to_dict()
    assert payload["promotion_decision"] == "not_evaluated"
    assert payload["dropped_features"] == [
        "context.session_position",
        "item.artist_affinity",
        "item.popularity",
    ]
    assert payload["context"]["dropped_features"][0]["source_rows"][0]["thresholds"] == {
        "max_abs_standardized_mean_shift": 0.2,
        "max_js_distance": 0.2,
    }
    json.dumps(payload, sort_keys=True)

    path = save_track_dcn_drift_mitigation_plan(plan, tmp_path / "mask_plan.json")
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_apply_plan_masks_dcn_temporal_dataset_and_preserves_metadata() -> None:
    dataset = DCNTemporalDataset(
        train=_split("train", time_start=1),
        validation=_split("validation", time_start=10),
        test=_split("test", time_start=20),
    )
    plan = build_track_dcn_drift_mitigation_plan(_report_dict())

    masked = apply_track_dcn_drift_mitigation(dataset, plan)

    np.testing.assert_allclose(masked.train.context_features, dataset.train.context_features[:, [0, 2]])
    np.testing.assert_allclose(masked.train.item_features, dataset.train.item_features[:, [1]])
    assert dataset.train.context_features.shape == (4, 3)
    assert dataset.train.item_features.shape == (4, 3)
    for split_name in ("train", "validation", "test"):
        original = getattr(dataset, split_name)
        observed = getattr(masked, split_name)
        np.testing.assert_array_equal(observed.labels, original.labels)
        np.testing.assert_array_equal(observed.query_ids, original.query_ids)
        np.testing.assert_array_equal(observed.candidate_ids, original.candidate_ids)
        np.testing.assert_array_equal(observed.event_times, original.event_times)
        np.testing.assert_array_equal(observed.sample_weights, original.sample_weights)


def test_plan_accepts_track_drift_evidence_report_dataclass() -> None:
    train = SimpleNamespace(
        context_features=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32"),
        item_features=np.asarray([[0.0, 2.0], [1.0, 2.0], [0.0, 3.0], [1.0, 3.0]], dtype="float32"),
    )
    validation = SimpleNamespace(
        context_features=np.asarray([[0.0, 9.0], [1.0, 9.0], [0.0, 10.0], [1.0, 10.0]], dtype="float32"),
        item_features=train.item_features.copy(),
    )
    report = build_track_drift_evidence(
        reference_split=train,
        comparison_splits={"validation": validation},
        context_feature_names=("stable_context", "drifty_context"),
        item_feature_names=("stable_item_a", "stable_item_b"),
        config=TrackDriftEvidenceConfig(
            max_abs_standardized_mean_shift=0.5,
            max_js_distance=0.5,
            histogram_bins=4,
        ),
    )

    plan = build_track_dcn_drift_mitigation_plan(report)

    assert plan.context.low_drift_mask == (True, False)
    assert plan.context.dropped_features[0].feature_name == "drifty_context"
    assert plan.item.low_drift_mask == (True, True)


def test_plan_rejects_masks_that_drop_all_context_or_item_features() -> None:
    report = _report_dict()
    groups = list(report["groups"])
    groups[0] = {**groups[0], "low_drift_mask": [False, False, False]}
    groups[1] = {**groups[1], "low_drift_mask": [False, False, False]}
    report["groups"] = groups

    with pytest.raises(ValueError, match="context low-drift mask retains 0 features"):
        build_track_dcn_drift_mitigation_plan(report)


def test_config_can_require_more_than_one_retained_item_feature() -> None:
    with pytest.raises(ValueError, match="item low-drift mask retains 1 features"):
        build_track_dcn_drift_mitigation_plan(
            _report_dict(),
            config=TrackDCNDriftMitigationConfig(minimum_item_features=2),
        )
