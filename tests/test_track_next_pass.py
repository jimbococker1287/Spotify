from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spotify.track_next_pass import (
    TrackNextPassConfig,
    _dcn_calibration_payload,
    _dcn_drift_payload,
    _prepare_dcn_training_dataset,
    _predict_dcn,
    _tuning_gate_manifest,
    expected_calibration_error,
    reranking_split_to_dcn,
)
from spotify.track_dcn_training import DCNCandidateSplit, DCNTemporalDataset
from spotify.track_reranking_data import TrackRerankingSplit


def _reranking_split() -> TrackRerankingSplit:
    return TrackRerankingSplit(
        query_ids=("query-a", "query-b"),
        example_ids=np.asarray([1, 1, 2, 2], dtype="int64"),
        group_ids=np.asarray([0, 0, 1, 1], dtype="int64"),
        group_offsets=np.asarray([0, 2, 4], dtype="int64"),
        candidate_track_uris=("a", "b", "c", "d"),
        candidate_item_ids=np.asarray([2, 3, 4, 5], dtype="int32"),
        context_features=np.ones((4, 2), dtype="float32"),
        candidate_features=np.ones((4, 3), dtype="float32"),
        labels=np.asarray([1, 0, 0, 1], dtype="float32"),
        source_example_count=2,
        skipped_oov_target_count=0,
    )


def test_reranking_split_to_dcn_expands_query_ids() -> None:
    converted = reranking_split_to_dcn(_reranking_split())

    converted.validate(name="converted", allow_empty=False)
    assert converted.query_ids.tolist() == [
        "query-a",
        "query-a",
        "query-b",
        "query-b",
    ]
    assert converted.candidate_ids.tolist() == ["a", "b", "c", "d"]


def test_expected_calibration_error_handles_perfect_and_bad_scores() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype="float32")

    assert expected_calibration_error(
        labels,
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        bins=2,
    ) == pytest.approx(0.0)
    assert expected_calibration_error(
        labels,
        np.asarray([1.0, 1.0, 0.0, 0.0]),
        bins=2,
    ) == pytest.approx(1.0)


def test_predict_dcn_uses_direct_batches_without_keras_predict() -> None:
    split = reranking_split_to_dcn(_reranking_split())
    batch_sizes: list[int] = []

    class Model:
        def __call__(self, inputs, *, training):
            assert training is False
            batch_sizes.append(len(inputs["context_input"]))
            return np.full((len(inputs["context_input"]), 1), 0.25)

        def predict(self, *_args, **_kwargs):
            raise AssertionError("Keras predict() must not be used")

    scores = _predict_dcn(Model(), split, batch_size=3)

    assert scores.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert batch_sizes == [3, 1]


def test_dcn_calibration_payload_exposes_gate_compatible_ece_keys() -> None:
    payload = _dcn_calibration_payload(
        validation_labels=np.asarray([1, 1, 0, 0, 0, 1], dtype="float32"),
        validation_scores=np.asarray([0.98, 0.97, 0.96, 0.02, 0.03, 0.04], dtype="float32"),
        test_labels=np.asarray([1, 0, 0, 1], dtype="float32"),
        test_scores=np.asarray([0.95, 0.05, 0.94, 0.06], dtype="float32"),
    )

    assert payload["status"] == "complete"
    assert payload["method"] == "temperature_scaling"
    assert payload["ece"] == payload["test_ece"]
    assert payload["expected_calibration_error"] == payload["test_ece"]
    assert payload["raw_test_ece"] == payload["test_ece_before"]
    assert payload["test_ece_after"] == payload["test_ece"]


def _dcn_split(context: np.ndarray, items: np.ndarray) -> DCNCandidateSplit:
    rows = int(context.shape[0])
    return DCNCandidateSplit(
        context_features=context.astype("float32"),
        item_features=items.astype("float32"),
        labels=np.tile(np.asarray([1, 0], dtype="float32"), rows // 2),
        query_ids=np.asarray([f"q-{idx // 2}" for idx in range(rows)], dtype=np.str_),
        candidate_ids=np.asarray([f"i-{idx}" for idx in range(rows)], dtype=np.str_),
    )


def test_dcn_drift_payload_writes_feature_level_artifact(tmp_path: Path) -> None:
    train_context = np.zeros((4, 8), dtype="float32")
    train_items = np.zeros((4, 9), dtype="float32")
    validation_context = train_context.copy()
    validation_context[:, 0] = 5.0
    validation_items = train_items.copy()
    test_context = train_context.copy()
    test_items = train_items.copy()
    test_items[:, 2] = 7.0
    dataset = DCNTemporalDataset(
        train=_dcn_split(train_context, train_items),
        validation=_dcn_split(validation_context, validation_items),
        test=_dcn_split(test_context, test_items),
    )

    payload = _dcn_drift_payload(dataset=dataset, output_dir=tmp_path)

    assert payload["status"] == "fail"
    assert payload["artifact_present"] is True
    artifact = Path(str(payload["artifact"]))
    assert artifact.exists()
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["validation_drift_score"] == payload["validation_drift_score"]
    assert saved["test_drift_score"] == payload["test_drift_score"]
    assert payload["validation_drift_score"] is not None
    assert payload["test_drift_score"] is not None
    assert "validation.context.log_history_length" in payload["failing_features"]
    assert "test.item.log_popularity_count" in payload["failing_features"]


def test_prepare_dcn_training_dataset_applies_validation_low_drift_mask(tmp_path: Path) -> None:
    train_context = np.zeros((4, 8), dtype="float32")
    train_items = np.zeros((4, 9), dtype="float32")
    validation_context = train_context.copy()
    validation_context[:, 0] = 5.0
    dataset = DCNTemporalDataset(
        train=_dcn_split(train_context, train_items),
        validation=_dcn_split(validation_context, train_items.copy()),
        test=_dcn_split(train_context.copy(), train_items.copy()),
    )
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        dcn_low_drift_mask=True,
    )

    prepared, context_names, item_names, mitigation = _prepare_dcn_training_dataset(
        dataset=dataset,
        config=config,
        output_dir=tmp_path,
    )

    assert prepared.train.context_features.shape == (4, 7)
    assert prepared.train.item_features.shape == (4, 9)
    assert "log_history_length" not in context_names
    assert item_names == (
        "cooccurrence_score",
        "cooccurrence_reciprocal_rank",
        "log_popularity_count",
        "popularity_reciprocal_rank",
        "retriever_agreement",
        "history_frequency_ratio",
        "history_reciprocal_recency",
        "same_as_last_track",
        "session_support_ratio",
    )
    plan = mitigation["low_drift_feature_mask"]
    assert isinstance(plan, dict)
    assert plan["selection_split"] == "validation"
    assert Path(str(plan["artifact"])).exists()


def test_prepare_dcn_training_dataset_applies_compact_recency_reweighting(tmp_path: Path) -> None:
    train_context = np.zeros((4, 8), dtype="float32")
    train_items = np.zeros((4, 9), dtype="float32")
    dataset = DCNTemporalDataset(
        train=_dcn_split(train_context, train_items),
        validation=_dcn_split(train_context.copy(), train_items.copy()),
        test=_dcn_split(train_context.copy(), train_items.copy()),
    )
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        dcn_recency_reweight=True,
        dcn_recency_min_weight=0.75,
        dcn_recency_max_weight=1.25,
        dcn_recency_fallback="row_order",
    )

    prepared, context_names, item_names, mitigation = _prepare_dcn_training_dataset(
        dataset=dataset,
        config=config,
        output_dir=tmp_path,
    )

    assert prepared.train.sample_weights is not None
    assert prepared.train.sample_weights[0] < prepared.train.sample_weights[-1]
    assert prepared.validation.sample_weights is None
    assert context_names == tuple(
        [
            "log_history_length",
            "history_unique_ratio",
            "history_repeat_ratio",
            "log_session_position",
            "log_mean_history_gap",
            "log_max_history_gap",
            "log_last_history_gap",
            "known_history_ratio",
        ]
    )
    assert len(item_names) == 9
    payload = mitigation["temporal_reweighting"]
    assert isinstance(payload, dict)
    assert payload["method"] == "fallback"
    assert payload["fallback"] == "row_order"
    assert payload["weight_count"] == 4
    assert "weights" not in payload
    assert Path(str(payload["artifact"])).exists()


def test_prepare_dcn_training_dataset_applies_manual_context_feature_drops(tmp_path: Path) -> None:
    train_context = np.zeros((4, 8), dtype="float32")
    train_items = np.zeros((4, 9), dtype="float32")
    dataset = DCNTemporalDataset(
        train=_dcn_split(train_context, train_items),
        validation=_dcn_split(train_context.copy(), train_items.copy()),
        test=_dcn_split(train_context.copy(), train_items.copy()),
    )
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        dcn_drop_context_features=("log_history_length", "log_session_position"),
    )

    prepared, context_names, item_names, mitigation = _prepare_dcn_training_dataset(
        dataset=dataset,
        config=config,
        output_dir=tmp_path,
    )

    assert prepared.train.context_features.shape == (4, 6)
    assert prepared.train.item_features.shape == (4, 9)
    assert "log_history_length" not in context_names
    assert "log_session_position" not in context_names
    assert len(item_names) == 9
    payload = mitigation["manual_feature_drop"]
    assert isinstance(payload, dict)
    assert payload["dropped_context_features"] == [
        "log_history_length",
        "log_session_position",
    ]
    assert Path(str(payload["artifact"])).exists()


def test_track_next_pass_config_requires_complete_public_pair(tmp_path: Path) -> None:
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        public_manifest=tmp_path / "manifest.json",
    )

    with pytest.raises(ValueError, match="supplied together"):
        config.validate()


def test_track_next_pass_config_rejects_unknown_manual_drop_feature(tmp_path: Path) -> None:
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        dcn_drop_context_features=("not_a_feature",),
    )

    with pytest.raises(ValueError, match="Unknown DCN context drop features"):
        config.validate()


def test_tuning_gate_manifest_extracts_best_trial_metadata() -> None:
    manifest = _tuning_gate_manifest(
        {
            "status": "complete",
            "studies": [
                {
                    "model_name": "dcn_v2",
                    "completed_trials": 2,
                    "total_trials": 2,
                    "best_trial": {
                        "metric_name": "validation_ndcg_at_10",
                        "value": 0.25,
                        "params": {"cross_layers": 2},
                    },
                }
            ],
        }
    )

    assert manifest["tuning_results"] == [
        {
            "model_name": "dcn_v2",
            "completed_trials": 2,
            "trial_count": 2,
            "best_params": {"cross_layers": 2},
            "parameters": {"cross_layers": 2},
            "tuning_metric": "validation_ndcg_at_10",
            "tuning_value": 0.25,
        }
    ]
