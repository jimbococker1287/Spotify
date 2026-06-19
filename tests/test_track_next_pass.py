from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spotify.track_next_pass import (
    TrackNextPassConfig,
    _predict_dcn,
    _tuning_gate_manifest,
    expected_calibration_error,
    reranking_split_to_dcn,
)
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


def test_track_next_pass_config_requires_complete_public_pair(tmp_path: Path) -> None:
    config = TrackNextPassConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        public_manifest=tmp_path / "manifest.json",
    )

    with pytest.raises(ValueError, match="supplied together"):
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
