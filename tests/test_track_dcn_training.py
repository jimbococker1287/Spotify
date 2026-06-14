from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from spotify.track_dcn_training import (
    DCNCandidateSplit,
    DCNTemporalDataset,
    DCNTrainingConfig,
    compute_dcn_training_weights,
    evaluate_dcn_scores,
    train_dcn_v2_reranker,
)


def _split(
    labels: list[int],
    query_ids: list[str],
    *,
    candidate_ids: list[str] | None = None,
    event_times: list[int] | None = None,
    sample_weights: list[float] | None = None,
) -> DCNCandidateSplit:
    row_count = len(labels)
    values = np.arange(row_count, dtype="float32")
    return DCNCandidateSplit(
        context_features=np.column_stack((values, values % 2)),
        item_features=np.column_stack((values / 10.0, np.ones(row_count))),
        labels=np.asarray(labels, dtype="float32"),
        query_ids=np.asarray(query_ids, dtype=object),
        candidate_ids=(
            None if candidate_ids is None else np.asarray(candidate_ids, dtype=object)
        ),
        event_times=(
            None if event_times is None else np.asarray(event_times, dtype="int64")
        ),
        sample_weights=(
            None
            if sample_weights is None
            else np.asarray(sample_weights, dtype="float32")
        ),
    )


def _empty_split() -> DCNCandidateSplit:
    return DCNCandidateSplit(
        context_features=np.empty((0, 2), dtype="float32"),
        item_features=np.empty((0, 2), dtype="float32"),
        labels=np.empty(0, dtype="float32"),
        query_ids=np.empty(0, dtype=object),
        candidate_ids=np.empty(0, dtype=object),
        event_times=np.empty(0, dtype="int64"),
    )


def test_grouped_ranking_and_pointwise_metrics_are_computed_per_query() -> None:
    split = _split(
        [1, 0, 1, 0, 1],
        ["q1", "q1", "q1", "q2", "q2"],
        candidate_ids=["a", "b", "c", "d", "e"],
    )

    result = evaluate_dcn_scores(
        split,
        np.asarray([0.9, 0.8, 0.1, 0.9, 0.8]),
        k_values=(1, 2),
    )

    assert result["status"] == "ok"
    assert result["pointwise"]["roc_auc"] == pytest.approx(1.0 / 3.0)
    assert result["pointwise"]["log_loss"] > 0.0
    assert result["ranking"]["query_count"] == 2
    assert result["ranking"]["evaluated_query_count"] == 2
    assert result["ranking"]["recall_at_k"]["1"] == pytest.approx(0.25)
    assert result["ranking"]["ndcg_at_k"]["1"] == pytest.approx(0.5)
    assert result["ranking"]["mrr_at_k"]["1"] == pytest.approx(0.5)
    assert result["ranking"]["recall_at_k"]["2"] == pytest.approx(0.75)
    assert result["ranking"]["mrr_at_k"]["2"] == pytest.approx(0.75)


def test_empty_and_single_class_evaluation_degrade_cleanly() -> None:
    empty = evaluate_dcn_scores(_empty_split(), np.empty(0), k_values=(5,))
    single_class = evaluate_dcn_scores(
        _split([0, 0], ["q1", "q1"]),
        np.asarray([0.2, 0.3]),
        k_values=(1,),
    )

    assert empty["status"] == "unavailable"
    assert empty["reason"] == "empty_split"
    assert empty["pointwise"]["roc_auc"] is None
    assert single_class["status"] == "ok"
    assert single_class["pointwise"]["roc_auc"] is None
    assert single_class["pointwise"]["log_loss"] > 0.0
    assert single_class["ranking"]["evaluated_query_count"] == 0
    assert single_class["ranking"]["recall_at_k"]["1"] is None


def test_training_weights_combine_sample_and_balanced_class_weights() -> None:
    split = _split(
        [0, 0, 0, 1],
        ["q1", "q1", "q2", "q2"],
        sample_weights=[1.0, 2.0, 1.0, 1.0],
    )

    weights, factors = compute_dcn_training_weights(split)

    assert factors["negative"] == pytest.approx(2.0 / 3.0)
    assert factors["positive"] == pytest.approx(2.0)
    np.testing.assert_allclose(weights, [2.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 2.0])


@pytest.mark.parametrize(
    ("split", "message"),
    [
        (
            DCNCandidateSplit(
                context_features=np.ones((2, 2)),
                item_features=np.ones((1, 2)),
                labels=np.asarray([0, 1]),
                query_ids=np.asarray(["q1", "q1"]),
            ),
            "item_features has 1 rows",
        ),
        (
            DCNCandidateSplit(
                context_features=np.ones((2, 2)),
                item_features=np.ones((2, 2)),
                labels=np.asarray([0, 2]),
                query_ids=np.asarray(["q1", "q1"]),
            ),
            "labels must be binary",
        ),
        (
            _split(
                [0, 1],
                ["q1", "q1"],
                candidate_ids=["same", "same"],
            ),
            "unique within each query",
        ),
    ],
)
def test_split_contract_rejects_invalid_candidate_arrays(
    split: DCNCandidateSplit,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split.validate(name="validation")


def test_temporal_dataset_rejects_split_leakage() -> None:
    dataset = DCNTemporalDataset(
        train=_split([0, 1], ["q1", "q1"], event_times=[10, 11]),
        validation=_split([0, 1], ["q2", "q2"], event_times=[9, 12]),
        test=_split([0, 1], ["q3", "q3"], event_times=[13, 14]),
    )

    with pytest.raises(ValueError, match="temporal split order"):
        dataset.validate()


def test_configuration_and_probability_validation_fail_early(tmp_path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DCNTrainingConfig(output_dir=tmp_path, batch_size=True).validate()
    with pytest.raises(ValueError, match="random_seed"):
        DCNTrainingConfig(output_dir=tmp_path, random_seed=-1).validate()
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        evaluate_dcn_scores(
            _split([0, 1], ["q1", "q1"]),
            np.asarray([-0.1, 1.1]),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
def test_runner_trains_with_empty_validation_and_writes_loadable_artifacts(
    tmp_path,
) -> None:
    import tensorflow as tf

    from spotify.dcn_v2_model import get_dcn_v2_custom_objects

    train = _split(
        [1, 0, 1, 0, 1, 0, 1, 0],
        ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"],
        candidate_ids=["a", "b", "c", "d", "e", "f", "g", "h"],
        event_times=list(range(8)),
    )
    test = _split(
        [1, 0, 0, 1],
        ["q5", "q5", "q6", "q6"],
        candidate_ids=["i", "j", "k", "l"],
        event_times=[20, 20, 21, 21],
    )
    result = train_dcn_v2_reranker(
        DCNTemporalDataset(train=train, validation=_empty_split(), test=test),
        DCNTrainingConfig(
            output_dir=tmp_path,
            epochs=2,
            batch_size=4,
            k_values=(1, 2),
            early_stopping_patience=0,
            model_params={
                "cross_layers": 1,
                "cross_parameterization": "vector",
                "deep_units": (4,),
                "dropout_rate": 0.0,
            },
        ),
    )

    checkpoint = tmp_path / "dcn_v2.keras"
    result_path = tmp_path / "dcn_v2_results.json"
    assert result["status"] == "completed"
    assert result["config"]["epochs_completed"] == 2
    assert result["input_split_summaries"]["train"]["row_count"] == 8
    assert result["metrics"]["validation"]["status"] == "unavailable"
    assert result["metrics"]["test"]["ranking"]["query_count"] == 2
    assert checkpoint.exists()
    assert json.loads(result_path.read_text())["model_name"] == "dcn_v2"

    restored = tf.keras.models.load_model(
        checkpoint,
        compile=False,
        custom_objects=get_dcn_v2_custom_objects(),
    )
    assert restored.output_shape == (None, 1)
