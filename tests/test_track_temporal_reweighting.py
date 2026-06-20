from __future__ import annotations

import json

import numpy as np
import pytest

from spotify.track_dcn_training import DCNCandidateSplit, DCNTemporalDataset
from spotify.track_temporal_reweighting import (
    DCNTemporalReweightingConfig,
    apply_temporal_reweighting,
    compute_normalized_recency_weights,
)


def _split(
    labels: list[int],
    query_ids: list[str],
    *,
    event_times: list[int] | None = None,
    sample_weights: list[float] | None = None,
) -> DCNCandidateSplit:
    row_count = len(labels)
    values = np.arange(row_count, dtype="float32")
    return DCNCandidateSplit(
        context_features=np.column_stack((values, values + 10.0)),
        item_features=np.column_stack((values / 10.0, np.ones(row_count))),
        labels=np.asarray(labels, dtype="float32"),
        query_ids=np.asarray(query_ids, dtype=object),
        candidate_ids=np.asarray([f"candidate-{index}" for index in range(row_count)], dtype=object),
        event_times=None if event_times is None else np.asarray(event_times, dtype="int64"),
        sample_weights=None if sample_weights is None else np.asarray(sample_weights, dtype="float32"),
    )


def test_event_times_produce_bounded_mean_normalized_weights() -> None:
    config = DCNTemporalReweightingConfig(min_weight=0.75, max_weight=1.5, half_life=10.0)

    result = compute_normalized_recency_weights(
        [10, 20, 30],
        row_count=3,
        query_ids=["q1", "q1", "q2"],
        config=config,
    )

    assert result.method == "event_times"
    assert result.fallback is None
    assert result.source_min == pytest.approx(10.0)
    assert result.source_max == pytest.approx(30.0)
    assert result.weights[0] < result.weights[1] < result.weights[2]
    assert float(result.weights.min()) >= 0.75
    assert float(result.weights.max()) <= 1.5
    assert result.weights.mean() == pytest.approx(1.0)
    json.dumps(result.to_dict())


def test_row_order_fallback_is_stable_and_explicit() -> None:
    config = DCNTemporalReweightingConfig(
        min_weight=0.5,
        max_weight=1.5,
        strength=2.0,
        fallback="row_order",
    )

    first = compute_normalized_recency_weights(None, row_count=4, config=config)
    second = compute_normalized_recency_weights(None, row_count=4, config=config)

    assert first.method == "fallback"
    assert first.fallback == "row_order"
    np.testing.assert_allclose(first.weights, second.weights)
    assert first.weights[0] < first.weights[-1]
    assert first.weights.mean() == pytest.approx(1.0)


def test_query_order_fallback_groups_candidates_by_first_query_appearance() -> None:
    result = compute_normalized_recency_weights(
        None,
        query_ids=["q1", "q1", "q2", "q3", "q3"],
        config=DCNTemporalReweightingConfig(fallback="query_order"),
    )

    assert result.method == "fallback"
    assert result.fallback == "query_order"
    assert result.weights[0] == pytest.approx(result.weights[1])
    assert result.weights[3] == pytest.approx(result.weights[4])
    assert result.weights[0] < result.weights[2] < result.weights[3]


def test_apply_temporal_reweighting_multiplies_existing_train_sample_weights_only() -> None:
    train = _split(
        [0, 1, 0],
        ["q1", "q1", "q2"],
        event_times=[100, 101, 102],
        sample_weights=[2.0, 1.0, 0.5],
    )
    validation = _split([0, 1], ["v1", "v1"], sample_weights=[3.0, 4.0])
    test = _split([1, 0], ["t1", "t1"], sample_weights=[5.0, 6.0])
    dataset = DCNTemporalDataset(train=train, validation=validation, test=test)
    config = DCNTemporalReweightingConfig(min_weight=0.75, max_weight=1.25)

    recency = compute_normalized_recency_weights(
        train.event_times,
        row_count=len(train),
        query_ids=train.query_ids,
        config=config,
    ).weights
    reweighted = apply_temporal_reweighting(dataset, config)

    assert reweighted is not dataset
    assert reweighted.train is not train
    np.testing.assert_allclose(
        reweighted.train.sample_weights,
        np.asarray([2.0, 1.0, 0.5], dtype="float32") * recency,
    )
    np.testing.assert_allclose(train.sample_weights, [2.0, 1.0, 0.5])
    np.testing.assert_allclose(reweighted.validation.context_features, validation.context_features)
    np.testing.assert_allclose(reweighted.validation.labels, validation.labels)
    np.testing.assert_allclose(reweighted.validation.sample_weights, validation.sample_weights)
    np.testing.assert_allclose(reweighted.test.item_features, test.item_features)
    np.testing.assert_allclose(reweighted.test.labels, test.labels)
    np.testing.assert_allclose(reweighted.test.sample_weights, test.sample_weights)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (DCNTemporalReweightingConfig(min_weight=0.0), "min_weight"),
        (DCNTemporalReweightingConfig(min_weight=1.2, max_weight=2.0), "bound"),
        (DCNTemporalReweightingConfig(max_weight=0.9), "bound"),
        (DCNTemporalReweightingConfig(half_life=0.0), "half_life"),
        (DCNTemporalReweightingConfig(strength=0.0), "strength"),
        (DCNTemporalReweightingConfig(fallback="unknown"), "fallback"),
    ],
)
def test_invalid_config_is_rejected(
    config: DCNTemporalReweightingConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_normalized_recency_weights([1, 2, 3], config=config)


def test_invalid_source_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="event_times has 2 rows; expected 3"):
        compute_normalized_recency_weights([1, 2], row_count=3)
    with pytest.raises(ValueError, match="query_ids are required"):
        compute_normalized_recency_weights(
            None,
            row_count=3,
            config=DCNTemporalReweightingConfig(fallback="query_order"),
        )
