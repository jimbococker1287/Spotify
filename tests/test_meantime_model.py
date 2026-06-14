from __future__ import annotations

import numpy as np
import pytest

from spotify.meantime_model import (
    build_meantime_model,
    get_meantime_custom_objects,
)

tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow is required for MEANTIME model tests",
)


def _small_model():
    return build_meantime_model(
        sequence_length=5,
        vocabulary_size=13,
        num_ctx=3,
        params={
            "embedding_dim": 16,
            "num_heads": 4,
            "feed_forward_dim": 32,
            "dropout_rate": 0.0,
            "num_blocks": 1,
            "num_time_buckets": 8,
            "max_time_gap": 100.0,
            "context_dim": 8,
        },
    )


def _inputs():
    sequences = np.array(
        [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]],
        dtype=np.int32,
    )
    time_gaps = np.array(
        [[0.0, 1.0, 5.0, 0.0, 0.0], [0.0, 2.0, 8.0, 20.0, 50.0]],
        dtype=np.float32,
    )
    context = np.array(
        [[0.25, 1.0, -0.5], [0.75, 0.0, 0.5]],
        dtype=np.float32,
    )
    return sequences, time_gaps, context


def test_meantime_model_shapes_and_temporal_layers() -> None:
    model = _small_model()

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == [
        "item_sequence",
        "time_gap_input",
        "context",
    ]
    assert model.input_shape == [(None, 5), (None, 5), (None, 3)]
    assert model.output_shape == (None, 13)
    assert model.output_names == ["next_item_output"]
    assert model.get_layer("item_embedding").input_dim == 13
    assert model.get_layer("absolute_position_embedding") is not None
    assert model.get_layer("temporal_gap_embedding").num_time_buckets == 8
    assert model.get_layer("temporal_causal_attention_1") is not None
    assert model.padding_token_id == 0


def test_meantime_forward_pass_masks_padding_prediction() -> None:
    tf.keras.utils.set_random_seed(17)
    model = _small_model()

    predictions = model(_inputs(), training=False).numpy()

    assert predictions.shape == (2, 13)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0.0).all()
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-6)
    np.testing.assert_allclose(predictions[:, 0], np.zeros(2), atol=1e-7)


def test_temporal_embeddings_and_predictions_respond_to_time_gaps() -> None:
    tf.keras.utils.set_random_seed(23)
    model = _small_model()
    temporal_layer = model.get_layer("temporal_gap_embedding")
    embedding_weights = np.repeat(
        np.arange(8, dtype=np.float32)[:, np.newaxis],
        16,
        axis=1,
    )
    temporal_layer.set_weights([embedding_weights])
    temporal_model = tf.keras.Model(
        inputs=model.inputs[:2],
        outputs=temporal_layer.output,
    )

    sequence = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    short_gaps = np.zeros((1, 5), dtype=np.float32)
    long_gaps = np.array([[0.0, 100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

    short_embedding = temporal_model(
        [sequence, short_gaps],
        training=False,
    ).numpy()
    long_embedding = temporal_model(
        [sequence, long_gaps],
        training=False,
    ).numpy()
    assert not np.allclose(short_embedding, long_embedding)

    context = np.zeros((1, 3), dtype=np.float32)
    short_prediction = model([sequence, short_gaps, context], training=False).numpy()
    long_prediction = model([sequence, long_gaps, context], training=False).numpy()
    assert not np.allclose(short_prediction, long_prediction)


def test_causal_attention_hides_future_items_and_time_gaps() -> None:
    tf.keras.utils.set_random_seed(29)
    model = _small_model()
    sequence_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("meantime_sequence_output").output,
    )
    context = np.zeros((1, 3), dtype=np.float32)
    first_items = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    changed_future_items = np.array([[1, 2, 3, 9, 10]], dtype=np.int32)
    first_gaps = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    changed_future_gaps = np.array(
        [[0.0, 1.0, 2.0, 80.0, 90.0]],
        dtype=np.float32,
    )

    first_output = sequence_model(
        [first_items, first_gaps, context],
        training=False,
    ).numpy()
    changed_output = sequence_model(
        [changed_future_items, changed_future_gaps, context],
        training=False,
    ).numpy()

    np.testing.assert_allclose(
        first_output[:, :3, :],
        changed_output[:, :3, :],
        atol=1e-6,
    )
    assert not np.allclose(first_output[:, 3:, :], changed_output[:, 3:, :])


def test_padding_positions_do_not_change_prediction() -> None:
    tf.keras.utils.set_random_seed(31)
    model = _small_model()
    sequence = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
    context = np.zeros((1, 3), dtype=np.float32)
    ordinary_padding_gaps = np.array(
        [[0.0, 2.0, 4.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    arbitrary_padding_gaps = np.array(
        [[0.0, 2.0, 4.0, 75.0, 100.0]],
        dtype=np.float32,
    )

    ordinary = model(
        [sequence, ordinary_padding_gaps, context],
        training=False,
    ).numpy()
    arbitrary = model(
        [sequence, arbitrary_padding_gaps, context],
        training=False,
    ).numpy()

    np.testing.assert_allclose(ordinary, arbitrary, atol=1e-6)


def test_meantime_validates_configuration_and_time_gaps() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        build_meantime_model(0, 10, 2)
    with pytest.raises(ValueError, match="vocabulary_size"):
        build_meantime_model(5, 2, 2)
    with pytest.raises(ValueError, match="divisible"):
        build_meantime_model(
            5,
            10,
            2,
            params={"embedding_dim": 10, "num_heads": 4},
        )

    model = _small_model()
    sequence, time_gaps, context = _inputs()
    time_gaps[0, 1] = -1.0
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonnegative"):
        model([sequence, time_gaps, context], training=False)


def test_meantime_round_trips_through_keras_artifact(tmp_path) -> None:
    tf.keras.utils.set_random_seed(37)
    model = _small_model()
    expected = model(_inputs(), training=False).numpy()
    path = tmp_path / "meantime.keras"
    model.save(path)

    restored = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_meantime_custom_objects(),
    )
    actual = restored(_inputs(), training=False).numpy()

    assert restored.output_shape == (None, 13)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
