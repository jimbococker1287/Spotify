from __future__ import annotations

import numpy as np
import pytest

from spotify.sasrec_model import build_sasrec_model

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for SASRec model tests")


def test_sasrec_model_has_standard_input_and_output_shapes() -> None:
    model = build_sasrec_model(sequence_length=6, num_artists=17, num_ctx=4)

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == ["seq_input", "ctx_input"]
    assert model.output_names == ["artist_output"]
    assert model.input_shape == [(None, 6), (None, 4)]
    assert model.output_shape == (None, 17)
    assert model.get_layer("item_embedding").output_dim == 128
    assert model.get_layer("position_embedding").input_dim == 6


def test_sasrec_uses_positions_and_causal_attention() -> None:
    tf.keras.utils.set_random_seed(7)
    model = build_sasrec_model(sequence_length=5, num_artists=12, num_ctx=3)

    item_embedding = model.get_layer("item_embedding")
    item_embedding.set_weights([np.zeros_like(item_embedding.get_weights()[0])])
    position_embedding = model.get_layer("position_embedding")
    position_weights = np.repeat(
        np.arange(5, dtype=np.float32)[:, np.newaxis],
        position_embedding.output_dim,
        axis=1,
    )
    position_embedding.set_weights([position_weights])

    embedding_model = tf.keras.Model(
        model.inputs,
        model.get_layer("item_position_embeddings").output,
    )
    repeated_items = np.full((1, 5), 3, dtype=np.int32)
    context = np.zeros((1, 3), dtype=np.float32)
    embedded = embedding_model([repeated_items, context], training=False).numpy()

    assert not np.allclose(embedded[:, 0, :], embedded[:, 1, :])

    item_weights = np.random.default_rng(7).normal(
        size=item_embedding.get_weights()[0].shape,
    ).astype(np.float32)
    item_embedding.set_weights([item_weights])

    sequence_model = tf.keras.Model(
        model.inputs,
        model.get_layer("sasrec_sequence_output").output,
    )
    shared_prefix = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    changed_future = np.array([[1, 2, 8, 9, 10]], dtype=np.int32)
    prefix_output = sequence_model([shared_prefix, context], training=False).numpy()
    changed_output = sequence_model([changed_future, context], training=False).numpy()

    np.testing.assert_allclose(prefix_output[:, :2, :], changed_output[:, :2, :], atol=1e-6)
    assert not np.allclose(prefix_output[:, 2:, :], changed_output[:, 2:, :])


def test_sasrec_forward_pass_returns_artist_probabilities() -> None:
    tf.keras.utils.set_random_seed(11)
    model = build_sasrec_model(sequence_length=4, num_artists=9, num_ctx=2)
    sequences = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    context = np.array([[0.25, 1.0], [0.75, 0.0]], dtype=np.float32)

    predictions = model([sequences, context], training=False).numpy()

    assert predictions.shape == (2, 9)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0.0).all()
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-6)
