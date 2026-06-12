from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from spotify.bert4rec_model import (
    bert4rec_mask_token_id,
    bert4rec_vocabulary_size,
    build_bert4rec_model,
    build_cloze_pretraining_batch,
)


def test_cloze_batch_is_deterministic_and_masks_only_reserved_token() -> None:
    sequences = np.array(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 1, 2, 2, 3],
        ],
        dtype="int32",
    )
    contexts = np.arange(9, dtype="float32").reshape(3, 3)

    first = build_cloze_pretraining_batch(
        sequences,
        contexts,
        num_artists=5,
        mask_probability=0.4,
        seed=17,
    )
    second = build_cloze_pretraining_batch(
        sequences,
        contexts,
        num_artists=5,
        mask_probability=0.4,
        seed=17,
    )

    for first_value, second_value in zip(first, second):
        np.testing.assert_array_equal(first_value, second_value)

    mask_token_id = bert4rec_mask_token_id(5)
    mask_locations = first.seq_input == mask_token_id
    assert mask_token_id == 5
    assert bert4rec_vocabulary_size(5) == 6
    assert np.all(mask_locations.sum(axis=1) == 1)
    assert np.all(first.artist_output >= 0)
    assert np.all(first.artist_output < 5)
    np.testing.assert_array_equal(
        first.seq_input[np.arange(len(first.seq_input)), first.masked_positions],
        np.full(len(first.seq_input), mask_token_id),
    )


def test_cloze_batch_expands_context_and_preserves_masked_targets() -> None:
    sequences = np.array([[0, 1, 2], [3, 2, 1]], dtype="int32")
    contexts = np.array([[0.25, 0.5], [0.75, 1.0]], dtype="float32")

    batch = build_cloze_pretraining_batch(
        sequences,
        contexts,
        num_artists=4,
        mask_probability=1.0,
        seed=9,
    )

    assert batch.seq_input.shape == (6, 3)
    assert batch.ctx_input.shape == (6, 2)
    assert batch.artist_output.shape == (6,)
    assert batch.masked_positions.shape == (6,)
    np.testing.assert_array_equal(batch.artist_output, sequences.reshape(-1))
    np.testing.assert_array_equal(batch.ctx_input[:3], np.repeat(contexts[:1], 3, axis=0))
    np.testing.assert_array_equal(batch.ctx_input[3:], np.repeat(contexts[1:], 3, axis=0))


def test_cloze_batch_rejects_reserved_or_invalid_artist_ids() -> None:
    contexts = np.zeros((1, 1), dtype="float32")

    with pytest.raises(ValueError, match="artist IDs"):
        build_cloze_pretraining_batch([[0, 3]], contexts, num_artists=3)
    with pytest.raises(ValueError, match="artist IDs"):
        build_cloze_pretraining_batch([[-1, 1]], contexts, num_artists=3)


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
def test_bert4rec_model_shapes_vocabulary_and_forward_pass() -> None:
    model = build_bert4rec_model(sequence_length=4, num_artists=7, num_ctx=3)

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == [
        "seq_input",
        "ctx_input",
    ]
    assert model.output_names == ["artist_output"]
    assert model.output_shape == (None, 7)
    assert model.get_layer("item_embedding").input_dim == 8
    assert model.get_layer("position_embedding").input_dim == 4
    assert model.mask_token_id == 7
    assert model.item_vocabulary_size == 8

    seq_input = np.array([[0, 1, 7, 3], [4, 7, 5, 6]], dtype="int32")
    ctx_input = np.zeros((2, 3), dtype="float32")
    predictions = model([seq_input, ctx_input], training=False).numpy()

    assert predictions.shape == (2, 7)
    assert np.all(np.isfinite(predictions))
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)
