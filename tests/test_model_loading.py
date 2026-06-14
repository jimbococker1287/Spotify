from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from spotify.model_loading import load_trusted_keras_model
from spotify.modeling import build_model_builders


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
def test_memory_network_checkpoint_round_trip(tmp_path) -> None:
    builders = dict(
        build_model_builders(
            sequence_length=4,
            num_artists=7,
            num_ctx=3,
            selected_names=("memory_net_artist",),
        )
    )
    model = builders["memory_net_artist"]()
    model_path = tmp_path / "best_memory_net_artist.keras"
    model.save(model_path)

    loaded = load_trusted_keras_model(
        model_path,
        model_name="memory_net_artist",
        compile=False,
    )
    seq_input = np.array([[0, 1, 2, 3], [3, 4, 5, 6]], dtype="int32")
    ctx_input = np.zeros((2, 3), dtype="float32")
    predictions = loaded([seq_input, ctx_input], training=False).numpy()

    assert predictions.shape == (2, 7)
    assert np.all(np.isfinite(predictions))
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
def test_sasrec_checkpoint_round_trip(tmp_path) -> None:
    builders = dict(
        build_model_builders(
            sequence_length=4,
            num_artists=7,
            num_ctx=3,
            selected_names=("sasrec",),
        )
    )
    model = builders["sasrec"]()
    model_path = tmp_path / "best_sasrec.keras"
    model.save(model_path)

    loaded = load_trusted_keras_model(
        model_path,
        model_name="sasrec",
        compile=False,
    )
    seq_input = np.array([[0, 1, 2, 3], [3, 4, 5, 6]], dtype="int32")
    ctx_input = np.zeros((2, 3), dtype="float32")
    predictions = loaded([seq_input, ctx_input], training=False).numpy()

    assert predictions.shape == (2, 7)
    assert np.all(np.isfinite(predictions))
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)
