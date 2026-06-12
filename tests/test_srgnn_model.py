from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from spotify.srgnn_model import build_session_graph, build_srgnn_model, get_srgnn_custom_objects


def test_module_keeps_tensorflow_import_lazy() -> None:
    source = __import__("spotify.srgnn_model", fromlist=["unused"])
    assert source is sys.modules["spotify.srgnn_model"]
    assert "tensorflow" not in source.__dict__


def test_session_graph_collapses_repeated_items_and_preserves_direction() -> None:
    graph = build_session_graph([4, 8, 4, 9])

    np.testing.assert_array_equal(graph.node_items, [4, 8, 9, 0])
    np.testing.assert_array_equal(graph.alias_inputs, [0, 1, 0, 2])
    np.testing.assert_allclose(
        graph.adjacency_out,
        [
            [0.0, 0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_allclose(
        graph.adjacency_in,
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )


def test_session_graph_accumulates_and_normalizes_transition_counts() -> None:
    graph = build_session_graph([1, 2, 1, 2, 1, 2, 3])

    np.testing.assert_array_equal(graph.alias_inputs, [0, 1, 0, 1, 0, 1, 2])
    np.testing.assert_allclose(graph.adjacency_out[0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(graph.adjacency_out[1], [2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(graph.adjacency_in[1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed")
def test_srgnn_model_has_expected_shapes_and_names() -> None:
    model = build_srgnn_model(sequence_length=5, num_artists=11, num_ctx=3)

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == ["seq_input", "ctx_input"]
    assert model.output_shape == (None, 11)
    assert model.output_names == ["artist_output"]
    assert model.get_layer("session_graph_message_passing") is not None
    assert model.get_layer("session_preference_attention") is not None


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed")
def test_srgnn_model_forward_pass_produces_artist_distribution() -> None:
    model = build_srgnn_model(sequence_length=5, num_artists=11, num_ctx=3)
    sequence = np.asarray([[1, 2, 1, 3, 4], [5, 6, 7, 6, 8]], dtype=np.int32)
    context = np.asarray([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]], dtype=np.float32)

    output = model([sequence, context], training=False).numpy()

    assert output.shape == (2, 11)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), np.ones(2), atol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed")
def test_srgnn_model_round_trips_through_keras_artifact(tmp_path) -> None:
    import tensorflow as tf

    model = build_srgnn_model(sequence_length=5, num_artists=11, num_ctx=3)
    path = tmp_path / "srgnn.keras"
    model.save(path)

    restored = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_srgnn_custom_objects(),
    )

    assert restored.output_shape == (None, 11)
