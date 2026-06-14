from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from spotify.dcn_v2_model import (
    build_dcn_v2_model,
    build_dcn_v2_reranker,
    get_dcn_v2_custom_objects,
)


def test_module_keeps_tensorflow_import_lazy() -> None:
    source = __import__("spotify.dcn_v2_model", fromlist=["unused"])

    assert source is sys.modules["spotify.dcn_v2_model"]
    assert "tensorflow" not in source.__dict__


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
@pytest.mark.parametrize("parameterization", ["matrix", "vector"])
def test_dcn_v2_model_has_expected_shapes_and_cross_parameters(
    parameterization: str,
) -> None:
    model = build_dcn_v2_model(
        num_context_features=4,
        num_item_features=3,
        params={
            "cross_layers": 2,
            "cross_parameterization": parameterization,
            "deep_units": (16, 8),
            "dropout_rate": 0.0,
        },
    )

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == [
        "context_input",
        "item_input",
    ]
    assert model.input_shape == [(None, 4), (None, 3)]
    assert model.output_shape == (None, 1)
    assert model.output_names == ["ranking_output"]
    assert model.get_layer("cross_layer_1").parameterization == parameterization
    assert model.get_layer("cross_layer_2").parameterization == parameterization
    assert model.get_layer("deep_dense_1").units == 16
    assert model.get_layer("deep_dense_2").units == 8

    kernel_shape = tuple(model.get_layer("cross_layer_1").kernel.shape)
    assert kernel_shape == ((7, 7) if parameterization == "matrix" else (7,))


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
@pytest.mark.parametrize("architecture", ["parallel", "stacked"])
def test_dcn_v2_forward_pass_returns_finite_ranking_probabilities(
    architecture: str,
) -> None:
    model = build_dcn_v2_reranker(
        num_context_features=3,
        num_item_features=2,
        params={
            "architecture": architecture,
            "deep_units": (12,),
            "dropout_rate": 0.0,
        },
    )
    context = np.array(
        [[0.1, 0.2, 0.3], [1.0, -0.5, 0.25], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    item = np.array(
        [[0.5, 1.0], [-1.0, 0.75], [0.25, 0.5]],
        dtype=np.float32,
    )

    predictions = model([context, item], training=False).numpy()

    assert predictions.shape == (3, 1)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0.0).all()
    assert (predictions <= 1.0).all()
    if architecture == "parallel":
        assert model.get_layer("cross_deep_fusion") is not None
    else:
        with pytest.raises(ValueError, match="No such layer"):
            model.get_layer("cross_deep_fusion")


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is not installed",
)
def test_cross_layer_config_and_model_artifact_round_trip(tmp_path) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(23)
    model = build_dcn_v2_model(
        num_context_features=3,
        num_item_features=2,
        params={
            "cross_layers": 2,
            "cross_parameterization": "vector",
            "deep_units": (10, 5),
            "activation": "tanh",
            "dropout_rate": 0.0,
            "architecture": "parallel",
            "l2_regularization": 0.001,
        },
    )
    cross_layer = model.get_layer("cross_layer_1")
    config = cross_layer.get_config()

    assert config["parameterization"] == "vector"
    assert config["use_bias"] is True
    assert config["kernel_regularizer"]["class_name"] == "L2"

    context = np.array([[0.1, 0.2, 0.3], [0.9, -0.4, 0.5]], dtype=np.float32)
    item = np.array([[0.25, 0.75], [-0.5, 1.0]], dtype=np.float32)
    expected = model([context, item], training=False).numpy()
    path = tmp_path / "dcn_v2.keras"
    model.save(path)

    restored = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_dcn_v2_custom_objects(),
    )
    actual = restored([context, item], training=False).numpy()

    assert restored.name == "dcn_v2_reranker"
    assert restored.get_layer("cross_layer_1").parameterization == "vector"
    assert restored.output_shape == (None, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("context_features", "item_features", "params", "message"),
    [
        (0, 2, None, "num_context_features"),
        (2, 0, None, "num_item_features"),
        (2, 2, {"cross_layers": 0}, "cross_layers"),
        (2, 2, {"cross_parameterization": "diagonal"}, "cross_parameterization"),
        (2, 2, {"deep_units": ()}, "deep_units"),
        (2, 2, {"deep_units": (8, 0)}, "deep_units"),
        (2, 2, {"dropout_rate": 1.0}, "dropout_rate"),
        (2, 2, {"architecture": "hybrid"}, "architecture"),
        (2, 2, {"l2_regularization": -0.1}, "l2_regularization"),
        (2, 2, {"surprise": True}, "unknown DCN-V2 parameters"),
    ],
)
def test_dcn_v2_rejects_invalid_configuration(
    context_features: int,
    item_features: int,
    params: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_dcn_v2_model(
            num_context_features=context_features,
            num_item_features=item_features,
            params=params,
        )
