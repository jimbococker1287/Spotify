from __future__ import annotations

import importlib
import importlib.util
import sys

import numpy as np
import pytest


def test_tensorflow_import_is_lazy() -> None:
    sys.modules.pop("spotify.multitask_model", None)
    module = importlib.import_module("spotify.multitask_model")

    assert "tensorflow" not in module.__dict__


def test_builder_validates_configuration_before_tensorflow_use() -> None:
    from spotify.multitask_model import build_multitask_recommender

    with pytest.raises(ValueError, match="num_items"):
        build_multitask_recommender(4, 1, 2)
    with pytest.raises(ValueError, match="architecture"):
        build_multitask_recommender(4, 8, 2, {"architecture": "shared_bottom"})
    with pytest.raises(ValueError, match="unknown multi-task parameters"):
        build_multitask_recommender(4, 8, 2, {"mystery": 4})
    with pytest.raises(ValueError, match="unknown loss weight tasks"):
        build_multitask_recommender(
            4,
            8,
            2,
            {"loss_weights": {"missing_output": 1.0}},
        )


tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow is required for multi-task model tests",
)

from spotify.multitask_model import (  # noqa: E402
    TASK_NAMES,
    build_multitask_recommender,
    get_multitask_custom_objects,
)


@pytest.mark.parametrize("architecture", ["mmoe", "ple"])
def test_multitask_architectures_have_named_task_outputs(architecture: str) -> None:
    model = build_multitask_recommender(
        sequence_length=5,
        num_items=13,
        num_ctx=3,
        params={
            "architecture": architecture,
            "embedding_dim": 8,
            "sequence_dim": 8,
            "context_dim": 4,
            "fusion_dim": 12,
            "num_experts": 2,
            "task_experts": 1,
            "expert_units": 8,
            "tower_units": 4,
            "dropout_rate": 0.0,
        },
        compile_model=False,
    )

    assert [tensor.name.split(":")[0] for tensor in model.inputs] == [
        "seq_input",
        "ctx_input",
    ]
    assert model.output_names == list(TASK_NAMES)
    assert model.output_shape == [
        (None, 13),
        (None, 1),
        (None, 1),
        (None, 1),
        (None, 1),
        (None, 1),
    ]
    assert model.get_layer(architecture) is not None


def test_forward_pass_produces_bounded_task_predictions() -> None:
    tf.keras.utils.set_random_seed(17)
    model = build_multitask_recommender(
        sequence_length=4,
        num_items=9,
        num_ctx=2,
        params={
            "sequence_encoder": "average",
            "embedding_dim": 8,
            "sequence_dim": 8,
            "context_dim": 4,
            "fusion_dim": 12,
            "num_experts": 2,
            "expert_units": 8,
            "tower_units": 4,
            "dropout_rate": 0.0,
        },
        compile_model=False,
    )
    sequences = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    context = np.asarray([[0.25, 1.0], [0.75, 0.0]], dtype=np.float32)

    predictions = model([sequences, context], training=False)

    assert len(predictions) == len(TASK_NAMES)
    next_item = predictions[0].numpy()
    assert next_item.shape == (2, 9)
    np.testing.assert_allclose(next_item.sum(axis=1), np.ones(2), atol=1e-6)
    for prediction in predictions[1:]:
        values = prediction.numpy()
        assert values.shape == (2, 1)
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_compiled_model_trains_all_task_heads() -> None:
    model = build_multitask_recommender(
        sequence_length=3,
        num_items=7,
        num_ctx=2,
        params={
            "embedding_dim": 4,
            "sequence_dim": 4,
            "context_dim": 3,
            "fusion_dim": 6,
            "num_experts": 2,
            "expert_units": 4,
            "tower_units": 3,
            "dropout_rate": 0.0,
        },
    )
    inputs = {
        "seq_input": np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int32),
        "ctx_input": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    }
    targets = {
        "next_item_output": np.asarray([4, 5], dtype=np.int32),
        "skip_output": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "dwell_output": np.asarray([[0.8], [0.2]], dtype=np.float32),
        "session_end_output": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "explicit_positive_output": np.asarray([[1.0], [0.0]], dtype=np.float32),
        "repeat_output": np.asarray([[0.0], [1.0]], dtype=np.float32),
    }
    sample_weights = {
        task_name: np.ones(2, dtype=np.float32) for task_name in TASK_NAMES
    }

    result = model.train_on_batch(
        inputs,
        [targets[task_name] for task_name in model.output_names],
        sample_weight=[
            sample_weights[task_name] for task_name in model.output_names
        ],
        return_dict=True,
    )

    assert model.run_eagerly is True
    assert np.isfinite(result["loss"])
    for task_name in TASK_NAMES:
        assert f"{task_name}_loss" in result


@pytest.mark.parametrize("architecture", ["mmoe", "ple"])
def test_routing_models_round_trip_through_keras_artifact(
    tmp_path,
    architecture: str,
) -> None:
    model = build_multitask_recommender(
        sequence_length=3,
        num_items=7,
        num_ctx=2,
        params={
            "architecture": architecture,
            "embedding_dim": 4,
            "sequence_dim": 4,
            "context_dim": 3,
            "fusion_dim": 6,
            "num_experts": 2,
            "task_experts": 1,
            "expert_units": 4,
            "tower_units": 3,
            "dropout_rate": 0.0,
        },
        compile_model=False,
    )
    path = tmp_path / f"{architecture}_multitask.keras"
    model.save(path)

    restored = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_multitask_custom_objects(),
    )

    assert restored.output_names == list(TASK_NAMES)
    assert restored.get_layer(architecture).num_tasks == len(TASK_NAMES)
