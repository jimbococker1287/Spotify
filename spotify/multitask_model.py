from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TASK_NAMES = (
    "next_item_output",
    "skip_output",
    "dwell_output",
    "session_end_output",
    "explicit_positive_output",
    "repeat_output",
)
BINARY_TASK_NAMES = (
    "skip_output",
    "session_end_output",
    "explicit_positive_output",
    "repeat_output",
)

_CUSTOM_OBJECTS: dict[str, object] | None = None
_DEFAULT_LOSS_WEIGHTS = {
    "next_item_output": 1.0,
    "skip_output": 0.35,
    "dwell_output": 0.25,
    "session_end_output": 0.25,
    "explicit_positive_output": 0.35,
    "repeat_output": 0.25,
}
_ALLOWED_PARAMS = {
    "architecture",
    "context_dim",
    "dropout_rate",
    "embedding_dim",
    "expert_units",
    "fusion_dim",
    "learning_rate",
    "loss_weights",
    "num_experts",
    "sequence_dim",
    "sequence_encoder",
    "task_experts",
    "tower_units",
}


def get_multitask_custom_objects() -> dict[str, object]:
    """Return lazily defined, serializable multi-task routing layers."""
    global _CUSTOM_OBJECTS
    if _CUSTOM_OBJECTS is not None:
        return dict(_CUSTOM_OBJECTS)

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MMoE(tf.keras.layers.Layer):
        """Multi-gate mixture-of-experts block with one gate per task."""

        def __init__(
            self,
            num_tasks: int,
            num_experts: int = 4,
            expert_units: int = 64,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.num_tasks = _positive_int("num_tasks", num_tasks)
            self.num_experts = _positive_int("num_experts", num_experts)
            self.expert_units = _positive_int("expert_units", expert_units)
            self.activation = activation
            self.dropout_rate = _dropout_rate(dropout_rate)
            self.experts = [
                tf.keras.layers.Dense(
                    self.expert_units,
                    activation=self.activation,
                    name=f"expert_{index + 1}",
                )
                for index in range(self.num_experts)
            ]
            self.expert_dropout = [
                tf.keras.layers.Dropout(self.dropout_rate, name=f"expert_dropout_{index + 1}")
                for index in range(self.num_experts)
            ]
            self.gates = [
                tf.keras.layers.Dense(
                    self.num_experts,
                    activation="softmax",
                    name=f"task_gate_{index + 1}",
                )
                for index in range(self.num_tasks)
            ]

        def build(self, input_shape):
            for expert in self.experts:
                expert.build(input_shape)
            for gate in self.gates:
                gate.build(input_shape)
            super().build(input_shape)

        def call(self, inputs, training=None):
            expert_outputs = tf.stack(
                [
                    dropout(expert(inputs), training=training)
                    for expert, dropout in zip(self.experts, self.expert_dropout)
                ],
                axis=1,
            )
            return [
                tf.reduce_sum(expert_outputs * gate(inputs)[:, :, None], axis=1)
                for gate in self.gates
            ]

        def get_config(self):
            return {
                **super().get_config(),
                "num_tasks": self.num_tasks,
                "num_experts": self.num_experts,
                "expert_units": self.expert_units,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class PLE(tf.keras.layers.Layer):
        """Single-level progressive layered extraction for shared/task experts."""

        def __init__(
            self,
            num_tasks: int,
            num_experts: int = 2,
            task_experts: int = 2,
            expert_units: int = 64,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.num_tasks = _positive_int("num_tasks", num_tasks)
            self.num_experts = _positive_int("num_experts", num_experts)
            self.task_experts = _positive_int("task_experts", task_experts)
            self.expert_units = _positive_int("expert_units", expert_units)
            self.activation = activation
            self.dropout_rate = _dropout_rate(dropout_rate)
            self.shared_experts = self._make_experts(tf, "shared", self.num_experts)
            self.specific_experts = [
                self._make_experts(tf, f"task_{task_index + 1}", self.task_experts)
                for task_index in range(self.num_tasks)
            ]
            gate_width = self.num_experts + self.task_experts
            self.gates = [
                tf.keras.layers.Dense(
                    gate_width,
                    activation="softmax",
                    name=f"task_gate_{index + 1}",
                )
                for index in range(self.num_tasks)
            ]

        def build(self, input_shape):
            for expert in self.shared_experts:
                expert.build(input_shape)
            for task_experts in self.specific_experts:
                for expert in task_experts:
                    expert.build(input_shape)
            for gate in self.gates:
                gate.build(input_shape)
            super().build(input_shape)

        def _make_experts(self, tf_module, prefix: str, count: int):
            return [
                tf_module.keras.Sequential(
                    [
                        tf_module.keras.layers.Dense(
                            self.expert_units,
                            activation=self.activation,
                        ),
                        tf_module.keras.layers.Dropout(self.dropout_rate),
                    ],
                    name=f"{prefix}_expert_{index + 1}",
                )
                for index in range(count)
            ]

        def call(self, inputs, training=None):
            shared = [expert(inputs, training=training) for expert in self.shared_experts]
            task_outputs = []
            for specific_experts, gate in zip(self.specific_experts, self.gates):
                candidates = shared + [
                    expert(inputs, training=training) for expert in specific_experts
                ]
                stacked = tf.stack(candidates, axis=1)
                weights = gate(inputs)[:, :, None]
                task_outputs.append(tf.reduce_sum(stacked * weights, axis=1))
            return task_outputs

        def get_config(self):
            return {
                **super().get_config(),
                "num_tasks": self.num_tasks,
                "num_experts": self.num_experts,
                "task_experts": self.task_experts,
                "expert_units": self.expert_units,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            }

    _CUSTOM_OBJECTS = {
        "MMoE": MMoE,
        "PLE": PLE,
        "spotify>MMoE": MMoE,
        "spotify>PLE": PLE,
    }
    return dict(_CUSTOM_OBJECTS)


def get_multitask_compile_config(
    num_items: int,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Create task-specific Keras losses, weights, and metrics."""
    num_items = _positive_int("num_items", num_items, minimum=2)
    validated_weights = _validate_loss_weights(loss_weights)

    import tensorflow as tf

    losses: dict[str, object] = {
        "next_item_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        "dwell_output": tf.keras.losses.Huber(),
    }
    losses.update(
        {
            task_name: tf.keras.losses.BinaryCrossentropy()
            for task_name in BINARY_TASK_NAMES
        }
    )
    metrics: dict[str, list[object]] = {
        "next_item_output": [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(5, num_items),
                name=f"top_{min(5, num_items)}_accuracy",
            ),
        ],
        "dwell_output": [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    }
    metrics.update(
        {
            task_name: [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ]
            for task_name in BINARY_TASK_NAMES
        }
    )
    return {
        "loss": losses,
        "loss_weights": validated_weights,
        "metrics": metrics,
    }


def build_multitask_recommender(
    sequence_length: int,
    num_items: int,
    num_ctx: int,
    params: Mapping[str, Any] | None = None,
    *,
    compile_model: bool = True,
):
    """Build an MMoE or PLE sequence/context recommender with six task heads."""
    sequence_length = _positive_int("sequence_length", sequence_length)
    num_items = _positive_int("num_items", num_items, minimum=2)
    num_ctx = _positive_int("num_ctx", num_ctx)
    config = _validate_params(params)

    import tensorflow as tf
    from tensorflow.keras import Model, layers

    seq_input = layers.Input(
        shape=(sequence_length,),
        dtype="int32",
        name="seq_input",
    )
    ctx_input = layers.Input(
        shape=(num_ctx,),
        dtype="float32",
        name="ctx_input",
    )
    sequence = layers.Embedding(
        input_dim=num_items,
        output_dim=config["embedding_dim"],
        name="item_embedding",
    )(seq_input)
    if config["sequence_encoder"] == "gru":
        sequence = layers.GRU(
            config["sequence_dim"],
            name="sequence_encoder",
        )(sequence)
    else:
        sequence = layers.GlobalAveragePooling1D(name="sequence_encoder")(sequence)
        sequence = layers.Dense(
            config["sequence_dim"],
            activation="relu",
            name="sequence_projection",
        )(sequence)

    context = layers.Dense(
        config["context_dim"],
        activation="relu",
        name="context_projection",
    )(ctx_input)
    shared = layers.Concatenate(name="sequence_context_fusion")([sequence, context])
    shared = layers.Dense(
        config["fusion_dim"],
        activation="relu",
        name="shared_projection",
    )(shared)
    shared = layers.Dropout(config["dropout_rate"], name="shared_dropout")(shared)

    custom_objects = get_multitask_custom_objects()
    if config["architecture"] == "mmoe":
        routed = custom_objects["MMoE"](
            num_tasks=len(TASK_NAMES),
            num_experts=config["num_experts"],
            expert_units=config["expert_units"],
            dropout_rate=config["dropout_rate"],
            name="mmoe",
        )(shared)
    else:
        routed = custom_objects["PLE"](
            num_tasks=len(TASK_NAMES),
            num_experts=config["num_experts"],
            task_experts=config["task_experts"],
            expert_units=config["expert_units"],
            dropout_rate=config["dropout_rate"],
            name="ple",
        )(shared)

    tower_outputs = [
        layers.Dense(
            config["tower_units"],
            activation="relu",
            name=f"{task_name.removesuffix('_output')}_tower",
        )(task_representation)
        for task_name, task_representation in zip(TASK_NAMES, routed)
    ]
    outputs = [
        layers.Dense(
            num_items,
            activation="softmax",
            dtype="float32",
            name="next_item_output",
        )(tower_outputs[0]),
        layers.Dense(1, activation="sigmoid", dtype="float32", name="skip_output")(
            tower_outputs[1]
        ),
        layers.Dense(1, activation="sigmoid", dtype="float32", name="dwell_output")(
            tower_outputs[2]
        ),
        layers.Dense(
            1,
            activation="sigmoid",
            dtype="float32",
            name="session_end_output",
        )(tower_outputs[3]),
        layers.Dense(
            1,
            activation="sigmoid",
            dtype="float32",
            name="explicit_positive_output",
        )(tower_outputs[4]),
        layers.Dense(1, activation="sigmoid", dtype="float32", name="repeat_output")(
            tower_outputs[5]
        ),
    ]
    model = Model(
        inputs=[seq_input, ctx_input],
        outputs=outputs,
        name=f"{config['architecture']}_multitask_recommender",
    )
    if compile_model:
        compile_config = get_multitask_compile_config(
            num_items,
            config["loss_weights"],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
            run_eagerly=True,
            **compile_config,
        )
    return model


def _validate_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping or None")
    unknown = sorted(set(params) - _ALLOWED_PARAMS)
    if unknown:
        raise ValueError(f"unknown multi-task parameters: {', '.join(unknown)}")

    config: dict[str, Any] = {
        "architecture": str(params.get("architecture", "mmoe")).lower(),
        "sequence_encoder": str(params.get("sequence_encoder", "gru")).lower(),
        "embedding_dim": _positive_int("embedding_dim", params.get("embedding_dim", 64)),
        "sequence_dim": _positive_int("sequence_dim", params.get("sequence_dim", 64)),
        "context_dim": _positive_int("context_dim", params.get("context_dim", 32)),
        "fusion_dim": _positive_int("fusion_dim", params.get("fusion_dim", 96)),
        "num_experts": _positive_int("num_experts", params.get("num_experts", 4)),
        "task_experts": _positive_int("task_experts", params.get("task_experts", 2)),
        "expert_units": _positive_int("expert_units", params.get("expert_units", 64)),
        "tower_units": _positive_int("tower_units", params.get("tower_units", 32)),
        "dropout_rate": _dropout_rate(params.get("dropout_rate", 0.1)),
        "learning_rate": _positive_float("learning_rate", params.get("learning_rate", 1e-3)),
        "loss_weights": _validate_loss_weights(params.get("loss_weights")),
    }
    if config["architecture"] not in {"mmoe", "ple"}:
        raise ValueError("architecture must be either 'mmoe' or 'ple'")
    if config["sequence_encoder"] not in {"gru", "average"}:
        raise ValueError("sequence_encoder must be either 'gru' or 'average'")
    return config


def _validate_loss_weights(
    loss_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if loss_weights is None:
        return dict(_DEFAULT_LOSS_WEIGHTS)
    if not isinstance(loss_weights, Mapping):
        raise TypeError("loss_weights must be a mapping or None")
    unknown = sorted(set(loss_weights) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"unknown loss weight tasks: {', '.join(unknown)}")
    weights = dict(_DEFAULT_LOSS_WEIGHTS)
    for task_name, raw_weight in loss_weights.items():
        weights[task_name] = _positive_float(
            f"loss_weights[{task_name!r}]",
            raw_weight,
            allow_zero=True,
        )
    if not any(weights.values()):
        raise ValueError("at least one loss weight must be positive")
    return weights


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = f"at least {minimum}" if minimum > 1 else "positive"
        raise ValueError(f"{name} must be {qualifier} integer")
    return value


def _positive_float(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    import math

    lower_bound_valid = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_bound_valid:
        adjective = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {adjective} number")
    return result


def _dropout_rate(value: Any) -> float:
    result = _positive_float("dropout_rate", value, allow_zero=True)
    if result >= 1.0:
        raise ValueError("dropout_rate must be less than 1")
    return result


__all__ = [
    "BINARY_TASK_NAMES",
    "TASK_NAMES",
    "build_multitask_recommender",
    "get_multitask_compile_config",
    "get_multitask_custom_objects",
]
