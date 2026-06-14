from __future__ import annotations

from collections.abc import Mapping, Sequence


_CUSTOM_OBJECTS: dict[str, object] | None = None

_DEFAULT_PARAMS: dict[str, object] = {
    "cross_layers": 3,
    "cross_parameterization": "matrix",
    "deep_units": (128, 64),
    "activation": "relu",
    "dropout_rate": 0.1,
    "architecture": "parallel",
    "output_activation": "sigmoid",
    "l2_regularization": 0.0,
}


def get_dcn_v2_custom_objects() -> dict[str, object]:
    """Return lazily defined custom objects required to load DCN-V2 models."""
    global _CUSTOM_OBJECTS
    if _CUSTOM_OBJECTS is not None:
        return dict(_CUSTOM_OBJECTS)

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class DCNCrossLayer(tf.keras.layers.Layer):
        """An explicit feature cross using matrix or vector parameterization."""

        def __init__(
            self,
            parameterization: str = "matrix",
            use_bias: bool = True,
            kernel_initializer: object = "glorot_uniform",
            bias_initializer: object = "zeros",
            kernel_regularizer: object | None = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            if parameterization not in {"matrix", "vector"}:
                raise ValueError("parameterization must be 'matrix' or 'vector'")
            self.parameterization = parameterization
            self.use_bias = bool(use_bias)
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
            self.bias_initializer = tf.keras.initializers.get(bias_initializer)
            self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
            self.kernel = None
            self.bias = None

        def build(self, input_shape):
            if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
                raise ValueError("DCNCrossLayer expects [base_features, crossed_features]")

            base_shape = tf.TensorShape(input_shape[0])
            crossed_shape = tf.TensorShape(input_shape[1])
            if base_shape.rank != 2 or crossed_shape.rank != 2:
                raise ValueError("DCNCrossLayer inputs must both be rank-2 tensors")

            input_dim = base_shape[-1]
            crossed_dim = crossed_shape[-1]
            if input_dim is None or crossed_dim is None:
                raise ValueError("DCNCrossLayer requires a known final feature dimension")
            if int(input_dim) != int(crossed_dim):
                raise ValueError("DCNCrossLayer inputs must have the same feature dimension")

            kernel_shape = (
                (int(input_dim), int(input_dim))
                if self.parameterization == "matrix"
                else (int(input_dim),)
            )
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
            )
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=(int(input_dim),),
                    initializer=self.bias_initializer,
                    trainable=True,
                )
            super().build(input_shape)

        def call(self, inputs):
            base_features, crossed_features = inputs
            if self.parameterization == "matrix":
                projected = tf.linalg.matmul(crossed_features, self.kernel)
                if self.bias is not None:
                    projected = tf.nn.bias_add(projected, self.bias)
                crossed = base_features * projected
            else:
                scale = tf.reduce_sum(crossed_features * self.kernel, axis=-1, keepdims=True)
                crossed = base_features * scale
                if self.bias is not None:
                    crossed = crossed + self.bias
            return crossed + crossed_features

        def compute_output_shape(self, input_shape):
            return input_shape[0]

        def get_config(self):
            return {
                **super().get_config(),
                "parameterization": self.parameterization,
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            }

    _CUSTOM_OBJECTS = {
        "DCNCrossLayer": DCNCrossLayer,
        "spotify>DCNCrossLayer": DCNCrossLayer,
    }
    return dict(_CUSTOM_OBJECTS)


def _validated_params(params: Mapping[str, object] | None) -> dict[str, object]:
    if params is None:
        raw_params: dict[str, object] = {}
    elif isinstance(params, Mapping):
        raw_params = dict(params)
    else:
        raise ValueError("params must be a mapping or None")

    unknown = sorted(set(raw_params) - set(_DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown DCN-V2 parameters: {', '.join(unknown)}")

    validated = {**_DEFAULT_PARAMS, **raw_params}

    cross_layers = validated["cross_layers"]
    if isinstance(cross_layers, bool) or not isinstance(cross_layers, int) or cross_layers < 1:
        raise ValueError("cross_layers must be a positive integer")

    parameterization = validated["cross_parameterization"]
    if parameterization not in {"matrix", "vector"}:
        raise ValueError("cross_parameterization must be 'matrix' or 'vector'")

    deep_units = validated["deep_units"]
    if (
        isinstance(deep_units, (str, bytes))
        or not isinstance(deep_units, Sequence)
        or not deep_units
    ):
        raise ValueError("deep_units must be a non-empty sequence of positive integers")
    normalized_units: list[int] = []
    for unit_count in deep_units:
        if isinstance(unit_count, bool) or not isinstance(unit_count, int) or unit_count < 1:
            raise ValueError("deep_units must contain only positive integers")
        normalized_units.append(int(unit_count))
    validated["deep_units"] = tuple(normalized_units)

    for name in ("activation", "output_activation"):
        value = validated[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    dropout_rate = validated["dropout_rate"]
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)):
        raise ValueError("dropout_rate must be a number in [0, 1)")
    if not 0.0 <= float(dropout_rate) < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    validated["dropout_rate"] = float(dropout_rate)

    architecture = validated["architecture"]
    if architecture not in {"parallel", "stacked"}:
        raise ValueError("architecture must be 'parallel' or 'stacked'")

    l2_regularization = validated["l2_regularization"]
    if isinstance(l2_regularization, bool) or not isinstance(l2_regularization, (int, float)):
        raise ValueError("l2_regularization must be a non-negative number")
    if float(l2_regularization) < 0.0:
        raise ValueError("l2_regularization must be a non-negative number")
    validated["l2_regularization"] = float(l2_regularization)

    return validated


def _positive_feature_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_dcn_v2_model(
    num_context_features: int,
    num_item_features: int,
    params: Mapping[str, object] | None = None,
):
    """Build a DCN-V2 contextual reranker with explicit cross and deep towers."""
    num_context_features = _positive_feature_count(
        num_context_features,
        name="num_context_features",
    )
    num_item_features = _positive_feature_count(
        num_item_features,
        name="num_item_features",
    )
    config = _validated_params(params)

    from tensorflow.keras import Model, layers, regularizers

    custom_objects = get_dcn_v2_custom_objects()
    DCNCrossLayer = custom_objects["DCNCrossLayer"]
    regularizer = (
        regularizers.l2(config["l2_regularization"])
        if config["l2_regularization"] > 0.0
        else None
    )

    context_input = layers.Input(
        shape=(num_context_features,),
        dtype="float32",
        name="context_input",
    )
    item_input = layers.Input(
        shape=(num_item_features,),
        dtype="float32",
        name="item_input",
    )
    base_features = layers.Concatenate(name="feature_fusion")(
        [context_input, item_input]
    )

    crossed_features = base_features
    for layer_index in range(config["cross_layers"]):
        crossed_features = DCNCrossLayer(
            parameterization=config["cross_parameterization"],
            kernel_regularizer=regularizer,
            name=f"cross_layer_{layer_index + 1}",
        )([base_features, crossed_features])

    deep_features = (
        crossed_features if config["architecture"] == "stacked" else base_features
    )
    for layer_index, unit_count in enumerate(config["deep_units"], start=1):
        deep_features = layers.Dense(
            unit_count,
            activation=config["activation"],
            kernel_regularizer=regularizer,
            name=f"deep_dense_{layer_index}",
        )(deep_features)
        if config["dropout_rate"] > 0.0:
            deep_features = layers.Dropout(
                config["dropout_rate"],
                name=f"deep_dropout_{layer_index}",
            )(deep_features)

    ranking_features = (
        layers.Concatenate(name="cross_deep_fusion")(
            [crossed_features, deep_features]
        )
        if config["architecture"] == "parallel"
        else deep_features
    )
    ranking_output = layers.Dense(
        1,
        activation=config["output_activation"],
        dtype="float32",
        kernel_regularizer=regularizer,
        name="ranking_output",
    )(ranking_features)

    model = Model(
        inputs=[context_input, item_input],
        outputs=ranking_output,
        name="dcn_v2_reranker",
    )
    model.dcn_v2_config = {
        "num_context_features": num_context_features,
        "num_item_features": num_item_features,
        **config,
    }
    return model


def build_dcn_v2_reranker(
    num_context_features: int,
    num_item_features: int,
    params: Mapping[str, object] | None = None,
):
    """Alias emphasizing the model's role as a contextual reranker."""
    return build_dcn_v2_model(
        num_context_features=num_context_features,
        num_item_features=num_item_features,
        params=params,
    )


__all__ = [
    "build_dcn_v2_model",
    "build_dcn_v2_reranker",
    "get_dcn_v2_custom_objects",
]
