from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


MODALITY_NAMES = ("audio", "metadata", "collaborative")

_CUSTOM_OBJECTS: dict[str, object] | None = None
_DEFAULT_PARAMS: dict[str, object] = {
    "activation": "gelu",
    "dropout_rate": 0.0,
    "modality_weights": (1.0, 1.0, 1.0),
    "trainable_modality_weights": True,
    "projection_use_bias": True,
    "audio_projection_trainable": True,
    "metadata_projection_trainable": True,
    "collaborative_projection_trainable": True,
    "l2_regularization": 0.0,
    "normalization_epsilon": 1e-8,
}


def infer_modality_mask(
    audio_embeddings: object | None = None,
    metadata_features: object | None = None,
    collaborative_embeddings: object | None = None,
    *,
    num_items: int | None = None,
) -> np.ndarray:
    """Return row-level availability for audio, metadata, and collaborative inputs.

    A modality is available only when its complete row is finite. Callers can use
    NaN rows for unavailable features and combine this inferred mask with a more
    restrictive explicit mask in :func:`fuse_track_representations`.
    """
    matrices = _validate_feature_matrices(
        audio_embeddings,
        metadata_features,
        collaborative_embeddings,
        num_items=num_items,
    )
    row_count = _resolve_row_count(matrices, num_items=num_items)
    mask = np.zeros((row_count, len(MODALITY_NAMES)), dtype=np.float32)
    for modality_index, matrix in enumerate(matrices):
        if matrix is not None:
            mask[:, modality_index] = np.all(np.isfinite(matrix), axis=1)
    return mask


def fuse_track_representations(
    audio_embeddings: object | None = None,
    metadata_features: object | None = None,
    collaborative_embeddings: object | None = None,
    *,
    projection_matrices: Mapping[str, object] | None = None,
    modality_mask: object | None = None,
    modality_weights: Mapping[str, float] | Sequence[float] | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Fuse projected track features into L2-normalized NumPy representations.

    Inputs may have different widths when a projection matrix shaped
    ``(input_width, output_width)`` is supplied for each differing modality.
    Missing rows may be represented by NaNs or by zeros in ``modality_mask``.
    A row with no available modality produces a zero representation.
    """
    matrices = _validate_feature_matrices(
        audio_embeddings,
        metadata_features,
        collaborative_embeddings,
    )
    if all(matrix is None for matrix in matrices):
        raise ValueError("at least one multimodal feature matrix is required")
    row_count = _resolve_row_count(matrices)
    projections = _validate_projection_matrices(projection_matrices)
    weights = _validate_modality_weights(modality_weights)
    epsilon = _positive_float("epsilon", epsilon)

    inferred_mask = infer_modality_mask(
        audio_embeddings,
        metadata_features,
        collaborative_embeddings,
        num_items=row_count,
    )
    if modality_mask is None:
        combined_mask = inferred_mask
    else:
        combined_mask = inferred_mask * _validate_numpy_mask(
            modality_mask,
            row_count=row_count,
        )

    projected: list[np.ndarray | None] = []
    output_dim: int | None = None
    for modality_name, matrix in zip(MODALITY_NAMES, matrices):
        if matrix is None:
            projected.append(None)
            continue
        cleaned = np.nan_to_num(
            matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        projection = projections.get(modality_name)
        if projection is not None:
            if projection.shape[0] != cleaned.shape[1]:
                raise ValueError(
                    f"{modality_name} projection input dimension "
                    f"{projection.shape[0]} does not match feature dimension "
                    f"{cleaned.shape[1]}"
                )
            cleaned = cleaned @ projection
        if output_dim is None:
            output_dim = int(cleaned.shape[1])
        elif cleaned.shape[1] != output_dim:
            raise ValueError(
                "all projected modalities must have the same output dimension; "
                "provide projection_matrices for differently sized features"
            )
        projected.append(cleaned.astype(np.float32, copy=False))

    if output_dim is None:
        raise ValueError("at least one multimodal feature matrix is required")

    fused = np.zeros((row_count, output_dim), dtype=np.float32)
    denominator = np.zeros((row_count, 1), dtype=np.float32)
    for modality_index, values in enumerate(projected):
        if values is None:
            continue
        row_weights = (
            combined_mask[:, modality_index : modality_index + 1]
            * weights[modality_index]
        )
        fused += values * row_weights
        denominator += row_weights
    fused = np.divide(
        fused,
        denominator,
        out=np.zeros_like(fused),
        where=denominator > 0.0,
    )
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return np.divide(
        fused,
        norms,
        out=np.zeros_like(fused),
        where=norms > epsilon,
    ).astype(np.float32, copy=False)


def score_cold_start_tracks(
    query_representations: object,
    candidate_representations: object,
    *,
    candidate_modality_mask: object | None = None,
    content_modalities: Sequence[str] = ("audio", "metadata"),
    minimum_content_modalities: int = 1,
    missing_modality_penalty: float = 0.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Score cold-start candidates by cosine similarity and content coverage.

    Collaborative availability is intentionally not required by default. This
    makes the API suitable for tracks that have lawful local/FMA/Music4All
    content features but no personal interaction history.
    """
    query = np.asarray(query_representations, dtype=np.float32)
    candidates = np.asarray(candidate_representations, dtype=np.float32)
    if query.ndim not in {1, 2}:
        raise ValueError("query_representations must be a rank-1 or rank-2 array")
    if candidates.ndim != 2 or candidates.shape[1] < 1:
        raise ValueError(
            "candidate_representations must be a non-empty rank-2 array"
        )
    if query.shape[-1] != candidates.shape[1]:
        raise ValueError(
            "query and candidate representation dimensions must match"
        )
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidates)):
        raise ValueError("query and candidate representations must be finite")

    modality_indices = _validate_content_modalities(content_modalities)
    minimum_content_modalities = _bounded_int(
        "minimum_content_modalities",
        minimum_content_modalities,
        minimum=0,
        maximum=len(modality_indices),
    )
    missing_modality_penalty = _non_negative_float(
        "missing_modality_penalty",
        missing_modality_penalty,
    )
    epsilon = _positive_float("epsilon", epsilon)

    query_was_vector = query.ndim == 1
    query_matrix = query[None, :] if query_was_vector else query
    query_matrix = _normalize_numpy_rows(query_matrix, epsilon=epsilon)
    candidate_matrix = _normalize_numpy_rows(candidates, epsilon=epsilon)
    scores = query_matrix @ candidate_matrix.T

    if candidate_modality_mask is not None:
        mask = _validate_numpy_mask(
            candidate_modality_mask,
            row_count=len(candidates),
        )
        content_count = np.sum(mask[:, modality_indices] > 0.0, axis=1)
        coverage = content_count.astype(np.float32) / float(len(modality_indices))
        scores -= missing_modality_penalty * (1.0 - coverage[None, :])
        scores[:, content_count < minimum_content_modalities] = -np.inf

    result = scores.astype(np.float32, copy=False)
    return result[0] if query_was_vector else result


def get_multimodal_custom_objects() -> dict[str, object]:
    """Return lazily defined Keras layers needed to load multimodal artifacts."""
    global _CUSTOM_OBJECTS
    if _CUSTOM_OBJECTS is not None:
        return dict(_CUSTOM_OBJECTS)

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MaskedMultimodalFusion(tf.keras.layers.Layer):
        """Project and fuse modalities while ignoring unavailable feature rows."""

        def __init__(
            self,
            output_dim: int,
            activation: str = "gelu",
            dropout_rate: float = 0.0,
            modality_weights: Sequence[float] = (1.0, 1.0, 1.0),
            trainable_modality_weights: bool = True,
            projection_use_bias: bool = True,
            projection_trainable: Sequence[bool] = (True, True, True),
            l2_regularization: float = 0.0,
            normalization_epsilon: float = 1e-8,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.output_dim = _positive_int("output_dim", output_dim)
            self.activation = str(activation)
            self.dropout_rate = _dropout_rate(dropout_rate)
            self.modality_weights = tuple(
                float(value)
                for value in _validate_modality_weights(modality_weights)
            )
            self.trainable_modality_weights = bool(trainable_modality_weights)
            self.projection_use_bias = bool(projection_use_bias)
            self.projection_trainable = _validate_projection_trainable(
                projection_trainable
            )
            self.l2_regularization = _non_negative_float(
                "l2_regularization",
                l2_regularization,
            )
            self.normalization_epsilon = _positive_float(
                "normalization_epsilon",
                normalization_epsilon,
            )
            regularizer = (
                tf.keras.regularizers.l2(self.l2_regularization)
                if self.l2_regularization > 0.0
                else None
            )
            self.projections = [
                tf.keras.layers.Dense(
                    self.output_dim,
                    activation=self.activation,
                    use_bias=self.projection_use_bias,
                    kernel_regularizer=regularizer,
                    trainable=self.projection_trainable[index],
                    name=f"{modality_name}_projection",
                )
                for index, modality_name in enumerate(MODALITY_NAMES)
            ]
            self.dropout = tf.keras.layers.Dropout(
                self.dropout_rate,
                name="fusion_dropout",
            )
            self.modality_logits = None

        def build(self, input_shape):
            if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 4:
                raise ValueError(
                    "MaskedMultimodalFusion expects "
                    "[audio, metadata, collaborative, modality_mask]"
                )
            feature_shapes = [
                tf.TensorShape(shape) for shape in input_shape[:3]
            ]
            mask_shape = tf.TensorShape(input_shape[3])
            if any(shape.rank != 2 for shape in feature_shapes):
                raise ValueError("multimodal feature inputs must be rank-2 tensors")
            if any(shape[-1] is None or int(shape[-1]) < 1 for shape in feature_shapes):
                raise ValueError(
                    "multimodal feature dimensions must be known and positive"
                )
            if mask_shape.rank != 2 or mask_shape[-1] != len(MODALITY_NAMES):
                raise ValueError("modality_mask must have shape (batch, 3)")
            for projection, shape in zip(self.projections, feature_shapes):
                projection.build(shape)
            if self.trainable_modality_weights:
                initial_logits = np.log(
                    np.maximum(
                        np.asarray(self.modality_weights, dtype=np.float32),
                        self.normalization_epsilon,
                    )
                )
                self.modality_logits = self.add_weight(
                    name="modality_logits",
                    shape=(len(MODALITY_NAMES),),
                    initializer=tf.keras.initializers.Constant(initial_logits),
                    trainable=True,
                )
            super().build(input_shape)

        def call(self, inputs, training=None):
            audio, metadata, collaborative, modality_mask = inputs
            mask = tf.clip_by_value(
                tf.cast(modality_mask, self.compute_dtype),
                0.0,
                1.0,
            )
            projected = tf.stack(
                [
                    projection(features)
                    for projection, features in zip(
                        self.projections,
                        (audio, metadata, collaborative),
                    )
                ],
                axis=1,
            )
            projected = self.dropout(projected, training=training)
            if self.modality_logits is not None:
                base_weights = tf.nn.softmax(self.modality_logits)
            else:
                base_weights = tf.cast(
                    tf.constant(self.modality_weights),
                    self.compute_dtype,
                )
                base_weights = tf.math.divide_no_nan(
                    base_weights,
                    tf.reduce_sum(base_weights),
                )
            available_weights = mask * base_weights[None, :]
            available_weights = tf.math.divide_no_nan(
                available_weights,
                tf.reduce_sum(available_weights, axis=1, keepdims=True),
            )
            fused = tf.reduce_sum(
                projected * available_weights[:, :, None],
                axis=1,
            )
            return tf.math.l2_normalize(
                fused,
                axis=-1,
                epsilon=self.normalization_epsilon,
            )

        def compute_output_shape(self, input_shape):
            return tf.TensorShape((input_shape[0][0], self.output_dim))

        def get_config(self):
            return {
                **super().get_config(),
                "output_dim": self.output_dim,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "modality_weights": self.modality_weights,
                "trainable_modality_weights": self.trainable_modality_weights,
                "projection_use_bias": self.projection_use_bias,
                "projection_trainable": self.projection_trainable,
                "l2_regularization": self.l2_regularization,
                "normalization_epsilon": self.normalization_epsilon,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class ColdStartCosineScore(tf.keras.layers.Layer):
        """Pairwise cosine score with optional missing-content penalty."""

        def __init__(
            self,
            content_modality_indices: Sequence[int] = (0, 1),
            minimum_content_modalities: int = 1,
            missing_modality_penalty: float = 0.0,
            normalization_epsilon: float = 1e-8,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.content_modality_indices = tuple(
                _bounded_int(
                    "content_modality_indices",
                    value,
                    minimum=0,
                    maximum=len(MODALITY_NAMES) - 1,
                )
                for value in content_modality_indices
            )
            if not self.content_modality_indices:
                raise ValueError(
                    "content_modality_indices must contain at least one index"
                )
            if len(set(self.content_modality_indices)) != len(
                self.content_modality_indices
            ):
                raise ValueError("content_modality_indices must be unique")
            self.minimum_content_modalities = _bounded_int(
                "minimum_content_modalities",
                minimum_content_modalities,
                minimum=0,
                maximum=len(self.content_modality_indices),
            )
            self.missing_modality_penalty = _non_negative_float(
                "missing_modality_penalty",
                missing_modality_penalty,
            )
            self.normalization_epsilon = _positive_float(
                "normalization_epsilon",
                normalization_epsilon,
            )

        def build(self, input_shape):
            if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
                raise ValueError(
                    "ColdStartCosineScore expects "
                    "[query, candidate, modality_mask]"
                )
            query_shape = tf.TensorShape(input_shape[0])
            candidate_shape = tf.TensorShape(input_shape[1])
            mask_shape = tf.TensorShape(input_shape[2])
            if query_shape.rank != 2 or candidate_shape.rank != 2:
                raise ValueError(
                    "query and candidate representations must be rank-2 tensors"
                )
            if query_shape[-1] != candidate_shape[-1]:
                raise ValueError(
                    "query and candidate representation dimensions must match"
                )
            if mask_shape.rank != 2 or mask_shape[-1] != len(MODALITY_NAMES):
                raise ValueError("modality_mask must have shape (batch, 3)")
            super().build(input_shape)

        def call(self, inputs):
            query, candidate, modality_mask = inputs
            query = tf.math.l2_normalize(
                query,
                axis=-1,
                epsilon=self.normalization_epsilon,
            )
            candidate = tf.math.l2_normalize(
                candidate,
                axis=-1,
                epsilon=self.normalization_epsilon,
            )
            score = tf.reduce_sum(query * candidate, axis=-1, keepdims=True)
            mask = tf.clip_by_value(
                tf.cast(modality_mask, score.dtype),
                0.0,
                1.0,
            )
            selected = tf.gather(
                mask,
                self.content_modality_indices,
                axis=1,
            )
            content_count = tf.reduce_sum(
                tf.cast(selected > 0.0, score.dtype),
                axis=1,
                keepdims=True,
            )
            coverage = content_count / float(len(self.content_modality_indices))
            score -= self.missing_modality_penalty * (1.0 - coverage)
            eligible = content_count >= float(self.minimum_content_modalities)
            return tf.where(
                eligible,
                score,
                tf.cast(float("-inf"), score.dtype),
            )

        def compute_output_shape(self, input_shape):
            return tf.TensorShape((input_shape[0][0], 1))

        def get_config(self):
            return {
                **super().get_config(),
                "content_modality_indices": self.content_modality_indices,
                "minimum_content_modalities": self.minimum_content_modalities,
                "missing_modality_penalty": self.missing_modality_penalty,
                "normalization_epsilon": self.normalization_epsilon,
            }

    _CUSTOM_OBJECTS = {
        "MaskedMultimodalFusion": MaskedMultimodalFusion,
        "ColdStartCosineScore": ColdStartCosineScore,
        "spotify>MaskedMultimodalFusion": MaskedMultimodalFusion,
        "spotify>ColdStartCosineScore": ColdStartCosineScore,
    }
    return dict(_CUSTOM_OBJECTS)


def build_multimodal_track_encoder(
    audio_dim: int,
    metadata_dim: int,
    collaborative_dim: int,
    embedding_dim: int = 64,
    params: Mapping[str, object] | None = None,
):
    """Build a track encoder over precomputed lawful audio/content features.

    The model consumes audio embeddings rather than an audio file or provider
    API. An upstream FMA, Music4All, or local audio encoder can therefore remain
    frozen while its outputs train jointly with metadata and collaborative
    projections.
    """
    audio_dim = _positive_int("audio_dim", audio_dim)
    metadata_dim = _positive_int("metadata_dim", metadata_dim)
    collaborative_dim = _positive_int("collaborative_dim", collaborative_dim)
    embedding_dim = _positive_int("embedding_dim", embedding_dim)
    config = _validate_params(params)

    from tensorflow.keras import Model, layers

    custom_objects = get_multimodal_custom_objects()
    MaskedMultimodalFusion = custom_objects["MaskedMultimodalFusion"]

    audio_input = layers.Input(
        shape=(audio_dim,),
        dtype="float32",
        name="audio_input",
    )
    metadata_input = layers.Input(
        shape=(metadata_dim,),
        dtype="float32",
        name="metadata_input",
    )
    collaborative_input = layers.Input(
        shape=(collaborative_dim,),
        dtype="float32",
        name="collaborative_input",
    )
    modality_mask = layers.Input(
        shape=(len(MODALITY_NAMES),),
        dtype="float32",
        name="modality_mask",
    )
    track_representation = MaskedMultimodalFusion(
        output_dim=embedding_dim,
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        modality_weights=config["modality_weights"],
        trainable_modality_weights=config["trainable_modality_weights"],
        projection_use_bias=config["projection_use_bias"],
        projection_trainable=(
            config["audio_projection_trainable"],
            config["metadata_projection_trainable"],
            config["collaborative_projection_trainable"],
        ),
        l2_regularization=config["l2_regularization"],
        normalization_epsilon=config["normalization_epsilon"],
        name="masked_multimodal_fusion",
    )(
        [
            audio_input,
            metadata_input,
            collaborative_input,
            modality_mask,
        ]
    )
    model = Model(
        inputs=[
            audio_input,
            metadata_input,
            collaborative_input,
            modality_mask,
        ],
        outputs=track_representation,
        name="multimodal_track_encoder",
    )
    model.multimodal_config = {
        "audio_dim": audio_dim,
        "metadata_dim": metadata_dim,
        "collaborative_dim": collaborative_dim,
        "embedding_dim": embedding_dim,
        **config,
    }
    return model


def build_multimodal_cold_start_scorer(
    audio_dim: int,
    metadata_dim: int,
    collaborative_dim: int,
    embedding_dim: int = 64,
    params: Mapping[str, object] | None = None,
    *,
    content_modalities: Sequence[str] = ("audio", "metadata"),
    minimum_content_modalities: int = 1,
    missing_modality_penalty: float = 0.0,
):
    """Build a pairwise Keras scorer for content-backed cold-start tracks."""
    modality_indices = _validate_content_modalities(content_modalities)
    minimum_content_modalities = _bounded_int(
        "minimum_content_modalities",
        minimum_content_modalities,
        minimum=0,
        maximum=len(modality_indices),
    )
    missing_modality_penalty = _non_negative_float(
        "missing_modality_penalty",
        missing_modality_penalty,
    )
    encoder = build_multimodal_track_encoder(
        audio_dim=audio_dim,
        metadata_dim=metadata_dim,
        collaborative_dim=collaborative_dim,
        embedding_dim=embedding_dim,
        params=params,
    )

    from tensorflow.keras import Model, layers

    custom_objects = get_multimodal_custom_objects()
    ColdStartCosineScore = custom_objects["ColdStartCosineScore"]
    query_input = layers.Input(
        shape=(embedding_dim,),
        dtype="float32",
        name="query_input",
    )
    score = ColdStartCosineScore(
        content_modality_indices=modality_indices,
        minimum_content_modalities=minimum_content_modalities,
        missing_modality_penalty=missing_modality_penalty,
        normalization_epsilon=encoder.multimodal_config[
            "normalization_epsilon"
        ],
        name="cold_start_score",
    )(
        [
            query_input,
            encoder.output,
            encoder.inputs[-1],
        ]
    )
    return Model(
        inputs=[query_input, *encoder.inputs],
        outputs=score,
        name="multimodal_cold_start_scorer",
    )


def _validate_feature_matrices(
    audio_embeddings: object | None,
    metadata_features: object | None,
    collaborative_embeddings: object | None,
    *,
    num_items: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    matrices: list[np.ndarray | None] = []
    expected_rows = (
        _positive_int("num_items", num_items) if num_items is not None else None
    )
    for modality_name, value in zip(
        MODALITY_NAMES,
        (audio_embeddings, metadata_features, collaborative_embeddings),
    ):
        if value is None:
            matrices.append(None)
            continue
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError(
                f"{modality_name} features must be a non-empty rank-2 array"
            )
        if expected_rows is None:
            expected_rows = int(matrix.shape[0])
        elif matrix.shape[0] != expected_rows:
            raise ValueError("all modality feature matrices must have equal rows")
        matrices.append(matrix)
    return tuple(matrices)  # type: ignore[return-value]


def _resolve_row_count(
    matrices: Sequence[np.ndarray | None],
    *,
    num_items: int | None = None,
) -> int:
    for matrix in matrices:
        if matrix is not None:
            return int(matrix.shape[0])
    if num_items is None:
        raise ValueError("num_items is required when all modalities are absent")
    return _positive_int("num_items", num_items)


def _validate_projection_matrices(
    projection_matrices: Mapping[str, object] | None,
) -> dict[str, np.ndarray]:
    if projection_matrices is None:
        return {}
    if not isinstance(projection_matrices, Mapping):
        raise TypeError("projection_matrices must be a mapping or None")
    unknown = sorted(set(projection_matrices) - set(MODALITY_NAMES))
    if unknown:
        raise ValueError(
            f"unknown projection modalities: {', '.join(unknown)}"
        )
    validated: dict[str, np.ndarray] = {}
    for modality_name, value in projection_matrices.items():
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError(
                f"{modality_name} projection must be a non-empty rank-2 array"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{modality_name} projection must contain only finite values"
            )
        validated[modality_name] = matrix
    return validated


def _validate_numpy_mask(value: object, *, row_count: int) -> np.ndarray:
    mask = np.asarray(value, dtype=np.float32)
    expected_shape = (row_count, len(MODALITY_NAMES))
    if mask.shape != expected_shape:
        raise ValueError(f"modality_mask must have shape {expected_shape}")
    if not np.all(np.isfinite(mask)) or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("modality_mask values must be finite and in [0, 1]")
    return mask


def _validate_modality_weights(
    value: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if value is None:
        raw_weights: Sequence[float] = (1.0, 1.0, 1.0)
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - set(MODALITY_NAMES))
        if unknown:
            raise ValueError(
                f"unknown modality weights: {', '.join(unknown)}"
            )
        raw_weights = tuple(float(value.get(name, 1.0)) for name in MODALITY_NAMES)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                "modality_weights must be a mapping or length-3 sequence"
            )
        raw_weights = value
    if len(raw_weights) != len(MODALITY_NAMES):
        raise ValueError("modality_weights must contain exactly three values")
    weights = np.asarray(raw_weights, dtype=np.float32)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            "modality_weights must be finite, non-negative, and not all zero"
        )
    return weights


def _validate_projection_trainable(value: object) -> tuple[bool, bool, bool]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != len(MODALITY_NAMES)
        or any(not isinstance(item, bool) for item in value)
    ):
        raise ValueError("projection_trainable must contain exactly three booleans")
    return tuple(value)  # type: ignore[return-value]


def _validate_content_modalities(value: Sequence[str]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("content_modalities must be a non-empty sequence")
    names = tuple(str(name) for name in value)
    unknown = sorted(set(names) - set(MODALITY_NAMES))
    if unknown:
        raise ValueError(f"unknown content modalities: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("content_modalities must be unique")
    return tuple(MODALITY_NAMES.index(name) for name in names)


def _normalize_numpy_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > epsilon,
    )


def _validate_params(
    params: Mapping[str, object] | None,
) -> dict[str, object]:
    if params is None:
        raw_params: dict[str, object] = {}
    elif isinstance(params, Mapping):
        raw_params = dict(params)
    else:
        raise TypeError("params must be a mapping or None")
    unknown = sorted(set(raw_params) - set(_DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown multimodal parameters: {', '.join(unknown)}")
    config = {**_DEFAULT_PARAMS, **raw_params}

    activation = config["activation"]
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError("activation must be a non-empty string")
    config["dropout_rate"] = _dropout_rate(config["dropout_rate"])
    config["modality_weights"] = tuple(
        float(value)
        for value in _validate_modality_weights(config["modality_weights"])
    )
    for name in (
        "trainable_modality_weights",
        "projection_use_bias",
        "audio_projection_trainable",
        "metadata_projection_trainable",
        "collaborative_projection_trainable",
    ):
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a boolean")
    config["l2_regularization"] = _non_negative_float(
        "l2_regularization",
        config["l2_regularization"],
    )
    config["normalization_epsilon"] = _positive_float(
        "normalization_epsilon",
        config["normalization_epsilon"],
    )
    return config


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bounded_int(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _non_negative_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _dropout_rate(value: Any) -> float:
    result = _non_negative_float("dropout_rate", value)
    if result >= 1.0:
        raise ValueError("dropout_rate must be less than 1")
    return result


__all__ = [
    "MODALITY_NAMES",
    "build_multimodal_cold_start_scorer",
    "build_multimodal_track_encoder",
    "fuse_track_representations",
    "get_multimodal_custom_objects",
    "infer_modality_mask",
    "score_cold_start_tracks",
]
