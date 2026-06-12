from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SessionGraph:
    node_items: np.ndarray
    alias_inputs: np.ndarray
    adjacency_in: np.ndarray
    adjacency_out: np.ndarray


def build_session_graph(sequence: Sequence[int]) -> SessionGraph:
    """Build a normalized directed item-transition graph for one session."""
    items = np.asarray(sequence, dtype=np.int64)
    if items.ndim != 1 or items.size == 0:
        raise ValueError("sequence must be a non-empty one-dimensional sequence")

    node_count = items.size
    unique_items: list[int] = []
    node_by_item: dict[int, int] = {}
    alias_inputs = np.empty(node_count, dtype=np.int64)

    for position, raw_item in enumerate(items):
        item = int(raw_item)
        if item not in node_by_item:
            node_by_item[item] = len(unique_items)
            unique_items.append(item)
        alias_inputs[position] = node_by_item[item]

    node_items = np.zeros(node_count, dtype=np.int64)
    node_items[: len(unique_items)] = unique_items

    transition_counts = np.zeros((node_count, node_count), dtype=np.float32)
    for source, destination in zip(alias_inputs[:-1], alias_inputs[1:]):
        transition_counts[source, destination] += 1.0

    return SessionGraph(
        node_items=node_items,
        alias_inputs=alias_inputs,
        adjacency_in=_normalize_rows(transition_counts.T),
        adjacency_out=_normalize_rows(transition_counts),
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=row_sums != 0,
    )


_CUSTOM_LAYER_TYPES: dict[str, object] | None = None


def get_srgnn_custom_objects() -> dict[str, object]:
    """Return lazily defined custom layers for loading saved SR-GNN models."""
    global _CUSTOM_LAYER_TYPES
    if _CUSTOM_LAYER_TYPES is not None:
        return dict(_CUSTOM_LAYER_TYPES)

    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Layer

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class SessionGraphMessagePassing(Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)
            self.in_projection = Dense(self.units, use_bias=False, name="incoming_projection")
            self.out_projection = Dense(self.units, use_bias=False, name="outgoing_projection")
            self.update_gate = Dense(self.units, activation="sigmoid", name="update_gate")
            self.reset_gate = Dense(self.units, activation="sigmoid", name="reset_gate")
            self.candidate = Dense(self.units, activation="tanh", name="candidate_state")

        def call(self, inputs):
            sequence_ids, position_embeddings = inputs
            seq_len = tf.shape(sequence_ids)[1]

            equality = tf.equal(sequence_ids[:, :, None], sequence_ids[:, None, :])
            earlier = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
            earlier = tf.linalg.set_diag(earlier, tf.zeros((seq_len,), dtype=tf.bool))
            node_mask = tf.logical_not(tf.reduce_any(equality & earlier[None, :, :], axis=2))

            representative_matches = equality & node_mask[:, None, :]
            alias_inputs = tf.argmax(
                tf.cast(representative_matches, tf.int32),
                axis=2,
                output_type=tf.int32,
            )

            node_states = position_embeddings * tf.cast(node_mask[:, :, None], position_embeddings.dtype)
            source = tf.one_hot(alias_inputs[:, :-1], depth=seq_len, dtype=position_embeddings.dtype)
            destination = tf.one_hot(alias_inputs[:, 1:], depth=seq_len, dtype=position_embeddings.dtype)
            transition_counts = tf.einsum("bti,btj->bij", source, destination)

            adjacency_out = tf.math.divide_no_nan(
                transition_counts,
                tf.reduce_sum(transition_counts, axis=2, keepdims=True),
            )
            incoming_counts = tf.transpose(transition_counts, perm=(0, 2, 1))
            adjacency_in = tf.math.divide_no_nan(
                incoming_counts,
                tf.reduce_sum(incoming_counts, axis=2, keepdims=True),
            )

            incoming = tf.matmul(adjacency_in, self.in_projection(node_states))
            outgoing = tf.matmul(adjacency_out, self.out_projection(node_states))
            message = tf.concat((incoming, outgoing), axis=-1)

            gate_inputs = tf.concat((message, node_states), axis=-1)
            update = self.update_gate(gate_inputs)
            reset = self.reset_gate(gate_inputs)
            candidate = self.candidate(tf.concat((message, reset * node_states), axis=-1))
            updated_nodes = (1.0 - update) * node_states + update * candidate
            updated_nodes *= tf.cast(node_mask[:, :, None], updated_nodes.dtype)
            return tf.gather(updated_nodes, alias_inputs, axis=1, batch_dims=1)

        def get_config(self):
            return {**super().get_config(), "units": self.units}

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class SessionPreference(Layer):
        def __init__(self, units: int, **kwargs):
            super().__init__(**kwargs)
            self.units = int(units)
            self.local_projection = Dense(self.units, use_bias=False, name="local_projection")
            self.global_projection = Dense(self.units, use_bias=True, name="global_projection")
            self.attention_score = Dense(1, use_bias=False, name="attention_score")
            self.preference_projection = Dense(self.units, activation="tanh", name="preference_projection")

        def call(self, session_states):
            local_preference = session_states[:, -1, :]
            query = self.local_projection(local_preference)[:, None, :]
            keys = self.global_projection(session_states)
            logits = self.attention_score(tf.nn.sigmoid(query + keys))
            weights = tf.nn.softmax(logits, axis=1)
            global_preference = tf.reduce_sum(weights * session_states, axis=1)
            return self.preference_projection(tf.concat((local_preference, global_preference), axis=-1))

        def get_config(self):
            return {**super().get_config(), "units": self.units}

    _CUSTOM_LAYER_TYPES = {
        "SessionGraphMessagePassing": SessionGraphMessagePassing,
        "SessionPreference": SessionPreference,
        "spotify>SessionGraphMessagePassing": SessionGraphMessagePassing,
        "spotify>SessionPreference": SessionPreference,
    }
    return dict(_CUSTOM_LAYER_TYPES)


def build_srgnn_model(
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
    params: dict[str, object] | None = None,
):
    """Build a session graph neural recommender with lazy TensorFlow imports."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if num_artists < 2:
        raise ValueError("num_artists must be at least 2")
    if num_ctx < 1:
        raise ValueError("num_ctx must be positive")

    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Concatenate, Dense, Embedding

    params = params or {}
    custom_objects = get_srgnn_custom_objects()
    SessionGraphMessagePassing = custom_objects["SessionGraphMessagePassing"]
    SessionPreference = custom_objects["SessionPreference"]
    embedding_dim = int(params.get("embedding_dim", 96))
    context_dim = int(params.get("context_dim", 64))
    fusion_dim = int(params.get("fusion_dim", 192))

    seq_input = Input(shape=(sequence_length,), dtype="int32", name="seq_input")
    ctx_input = Input(shape=(num_ctx,), dtype="float32", name="ctx_input")

    item_embeddings = Embedding(
        input_dim=num_artists,
        output_dim=embedding_dim,
        name="artist_embedding",
    )(seq_input)
    session_states = SessionGraphMessagePassing(
        embedding_dim,
        name="session_graph_message_passing",
    )((seq_input, item_embeddings))
    session_preference = SessionPreference(
        embedding_dim,
        name="session_preference_attention",
    )(session_states)

    context = Dense(context_dim, activation="relu", name="context_projection")(ctx_input)
    fused = Concatenate(name="session_context_fusion")((session_preference, context))
    fused = Dense(fusion_dim, activation="relu", name="fusion_projection")(fused)
    artist_output = Dense(
        num_artists,
        activation="softmax",
        dtype="float32",
        name="artist_output",
    )(fused)
    return Model(
        inputs=[seq_input, ctx_input],
        outputs=artist_output,
        name="srgnn_recommender",
    )


__all__ = [
    "SessionGraph",
    "build_session_graph",
    "build_srgnn_model",
    "get_srgnn_custom_objects",
]
