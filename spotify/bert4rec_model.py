from __future__ import annotations

from typing import NamedTuple


class ClozePretrainingBatch(NamedTuple):
    """Arrays for fitting BERT4Rec with one masked artist per example."""

    seq_input: object
    ctx_input: object
    artist_output: object
    masked_positions: object

    @property
    def keras_inputs(self) -> list[object]:
        return [self.seq_input, self.ctx_input]


def bert4rec_mask_token_id(num_artists: int) -> int:
    """Return the reserved mask token, immediately outside artist IDs."""
    if num_artists <= 0:
        raise ValueError("num_artists must be positive")
    return int(num_artists)


def bert4rec_vocabulary_size(num_artists: int) -> int:
    """Return the item vocabulary size, including the reserved mask token."""
    return bert4rec_mask_token_id(num_artists) + 1


def build_cloze_pretraining_batch(
    sequences,
    contexts,
    num_artists: int,
    *,
    mask_probability: float = 0.15,
    seed: int = 0,
) -> ClozePretrainingBatch:
    """Expand sequences into deterministic, single-mask Cloze examples.

    Every position selected for pretraining produces one example. If random
    selection chooses no positions for a source sequence, one position is
    selected so that every source sequence contributes to the batch.
    """
    import numpy as np

    mask_token_id = bert4rec_mask_token_id(num_artists)
    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be in the interval (0, 1]")

    sequence_values = np.asarray(sequences)
    context_values = np.asarray(contexts)
    if sequence_values.ndim != 2:
        raise ValueError("sequences must be a rank-2 array")
    if context_values.ndim != 2:
        raise ValueError("contexts must be a rank-2 array")
    if sequence_values.shape[0] != context_values.shape[0]:
        raise ValueError("sequences and contexts must have the same batch size")
    if sequence_values.shape[1] == 0:
        raise ValueError("sequences must contain at least one position")
    if not np.issubdtype(sequence_values.dtype, np.integer):
        raise ValueError("sequences must contain integer artist IDs")
    if np.any(sequence_values < 0) or np.any(sequence_values >= num_artists):
        raise ValueError(f"artist IDs must be in [0, {num_artists})")

    rng = np.random.default_rng(seed)
    selected = rng.random(sequence_values.shape) < mask_probability
    for row_index in range(sequence_values.shape[0]):
        if not selected[row_index].any():
            selected[row_index, rng.integers(sequence_values.shape[1])] = True

    source_rows, masked_positions = np.nonzero(selected)
    masked_sequences = sequence_values[source_rows].astype("int32", copy=True)
    artist_targets = masked_sequences[
        np.arange(masked_sequences.shape[0]), masked_positions
    ].copy()
    masked_sequences[np.arange(masked_sequences.shape[0]), masked_positions] = mask_token_id

    return ClozePretrainingBatch(
        seq_input=masked_sequences,
        ctx_input=context_values[source_rows].copy(),
        artist_output=artist_targets.astype("int32", copy=False),
        masked_positions=masked_positions.astype("int32", copy=False),
    )


def build_bert4rec_model(
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
    params: dict[str, object] | None = None,
):
    """Build a BERT4Rec-style masked-item classifier."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_ctx <= 0:
        raise ValueError("num_ctx must be positive")

    mask_token_id = bert4rec_mask_token_id(num_artists)
    vocabulary_size = bert4rec_vocabulary_size(num_artists)

    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
        Add,
        Concatenate,
        Dense,
        Dropout,
        Embedding,
        Lambda,
        LayerNormalization,
        MultiHeadAttention,
    )

    params = params or {}
    embedding_dim = int(params.get("embedding_dim", 128))
    num_heads = int(params.get("num_heads", 4))
    feed_forward_multiplier = int(params.get("feed_forward_multiplier", 4))
    dropout_rate = float(params.get("dropout_rate", 0.1))
    num_blocks = int(params.get("num_blocks", 2))

    seq_input = Input(shape=(sequence_length,), dtype="int32", name="seq_input")
    ctx_input = Input(shape=(num_ctx,), dtype="float32", name="ctx_input")

    item_embeddings = Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name="item_embedding",
    )(seq_input)
    positions = Lambda(
        lambda tokens: tf.broadcast_to(
            tf.range(tf.shape(tokens)[1])[tf.newaxis, :],
            tf.shape(tokens),
        ),
        output_shape=(sequence_length,),
        name="position_indices",
    )(seq_input)
    position_embeddings = Embedding(
        input_dim=sequence_length,
        output_dim=embedding_dim,
        name="position_embedding",
    )(positions)
    encoded = Add(name="item_position_embeddings")([item_embeddings, position_embeddings])
    encoded = Dropout(dropout_rate, name="embedding_dropout")(encoded)

    for block_index in range(num_blocks):
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name=f"bidirectional_attention_{block_index + 1}",
        )(encoded, encoded)
        encoded = LayerNormalization(
            epsilon=1e-6,
            name=f"attention_norm_{block_index + 1}",
        )(encoded + attention)

        feed_forward = Dense(
            embedding_dim * feed_forward_multiplier,
            activation="gelu",
            name=f"feed_forward_expand_{block_index + 1}",
        )(encoded)
        feed_forward = Dropout(dropout_rate, name=f"feed_forward_dropout_{block_index + 1}")(
            feed_forward
        )
        feed_forward = Dense(
            embedding_dim,
            name=f"feed_forward_project_{block_index + 1}",
        )(feed_forward)
        encoded = LayerNormalization(
            epsilon=1e-6,
            name=f"feed_forward_norm_{block_index + 1}",
        )(encoded + feed_forward)

    def pool_masked_positions(inputs):
        token_states, token_ids = inputs
        weights = tf.cast(tf.equal(token_ids, mask_token_id), token_states.dtype)
        has_mask = tf.reduce_any(tf.cast(weights, tf.bool), axis=1, keepdims=True)
        fallback = tf.one_hot(
            tf.fill([tf.shape(token_ids)[0]], tf.shape(token_ids)[1] - 1),
            depth=tf.shape(token_ids)[1],
            dtype=token_states.dtype,
        )
        weights = tf.where(has_mask, weights, fallback)
        weights = weights[..., tf.newaxis]
        return tf.math.divide_no_nan(
            tf.reduce_sum(token_states * weights, axis=1),
            tf.reduce_sum(weights, axis=1),
        )

    masked_state = Lambda(
        pool_masked_positions,
        output_shape=(embedding_dim,),
        name="masked_position_pooling",
    )([encoded, seq_input])
    context_state = Dense(64, activation="relu", name="context_projection")(ctx_input)
    merged = Concatenate(name="sequence_context")([masked_state, context_state])
    merged = Dense(256, activation="gelu", name="prediction_projection")(merged)
    merged = Dropout(dropout_rate, name="prediction_dropout")(merged)
    artist_output = Dense(
        num_artists,
        activation="softmax",
        dtype="float32",
        name="artist_output",
    )(merged)

    model = Model(
        inputs=[seq_input, ctx_input],
        outputs=artist_output,
        name="bert4rec",
    )
    model.mask_token_id = mask_token_id
    model.item_vocabulary_size = vocabulary_size
    return model
