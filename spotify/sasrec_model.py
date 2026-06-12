from __future__ import annotations


def build_sasrec_model(
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
    params: dict[str, object] | None = None,
):
    """Build a SASRec-style next-artist prediction model."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_artists <= 1:
        raise ValueError("num_artists must be greater than 1")
    if num_ctx <= 0:
        raise ValueError("num_ctx must be positive")

    import tensorflow as tf
    from tensorflow.keras import Model, layers

    params = params or {}
    embedding_dim = int(params.get("embedding_dim", 128))
    num_heads = int(params.get("num_heads", 4))
    feed_forward_dim = int(params.get("feed_forward_dim", 256))
    dropout_rate = float(params.get("dropout_rate", 0.1))
    num_blocks = int(params.get("num_blocks", 2))

    seq_input = layers.Input(shape=(sequence_length,), dtype="int32", name="seq_input")
    item_embeddings = layers.Embedding(
        input_dim=num_artists,
        output_dim=embedding_dim,
        name="item_embedding",
    )(seq_input)
    position_indices = layers.Lambda(
        lambda token_ids: tf.tile(
            tf.range(sequence_length, dtype=tf.int32)[tf.newaxis, :],
            [tf.shape(token_ids)[0], 1],
        ),
        output_shape=(sequence_length,),
        name="position_indices",
    )(seq_input)
    position_embeddings = layers.Embedding(
        input_dim=sequence_length,
        output_dim=embedding_dim,
        name="position_embedding",
    )(position_indices)
    x = layers.Add(name="item_position_embeddings")([item_embeddings, position_embeddings])
    x = layers.Dropout(dropout_rate, name="embedding_dropout")(x)

    for block_index in range(num_blocks):
        block_number = block_index + 1
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name=f"causal_self_attention_{block_number}",
        )(x, x, use_causal_mask=True)
        attention = layers.Dropout(
            dropout_rate,
            name=f"attention_dropout_{block_number}",
        )(attention)
        x = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"attention_norm_{block_number}",
        )(layers.Add(name=f"attention_residual_{block_number}")([x, attention]))

        feed_forward = layers.Dense(
            feed_forward_dim,
            activation="relu",
            name=f"feed_forward_expand_{block_number}",
        )(x)
        feed_forward = layers.Dropout(
            dropout_rate,
            name=f"feed_forward_dropout_{block_number}",
        )(feed_forward)
        feed_forward = layers.Dense(
            embedding_dim,
            name=f"feed_forward_project_{block_number}",
        )(feed_forward)
        x = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"sequence_norm_{block_number}",
        )(layers.Add(name=f"feed_forward_residual_{block_number}")([x, feed_forward]))

    sequence_output = layers.Activation("linear", name="sasrec_sequence_output")(x)
    last_position = layers.Lambda(
        lambda encoded_sequence: encoded_sequence[:, -1, :],
        output_shape=(embedding_dim,),
        name="last_position",
    )(sequence_output)

    ctx_input = layers.Input(shape=(num_ctx,), dtype="float32", name="ctx_input")
    context = layers.Dense(64, activation="relu", name="context_projection")(ctx_input)
    fused = layers.Concatenate(name="sequence_context_fusion")([last_position, context])
    fused = layers.Dense(256, activation="relu", name="fusion_projection")(fused)
    fused = layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    artist_output = layers.Dense(
        num_artists,
        activation="softmax",
        dtype="float32",
        name="artist_output",
    )(fused)

    return Model(
        inputs=[seq_input, ctx_input],
        outputs=artist_output,
        name="sasrec",
    )
