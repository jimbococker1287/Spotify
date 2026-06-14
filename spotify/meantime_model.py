from __future__ import annotations


_CUSTOM_OBJECTS: dict[str, object] | None = None


def get_meantime_custom_objects() -> dict[str, object]:
    """Return the custom Keras layers used by the temporal recommender."""
    global _CUSTOM_OBJECTS
    if _CUSTOM_OBJECTS is not None:
        return dict(_CUSTOM_OBJECTS)

    import tensorflow as tf

    def bucketize_time_gaps(time_gaps, num_buckets: int, max_time_gap: float):
        clipped = tf.clip_by_value(
            tf.cast(time_gaps, tf.float32),
            0.0,
            float(max_time_gap),
        )
        scale = tf.math.log1p(clipped) / tf.math.log1p(
            tf.cast(max_time_gap, tf.float32)
        )
        return tf.cast(
            tf.floor(scale * tf.cast(num_buckets - 1, tf.float32)),
            tf.int32,
        )

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeItemMask(tf.keras.layers.Layer):
        def __init__(self, vocabulary_size: int, padding_token_id: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.vocabulary_size = int(vocabulary_size)
            self.padding_token_id = int(padding_token_id)

        def call(self, item_ids):
            tf.debugging.assert_greater_equal(
                item_ids,
                tf.cast(0, item_ids.dtype),
                message="item IDs must be nonnegative",
            )
            tf.debugging.assert_less(
                item_ids,
                tf.cast(self.vocabulary_size, item_ids.dtype),
                message="item IDs must be smaller than vocabulary_size",
            )
            return tf.not_equal(item_ids, self.padding_token_id)

        def get_config(self):
            return {
                **super().get_config(),
                "vocabulary_size": self.vocabulary_size,
                "padding_token_id": self.padding_token_id,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeAbsolutePositionEmbedding(tf.keras.layers.Layer):
        def __init__(self, sequence_length: int, embedding_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = int(sequence_length)
            self.embedding_dim = int(embedding_dim)

        def build(self, input_shape):
            self.embeddings = self.add_weight(
                name="embeddings",
                shape=(self.sequence_length, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, item_ids):
            positions = tf.range(tf.shape(item_ids)[1])
            embedded = tf.gather(self.embeddings, positions)
            return tf.broadcast_to(
                embedded[tf.newaxis, :, :],
                [tf.shape(item_ids)[0], tf.shape(item_ids)[1], self.embedding_dim],
            )

        def get_config(self):
            return {
                **super().get_config(),
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeTemporalGapEmbedding(tf.keras.layers.Layer):
        def __init__(
            self,
            num_time_buckets: int,
            embedding_dim: int,
            max_time_gap: float,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.num_time_buckets = int(num_time_buckets)
            self.embedding_dim = int(embedding_dim)
            self.max_time_gap = float(max_time_gap)

        def build(self, input_shape):
            self.embeddings = self.add_weight(
                name="embeddings",
                shape=(self.num_time_buckets, self.embedding_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            time_gaps, item_mask = inputs
            tf.debugging.assert_all_finite(
                time_gaps,
                "time_gaps must contain only finite values",
            )
            tf.debugging.assert_greater_equal(
                time_gaps,
                tf.cast(0.0, time_gaps.dtype),
                message="time_gaps must be nonnegative",
            )
            bucket_ids = bucketize_time_gaps(
                time_gaps,
                self.num_time_buckets,
                self.max_time_gap,
            )
            embedded = tf.gather(self.embeddings, bucket_ids)
            return embedded * tf.cast(item_mask[..., tf.newaxis], embedded.dtype)

        def get_config(self):
            return {
                **super().get_config(),
                "num_time_buckets": self.num_time_buckets,
                "embedding_dim": self.embedding_dim,
                "max_time_gap": self.max_time_gap,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeSequenceMask(tf.keras.layers.Layer):
        def call(self, inputs):
            sequence, item_mask = inputs
            return sequence * tf.cast(item_mask[..., tf.newaxis], sequence.dtype)

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeTemporalCausalAttention(tf.keras.layers.Layer):
        def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            num_time_buckets: int,
            max_time_gap: float,
            dropout_rate: float = 0.0,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.embedding_dim = int(embedding_dim)
            self.num_heads = int(num_heads)
            self.num_time_buckets = int(num_time_buckets)
            self.max_time_gap = float(max_time_gap)
            self.dropout_rate = float(dropout_rate)
            self.head_dim = self.embedding_dim // self.num_heads
            self.query_projection = tf.keras.layers.Dense(
                self.embedding_dim,
                use_bias=False,
                name="query_projection",
            )
            self.key_projection = tf.keras.layers.Dense(
                self.embedding_dim,
                use_bias=False,
                name="key_projection",
            )
            self.value_projection = tf.keras.layers.Dense(
                self.embedding_dim,
                use_bias=False,
                name="value_projection",
            )
            self.output_projection = tf.keras.layers.Dense(
                self.embedding_dim,
                use_bias=False,
                name="output_projection",
            )
            self.attention_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        def build(self, input_shape):
            self.relative_time_bias = self.add_weight(
                name="relative_time_bias",
                shape=(self.num_time_buckets, self.num_heads),
                initializer="zeros",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs, training=None):
            sequence, time_gaps, item_mask = inputs
            batch_size = tf.shape(sequence)[0]
            sequence_length = tf.shape(sequence)[1]

            def split_heads(values):
                return tf.reshape(
                    values,
                    [batch_size, sequence_length, self.num_heads, self.head_dim],
                )

            queries = split_heads(self.query_projection(sequence))
            keys = split_heads(self.key_projection(sequence))
            values = split_heads(self.value_projection(sequence))
            scores = tf.einsum("bihd,bjhd->bhij", queries, keys)
            scores = scores / tf.math.sqrt(tf.cast(self.head_dim, scores.dtype))

            masked_gaps = tf.where(item_mask, time_gaps, tf.zeros_like(time_gaps))
            event_times = tf.cumsum(masked_gaps, axis=1)
            relative_gaps = tf.abs(
                event_times[:, :, tf.newaxis] - event_times[:, tf.newaxis, :]
            )
            relative_bucket_ids = bucketize_time_gaps(
                relative_gaps,
                self.num_time_buckets,
                self.max_time_gap,
            )
            relative_bias = tf.gather(
                self.relative_time_bias,
                relative_bucket_ids,
            )
            relative_bias = tf.transpose(relative_bias, [0, 3, 1, 2])
            scores = scores + tf.cast(relative_bias, scores.dtype)

            causal_mask = tf.linalg.band_part(
                tf.ones((sequence_length, sequence_length), dtype=tf.bool),
                -1,
                0,
            )
            key_mask = item_mask[:, tf.newaxis, tf.newaxis, :]
            allowed = causal_mask[tf.newaxis, tf.newaxis, :, :] & key_mask
            scores = tf.where(
                allowed,
                scores,
                tf.cast(-1e9, scores.dtype),
            )
            attention_weights = tf.nn.softmax(scores, axis=-1)
            attention_weights = self.attention_dropout(
                attention_weights,
                training=training,
            )
            attended = tf.einsum("bhij,bjhd->bihd", attention_weights, values)
            attended = tf.reshape(
                attended,
                [batch_size, sequence_length, self.embedding_dim],
            )
            attended = self.output_projection(attended)
            return attended * tf.cast(item_mask[..., tf.newaxis], attended.dtype)

        def get_config(self):
            return {
                **super().get_config(),
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_time_buckets": self.num_time_buckets,
                "max_time_gap": self.max_time_gap,
                "dropout_rate": self.dropout_rate,
            }

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimeLastValidPosition(tf.keras.layers.Layer):
        def call(self, inputs):
            sequence, item_mask = inputs
            positions = tf.range(tf.shape(sequence)[1], dtype=tf.int32)
            masked_positions = tf.where(
                item_mask,
                positions[tf.newaxis, :],
                tf.fill(tf.shape(item_mask), -1),
            )
            last_positions = tf.reduce_max(masked_positions, axis=1)
            safe_positions = tf.maximum(last_positions, 0)
            pooled = tf.gather(sequence, safe_positions, axis=1, batch_dims=1)
            has_items = last_positions >= 0
            return pooled * tf.cast(has_items[:, tf.newaxis], pooled.dtype)

    @tf.keras.utils.register_keras_serializable(package="spotify")
    class MeantimePaddingMaskedSoftmax(tf.keras.layers.Layer):
        def __init__(self, padding_token_id: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.padding_token_id = int(padding_token_id)

        def call(self, logits):
            padding_penalty = tf.one_hot(
                self.padding_token_id,
                depth=tf.shape(logits)[-1],
                on_value=tf.cast(-1e9, logits.dtype),
                off_value=tf.cast(0.0, logits.dtype),
            )
            return tf.nn.softmax(logits + padding_penalty, axis=-1)

        def get_config(self):
            return {
                **super().get_config(),
                "padding_token_id": self.padding_token_id,
            }

    classes = [
        MeantimeItemMask,
        MeantimeAbsolutePositionEmbedding,
        MeantimeTemporalGapEmbedding,
        MeantimeSequenceMask,
        MeantimeTemporalCausalAttention,
        MeantimeLastValidPosition,
        MeantimePaddingMaskedSoftmax,
    ]
    _CUSTOM_OBJECTS = {}
    for layer_class in classes:
        _CUSTOM_OBJECTS[layer_class.__name__] = layer_class
        _CUSTOM_OBJECTS[f"spotify>{layer_class.__name__}"] = layer_class
    return dict(_CUSTOM_OBJECTS)


def build_meantime_model(
    sequence_length: int,
    vocabulary_size: int,
    num_ctx: int,
    params: dict[str, object] | None = None,
):
    """Build a time-aware causal next-item recommender.

    Item ID 0 is reserved for padding. ``vocabulary_size`` includes that
    padding token, and the output probability assigned to it is forced to zero.
    ``time_gap_input`` contains the nonnegative elapsed time since the previous
    event in any consistent unit.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if vocabulary_size <= 2:
        raise ValueError("vocabulary_size must be greater than 2")
    if num_ctx <= 0:
        raise ValueError("num_ctx must be positive")

    params = params or {}
    embedding_dim = int(params.get("embedding_dim", 128))
    num_heads = int(params.get("num_heads", 4))
    feed_forward_dim = int(params.get("feed_forward_dim", embedding_dim * 4))
    dropout_rate = float(params.get("dropout_rate", 0.1))
    num_blocks = int(params.get("num_blocks", 2))
    num_time_buckets = int(params.get("num_time_buckets", 64))
    max_time_gap = float(params.get("max_time_gap", 30.0 * 24.0 * 60.0 * 60.0))
    context_dim = int(params.get("context_dim", min(128, embedding_dim)))

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")
    if feed_forward_dim <= 0:
        raise ValueError("feed_forward_dim must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_time_buckets < 2:
        raise ValueError("num_time_buckets must be at least 2")
    if max_time_gap <= 0.0:
        raise ValueError("max_time_gap must be positive")
    if context_dim <= 0:
        raise ValueError("context_dim must be positive")

    import tensorflow as tf

    custom_objects = get_meantime_custom_objects()
    ItemMask = custom_objects["MeantimeItemMask"]
    AbsolutePositionEmbedding = custom_objects[
        "MeantimeAbsolutePositionEmbedding"
    ]
    TemporalGapEmbedding = custom_objects["MeantimeTemporalGapEmbedding"]
    SequenceMask = custom_objects["MeantimeSequenceMask"]
    TemporalCausalAttention = custom_objects[
        "MeantimeTemporalCausalAttention"
    ]
    LastValidPosition = custom_objects["MeantimeLastValidPosition"]
    PaddingMaskedSoftmax = custom_objects["MeantimePaddingMaskedSoftmax"]

    item_input = tf.keras.Input(
        shape=(sequence_length,),
        dtype="int32",
        name="item_sequence",
    )
    time_gap_input = tf.keras.Input(
        shape=(sequence_length,),
        dtype="float32",
        name="time_gap_input",
    )
    context_input = tf.keras.Input(
        shape=(num_ctx,),
        dtype="float32",
        name="context",
    )

    item_mask = ItemMask(
        vocabulary_size=vocabulary_size,
        name="item_mask",
    )(item_input)
    item_embeddings = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name="item_embedding",
    )(item_input)
    position_embeddings = AbsolutePositionEmbedding(
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        name="absolute_position_embedding",
    )(item_input)
    temporal_embeddings = TemporalGapEmbedding(
        num_time_buckets=num_time_buckets,
        embedding_dim=embedding_dim,
        max_time_gap=max_time_gap,
        name="temporal_gap_embedding",
    )([time_gap_input, item_mask])
    encoded = tf.keras.layers.Add(name="item_position_time_embeddings")(
        [item_embeddings, position_embeddings, temporal_embeddings]
    )
    encoded = tf.keras.layers.Dropout(
        dropout_rate,
        name="embedding_dropout",
    )(encoded)
    encoded = SequenceMask(name="initial_sequence_mask")([encoded, item_mask])

    for block_index in range(num_blocks):
        block_number = block_index + 1
        attention = TemporalCausalAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_time_buckets=num_time_buckets,
            max_time_gap=max_time_gap,
            dropout_rate=dropout_rate,
            name=f"temporal_causal_attention_{block_number}",
        )([encoded, time_gap_input, item_mask])
        attention = tf.keras.layers.Dropout(
            dropout_rate,
            name=f"attention_dropout_{block_number}",
        )(attention)
        encoded = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"attention_norm_{block_number}",
        )(
            tf.keras.layers.Add(name=f"attention_residual_{block_number}")(
                [encoded, attention]
            )
        )
        encoded = SequenceMask(name=f"attention_mask_{block_number}")(
            [encoded, item_mask]
        )

        feed_forward = tf.keras.layers.Dense(
            feed_forward_dim,
            activation="gelu",
            name=f"feed_forward_expand_{block_number}",
        )(encoded)
        feed_forward = tf.keras.layers.Dropout(
            dropout_rate,
            name=f"feed_forward_dropout_{block_number}",
        )(feed_forward)
        feed_forward = tf.keras.layers.Dense(
            embedding_dim,
            name=f"feed_forward_project_{block_number}",
        )(feed_forward)
        encoded = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"feed_forward_norm_{block_number}",
        )(
            tf.keras.layers.Add(name=f"feed_forward_residual_{block_number}")(
                [encoded, feed_forward]
            )
        )
        encoded = SequenceMask(name=f"feed_forward_mask_{block_number}")(
            [encoded, item_mask]
        )

    sequence_output = tf.keras.layers.Activation(
        "linear",
        name="meantime_sequence_output",
    )(encoded)
    sequence_state = LastValidPosition(name="last_valid_position")(
        [sequence_output, item_mask]
    )
    context_state = tf.keras.layers.Dense(
        context_dim,
        activation="gelu",
        name="context_projection",
    )(context_input)
    fused = tf.keras.layers.Concatenate(name="sequence_context_fusion")(
        [sequence_state, context_state]
    )
    fused = tf.keras.layers.Dense(
        embedding_dim * 2,
        activation="gelu",
        name="fusion_projection",
    )(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    logits = tf.keras.layers.Dense(
        vocabulary_size,
        dtype="float32",
        name="next_item_logits",
    )(fused)
    output = PaddingMaskedSoftmax(name="next_item_output")(logits)

    model = tf.keras.Model(
        inputs=[item_input, time_gap_input, context_input],
        outputs=output,
        name="meantime",
    )
    model.padding_token_id = 0
    model.item_vocabulary_size = vocabulary_size
    model.num_time_buckets = num_time_buckets
    return model
