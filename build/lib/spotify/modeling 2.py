from __future__ import annotations

from typing import Callable


def build_model_builders(
    sequence_length: int,
    num_artists: int,
    num_ctx: int,
    selected_names: tuple[str, ...],
):
    import tensorflow as tf
    from tensorflow.keras import Model, Input, regularizers
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Add,
        AdditiveAttention,
        Bidirectional,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        GRU,
        GlobalAveragePooling1D,
        Lambda,
        LayerNormalization,
        LSTM,
        MaxPooling1D,
        MultiHeadAttention,
    )

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
        dim = K.int_shape(inputs)[-1]
        x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization()(x + inputs)

        x_ff = Dense(ff_dim, activation="relu")(x)
        x_ff = Dropout(dropout)(x_ff)
        x_ff = Dense(dim)(x_ff)
        return LayerNormalization()(x_ff + x)

    class MemoryLayer(tf.keras.layers.Layer):
        def __init__(self, slots, dim):
            super().__init__()
            self.slots = slots
            self.dim = dim

        def build(self, input_shape):
            self.memory = self.add_weight(
                shape=(self.slots, self.dim),
                initializer="random_normal",
                trainable=True,
                name="memory",
            )

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            return tf.tile(tf.expand_dims(self.memory, 0), [batch_size, 1, 1])

    def build_transformer():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=128, dropout=0.1)
        x = GlobalAveragePooling1D()(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)
        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_lstm():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Bidirectional(LSTM(64))(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_dense():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = GlobalAveragePooling1D()(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        merged = Concatenate()([x, ctx_in])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_tcn():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=64)(seq_in)

        for dilation in (1, 2, 4, 8):
            prev = x
            x = Conv1D(64, kernel_size=3, padding="causal", dilation_rate=dilation, activation="relu")(x)
            x = LayerNormalization()(x)
            x = Add()([x, prev])

        x = GlobalAveragePooling1D()(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_cnn_lstm():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(LSTM(64))(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_gru():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Bidirectional(GRU(64))(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_gru_artist():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=256)(seq_in)
        x = Bidirectional(GRU(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.0))(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dropout(0.2)(merged)
        merged = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(merged)

        return Model(
            [seq_in, ctx_in],
            Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged),
        )

    def build_cnn():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GlobalAveragePooling1D()(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        merged = Concatenate()([x, ctx_in])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_attention_rnn():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)

        last = Lambda(lambda t: t[:, -1, :])(x)
        query = Lambda(lambda t: tf.expand_dims(t, axis=1))(last)
        attn_out = AdditiveAttention()([query, x])
        context_vec = GlobalAveragePooling1D(name="context_vec")(attn_out)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([context_vec, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_tft_model():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        embed = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embed)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
        attn_vec = GlobalAveragePooling1D()(attn_out)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        gate = Dense(attn_vec.shape[-1], activation="sigmoid")(ctx_in)
        gated = tf.keras.layers.Multiply()([attn_vec, gate])

        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([gated, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_transformer_xl():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)

        for _ in range(4):
            x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=256, dropout=0.1)

        x = GlobalAveragePooling1D()(x)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_memory_network():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        x = Bidirectional(LSTM(64))(x)

        mem = MemoryLayer(slots=10, dim=128)(x)
        query = Lambda(lambda t: K.expand_dims(t, axis=1))(x)
        attn_mem = AdditiveAttention()([query, mem])
        mem_vec = Lambda(lambda t: t[:, 0, :])(attn_mem)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([x, mem_vec, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    def build_memory_network_artist():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=256)(seq_in)
        x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1))(x)

        mem = MemoryLayer(slots=16, dim=256)(x)
        query = Lambda(lambda t: K.expand_dims(t, axis=1))(x)
        attn_mem = AdditiveAttention()([query, mem])
        mem_vec = Lambda(lambda t: t[:, 0, :])(attn_mem)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(ctx_in)
        merged = Concatenate()([x, mem_vec, c])
        merged = Dropout(0.2)(merged)
        merged = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(merged)

        return Model(
            [seq_in, ctx_in],
            Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged),
        )

    def build_graph_seq_model():
        seq_in = Input(shape=(sequence_length,), name="seq_input")
        x = Embedding(input_dim=num_artists, output_dim=128)(seq_in)
        gat = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        gat = LayerNormalization()(gat + x)
        gat = GlobalAveragePooling1D()(gat)

        ctx_in = Input(shape=(num_ctx,), name="ctx_input")
        c = Dense(64, activation="relu")(ctx_in)
        merged = Concatenate()([gat, c])
        merged = Dense(256, activation="relu")(merged)

        art = Dense(num_artists, activation="softmax", name="artist_output", dtype="float32")(merged)
        skip = Dense(1, activation="sigmoid", name="skip_output", dtype="float32")(merged)
        return Model([seq_in, ctx_in], [art, skip])

    registry: dict[str, Callable[[], object]] = {
        "gru_artist": build_gru_artist,
        "memory_net_artist": build_memory_network_artist,
        "lstm": build_lstm,
        "transformer": build_transformer,
        "dense": build_dense,
        "tcn": build_tcn,
        "cnn_lstm": build_cnn_lstm,
        "gru": build_gru,
        "cnn": build_cnn,
        "attention_rnn": build_attention_rnn,
        "tft": build_tft_model,
        "transformer_xl": build_transformer_xl,
        "memory_net": build_memory_network,
        "graph_seq": build_graph_seq_model,
    }

    unknown = [name for name in selected_names if name not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model names: {', '.join(unknown)}. Known models: {known}")

    return [(name, registry[name]) for name in selected_names]
