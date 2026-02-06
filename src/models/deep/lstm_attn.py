# src/models/deep/lstm_attn.py
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer,
)
import keras.backend as K


class AttentionLayer(Layer):
    """
    自定义 Attention 层 (Bahdanau Attention 简化版)
    输入: (batch_size, time_steps, features)
    输出: (batch_size, features) - 加权求和后的上下文向量
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 权重矩阵 W: (features, 1)
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        # 偏置 b: (time_steps, 1)
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, step, hidden)

        # 1. 计算注意力分数 e = tanh(W*x + b)
        # e shape: (batch, step, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)

        # 2. 计算权重 a = softmax(e)
        a = K.softmax(e, axis=1)

        # 3. 加权求和 output = sum(a * x)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_lstm_attn_model(vocab_size, max_len, embedding_dim=100):
    input_layer = Input(shape=(max_len,), name="input_ids")

    # 1. Embedding
    embedding = Embedding(
        input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len
    )(input_layer)

    # 2. Bi-LSTM (return_sequences=True 是必须的，为了给 Attention 用)
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    lstm_out = Dropout(0.5)(lstm_out)

    # 3. Attention Layer [核心差异]
    # 它把时序数据 (Batch, Time, Feat) 压缩成 (Batch, Feat)
    attn_out = AttentionLayer()(lstm_out)

    # 4. Dense
    x = Dense(64, activation="relu")(attn_out)
    x = Dropout(0.5)(x)

    # 5. Output
    output = Dense(2, activation="softmax", name="output")(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model
