# src/models/deep/textcnn.py
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    Concatenate,
)


def build_textcnn_model(vocab_size, max_len, embedding_dim=100, num_filters=128):
    """
    经典的 TextCNN 模型复现
    Paper: Convolutional Neural Networks for Sentence Classification (Kim, 2014)
    """
    input_layer = Input(shape=(max_len,), name="input_ids")

    # 1. Embedding
    embedding = Embedding(
        input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len
    )(input_layer)

    # 2. Multi-scale Convolution (类似 N-Gram)
    # 使用不同尺寸的卷积核捕捉不同长度的攻击模式 (e.g., 2-gram, 3-gram, 4-gram)
    kernel_sizes = [2, 3, 4, 5]
    pooled_outputs = []

    for kernel_size in kernel_sizes:
        # Conv1D
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="valid",
        )(embedding)
        # Global Max Pooling (提取该窗口下最显著的特征)
        x = GlobalMaxPooling1D()(x)
        pooled_outputs.append(x)

    # 3. Concatenate
    merged = Concatenate(axis=-1)(pooled_outputs)

    # 4. Dropout & Dense
    x = Dropout(0.5)(merged)
    x = Dense(128, activation="relu")(x)

    # 5. Output
    output = Dense(2, activation="softmax", name="output")(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model
