# src/models/deep/cnn_bilstm.py
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)


def build_cnn_bilstm_model(vocab_size, max_len, embedding_dim=100):
    """
    复现提供的 CNN-BiLSTM 论文结构
    """
    input_layer = Input(shape=(max_len,), name="input_ids")

    # 1. Embedding Layer
    # vocab_size + 1 是因为 padding 的 0 索引
    embedding = Embedding(
        input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len
    )(input_layer)

    # 2. Convolution Layers (Parallel)
    # 卷积核大小 (3, 4, 5)
    conv1 = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(
        embedding
    )
    conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(filters=256, kernel_size=4, activation="relu", padding="same")(
        embedding
    )
    conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(filters=512, kernel_size=5, activation="relu", padding="same")(
        embedding
    )
    conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # 合并特征
    merged = Concatenate(axis=-1)([pool1, pool2, pool3])

    # 3. Bi-LSTM Layer
    bilstm = Bidirectional(LSTM(128, return_sequences=False))(merged)

    # 4. Fully Connected Layers
    dense1 = Dense(128, activation="relu")(bilstm)
    dense2 = Dense(64, activation="relu")(dense1)

    # 5. Output Layer (Softmax for 2 classes)
    # 也可以用 Sigmoid (1 unit)，这里保持原代码逻辑用 Softmax (2 units)
    output = Dense(2, activation="softmax", name="output")(dense2)

    model = Model(inputs=input_layer, outputs=output)

    # 编译模型
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model
