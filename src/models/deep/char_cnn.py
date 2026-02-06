# src/models/deep/char_cnn.py
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


def build_char_cnn_model(
    vocab_size,
    max_len=1000,  # 字符级长度通常较长
    embedding_dim=64,  # 字符嵌入通常不需要太大
    num_filters=128,
):
    input_layer = Input(shape=(max_len,), name="input_char_ids")

    # 1. Embedding (Alphabet size -> Dense vector)
    embedding = Embedding(
        input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_len
    )(input_layer)

    # 2. Convolution (捕捉 n-gram 字符组合)
    # 比如 kernel_size=3 捕捉 'uni', kernel_size=7 捕捉 'select'
    kernel_sizes = [3, 5, 7, 9]
    pooled_outputs = []

    for kernel_size in kernel_sizes:
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="valid",
        )(embedding)
        x = GlobalMaxPooling1D()(x)
        pooled_outputs.append(x)

    # 3. Merge & Dense
    merged = Concatenate(axis=-1)(pooled_outputs)
    x = Dropout(0.5)(merged)
    x = Dense(128, activation="relu")(x)

    # 4. Output
    output = Dense(2, activation="softmax", name="output")(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model
