import tensorflow as tf


def build_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_dim, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
