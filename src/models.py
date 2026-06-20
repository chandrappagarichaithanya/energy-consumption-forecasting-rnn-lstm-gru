"""
Model architectures.

The original scripts used `CuDNNLSTM`, a GPU-only layer that was removed
in TensorFlow 2.x. These versions use the standard `LSTM`/`GRU` layers,
which TensorFlow automatically routes to the cuDNN kernel when a GPU is
available and falls back to a plain CPU implementation otherwise — so
this runs anywhere, not just on a GPU-equipped Colab instance.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(seq_len: int, n_features: int, lstm_units: int = 36) -> keras.Model:
    """Single-step LSTM regressor: a window of history -> one future value."""
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(lstm_units, return_sequences=True, activation="tanh"),
        layers.LSTM(lstm_units, return_sequences=False, activation="tanh"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae", metrics=["mse", "mae"])
    return model


def build_seq2seq_model(input_len: int, target_len: int, hidden_units=(30, 30), learning_rate=0.001):
    """Encoder-decoder GRU model for multi-step forecasting.

    Returns the trainable model plus the separate encoder/decoder
    inference models needed to generate forecasts step by step after
    training (Keras requires these as distinct graphs for autoregressive
    decoding).
    """
    num_features = 1

    encoder_inputs = layers.Input(shape=(input_len, num_features))
    encoder_cells = [keras.layers.GRUCell(u) for u in hidden_units]
    encoder = layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

    decoder_inputs = layers.Input(shape=(None, num_features))
    decoder_cells = [keras.layers.GRUCell(u) for u in hidden_units]
    decoder_rnn = layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder_rnn(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(num_features)
    decoder_outputs = decoder_dense(decoder_outputs_and_states[0])

    model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    # --- inference-time encoder/decoder, sharing trained weights ---
    encoder_predict_model = keras.models.Model(encoder_inputs, encoder_states)

    decoder_states_inputs = [layers.Input(shape=(u,)) for u in hidden_units[::-1]]
    decoder_outputs_and_states2 = decoder_rnn(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states2 = decoder_outputs_and_states2[1:]
    decoder_outputs2 = decoder_dense(decoder_outputs_and_states2[0])
    decoder_predict_model = keras.models.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + list(decoder_states2)
    )

    return model, encoder_predict_model, decoder_predict_model


def seq2seq_predict(x, encoder_model, decoder_model, num_steps: int, batch_size: int = 64):
    """Autoregressively roll out `num_steps` predictions from a trained encoder/decoder pair."""
    states = encoder_model.predict(x, verbose=0)
    if not isinstance(states, list):
        states = [states]

    decoder_input = tf.zeros((x.shape[0], 1, 1))
    outputs = []
    for _ in range(num_steps):
        result = decoder_model.predict([decoder_input] + states, batch_size=batch_size, verbose=0)
        output, states = result[0], result[1:]
        outputs.append(output)
        decoder_input = output

    return tf.concat(outputs, axis=1).numpy()
