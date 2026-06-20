"""
Streamlit demo app: loads the trained LSTM model and lets a user pick a
window from the test set to see the model's 6-hour-ahead forecast plotted
against the real value.

Run from the project root:
    streamlit run app/streamlit_app.py

This is the piece the original repo was missing entirely: an actual
interface a person can open in a browser to see the model make a
prediction, rather than only Python scripts meant for a Colab cell.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LSTM_CONFIG, DATASET_PATH
from src.data import load_raw_data, preprocess, make_supervised, make_sequences, train_val_test_split

st.set_page_config(page_title="Energy Consumption Forecaster", page_icon="⚡", layout="centered")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(LSTM_CONFIG.model_path)
    with open(LSTM_CONFIG.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_test_data():
    cfg = LSTM_CONFIG
    df = load_raw_data(DATASET_PATH)
    df = df[[cfg.target_column]]
    df = preprocess(df, cfg.target_column)
    df_supervised = make_supervised(df, cfg.target_column, cfg.future_period)
    _, _, test_df = train_val_test_split(df_supervised, cfg.test_fraction, cfg.valid_fraction)
    return test_df


def main():
    st.title("⚡ Household Energy Consumption Forecaster")
    st.caption(
        "Trained LSTM model — predicts Global Active Power "
        f"{LSTM_CONFIG.future_period} hours ahead, from the preceding "
        f"{LSTM_CONFIG.seq_len} hours of readings."
    )

    model, scaler = load_artifacts()
    test_df = load_test_data()

    feature_columns = [LSTM_CONFIG.target_column]
    scaled_test_df = test_df.copy()
    scaled_test_df[feature_columns] = scaler.transform(scaled_test_df[feature_columns])

    test_x, test_y = make_sequences(scaled_test_df, LSTM_CONFIG.seq_len, shuffle=False)
    n_windows = len(test_x)

    st.subheader("Pick a window from the held-out test set")
    idx = st.slider("Window index", 0, n_windows - 1, n_windows // 2)

    if st.button("Run forecast", type="primary"):
        x = test_x[idx: idx + 1]
        pred_scaled = model.predict(x, verbose=0).flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        actual_scaled = test_y[idx]
        actual = scaler.inverse_transform(np.array([[actual_scaled]])).flatten()[0]

        history = scaler.inverse_transform(x[0]).flatten()

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted (kW)", f"{pred:.3f}")
        col2.metric("Actual (kW)", f"{actual:.3f}")
        col3.metric("Error", f"{abs(pred - actual):.3f}")

        chart_df = pd.DataFrame({
            "hour": list(range(-len(history), 0)) + [LSTM_CONFIG.future_period],
            "value": list(history) + [actual],
            "series": ["history"] * len(history) + ["actual_future"],
        })
        pred_point = pd.DataFrame({
            "hour": [LSTM_CONFIG.future_period],
            "value": [pred],
            "series": ["predicted"],
        })
        chart_df = pd.concat([chart_df, pred_point], ignore_index=True)

        st.line_chart(chart_df.pivot(index="hour", columns="series", values="value"))

    with st.expander("About this model"):
        st.write(
            f"""
            - Architecture: 2-layer LSTM, {LSTM_CONFIG.lstm_units} units each
            - Input: {LSTM_CONFIG.seq_len} hours of `Global_active_power` history
            - Output: power draw {LSTM_CONFIG.future_period} hours ahead
            - Trained with walk-forward (`TimeSeriesSplit`) cross-validation
            - Dataset: UCI/Kaggle household power consumption, resampled to 1h
            """
        )


if __name__ == "__main__":
    main()
