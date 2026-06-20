"""
Evaluate a saved LSTM model on held-out test data and print metrics.

Run from the project root:
    python -m src.evaluate_lstm
"""
import pickle

import numpy as np
import tensorflow as tf

from src.config import LSTM_CONFIG, DATASET_PATH
from src.data import load_raw_data, preprocess, make_supervised, make_sequences, train_val_test_split, scale_splits
from src.metrics import regression_report


def main(cfg=LSTM_CONFIG):
    df = load_raw_data(DATASET_PATH)
    df = df[[cfg.target_column]]
    df = preprocess(df, cfg.target_column)
    df = make_supervised(df, cfg.target_column, cfg.future_period)

    train_df, valid_df, test_df = train_val_test_split(df, cfg.test_fraction, cfg.valid_fraction)
    with open(cfg.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_columns = [cfg.target_column]
    test_df = test_df.copy()
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    test_x, test_y = make_sequences(test_df, cfg.seq_len, shuffle=False)

    model = tf.keras.models.load_model(cfg.model_path)
    pred = model.predict(test_x, verbose=0).flatten()

    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true_inv = scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()

    report = regression_report(true_inv, pred_inv)
    print("Test set metrics (kWh scale):")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")
    return report


if __name__ == "__main__":
    main()
