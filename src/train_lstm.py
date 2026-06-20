"""
Train the single-step LSTM forecaster and save the model + scaler to disk.

Run from the project root:
    python -m src.train_lstm

This replaces the old lstm/lstm.py notebook-cell script. Key fixes versus
the original:
  - `epochs=epochs` referenced an undefined variable (only `EPOCHS` existed)
    -> now uses the config value directly.
  - `evaluate_theil` was referenced in an average() call but never appended
    to -> theil_u is now computed correctly per split.
  - CuDNNLSTM (GPU-only, removed in TF2) -> standard LSTM layer.
  - keras.experimental.export_saved_model (removed API) -> model.save(),
    using the modern .keras format, which is what the Streamlit app loads.
  - Scaler is persisted alongside the model (it wasn't saved at all
    before), since you can't invert predictions back to kWh without it.
"""
import argparse
import pickle

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.config import LSTM_CONFIG
from src.data import load_raw_data, preprocess, make_supervised, make_sequences, scale_splits, train_val_test_split
from src.models import build_lstm_model
from src.metrics import regression_report


def main(cfg=LSTM_CONFIG, n_splits: int = 3, verbose: int = 1):
    print(f"Loading data from {cfg.scaler_path.parent.parent / 'dataset' / 'kaggle_data_1h.csv'} ...")
    from src.config import DATASET_PATH
    df = load_raw_data(DATASET_PATH)
    df = df[[cfg.target_column]]
    df = preprocess(df, cfg.target_column)
    df = make_supervised(df, cfg.target_column, cfg.future_period)
    print(f"Usable rows after preprocessing: {len(df)}")

    feature_columns = [cfg.target_column]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    reports = []
    last_model, last_scaler = None, None

    for split_i, (train_index, test_index) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df, valid_df, _ = train_val_test_split(train_df, test_fraction=0, valid_fraction=cfg.valid_fraction)

        train_df, valid_df, scaled_test_df, scaler = scale_splits(train_df, valid_df, test_df, feature_columns)

        train_x, train_y = make_sequences(train_df, cfg.seq_len, shuffle=True)
        valid_x, valid_y = make_sequences(valid_df, cfg.seq_len, shuffle=False)
        test_x, test_y = make_sequences(scaled_test_df, cfg.seq_len, shuffle=False)

        if len(train_x) == 0 or len(test_x) == 0:
            print(f"Split {split_i}: not enough data yet, skipping.")
            continue

        print(f"Split {split_i}: train={train_x.shape}, valid={valid_x.shape}, test={test_x.shape}")

        model = build_lstm_model(cfg.seq_len, train_x.shape[2], cfg.lstm_units)
        model.fit(
            train_x, train_y,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_data=(valid_x, valid_y) if len(valid_x) else None,
            verbose=verbose,
        )

        pred = model.predict(test_x, verbose=0).flatten()
        # invert scaling (single-feature scaler here, so this is direct)
        pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        true_inv = scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()

        report = regression_report(true_inv, pred_inv)
        print(f"Split {split_i} metrics: {report}")
        reports.append(report)

        last_model, last_scaler = model, scaler

    if not reports:
        raise RuntimeError("No split produced enough data to train on — reduce seq_len or n_splits.")

    avg_report = {k: float(np.mean([r[k] for r in reports])) for k in reports[0]}
    print("\n=== Average metrics across walk-forward splits ===")
    for k, v in avg_report.items():
        print(f"  {k}: {v:.4f}")

    last_model.save(cfg.model_path)
    with open(cfg.scaler_path, "wb") as f:
        pickle.dump(last_scaler, f)
    print(f"\nSaved model to {cfg.model_path}")
    print(f"Saved scaler to {cfg.scaler_path}")

    return avg_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=LSTM_CONFIG.epochs)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    import dataclasses
    cfg = dataclasses.replace(LSTM_CONFIG, epochs=args.epochs)
    main(cfg, n_splits=args.n_splits)
