"""
Data loading and preprocessing for the household power consumption dataset.

Consolidates the preprocessing logic that was duplicated (with small
inconsistencies) across the original lstm.py and seq2seq.py scripts:
missing-value handling, exponential smoothing, outlier removal, and
sequence windowing for supervised learning.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(csv_path) -> pd.DataFrame:
    """Load the raw CSV, parse the datetime index, and resample to 1h bins."""
    df = pd.read_csv(
        csv_path,
        sep=",",
        index_col="time",
        low_memory=False,
        encoding="utf-8",
    )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df = df.resample("1h").mean()
    df.dropna(inplace=True)
    return df


def remove_outliers(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """Drop points more than `factor` standard deviations from the mean."""
    upper = series.mean() + series.std() * factor
    lower = series.mean() - series.std() * factor
    return series[(series < upper) & (series > lower)]


def preprocess(df: pd.DataFrame, target_column: str, smoothing_alpha: float = 0.2) -> pd.DataFrame:
    """Apply exponential smoothing and outlier removal to the target column.

    Equivalent to the cleaning steps in the original scripts, minus the
    hardcoded 2007/2008 "fill values" patch, which was specific to known
    gaps in that exact Kaggle dataset snapshot and doesn't generalize.
    """
    df = df.copy()
    df = df.ewm(alpha=smoothing_alpha).mean()
    df.dropna(inplace=True)

    mask = remove_outliers(df[target_column])
    df = df.loc[mask.index]
    return df


def make_supervised(df: pd.DataFrame, target_column: str, future_period: int) -> pd.DataFrame:
    """Create a `future` column shifted `future_period` steps ahead (the label)."""
    df = df.copy()
    df["future"] = df[target_column].shift(-future_period)
    df.dropna(inplace=True)
    return df


def train_val_test_split(df: pd.DataFrame, test_fraction: float, valid_fraction: float):
    """Chronological split — no shuffling, since order matters for time series."""
    n = len(df)
    test_n = int(n * test_fraction)
    valid_n = int(n * valid_fraction)

    test_df = df.iloc[n - test_n:]
    valid_df = df.iloc[n - test_n - valid_n: n - test_n]
    train_df = df.iloc[: n - test_n - valid_n]
    return train_df, valid_df, test_df


def scale_splits(train_df, valid_df, test_df, feature_columns):
    """Fit a MinMaxScaler on train only, then apply it to all three splits.

    Fitting on train-only avoids leaking validation/test distribution
    information into the scaler — fitting on the whole dataset (as the
    original scripts effectively risked doing in places) is a common
    time-series-specific bug.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    valid_df[feature_columns] = scaler.transform(valid_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    return train_df, valid_df, test_df, scaler


def make_sequences(df: pd.DataFrame, seq_len: int, shuffle: bool = False):
    """Turn a dataframe of [features..., future] rows into (X, y) windows.

    X[i] is `seq_len` consecutive rows of features; y[i] is the
    corresponding `future` value at the end of that window.
    """
    feature_cols = [c for c in df.columns if c != "future"]
    values = df[feature_cols].values
    targets = df["future"].values

    window = deque(maxlen=seq_len)
    X, y = [], []
    for i in range(len(values)):
        window.append(values[i])
        if len(window) == seq_len:
            X.append(np.array(window))
            y.append(targets[i])

    X, y = np.array(X), np.array(y)
    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
    return X, y


def make_seq2seq_windows(series: np.ndarray, input_len: int, target_len: int, shuffle: bool = False):
    """Build encoder/decoder windows for the Seq2Seq model from a 1D scaled series."""
    seq_len = input_len + target_len
    window = deque(maxlen=seq_len)
    windows = []
    for value in series:
        window.append(value)
        if len(window) == seq_len:
            windows.append(np.array(window))

    windows = np.array(windows)
    if shuffle:
        idx = np.random.permutation(len(windows))
        windows = windows[idx]

    windows = windows.reshape(windows.shape[0], windows.shape[1], 1)
    encoder_input = windows[:, :input_len, :]
    decoder_target = windows[:, input_len:, :]
    decoder_input = np.zeros((decoder_target.shape[0], decoder_target.shape[1], 1))
    return encoder_input, decoder_input, decoder_target
