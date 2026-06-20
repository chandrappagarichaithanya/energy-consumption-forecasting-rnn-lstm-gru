"""
Forecast evaluation metrics.

Pulls the metric functions out of the original performance.py scripts
(which relied on globals like `scaler`, `test_x`, `model` existing in
the calling scope from a prior script having "already run" in the same
notebook session) into pure functions that take their inputs explicitly.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # avoid divide-by-zero for near-zero true values
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def theil_u(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Theil's U statistic: compares forecast error to a naive no-change forecast.

    U < 1 means the model beats the naive "tomorrow = today" baseline;
    U >= 1 means it doesn't. (The original implementation referenced this
    function but never actually called it / populated its results list.)
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    num = np.sum(((y_pred[1:] - y_true[1:]) / y_true[:-1]) ** 2)
    den = np.sum(((y_true[1:] - y_true[:-1]) / y_true[:-1]) ** 2)
    return float(np.sqrt(num / den)) if den != 0 else float("nan")


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All key metrics in one call, for logging or display."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "theil_u": theil_u(y_pred, y_true),
    }
