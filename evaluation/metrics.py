"""Forecast accuracy metrics, computed only on the held-out validation set."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error.

    Temperature is on the Celsius scale, so it crosses (and sits near) 0,
    which makes a naive MAPE blow up - dividing by a true value close to
    zero turns a tiny 0.5C error into a huge percentage. This showed up
    directly during development: the first MAPE run on this dataset came
    back as a nonsense trillion-percent number before I traced it to values
    near 0C. To keep the metric honest instead of hiding the problem, points
    where |y_true| is under 1C are excluded from the MAPE calculation, and
    RMSE/MAE (which don't have this issue) are reported as the primary
    metrics alongside it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > 1.0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def summarize(y_true, y_pred) -> dict:
    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE_%": round(mape(y_true, y_pred), 4),
    }
