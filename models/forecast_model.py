"""
Model choice: XGBoost regressor on lagged/rolling/cyclical features, not
Prophet or SARIMAX.

Why (this is the reasoning behind the choice, also in README.md):
    - The Jena dataset is large (~70k hourly rows after resampling) and noisy
      at the hourly level - pressure and humidity add useful signal on top
      of temperature's own history, which a univariate tool like Prophet
      isn't built to use directly.
    - With the rolling/lag/cyclical features already engineered, a tree
      model captures both the short-term momentum (lags, EWM) and the
      seasonal shape (day/hour sin-cos) without needing an ARIMA order to be
      chosen by hand.
    - It trains fast even at this row count, which mattered given the 1-week
      timeline - SARIMAX gets slow on ~70k rows once you add seasonal terms.
    - SHAP interpretability comes for free with a tree model.

This is not a plain regression on same-timestep features - every predictor
fed to the model is a lag, a rolling/smoothed statistic, or a cyclical
encoding, so the model has real temporal context.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from features.engineering import MODEL_FEATURES, TARGET

CUTOFF_DATE = "2015-01-01"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series


def time_based_split(features: pd.DataFrame, cutoff: str = CUTOFF_DATE) -> SplitData:
    """Strict chronological split - train on everything before the cutoff,
    validate on everything after. No shuffling anywhere.

    Why a random split would leak here specifically: several of our features
    (rolling means/stds, lags) are computed across the whole series before
    the split happens. If rows were shuffled and randomly assigned to
    train/validation, a training row could end up with a rolling-mean value
    that was partly computed from rows that are now sitting in "validation",
    and a validation row could land right next to a training row in time -
    meaning the model would already have seen almost-identical recent
    history for it. Both let future information leak backward into
    training, which makes validation accuracy look better than it would on
    genuinely unseen future data.
    """
    train = features[features.index < cutoff]
    val = features[features.index >= cutoff]
    return SplitData(
        X_train=train[MODEL_FEATURES],
        X_val=val[MODEL_FEATURES],
        y_train=train[TARGET],
        y_val=val[TARGET],
    )


def build_model() -> XGBRegressor:
    return XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)


def train(split: SplitData) -> XGBRegressor:
    model = build_model()
    model.fit(split.X_train, split.y_train)
    return model


def time_series_cv_scores(features: pd.DataFrame, n_splits: int = 5) -> list[float]:
    """Expanding-window CV via TimeSeriesSplit, as a sanity check on top of
    the single time-based split above. Never standard k-fold - that would
    shuffle time order and mix future rows into earlier training folds."""
    X, y = features[MODEL_FEATURES], features[TARGET]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmses = []
    for train_idx, test_idx in tscv.split(X):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        rmse = float(np.sqrt(np.mean((y.iloc[test_idx].values - preds) ** 2)))
        rmses.append(rmse)
    return rmses
