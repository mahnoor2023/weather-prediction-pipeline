"""
Custom feature engineering, built by hand with pandas .rolling()/.shift()/.ewm()
(no auto-feature-generation library), following the same feature set worked
out in the exploration notebook.

Target variable is T (degC) - hourly air temperature.
"""

import numpy as np
import pandas as pd

TARGET = "T (degC)"


def add_rolling_features(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    df["temp_roll_mean_24h"] = df[col].rolling(window=24).mean()
    df["temp_roll_mean_7d"] = df[col].rolling(window=24 * 7).mean()
    df["temp_roll_std_24h"] = df[col].rolling(window=24).std()
    df["temp_roll_std_7d"] = df[col].rolling(window=24 * 7).std()
    return df


def add_exp_smoothing(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    # span=24 -> roughly a day's worth of memory, more weight on recent hours
    df["temp_ewm"] = df[col].ewm(span=24).mean()
    return df


def add_lag_features(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    df["temp_lag_24h"] = df[col].shift(24)
    df["temp_lag_7d"] = df[col].shift(24 * 7)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """day-of-year and hour-of-day as sin/cos pairs, so the model doesn't
    see Dec 31 -> Jan 1 as 364 days apart, or 23:00 -> 00:00 as 23 hours
    apart, when they're actually right next to each other."""
    df = df.copy()
    df["day_of_year"] = df.index.dayofyear
    df["hour"] = df.index.hour

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_feature_set(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Runs the full feature pipeline. Rows in the first 7 days get dropped
    since the 7-day rolling/lag features can't be computed for them yet."""
    out = df.copy()
    out = add_rolling_features(out, target)
    out = add_exp_smoothing(out, target)
    out = add_lag_features(out, target)
    out = add_cyclical_time_features(out)
    out = out.dropna()
    return out


# feature columns actually fed to the model - pressure and humidity come
# straight from the sensor, everything else is engineered above
MODEL_FEATURES = [
    "p (mbar)",
    "rh (%)",
    "temp_roll_mean_24h",
    "temp_roll_std_24h",
    "temp_roll_mean_7d",
    "temp_roll_std_7d",
    "temp_ewm",
    "temp_lag_24h",
    "temp_lag_7d",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
]


if __name__ == "__main__":
    from data.loader import load_clean

    _, clean = load_clean()
    feats = build_feature_set(clean)
    print("Shape after feature engineering:", feats.shape)
    print(feats[MODEL_FEATURES].head())
