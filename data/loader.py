"""
Loads the Jena Climate dataset (2009-2016) and gets it ready for feature
engineering.

Steps (matches the exploration notebook in notebooks/Project_Weather_exploration.ipynb):
    1. Read the raw CSV - it's recorded every 10 minutes, which is more
       granular than we need and makes 7-day rolling windows huge (1008 rows
       instead of 168), so we resample to hourly first.
    2. Set Date Time as the index so all the later .rolling()/.shift()/.ewm()
       calls are time-aware.
    3. Resample to hourly means.
    4. Resampling itself can reintroduce a few gaps (hours where none of the
       10-min readings existed), so we interpolate again after resampling.

Interpolation choice: time-based linear interpolation, not mean-imputation.
Weather variables move smoothly hour-to-hour, so a straight line between the
last known reading and the next known reading is a far better estimate than
replacing a gap with the column's global mean - mean-imputation would flatten
every gap to the same flat value regardless of season or time of day, which
destroys exactly the seasonal/daily shape this project is trying to model.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/jena_climate_2009_2016.csv")

# Columns we actually use downstream - the raw file has a few more
# (Tpot, Tdew, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max. wv, wd) that
# aren't needed for this pipeline.
KEEP_COLUMNS = ["p (mbar)", "T (degC)", "rh (%)"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time")
    return df[KEEP_COLUMNS]


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("h").mean()


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].interpolate(method="time", limit_direction="both")
    return df


def load_clean(path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hourly_raw, hourly_interpolated) so before/after gaps can be
    plotted for the write-up."""
    raw_10min = load_raw(path)
    hourly = resample_hourly(raw_10min)
    hourly_clean = interpolate_missing(hourly)
    return hourly, hourly_clean


if __name__ == "__main__":
    raw, clean = load_clean()
    print("Hourly shape:", raw.shape)
    print("Missing before interpolation:\n", raw.isna().sum())
    print("\nMissing after interpolation:\n", clean.isna().sum())
