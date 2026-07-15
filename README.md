# Intelligent Weather Prediction Pipeline

Time-series forecasting pipeline built to predict temperature using the Jena
Climate dataset (2009-2016), developed for the Khizex Data Science
Internship. This is the modular version of `Project_Weather.ipynb` - same
approach and results, split into reusable `data/` / `features/` / `models/`
/ `evaluation/` modules instead of one long notebook, per the assignment's
code-structure requirement.

## Project structure
```
weather-prediction-pipeline/
├── data/
│   ├── raw/
│   │   └── jena_climate_2009_2016.csv
│   └── loader.py                # CSV load, hourly resample, interpolation
├── features/
│   └── engineering.py           # rolling stats, lags, EWM, cyclical encoding
├── models/
│   └── forecast_model.py        # time-based split, TimeSeriesSplit CV, XGBoost
├── evaluation/
│   ├── metrics.py                # RMSE / MAE / MAPE
│   └── interpret.py              # SHAP interpretability
├── notebooks/
│   └── Project_Weather_exploration.ipynb   # original EDA/exploration notebook
├── outputs/                      # plots + results_report.md land here after a run
├── main.py                       # runs the whole pipeline
└── requirements.txt
```

## 1. Dataset

The pipeline reads from **`data/raw/jena_climate_2009_2016.csv`** - the raw
Jena Climate dataset, recorded every 10 minutes from 2009-2016. It's already
in this repo, so no download step is needed. `data/loader.py` resamples it
to hourly and interpolates the small number of gaps that resampling
introduces.

If you swap in a different weather CSV later, `data/loader.py` expects a
`Date Time` column plus `T (degC)`, `rh (%)`, and `p (mbar)` - update
`KEEP_COLUMNS` in that file if your column names differ.

## 2. How to run this in VS Code

1. Open the `weather-prediction-pipeline` folder in VS Code.
2. Open a terminal (`` Ctrl+` ``), making sure it's in the project root
   (the folder with `main.py` in it).
3. Set up the environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```bash
   python main.py
   ```
5. Check `outputs/` for:
   - `results_report.md` - RMSE/MAE/MAPE + SHAP interpretation
   - `predicted_vs_actual.png`
   - `shap_summary.png`
   - `interpolation_before_after.png`

If VS Code says "python not recognized," check that Python is installed and
selected as the interpreter (bottom-right corner, or `Ctrl+Shift+P` ->
"Python: Select Interpreter").

## 3. Feature engineering approach

Built by hand in `features/engineering.py` with `.rolling()`, `.shift()`,
and `.ewm()` - not an auto-feature library:

- **Rolling mean & std** - 24-hour and 7-day windows (`temp_roll_mean_24h`,
  `temp_roll_mean_7d`, `temp_roll_std_24h`, `temp_roll_std_7d`), for
  short-term trend and volatility.
- **Exponential smoothing** - `temp_ewm` (span=24), weights recent hours more
  heavily than a flat rolling mean does.
- **Lag features** - `temp_lag_24h` and `temp_lag_7d`, giving the model
  direct access to what the temperature was exactly a day and a week ago.
- **Cyclical encoding** - `day_of_year` and `hour` are each turned into a
  sin/cos pair instead of raw integers, so December and January (or 11 PM
  and midnight) read as adjacent to the model instead of maximally far
  apart.

Pressure (`p (mbar)`) and humidity (`rh (%)`) are also fed in directly, since
temperature doesn't move independently of the other two.

## 4. Model choice and why

**XGBoost regressor**, trained on the features above, rather than Prophet
or SARIMAX:

- The dataset is large (~70,000 hourly rows) and noisy at the hourly level.
  Pressure and humidity add useful signal on top of temperature's own
  history, which Prophet (a univariate-first tool) isn't built to use
  directly, and SARIMAX gets slow once you add seasonal terms at this row
  count.
- With the lag/rolling/EWM/cyclical features already engineered, a tree
  model picks up both short-term momentum and the seasonal shape without
  needing an ARIMA order chosen by hand.
- SHAP interpretability comes essentially free with a tree model.

Trade-off, stated honestly: XGBoost doesn't hand you a clean additive
trend/seasonality decomposition the way Prophet does - that signal comes
back through the sin/cos features and the SHAP ranking instead (see
`outputs/results_report.md`).

This isn't a plain same-timestep regression - every feature is either a lag,
a rolling/smoothed statistic, or a cyclical encoding, so the model has real
temporal context.

## 5. Leakage prevention

- **Strict time-based split** (`models/forecast_model.py::time_based_split`):
  training on everything before `2015-01-01`, validating on everything after.
  No `train_test_split(..., shuffle=True)` anywhere.
- **Why a random split would leak**: several features (rolling means/stds,
  lags) are computed across the whole series before the split. If rows were
  shuffled and randomly assigned, a "training" row could carry a rolling-mean
  value partly computed from rows now sitting in "validation," and a
  validation row could land right next to a training row in time - meaning
  the model would already have seen near-identical recent history for it.
  Both leak future information backward into training and make validation
  accuracy look better than it would on genuinely unseen future data.
- **Cross-validation**: `time_series_cv_scores()` uses `TimeSeriesSplit`
  (expanding window), never `KFold`, as an extra sanity check on top of the
  single time-based split.

## 6. Missing data handling

Resampling from 10-minute to hourly reintroduces a small number of gaps (88
hourly rows per column, out of ~70,129). These are filled with time-based
linear interpolation, not mean-imputation - weather variables move smoothly
hour to hour, so a straight line between the two nearest real readings keeps
the seasonal/daily shape intact through the gap, whereas mean-imputation
would flatten every gap to the same flat value regardless of season or time
of day. A before/after plot is written to
`outputs/interpolation_before_after.png` on every run.

## 7. Results (held-out, time-ordered validation set)

| Metric | Value |
|---|---|
| RMSE | ~1.31 °C |
| MAE | ~1.00 °C |
| MAPE | reported with a caveat - see `outputs/results_report.md` |

MAPE note: temperature crosses/sits near 0°C, and dividing by an actual value
near zero can blow the percentage error up to a meaningless number - this
came up directly while building the metrics (an early run produced an
absurd trillion-percent MAPE before I traced it to a handful of near-zero
readings). Points where `|actual| <= 1°C` are excluded from the MAPE
calculation so it stays meaningful, and RMSE/MAE are used as the primary
metrics.

## Notes on AI use

Per the brief's own allowance, AI was used to speed up boilerplate (SHAP
plotting call syntax, restructuring the original notebook cells into
separate files). The feature choices, model choice, split logic, and the
reasoning above are the same ones worked out in
`notebooks/Project_Weather_exploration.ipynb` - that notebook is the actual
development history and is kept in `notebooks/` per the assignment's
structure (exploration only, not the final deliverable).
