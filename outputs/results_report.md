# Weather Forecasting Pipeline - Results Report

**Data Source:** Jena Climate Dataset (2009-2016), originally 10-minute
interval, resampled to hourly frequency for computational efficiency.
Variables used: temperature (T degC), humidity (rh %), pressure (p mbar).

## Validation Metrics (held-out, time-ordered split, cutoff = 2015-01-01)
- RMSE: 1.3073
- MAE: 1.0037
- MAPE: 16.1196%  (see note below)

MAPE note: temperature values near/crossing 0C inflate percentage error, a
known limitation of MAPE on Celsius-scale data (points with |actual| <= 1C
are excluded from the MAPE calculation above to keep it meaningful).
RMSE and MAE don't have this problem and are the primary metrics here.

These numbers are reported strictly on data after the cutoff date - the
model never saw this period during training, and the split was done
chronologically (no shuffling), so there's no leakage from future rows
into training.

## Time-Series Cross-Validation (expanding window, TimeSeriesSplit)
- Fold 1: RMSE = 1.5012
- Fold 2: RMSE = 1.4711
- Fold 3: RMSE = 1.2770
- Fold 4: RMSE = 1.3259
- Fold 5: RMSE = 1.2555

## Predicted vs Actual
See `outputs/predicted_vs_actual.png`.

## Feature Importance (mean |SHAP value|, computed on a 1,000-row validation sample)
- temp_ewm: 7.2341
- hour_cos: 0.8480
- rh (%): 0.4325
- hour_sin: 0.3485
- temp_roll_std_24h: 0.2618
- temp_lag_24h: 0.2244
- temp_roll_mean_7d: 0.1139
- p (mbar): 0.1124

Full SHAP summary plot: `outputs/shap_summary.png`

### Interpretation
`temp_ewm` (the exponential smoothing feature) comes out as the most
influential predictor, which lines up with intuition - it's the smoothed,
recent version of temperature and reacts quickly to what's already
happening. Hour-of-day (`hour_sin`/`hour_cos`) and humidity (`rh (%)`) are
the next biggest drivers, capturing the day/night temperature swing and the
real inverse relationship between temperature and humidity. The longer-range
`temp_lag_7d` feature has comparatively low importance next to the shorter
lags/rolling stats, which suggests the model is leaning mostly on recent
trend rather than what happened a full week ago - day-of-year still
contributes, but mostly for the slower seasonal drift rather than short-term
prediction accuracy.

## Missing Data Handling
Resampling the original 10-minute readings to hourly reintroduced a small
number of gaps (88 hourly rows per column). These were filled using
time-based linear interpolation rather than mean-imputation, since weather
variables move smoothly hour to hour and a straight line between the last
known reading and the next preserves the seasonal/daily shape through the
gap, whereas mean-imputation would flatten it to a single flat value
regardless of season or time of day. See
`outputs/interpolation_before_after.png` for a before/after example.
