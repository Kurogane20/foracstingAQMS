# ML Quality Improvements — Sub-project A Design

## Goal

Improve prediction quality through three backend enhancements: anomaly detection before training, per-sensor hyperparameter tuning with Optuna, and confidence intervals (lower/upper bounds) from quantile LightGBM — surfaced in both the forecast table and chart.

## Architecture

All changes follow an incremental approach: modify existing pipeline files (`preprocessor.py`, `lgbm.py`, `retrain.py`) and add one new file (`app/models/tuning.py`). No new pipeline wrappers or feature flags. The DB gets two JSON columns added to `predictions`. The frontend TSX gets updated cell rendering and a Highcharts arearange series.

## Tech Stack

Python 3.11, FastAPI, TensorFlow/Keras (BiLSTM), LightGBM, Optuna, MySQL, TypeScript, Highcharts 12.x

---

## Section 1: Anomaly Detection

### Where

`app/models/preprocessor.py` — new function `detect_and_remove_anomalies(df)` called between `resample_hourly()` and `clean()`.

### Method

Rolling Z-score with a 24-hour window:

```
For each sensor column:
  rolling_median = df[col].rolling(24, min_periods=1, center=True).median()
  rolling_std    = df[col].rolling(24, min_periods=1, center=True).std().fillna(1)
  anomaly_mask   = |df[col] - rolling_median| > 3 × rolling_std
  df.loc[anomaly_mask, col] = NaN
  df[col] = df[col].interpolate(method='linear').ffill().bfill()
```

### Updated Pipeline Order

```
fetch_sensor_data → resample_hourly → detect_and_remove_anomalies → clean → add_time_features → normalize
```

This applies to both `preprocess_for_training` and `preprocess_for_predict`.

### Why Rolling Z-score

The existing `clean()` IQR clipping handles global distribution but misses sudden temporal spikes (e.g., PM2.5 jumping from 50 to 5000 for one hour due to sensor fault). Rolling Z-score detects these by comparing each value against its local 24h context. IQR clipping still runs afterward as a second smoothing pass.

---

## Section 2: Hyperparameter Tuning with Optuna (Per-Sensor)

### New File

`app/models/tuning.py` — two functions: `tune_bilstm(X, y, uid)` and `tune_lgbm(bilstm_preds, y_true, base_times, uid)`.

### Parameters Tuned

| Model   | Parameter      | Range      |
|---------|---------------|------------|
| BiLSTM  | units          | 64–256     |
| BiLSTM  | dropout        | 0.1–0.4    |
| BiLSTM  | batch_size     | 16–64      |
| LightGBM | n_estimators  | 200–800    |
| LightGBM | learning_rate | 0.01–0.1   |
| LightGBM | num_leaves    | 31–127     |

### Trial Count

- BiLSTM: 15 trials, max 30 epochs each with EarlyStopping(patience=5) during tuning
- LightGBM: 30 trials
- Estimated time: 45–90 minutes per sensor

### Persistence

Best params saved to `saved_models/{uid}/best_params.json`:
```json
{
  "bilstm": {"units": 192, "dropout": 0.2, "batch_size": 32},
  "lgbm":   {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63}
}
```

### Integration into retrain flow

`retrain_sensor(uid)` in `services/retrain.py`:
- If `best_params.json` exists → use saved params, skip Optuna
- If `best_params.json` does not exist → run tuning first, save params, then train

### New API Endpoint

`POST /tune/{uid}` — triggers tuning + training as background task. Returns 202.

### Model Status

New status value: `tuning`. Added to `model_metadata.status` ENUM (or varchar — no schema change needed if varchar).

### UI Changes (Laravel + TSX)

- New "Tune" button per row (beside existing Train button)
- `statusBadge()` in TSX handles new `tuning` status: blue spinner badge labeled "Tuning"
- Laravel controller adds `triggerMlTuneOne(string $uid)` method
- Route: `POST ml-forecast/tune/{uid}`

---

## Section 3: Confidence Intervals — Backend

### LightGBM Changes (`lgbm.py`)

Train 3 models per sensor in `train_lgbm()`:

```python
models = {
    "point": lgb.LGBMRegressor(objective='regression', ...),
    "lower": lgb.LGBMRegressor(objective='quantile', alpha=0.1, ...),
    "upper": lgb.LGBMRegressor(objective='quantile', alpha=0.9, ...),
}
```

Saved as `lgbm.pkl`, `lgbm_lower.pkl`, `lgbm_upper.pkl` in `saved_models/{uid}/`.

`predict_lgbm()` returns:
```python
{
    "point": np.ndarray,  # shape (6, 14)
    "lower": np.ndarray,  # shape (6, 14)
    "upper": np.ndarray,  # shape (6, 14)
}
```

### Pipeline Changes (`pipeline.py`)

`run_predict(uid)` passes lower/upper arrays to `save_predictions()`.

### DB Migration

```sql
ALTER TABLE predictions
  ADD COLUMN lower_bounds JSON NULL,
  ADD COLUMN upper_bounds JSON NULL;
```

Format per row (one row = one step):
```json
{"pm_25": 68.2, "pm_10": 71.5, "aqi_index": 64.1, ...}
```

### API Response

Each prediction step object gains two new nullable fields:
```json
{
  "step": 1,
  "target_time": "2026-05-25 09:00:00",
  "aqi_index": 75.3,
  ...
  "lower_bounds": {"aqi_index": 63.1, "pm_25": 42.0, ...},
  "upper_bounds": {"aqi_index": 87.5, "pm_25": 61.3, ...}
}
```

---

## Section 4: Confidence Intervals — Frontend

### TypeScript Interface Update

```ts
interface Prediction {
  // ... existing fields ...
  lower_bounds: Record<string, number> | null
  upper_bounds: Record<string, number> | null
}
```

### Table Cell Rendering

Each +1h–+6h cell:
```
75.3          ← point prediction (large)
63–88         ← lower–upper range (small, gray text below)
```

If `lower_bounds` or `upper_bounds` is null, render point value only (backward compatible with old predictions).

### Chart: Highcharts Arearange

```ts
const bandData: [number, number, number][] = sensor.predictions
  .sort((a, b) => a.step - b.step)
  .filter(p => p.lower_bounds && p.upper_bounds)
  .map(p => [
    toUtcMs(p.target_time),
    p.lower_bounds![param],
    p.upper_bounds![param],
  ])

series: [
  {
    type: 'arearange',
    data: bandData,
    color: '#7c3aed',
    fillOpacity: 0.15,
    lineWidth: 0,
    enableMouseTracking: false,
  },
  {
    type: 'spline',
    name: label,
    data: seriesData,
    color: '#7c3aed',
  }
]
```

---

## Files to Create / Modify

| Action | File |
|--------|------|
| Create | `app/models/tuning.py` |
| Modify | `app/models/preprocessor.py` — add `detect_and_remove_anomalies()`, update pipeline functions |
| Modify | `app/models/lgbm.py` — train/predict 3 quantile models |
| Modify | `app/models/pipeline.py` — pass lower/upper to save_predictions |
| Modify | `app/services/retrain.py` — check best_params.json, add tuning status |
| Modify | `app/db.py` — update `save_predictions()` to write lower/upper_bounds |
| Modify | `app/routers/training.py` — add `POST /tune/{uid}` endpoint |
| Migrate | `predictions` table — add `lower_bounds JSON`, `upper_bounds JSON` |
| Modify | `bc-enviro-web/.../ml-forecast/index.tsx` — update Prediction interface, cell rendering, chart |
| Modify | `bc-enviro-web/.../PlatformAirQualityController.php` — add `triggerMlTuneOne()` |
| Modify | `bc-enviro-web/routes/web.php` — add tune route |
