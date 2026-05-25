# ML Quality Improvements (Sub-project A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add anomaly detection before training, per-sensor Optuna hyperparameter tuning, and 80% confidence intervals (quantile LightGBM) to the air quality forecast pipeline — surfaced in both the forecast table and chart.

**Architecture:** Incremental modifications to existing pipeline files: add `detect_and_remove_anomalies()` to `preprocessor.py`, extend `bilstm.py`/`lgbm.py` to accept external hyperparams and train quantile models, add new `app/models/tuning.py` for Optuna, update `retrain_sensor()` to orchestrate tuning, and update the Laravel+TSX frontend for tune button, range cells, and arearange chart bands.

**Tech Stack:** Python 3.11, FastAPI, TensorFlow/Keras, LightGBM, Optuna, MySQL, TypeScript, Highcharts 12.x, Laravel 11, Tailwind CSS

---

## File Map

| Action | Path |
|--------|------|
| Migrate | `predictions` table — add `lower_bounds JSON NULL`, `upper_bounds JSON NULL` |
| Modify | `requirements.txt` — add `optuna>=3.6` |
| Modify | `app/models/preprocessor.py` — add `detect_and_remove_anomalies()`, update pipeline functions |
| Modify | `app/models/bilstm.py` — accept `units`, `dropout`, `batch_size`, `patience` params |
| Modify | `app/models/lgbm.py` — accept `lgbm_params`, train 3 quantile models, `predict_lgbm()` returns dict |
| Modify | `app/models/pipeline.py` — handle dict from `predict_lgbm`, thread lower/upper through |
| Modify | `app/db.py` — `save_predictions()` writes JSON bounds, getters parse JSON |
| Create | `app/models/tuning.py` — Optuna objective functions + save/load best_params.json |
| Modify | `app/services/retrain.py` — tuning status, `run_tuning` flag, load best_params |
| Modify | `app/routers/training.py` — add `POST /tune/{uid}` |
| Modify | `tests/test_preprocessor.py` — tests for anomaly detection |
| Modify | `tests/test_lgbm.py` — tests for quantile predict dict return |
| Create | `tests/test_tuning.py` — tests for save/load best_params |
| Modify | `bc-enviro-web/app/Services/ForecastService.php` — add `triggerTuneOne()` |
| Modify | `bc-enviro-web/app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php` — add `triggerMlTuneOne()` |
| Modify | `bc-enviro-web/routes/web.php` — add tune route |
| Modify | `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx` — interface, table cells, chart band, tune button |

---

## Task 1: DB Migration + Install Optuna

**Files:**
- Migrate: `predictions` table on RESULT_DB (103.150.194.228:3306)
- Modify: `requirements.txt`

- [ ] **Step 1: Run DB migration on RESULT_DB**

Connect to 103.150.194.228:3306 (aqms_db) and run:

```sql
ALTER TABLE predictions
  ADD COLUMN lower_bounds JSON NULL,
  ADD COLUMN upper_bounds JSON NULL;
```

Verify:
```sql
DESCRIBE predictions;
-- Should show lower_bounds and upper_bounds as JSON columns
```

- [ ] **Step 2: Add optuna to requirements.txt**

Open `requirements.txt` and add after the existing dependencies:

```
optuna>=3.6
```

- [ ] **Step 3: Install optuna in venv**

```bash
cd air_quality_forecast
venv/Scripts/pip install optuna>=3.6
```

Expected: `Successfully installed optuna-X.X.X`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add optuna dependency and predictions confidence columns"
```

---

## Task 2: Anomaly Detection in preprocessor.py

**Files:**
- Modify: `app/models/preprocessor.py`
- Modify: `tests/test_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/test_preprocessor.py` and add at the bottom:

```python
from app.models.preprocessor import detect_and_remove_anomalies


def test_detect_and_remove_anomalies_replaces_spike(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[12, 0] = 99999.0  # extreme temporal spike
    result = detect_and_remove_anomalies(df)
    assert result.iloc[12, 0] < 99999.0


def test_detect_and_remove_anomalies_no_nans(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[5, 0] = 99999.0
    result = detect_and_remove_anomalies(df)
    assert not result.isnull().any().any()


def test_detect_and_remove_anomalies_preserves_normal(sample_hourly_df):
    result = detect_and_remove_anomalies(sample_hourly_df)
    # no more than 10% of values should change on normal data
    changed = (result.values != sample_hourly_df.values).sum()
    total = sample_hourly_df.size
    assert changed / total < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd air_quality_forecast
venv/Scripts/pytest tests/test_preprocessor.py::test_detect_and_remove_anomalies_replaces_spike -v
```

Expected: `FAILED` — `ImportError: cannot import name 'detect_and_remove_anomalies'`

- [ ] **Step 3: Implement detect_and_remove_anomalies in preprocessor.py**

Open `app/models/preprocessor.py`. Add the new function after the `resample_hourly` function (after line 43):

```python
def detect_and_remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        rolling_median = df[col].rolling(24, min_periods=1, center=True).median()
        rolling_std = df[col].rolling(24, min_periods=1, center=True).std().fillna(1.0)
        anomaly_mask = (df[col] - rolling_median).abs() > 3 * rolling_std
        df.loc[anomaly_mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
    return df
```

- [ ] **Step 4: Update preprocess_for_predict to call detect_and_remove_anomalies**

Find `preprocess_for_predict` in `app/models/preprocessor.py` (currently at line 112). Replace the entire function:

```python
def preprocess_for_predict(uid: str) -> tuple:
    df_raw    = fetch_sensor_data(uid, hours=N_INPUT_HOURS * 3)
    df_hourly = resample_hourly(df_raw)
    if len(df_hourly) < N_INPUT_HOURS:
        raise ValueError(
            f"Insufficient hourly data for uid={uid}: got {len(df_hourly)} rows, need {N_INPUT_HOURS}"
        )
    df_clean = detect_and_remove_anomalies(df_hourly)
    df_clean = clean(df_clean)
    df_timed = add_time_features(df_clean)
    df_norm, _ = normalize(df_timed, uid, fit=False)
    X = df_norm.values[-N_INPUT_HOURS:][np.newaxis, :, :]  # (1, 24, 18)
    return X, df_norm.index[-N_INPUT_HOURS:]
```

- [ ] **Step 5: Update preprocess_for_training to call detect_and_remove_anomalies**

Find `preprocess_for_training` in `app/models/preprocessor.py` (currently at line 126). Replace the entire function:

```python
def preprocess_for_training(uid: str) -> tuple:
    df_raw    = fetch_sensor_data(uid, hours=TRAIN_HISTORY_HOURS + 2)
    df_hourly = resample_hourly(df_raw)
    df_clean  = detect_and_remove_anomalies(df_hourly)
    df_clean  = clean(df_clean)
    df_timed  = add_time_features(df_clean)
    df_norm, _ = normalize(df_timed, uid, fit=True)
    return create_sequences(df_norm.values, N_INPUT_HOURS, N_FORECAST_HOURS,
                            n_target_cols=N_FEATURES)
```

- [ ] **Step 6: Run all preprocessor tests to verify they pass**

```bash
venv/Scripts/pytest tests/test_preprocessor.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/models/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: add rolling z-score anomaly detection to preprocessing pipeline"
```

---

## Task 3: Bilstm + LightGBM accept external params + quantile models

**Files:**
- Modify: `app/models/bilstm.py`
- Modify: `app/models/lgbm.py`
- Modify: `tests/test_lgbm.py`

- [ ] **Step 1: Write failing test for quantile predict_lgbm return**

Open `tests/test_lgbm.py`. Add imports at the top:

```python
import os
from app.models.lgbm import build_lgbm_features, train_lgbm, predict_lgbm
```

Add tests at the bottom:

```python
def test_predict_lgbm_returns_dict_with_three_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    N = 30
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]
    train_lgbm(bilstm_preds, y_true, base_times, "uid_test")

    bilstm_out = np.random.randn(N_FORECAST_HOURS, N_FEATURES)
    result = predict_lgbm(bilstm_out, pd.Timestamp("2024-06-15 14:00:00"), "uid_test")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"point", "lower", "upper"}
    assert result["point"].shape == (N_FORECAST_HOURS, N_FEATURES)
    assert result["lower"].shape == (N_FORECAST_HOURS, N_FEATURES)
    assert result["upper"].shape == (N_FORECAST_HOURS, N_FEATURES)


def test_train_lgbm_creates_three_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    N = 20
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]
    train_lgbm(bilstm_preds, y_true, base_times, "uid_test")

    uid_dir = tmp_path / "uid_test"
    assert (uid_dir / "lgbm.pkl").exists()
    assert (uid_dir / "lgbm_lower.pkl").exists()
    assert (uid_dir / "lgbm_upper.pkl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/pytest tests/test_lgbm.py::test_predict_lgbm_returns_dict_with_three_keys -v
```

Expected: `FAILED` — dict assertion fails because current `predict_lgbm` returns ndarray.

- [ ] **Step 3: Update bilstm.py to accept external hyperparams**

Open `app/models/bilstm.py`. Replace `build_bilstm` and `train_bilstm` with:

```python
def build_bilstm(units: int = BILSTM_UNITS, dropout: float = 0.2) -> tf.keras.Model:
    model = Sequential([
        Bidirectional(
            LSTM(units, return_sequences=True),
            input_shape=(N_INPUT_HOURS, N_SSA_FEATURES),
        ),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2)),
        Dropout(dropout),
        Dense(N_FORECAST_HOURS * N_FEATURES),
        Reshape((N_FORECAST_HOURS, N_FEATURES)),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_bilstm(
    X: np.ndarray,
    y: np.ndarray,
    uid: str,
    units: int = BILSTM_UNITS,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 20,
) -> tf.keras.Model:
    X_prepared = _prepare_input(X)
    model = build_bilstm(units=units, dropout=dropout)
    if len(X_prepared) < 10:
        raise ValueError(
            f"Need at least 10 training samples for validation_split=0.1, got {len(X_prepared)}"
        )
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(
        X_prepared, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model
```

- [ ] **Step 4: Update lgbm.py — train 3 quantile models + predict returns dict**

Open `app/models/lgbm.py`. Replace the entire file content:

```python
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from app.config import N_FORECAST_HOURS, N_FEATURES, MODELS_DIR


def build_lgbm_features(bilstm_output: np.ndarray, base_time: pd.Timestamp) -> np.ndarray:
    rows = []
    for step in range(N_FORECAST_HOURS):
        t = base_time + pd.Timedelta(hours=step + 1)
        time_feats = [t.hour, t.dayofweek, t.month]
        rows.append(np.concatenate([bilstm_output[step], time_feats]))
    return np.array(rows)


def train_lgbm(
    bilstm_preds: np.ndarray,
    y_true: np.ndarray,
    base_times: list,
    uid: str,
    lgbm_params: dict = None,
) -> None:
    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        feats = build_lgbm_features(bilstm_preds[i], bt)
        X_all.append(feats)
        y_all.append(y_true[i])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    base = dict(
        n_estimators=400, learning_rate=0.03, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    if lgbm_params:
        base.update(lgbm_params)

    uid_dir = os.path.join(MODELS_DIR, uid)
    os.makedirs(uid_dir, exist_ok=True)

    for suffix, extra in [
        ("",       {}),
        ("_lower", {"objective": "quantile", "alpha": 0.1}),
        ("_upper", {"objective": "quantile", "alpha": 0.9}),
    ]:
        params = {**base, **extra}
        model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        model.fit(X_all, y_all)
        joblib.dump(model, os.path.join(uid_dir, f"lgbm{suffix}.pkl"))


def predict_lgbm(
    bilstm_output: np.ndarray,
    base_time: pd.Timestamp,
    uid: str,
) -> dict:
    X = build_lgbm_features(bilstm_output, base_time)
    result = {}
    for key, suffix in [("point", ""), ("lower", "_lower"), ("upper", "_upper")]:
        path = os.path.join(MODELS_DIR, uid, f"lgbm{suffix}.pkl")
        result[key] = joblib.load(path).predict(X) if os.path.exists(path) else None
    return result
```

- [ ] **Step 5: Run lgbm tests to verify they pass**

```bash
venv/Scripts/pytest tests/test_lgbm.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/models/bilstm.py app/models/lgbm.py tests/test_lgbm.py
git commit -m "feat: bilstm/lgbm accept external hyperparams; lgbm trains quantile models"
```

---

## Task 4: Update pipeline.py — handle dict return + lower/upper in results

**Files:**
- Modify: `app/models/pipeline.py`

- [ ] **Step 1: Replace run_prediction in pipeline.py**

Open `app/models/pipeline.py`. Replace the entire file:

```python
import numpy as np
import pandas as pd
from app.config import FEATURE_COLS, N_FORECAST_HOURS, N_INPUT_HOURS
from app.models.preprocessor import (
    preprocess_for_predict,
    preprocess_for_training,
    denormalize,
)
from app.models.bilstm import predict_bilstm, train_bilstm
from app.models.lgbm import predict_lgbm, train_lgbm


def run_prediction(uid: str) -> list:
    X, timestamps = preprocess_for_predict(uid)
    last_time = timestamps[-1]

    bilstm_out = predict_bilstm(X, uid)                    # (1, 6, 14)
    lgbm_result = predict_lgbm(bilstm_out[0], last_time, uid)  # dict

    final_point = denormalize(lgbm_result["point"], uid)   # (6, 14)
    final_lower = denormalize(lgbm_result["lower"], uid) if lgbm_result.get("lower") is not None else None
    final_upper = denormalize(lgbm_result["upper"], uid) if lgbm_result.get("upper") is not None else None

    results = []
    for step in range(N_FORECAST_HOURS):
        target_time = last_time + pd.Timedelta(hours=step + 1)
        row = {"step": step + 1, "target_time": target_time.isoformat()}
        for i, col in enumerate(FEATURE_COLS):
            row[col] = float(final_point[step, i])
        row["lower_bounds"] = (
            {col: float(final_lower[step, i]) for i, col in enumerate(FEATURE_COLS)}
            if final_lower is not None else None
        )
        row["upper_bounds"] = (
            {col: float(final_upper[step, i]) for i, col in enumerate(FEATURE_COLS)}
            if final_upper is not None else None
        )
        results.append(row)
    return results


def run_training(uid: str, bilstm_params: dict = None, lgbm_params: dict = None) -> dict:
    X, y = preprocess_for_training(uid)

    if len(X) < 10:
        raise ValueError(f"Insufficient data for {uid}: only {len(X)} sequences")

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    bilstm_kw = bilstm_params or {}
    train_bilstm(X_train, y_train, uid, **bilstm_kw)

    bilstm_preds_train = np.array([
        predict_bilstm(X_train[i : i + 1], uid)[0]
        for i in range(len(X_train))
    ])
    base_times_train = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        for i in range(len(X_train))
    ]
    train_lgbm(bilstm_preds_train, y_train, base_times_train, uid, lgbm_params=lgbm_params)

    val_bilstm = np.array([
        predict_bilstm(X_val[i : i + 1], uid)[0]
        for i in range(len(X_val))
    ])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i], pd.Timestamp("2024-01-01") + pd.Timedelta(hours=split + i), uid)["point"]
        for i in range(len(X_val))
    ])
    mae = float(np.mean(np.abs(val_lgbm - y_val)))

    return {"training_samples": len(X), "mae_score": round(mae, 6)}
```

- [ ] **Step 2: Run existing pipeline tests to verify nothing broke**

```bash
venv/Scripts/pytest tests/test_pipeline.py -v
```

Expected: All existing tests PASS. (They call `run_training`/`run_prediction` with mocked DB — the optional param defaults preserve backward compat.)

- [ ] **Step 3: Commit**

```bash
git add app/models/pipeline.py
git commit -m "feat: pipeline threads lower/upper bounds through prediction; run_training accepts hyperparams"
```

---

## Task 5: Update db.py — save bounds + parse JSON

**Files:**
- Modify: `app/db.py`

- [ ] **Step 1: Update save_predictions to write lower_bounds/upper_bounds**

Open `app/db.py`. Add `import json` at the top (after `from datetime import datetime`):

```python
import json
```

Replace the entire `save_predictions` function:

```python
def save_predictions(uid: str, predicted_at: datetime, predictions: list[dict]) -> None:
    from app.config import FEATURE_COLS
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM predictions WHERE uid = %s AND predicted_at = %s",
            (uid, predicted_at),
        )
        cols_sql = ", ".join(f"`{c}`" for c in FEATURE_COLS)
        placeholders = ", ".join(["%s"] * len(FEATURE_COLS))
        for pred in predictions:
            values = tuple(pred.get(c) for c in FEATURE_COLS)
            lower_json = json.dumps(pred["lower_bounds"]) if pred.get("lower_bounds") else None
            upper_json = json.dumps(pred["upper_bounds"]) if pred.get("upper_bounds") else None
            cursor.execute(
                f"""
                INSERT INTO predictions
                    (uid, predicted_at, target_time, step, {cols_sql}, lower_bounds, upper_bounds)
                VALUES (%s, %s, %s, %s, {placeholders}, %s, %s)
                """,
                (uid, predicted_at, pred["target_time"], pred["step"], *values,
                 lower_json, upper_json),
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 2: Add _parse_bounds helper and update getters**

Add this helper function just before `get_latest_predictions`:

```python
def _parse_bounds(row: dict) -> dict:
    for field in ("lower_bounds", "upper_bounds"):
        val = row.get(field)
        if isinstance(val, str):
            try:
                row[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                row[field] = None
    return row
```

In `get_latest_predictions`, change the return line from:
```python
        return [dict(r) for r in cursor.fetchall()]
```
to:
```python
        return [_parse_bounds(dict(r)) for r in cursor.fetchall()]
```

In `get_all_latest_predictions`, change:
```python
            result[uid].append(dict(row))
```
to:
```python
            result[uid].append(_parse_bounds(dict(row)))
```

- [ ] **Step 3: Run API test to verify save/load round-trip**

```bash
venv/Scripts/pytest tests/test_api.py -v
```

Expected: All existing API tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/db.py
git commit -m "feat: save_predictions writes lower/upper JSON bounds; getters parse JSON on read"
```

---

## Task 6: Optuna Tuning Module (tuning.py)

**Files:**
- Create: `app/models/tuning.py`
- Create: `tests/test_tuning.py`

- [ ] **Step 1: Write failing tests for save/load best_params**

Create `tests/test_tuning.py`:

```python
import json
import os
import pytest
from app.models.tuning import save_best_params, load_best_params


def test_save_and_load_best_params_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    bilstm = {"units": 192, "dropout": 0.2, "batch_size": 32}
    lgbm   = {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 63}
    save_best_params("uid_x", bilstm, lgbm)
    result = load_best_params("uid_x")
    assert result["bilstm"] == bilstm
    assert result["lgbm"] == lgbm


def test_load_best_params_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    assert load_best_params("nonexistent_uid") is None


def test_save_best_params_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    save_best_params("uid_y", {"units": 64}, {"n_estimators": 200})
    assert (tmp_path / "uid_y" / "best_params.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/pytest tests/test_tuning.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'app.models.tuning'`

- [ ] **Step 3: Create app/models/tuning.py**

Create `app/models/tuning.py` with this complete content:

```python
import json
import logging
import os

import numpy as np
import optuna
import pandas as pd

from app.config import MODELS_DIR

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _bilstm_objective(trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from app.models.bilstm import _prepare_input, build_bilstm

    units      = trial.suggest_int("units", 64, 256, step=64)
    dropout    = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    X_prepared = _prepare_input(X_train)
    model = build_bilstm(units=units, dropout=dropout)
    split = max(1, int(len(X_prepared) * 0.1))
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(
        X_prepared[:-split], y_train[:-split],
        validation_data=(X_prepared[-split:], y_train[-split:]),
        epochs=30,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0,
    )
    tf.keras.backend.clear_session()
    return float(min(hist.history["val_loss"]))


def _lgbm_objective(
    trial,
    bilstm_preds: np.ndarray,
    y_true: np.ndarray,
    base_times: list,
) -> float:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    from app.models.lgbm import build_lgbm_features

    n_estimators  = trial.suggest_int("n_estimators", 200, 800, step=100)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    num_leaves    = trial.suggest_int("num_leaves", 31, 127)

    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        X_all.append(build_lgbm_features(bilstm_preds[i], bt))
        y_all.append(y_true[i])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    split = max(1, int(len(X_all) * 0.1))
    model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=num_leaves, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
    ))
    model.fit(X_all[:-split], y_all[:-split])
    preds = model.predict(X_all[-split:])
    return float(np.mean(np.abs(preds - y_all[-split:])))


def tune_bilstm(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 15) -> dict:
    logger.info(f"[tuning] BiLSTM — {n_trials} trials")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _bilstm_objective(trial, X_train, y_train),
        n_trials=n_trials,
    )
    logger.info(f"[tuning] BiLSTM best: {study.best_params}")
    return study.best_params


def tune_lgbm(
    bilstm_preds: np.ndarray,
    y_true: np.ndarray,
    base_times: list,
    n_trials: int = 30,
) -> dict:
    logger.info(f"[tuning] LightGBM — {n_trials} trials")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _lgbm_objective(trial, bilstm_preds, y_true, base_times),
        n_trials=n_trials,
    )
    logger.info(f"[tuning] LightGBM best: {study.best_params}")
    return study.best_params


def save_best_params(uid: str, bilstm_params: dict, lgbm_params: dict) -> None:
    path = os.path.join(MODELS_DIR, uid, "best_params.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"bilstm": bilstm_params, "lgbm": lgbm_params}, f, indent=2)


def load_best_params(uid: str) -> dict | None:
    path = os.path.join(MODELS_DIR, uid, "best_params.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tuning tests to verify they pass**

```bash
venv/Scripts/pytest tests/test_tuning.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/tuning.py tests/test_tuning.py
git commit -m "feat: add Optuna tuning module with per-sensor BiLSTM/LightGBM hyperparameter search"
```

---

## Task 7: Update retrain_sensor() for tuning workflow

**Files:**
- Modify: `app/services/retrain.py`

- [ ] **Step 1: Replace retrain.py with tuning-aware version**

Open `app/services/retrain.py`. Replace the entire file:

```python
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.db import upsert_metadata
from app.models.pipeline import run_training
from app.models.tuning import load_best_params, save_best_params, tune_bilstm, tune_lgbm

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str, run_tuning: bool = False) -> dict:
    upsert_metadata(uid, status="training")
    try:
        bilstm_kw: dict = {}
        lgbm_kw: dict   = {}

        if run_tuning:
            upsert_metadata(uid, status="tuning")
            logger.info(f"[{uid}] Starting hyperparameter tuning...")

            from app.models.preprocessor import preprocess_for_training
            from app.models.bilstm import train_bilstm, predict_bilstm

            X, y = preprocess_for_training(uid)
            split = int(len(X) * 0.9)
            X_tr, y_tr = X[:split], y[:split]

            bilstm_kw = tune_bilstm(X_tr, y_tr)

            # Temp train BiLSTM with best params to produce inputs for LightGBM tuning
            train_bilstm(X_tr, y_tr, uid, **bilstm_kw)
            bilstm_preds = np.array([
                predict_bilstm(X_tr[i : i + 1], uid)[0]
                for i in range(len(X_tr))
            ])
            base_times = [
                pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
                for i in range(len(X_tr))
            ]
            lgbm_kw = tune_lgbm(bilstm_preds, y_tr, base_times)
            save_best_params(uid, bilstm_kw, lgbm_kw)
            logger.info(f"[{uid}] Tuning complete: bilstm={bilstm_kw}, lgbm={lgbm_kw}")
            upsert_metadata(uid, status="training")
        else:
            best = load_best_params(uid)
            if best:
                bilstm_kw = best.get("bilstm", {})
                lgbm_kw   = best.get("lgbm", {})

        result = run_training(uid, bilstm_params=bilstm_kw, lgbm_params=lgbm_kw)
        trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_metadata(
            uid,
            status="ready",
            last_trained_at=trained_at,
            training_samples=result["training_samples"],
            mae_score=result["mae_score"],
            error_message=None,
        )
        logger.info(f"[{uid}] Training complete — samples={result['training_samples']} mae={result['mae_score']}")
        return {"uid": uid, "status": "ready", **result}

    except Exception as e:
        logger.error(f"[{uid}] Training failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
```

- [ ] **Step 2: Run all tests to verify nothing broke**

```bash
venv/Scripts/pytest tests/ -v --ignore=tests/test_api.py
```

Expected: All tests PASS. (`test_api.py` requires live DB, skip for unit tests.)

- [ ] **Step 3: Commit**

```bash
git add app/services/retrain.py
git commit -m "feat: retrain_sensor supports run_tuning flag; loads best_params.json when available"
```

---

## Task 8: Add /tune/{uid} FastAPI endpoint + Laravel wiring

**Files:**
- Modify: `app/routers/training.py`
- Modify: `bc-enviro-web/app/Services/ForecastService.php`
- Modify: `bc-enviro-web/app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php`
- Modify: `bc-enviro-web/routes/web.php`

- [ ] **Step 1: Add POST /tune/{uid} to FastAPI training router**

Open `app/routers/training.py`. Replace the entire file:

```python
from fastapi import APIRouter, BackgroundTasks, Depends
from app.auth import require_api_key
from app.db import get_all_uids
from app.services.retrain import retrain_sensor

router = APIRouter(tags=["training"])


@router.post("/retrain/all", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_all(background_tasks: BackgroundTasks):
    uids = get_all_uids()
    for uid in uids:
        background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {len(uids)} sensors"}


@router.post("/retrain/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_one(uid: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {uid}"}


@router.post("/tune/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
def tune_one(uid: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_sensor, uid, True)
    return {"message": f"Tuning + training queued for {uid}"}
```

- [ ] **Step 2: Add triggerTuneOne to ForecastService.php**

Open `bc-enviro-web/app/Services/ForecastService.php`. Add this method before the closing `}` of the class (before line 87):

```php
        public function triggerTuneOne(string $uid): bool {
            try {
                $response = Http::withHeaders($this->headers)->timeout(30)->post("{$this->baseUrl}/tune/{$uid}");
                return $response->successful();
            } catch (\Exception $e) {
                Log::error("ForecastService::triggerTuneOne failed for {$uid}: {$e->getMessage()}");
                return false;
            }
        }
```

- [ ] **Step 3: Add triggerMlTuneOne to PlatformAirQualityController.php**

Open the controller file. Find the closing `}` of `triggerMlRetrainOne` (around line 735). Add the new method immediately after it, before the outer closing `}`:

```php
        /**
         * Route: POST /aqms/dashboard/ml-forecast/tune/{uid}
         */
        public function triggerMlTuneOne(Request $request, string $uid) {
            try {
                $this->forecast->triggerTuneOne($uid);
                return response()->json(['message' => "Tuning queued for {$uid}"], 202);
            } catch (Exception $e) {
                return response()->json([
                    'message' => 'Failed to trigger tuning',
                    'error'   => config('app.debug') ? $e->getMessage() : 'Internal server error',
                ], 500);
            }
        }
```

- [ ] **Step 4: Add tune route to web.php**

Open `bc-enviro-web/routes/web.php`. Find line 264 (the retrain/{uid} route). Add the tune route immediately after it:

```php
                Route::post('ml-forecast/tune/{uid}', [PlatformAirQualityController::class, 'triggerMlTuneOne']);
```

- [ ] **Step 5: Verify FastAPI server starts without errors**

```bash
cd air_quality_forecast
venv/Scripts/uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Expected: Server starts, no import errors. Check `GET /docs` shows `/tune/{uid}` endpoint.

- [ ] **Step 6: Commit both projects**

```bash
cd air_quality_forecast
git add app/routers/training.py
git commit -m "feat: add POST /tune/{uid} endpoint for Optuna tuning + training"

cd ..\bc-enviro-web
git add app/Services/ForecastService.php
git add "app/Http/Controllers/BeAqms/Dashboard/PlatformAirQualityController.php"
git add routes/web.php
git commit -m "feat: add tune endpoint wiring (ForecastService, Controller, route)"
```

---

## Task 9: Frontend TSX — range cells, arearange chart, tune button

**Files:**
- Modify: `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx`

- [ ] **Step 1: Add Highcharts More import for arearange support**

Open `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx`.

Replace the top imports block:

```ts
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more'
import { getMetaContent } from '@/js/plugins/functions'
import { showModalDialog, closeModalDialog } from '@/js/plugins/modal'

HighchartsMore(Highcharts)

const showModal = showModalDialog
```

- [ ] **Step 2: Update Prediction interface to include bounds**

Find the `interface Prediction` block. Replace it:

```ts
interface Prediction {
    step: number
    target_time: string
    pm_25: number; pm_10: number; tsp: number; aqi_index: number
    noise: number; temp: number; humidity: number; mmhg: number
    aqi_index_pm25: number; aqi_index_pm10: number; aqi_index_tsp: number
    lower_bounds: Record<string, number> | null
    upper_bounds: Record<string, number> | null
}
```

- [ ] **Step 3: Add 'tuning' to statusBadge**

Find the `statusBadge` function. Replace the `map` object inside it:

```ts
    const map: Record<string, string> = {
        ready:     '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-green-100 text-green-700"><i class="fas fa-circle-check"></i> Ready</span>',
        training:  '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-blue-100 text-blue-700"><i class="fas fa-spinner fa-pulse"></i> Training</span>',
        tuning:    '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-indigo-100 text-indigo-700"><i class="fas fa-flask fa-pulse"></i> Tuning</span>',
        error:     '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-red-100 text-red-700"><i class="fas fa-circle-exclamation"></i> Error</span>',
        untrained: '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-gray-100 text-gray-600"><i class="fas fa-circle-minus"></i> Untrained</span>',
    }
```

- [ ] **Step 4: Update table cells to show confidence range + add Tune button**

Find the `renderTable` function. Locate the `cells` map block and the `chartBtn`/`trainBtn` definitions. Replace from the `const cells = [1,2,3,4,5,6].map(...)` through to the end of `trainBtn`:

```ts
            const cells = [1, 2, 3, 4, 5, 6].map(step => {
                const p = preds.find(x => x.step === step)
                if (!p) return `<td class="border-b px-3 py-2 text-center bg-purple-50/50 text-gray-300">—</td>`
                const val = +(p[param as keyof Prediction] as number ?? 0)
                const color = aqiColor(param, val)
                const cat = aqiCategory(param, val)
                const lb = p.lower_bounds?.[param]
                const ub = p.upper_bounds?.[param]
                const range = (lb != null && ub != null)
                    ? `<div class="text-[10px] text-gray-400 leading-none mt-0.5">${lb.toFixed(0)}–${ub.toFixed(0)}</div>`
                    : ''
                return `<td class="border-b px-3 py-2 text-center bg-purple-50/30">
                    <span class="inline-block px-2 py-0.5 rounded text-[11px] font-semibold ${color}">${val.toFixed(1)}</span>
                    ${cat ? `<div>${cat}</div>` : ''}
                    ${range}
                </td>`
            }).join('')

            const trend = trendArrow(preds, param)
            const staleness = stalenessInfo(sensor.last_predicted_at)

            const chartBtn = preds.length
                ? `<button class="btnShowChart inline-flex items-center gap-1 text-[11px] px-2 py-0.5 bg-purple-50 hover:bg-purple-100 text-purple-600 border border-purple-200 rounded transition-colors" data-uid="${sensor.uid}" data-alias="${sensor.uid_alias}">
                    <i class="fas fa-chart-line"></i>
                </button>`
                : ''

            const trainBtn = `<button class="btnTrainOne inline-flex items-center gap-1 text-[11px] px-2 py-0.5 bg-gray-50 hover:bg-gray-100 text-gray-600 border border-gray-200 rounded transition-colors" data-uid="${sensor.uid}">
                <i class="fas fa-dumbbell"></i>
            </button>`

            const tuneBtn = `<button class="btnTuneOne inline-flex items-center gap-1 text-[11px] px-2 py-0.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 border border-indigo-200 rounded transition-colors" data-uid="${sensor.uid}" title="Tune hyperparameters">
                <i class="fas fa-flask"></i>
            </button>`
```

Find the actions cell in the row template. Replace it:

```ts
                <td class="border-b px-4 py-2 text-center">
                    <div class="flex items-center justify-center gap-1">${tuneBtn}${trainBtn}${chartBtn}</div>
                </td>
```

- [ ] **Step 5: Wire up tune buttons in renderTable**

Find the section that wires up `.btnTrainOne` click handlers (after `// Wire up per-row train buttons`). Add the tune button wiring immediately after the closing `})` of the train button block:

```ts
        // Wire up per-row tune buttons
        tableBody.querySelectorAll<HTMLButtonElement>('.btnTuneOne').forEach(btn => {
            btn.addEventListener('click', async () => {
                const uid = btn.dataset.uid!
                btn.disabled = true
                btn.innerHTML = '<i class="fas fa-spinner fa-pulse"></i>'

                await fetch(`/aqms/dashboard/ml-forecast/tune/${uid}`, {
                    method: 'POST',
                    headers: { 'X-CSRF-TOKEN': csrfToken, 'Content-Type': 'application/json' }
                })

                btn.innerHTML = '<i class="fas fa-check"></i>'
                setTimeout(() => {
                    btn.disabled = false
                    btn.innerHTML = '<i class="fas fa-flask"></i>'
                    loadAll()
                }, 3000)
            })
        })
```

- [ ] **Step 6: Add arearange band to showForecastChart**

Find `showForecastChart`. Replace the `series` array and `seriesData` definition inside it:

```ts
        const seriesData: [number, number][] = sensor.predictions
            .sort((a, b) => a.step - b.step)
            .map(p => [toUtcMs(p.target_time), +(p[param as keyof Prediction] as number ?? 0)])

        const bandData: [number, number, number][] = sensor.predictions
            .sort((a, b) => a.step - b.step)
            .filter(p => p.lower_bounds != null && p.upper_bounds != null)
            .map(p => [
                toUtcMs(p.target_time),
                +(p.lower_bounds![param] ?? 0),
                +(p.upper_bounds![param] ?? 0),
            ])
```

And replace the `series` property in the Highcharts config:

```ts
                series: [
                    {
                        type: 'arearange',
                        name: 'Confidence',
                        data: bandData,
                        color: '#7c3aed',
                        fillOpacity: 0.12,
                        lineWidth: 0,
                        enableMouseTracking: false,
                        zIndex: 0,
                    } as Highcharts.SeriesArearangeOptions,
                    {
                        type: 'spline',
                        name: label,
                        data: seriesData,
                        color: '#7c3aed',
                        zIndex: 1,
                    } as Highcharts.SeriesSplineOptions,
                ]
```

- [ ] **Step 7: Build and verify**

```bash
cd bc-enviro-web
npm run build
```

Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 8: Commit**

```bash
cd bc-enviro-web
git add resources/js/main/be-aqms/ml-forecast/index.tsx
git commit -m "feat: confidence interval range in table cells, arearange chart band, tune button per sensor"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Anomaly detection ✓ (Task 2), Quantile confidence intervals ✓ (Tasks 3–5), Optuna tuning per-sensor ✓ (Tasks 6–8), Frontend table + chart bounds ✓ (Task 9), Tune button ✓ (Task 8–9)
- [x] **No placeholders:** All steps have concrete code blocks
- [x] **Type consistency:** `predict_lgbm()` returns `dict` in Task 3 and is consumed as `dict` in Task 4. `train_bilstm(**bilstm_kw)` uses dict unpacking — `bilstm_kw` keys match the new `train_bilstm` kwargs (`units`, `dropout`, `batch_size`). `lower_bounds` field name is consistent across db.py, pipeline.py, and TSX interface.
- [x] **Backward compat:** Old predictions without bounds show `None` for lower/upper — TSX only renders range when both are non-null. Old lgbm.pkl without lower/upper variants returns `None` in `predict_lgbm` — pipeline handles None before calling `denormalize`.
