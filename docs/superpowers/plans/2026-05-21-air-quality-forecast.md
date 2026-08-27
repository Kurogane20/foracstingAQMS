# Air Quality Forecast Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI microservice that predicts all air quality parameters 6 hours ahead for 9 sensors using SSA-BiLSTM-LightGBM, integrated with an existing Laravel + MySQL system.

**Architecture:** FastAPI service (port 8001) reads from existing `t_loggers` MySQL table, runs per-sensor SSA→BiLSTM→LightGBM pipeline, writes results to `predictions` table. Laravel calls the API hourly via scheduler and renders results as chart + table on the dashboard.

**Tech Stack:** Python 3.11, FastAPI 0.111, TensorFlow 2.16, LightGBM 4.3, scikit-learn 1.5, mysql-connector-python 8.4, pytest 8.2; Laravel 10.x (existing), Blade, ApexCharts

---

## File Map

### Python — `air_quality_forecast/`
| File | Responsibility |
|------|----------------|
| `app/config.py` | Constants, DB config, env loading |
| `app/db.py` | DB utilities: get UIDs, get predictions, get model status |
| `app/models/preprocessor.py` | Fetch from MySQL, resample hourly, clean outliers, normalize |
| `app/models/ssa.py` | SSA decomposition per feature column |
| `app/models/bilstm.py` | BiLSTM architecture, training, inference |
| `app/models/lgbm.py` | LightGBM post-processor, training, inference |
| `app/models/pipeline.py` | Orchestrate full predict/train pipeline per sensor |
| `app/services/predict.py` | Run prediction pipeline + save to DB + update metadata |
| `app/services/retrain.py` | Run training pipeline + save models + update metadata |
| `app/routers/prediction.py` | `/predict/*` and `/predictions/*` endpoints |
| `app/routers/training.py` | `/retrain/*` endpoints |
| `app/main.py` | FastAPI app entry point, `/status`, `/health` |
| `scripts/init_db.py` | Create `predictions` and `model_metadata` tables |
| `tests/conftest.py` | Shared pytest fixtures and mock data |
| `tests/test_preprocessor.py` | Unit tests for preprocessor pure functions |
| `tests/test_ssa.py` | Unit tests for SSA decomposition |
| `tests/test_bilstm.py` | Unit tests for BiLSTM model output shapes |
| `tests/test_lgbm.py` | Unit tests for LightGBM feature builder |
| `tests/test_api.py` | Integration tests for API endpoints |
| `requirements.txt` | Python dependencies with pinned versions |
| `.env.example` | Environment variable template |

### Laravel — additions to existing project
| File | Responsibility |
|------|----------------|
| `app/Services/ForecastService.php` | HTTP calls to FastAPI microservice |
| `app/Http/Controllers/ForecastController.php` | Handle `/forecast` routes |
| `resources/views/forecast/index.blade.php` | All-sensor overview dashboard |
| `resources/views/forecast/show.blade.php` | Single sensor chart + prediction table |
| `routes/web.php` | Add 3 forecast routes |
| `app/Console/Kernel.php` | Hourly predict + nightly retrain cron |

---

## Task 1: Project Scaffold

**Files:**
- Create: `air_quality_forecast/requirements.txt`
- Create: `air_quality_forecast/.env.example`
- Create: `air_quality_forecast/app/__init__.py`
- Create: `air_quality_forecast/app/models/__init__.py`
- Create: `air_quality_forecast/app/services/__init__.py`
- Create: `air_quality_forecast/app/routers/__init__.py`
- Create: `air_quality_forecast/tests/__init__.py`
- Create: `air_quality_forecast/saved_models/.gitkeep`
- Create: `air_quality_forecast/logs/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd air_quality_forecast
mkdir -p app/models app/services app/routers scripts tests saved_models logs
touch app/__init__.py app/models/__init__.py app/services/__init__.py app/routers/__init__.py tests/__init__.py
touch saved_models/.gitkeep logs/.gitkeep
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
tensorflow==2.16.1
lightgbm==4.3.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
mysql-connector-python==8.4.0
python-dotenv==1.0.1
joblib==1.4.2
pytest==8.2.1
httpx==0.27.0
```

- [ ] **Step 3: Create `.env.example`**

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

- [ ] **Step 4: Copy `.env.example` to `.env` and fill in real credentials**

```bash
cp .env.example .env
# Edit .env with real credentials
```

- [ ] **Step 5: Create virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .env.example app/ tests/ scripts/ saved_models/.gitkeep logs/.gitkeep
echo ".env" >> .gitignore
echo "saved_models/**/*.h5" >> .gitignore
echo "saved_models/**/*.pkl" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "venv/" >> .gitignore
git add .gitignore
git commit -m "feat: project scaffold with dependencies"
```

---

## Task 2: Config Module

**Files:**
- Create: `app/config.py`

- [ ] **Step 1: Create `app/config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

FEATURE_COLS = [
    "pm_25", "pm_25_correction", "pm_10", "pm_10_correction",
    "tsp", "tsp_correction", "noise", "temp", "mmhg", "humidity",
    "aqi_index_pm25", "aqi_index_pm10", "aqi_index_tsp", "aqi_index",
]

N_FEATURES = len(FEATURE_COLS)   # 14
N_INPUT_HOURS = 24
N_FORECAST_HOURS = 6
SSA_WINDOW = 12
BILSTM_UNITS = 64
TRAIN_HISTORY_HOURS = 24 * 90    # 90 days

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
```

- [ ] **Step 2: Verify config loads without error**

```bash
python3 -c "from app.config import DB_CONFIG, FEATURE_COLS, N_FEATURES; print('N_FEATURES:', N_FEATURES)"
```

Expected output: `N_FEATURES: 14`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add config module with constants and DB config"
```

---

## Task 3: Database Init Script

**Files:**
- Create: `scripts/init_db.py`

- [ ] **Step 1: Create `scripts/init_db.py`**

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from app.config import DB_CONFIG

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id               CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    uid              VARCHAR(100) NOT NULL,
    predicted_at     DATETIME NOT NULL,
    target_time      DATETIME NOT NULL,
    step             TINYINT NOT NULL,
    pm_25            DOUBLE,
    pm_25_correction DOUBLE,
    pm_10            DOUBLE,
    pm_10_correction DOUBLE,
    tsp              DOUBLE,
    tsp_correction   DOUBLE,
    noise            DOUBLE,
    temp             DOUBLE,
    mmhg             DOUBLE,
    humidity         DOUBLE,
    aqi_index_pm25   DOUBLE,
    aqi_index_pm10   DOUBLE,
    aqi_index_tsp    DOUBLE,
    aqi_index        DOUBLE,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_uid_target (uid, target_time),
    INDEX idx_predicted_at (predicted_at)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS model_metadata (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    uid               VARCHAR(100) NOT NULL UNIQUE,
    last_trained_at   DATETIME,
    last_predicted_at DATETIME,
    training_samples  INT,
    mae_score         DOUBLE,
    status            ENUM('untrained','ready','training','error') DEFAULT 'untrained',
    error_message     TEXT,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print("Tables created: predictions, model_metadata")
```

- [ ] **Step 2: Run the init script**

```bash
python3 scripts/init_db.py
```

Expected output: `Tables created: predictions, model_metadata`

- [ ] **Step 3: Commit**

```bash
git add scripts/init_db.py
git commit -m "feat: add DB init script for predictions and model_metadata tables"
```

---

## Task 4: DB Utilities

**Files:**
- Create: `app/db.py`

- [ ] **Step 1: Create `app/db.py`**

```python
from datetime import datetime
import mysql.connector
from app.config import DB_CONFIG


def get_all_uids() -> list[str]:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT uid FROM t_loggers WHERE deleted_at IS NULL ORDER BY uid"
    )
    uids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return uids


def get_latest_predictions(uid: str) -> list[dict]:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT * FROM predictions
        WHERE uid = %s
          AND predicted_at = (
              SELECT MAX(predicted_at) FROM predictions WHERE uid = %s
          )
        ORDER BY step ASC
        """,
        (uid, uid),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_model_status() -> list[dict]:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM model_metadata ORDER BY uid")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def upsert_metadata(
    uid: str,
    status: str,
    last_trained_at: datetime = None,
    last_predicted_at: datetime = None,
    training_samples: int = None,
    mae_score: float = None,
    error_message: str = None,
) -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO model_metadata
            (uid, status, last_trained_at, last_predicted_at,
             training_samples, mae_score, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            status           = VALUES(status),
            last_trained_at  = COALESCE(VALUES(last_trained_at), last_trained_at),
            last_predicted_at= COALESCE(VALUES(last_predicted_at), last_predicted_at),
            training_samples = COALESCE(VALUES(training_samples), training_samples),
            mae_score        = COALESCE(VALUES(mae_score), mae_score),
            error_message    = VALUES(error_message)
        """,
        (uid, status, last_trained_at, last_predicted_at,
         training_samples, mae_score, error_message),
    )
    conn.commit()
    conn.close()


def save_predictions(uid: str, predicted_at: datetime, predictions: list[dict]) -> None:
    from app.config import FEATURE_COLS
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM predictions WHERE uid = %s AND predicted_at = %s",
        (uid, predicted_at),
    )
    cols_sql = ", ".join(FEATURE_COLS)
    placeholders = ", ".join(["%s"] * len(FEATURE_COLS))
    for pred in predictions:
        values = tuple(pred.get(c) for c in FEATURE_COLS)
        cursor.execute(
            f"""
            INSERT INTO predictions
                (uid, predicted_at, target_time, step, {cols_sql})
            VALUES (%s, %s, %s, %s, {placeholders})
            """,
            (uid, predicted_at, pred["target_time"], pred["step"], *values),
        )
    conn.commit()
    conn.close()
```

- [ ] **Step 2: Commit**

```bash
git add app/db.py
git commit -m "feat: add DB utility functions"
```

---

## Task 5: Preprocessor

**Files:**
- Create: `app/models/preprocessor.py`
- Create: `tests/conftest.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write failing tests for preprocessor pure functions**

Create `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS


@pytest.fixture
def sample_hourly_df():
    """24 rows of hourly sensor data, all columns present."""
    np.random.seed(42)
    index = pd.date_range("2024-01-01", periods=N_INPUT_HOURS, freq="h")
    data = {col: np.random.uniform(0, 100, N_INPUT_HOURS) for col in FEATURE_COLS}
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_minute_df():
    """1440 rows of per-minute sensor data."""
    np.random.seed(42)
    n = 1440
    timestamps = [int(pd.Timestamp("2024-01-01").timestamp()) + i * 60 for i in range(n)]
    data = {"datetime_unix": timestamps}
    for col in FEATURE_COLS:
        data[col] = np.random.uniform(0, 100, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_sequence_data():
    """Array of shape (100, 14) for sequence creation tests."""
    np.random.seed(42)
    return np.random.uniform(0, 1, (100, len(FEATURE_COLS)))
```

Create `tests/test_preprocessor.py`:

```python
import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS
from app.models.preprocessor import resample_hourly, clean, create_sequences


def test_resample_hourly_returns_24_rows(sample_minute_df):
    result = resample_hourly(sample_minute_df)
    assert len(result) == 24
    assert list(result.columns) == FEATURE_COLS


def test_resample_hourly_index_is_datetime(sample_minute_df):
    result = resample_hourly(sample_minute_df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_clean_fills_missing_values(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[5, 0] = np.nan
    result = clean(df)
    assert not result.isnull().any().any()


def test_clean_caps_outliers(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[0, 0] = 999999.0
    result = clean(df)
    assert result.iloc[0, 0] < 999999.0


def test_create_sequences_output_shapes(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    n_samples = len(sample_sequence_data) - N_INPUT_HOURS - N_FORECAST_HOURS + 1
    assert X.shape == (n_samples, N_INPUT_HOURS, len(FEATURE_COLS))
    assert y.shape == (n_samples, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_create_sequences_values_are_contiguous(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    np.testing.assert_array_equal(X[0], sample_sequence_data[:N_INPUT_HOURS])
    np.testing.assert_array_equal(y[0], sample_sequence_data[N_INPUT_HOURS:N_INPUT_HOURS + N_FORECAST_HOURS])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — functions not yet defined.

- [ ] **Step 3: Create `app/models/preprocessor.py`**

```python
import os
import numpy as np
import pandas as pd
import joblib
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from app.config import (
    DB_CONFIG, FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS,
    TRAIN_HISTORY_HOURS, MODELS_DIR,
)


def fetch_sensor_data(uid: str, hours: int = N_INPUT_HOURS) -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    cutoff_unix = int(pd.Timestamp.now().timestamp()) - hours * 3600
    query = """
        SELECT datetime_unix, pm_25, pm_25_correction, pm_10, pm_10_correction,
               tsp, tsp_correction, noise, temp, mmhg, humidity,
               aqi_index_pm25, aqi_index_pm10, aqi_index_tsp, aqi_index
        FROM t_loggers
        WHERE uid = %s AND datetime_unix >= %s AND deleted_at IS NULL
        ORDER BY datetime_unix ASC
    """
    df = pd.read_sql(query, conn, params=(uid, cutoff_unix))
    conn.close()
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["datetime_unix"], unit="s")
    df = df.set_index("timestamp")
    return df[FEATURE_COLS].resample("h").mean()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


def normalize(df: pd.DataFrame, uid: str, fit: bool = False) -> tuple:
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(df.values)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def denormalize(arr: np.ndarray, uid: str) -> np.ndarray:
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(arr)


def create_sequences(data: np.ndarray, n_in: int, n_out: int) -> tuple:
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i : i + n_in])
        y.append(data[i + n_in : i + n_in + n_out])
    return np.array(X), np.array(y)


def preprocess_for_predict(uid: str) -> tuple:
    df_raw = fetch_sensor_data(uid, hours=N_INPUT_HOURS + 2)
    df_hourly = resample_hourly(df_raw)
    df_clean = clean(df_hourly)
    df_norm, _ = normalize(df_clean, uid, fit=False)
    X = df_norm.values[-N_INPUT_HOURS:][np.newaxis, :, :]  # (1, 24, 14)
    return X, df_norm.index[-N_INPUT_HOURS:]


def preprocess_for_training(uid: str) -> tuple:
    df_raw = fetch_sensor_data(uid, hours=TRAIN_HISTORY_HOURS + 2)
    df_hourly = resample_hourly(df_raw)
    df_clean = clean(df_hourly)
    df_norm, _ = normalize(df_clean, uid, fit=True)
    return create_sequences(df_norm.values, N_INPUT_HOURS, N_FORECAST_HOURS)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/preprocessor.py tests/conftest.py tests/test_preprocessor.py
git commit -m "feat: add preprocessor with resample, clean, normalize, sequence creation"
```

---

## Task 6: SSA Decomposition

**Files:**
- Create: `app/models/ssa.py`
- Create: `tests/test_ssa.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ssa.py`:

```python
import numpy as np
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, SSA_WINDOW
from app.models.ssa import ssa_decompose, apply_ssa_to_dataframe


def test_ssa_decompose_returns_two_arrays():
    series = np.sin(np.linspace(0, 4 * np.pi, N_INPUT_HOURS)) + np.random.randn(N_INPUT_HOURS) * 0.1
    trend, oscillation = ssa_decompose(series, SSA_WINDOW)
    assert trend.shape == (N_INPUT_HOURS,)
    assert oscillation.shape == (N_INPUT_HOURS,)


def test_ssa_decompose_trend_plus_oscillation_approximates_series():
    series = np.sin(np.linspace(0, 4 * np.pi, N_INPUT_HOURS))
    trend, oscillation = ssa_decompose(series, SSA_WINDOW)
    np.testing.assert_allclose(trend + oscillation, series, atol=1e-10)


def test_apply_ssa_doubles_feature_count():
    n_features = len(FEATURE_COLS)
    data = np.random.randn(N_INPUT_HOURS, n_features)
    result = apply_ssa_to_dataframe(data, SSA_WINDOW)
    assert result.shape == (N_INPUT_HOURS, n_features * 2)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ssa.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `app/models/ssa.py`**

```python
import numpy as np
from app.config import SSA_WINDOW


def ssa_decompose(series: np.ndarray, window: int = SSA_WINDOW) -> tuple:
    n = len(series)
    k = n - window + 1
    trajectory = np.array([series[i : i + window] for i in range(k)])
    U, sigma, Vt = np.linalg.svd(trajectory, full_matrices=False)
    trend = _diagonal_average(sigma[0] * np.outer(U[:, 0], Vt[0, :]), window, n)
    oscillation = series - trend
    return trend, oscillation


def _diagonal_average(mat: np.ndarray, window: int, n: int) -> np.ndarray:
    k = n - window + 1
    result = np.zeros(n)
    counts = np.zeros(n)
    for i in range(k):
        for j in range(window):
            result[i + j] += mat[i, j]
            counts[i + j] += 1
    return result / counts


def apply_ssa_to_dataframe(data: np.ndarray, window: int = SSA_WINDOW) -> np.ndarray:
    n_timesteps, n_features = data.shape
    output = np.zeros((n_timesteps, n_features * 2))
    for i in range(n_features):
        trend, oscillation = ssa_decompose(data[:, i], window)
        output[:, i] = trend
        output[:, i + n_features] = oscillation
    return output
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_ssa.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/ssa.py tests/test_ssa.py
git commit -m "feat: add SSA decomposition with diagonal averaging"
```

---

## Task 7: BiLSTM Model

**Files:**
- Create: `app/models/bilstm.py`
- Create: `tests/test_bilstm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bilstm.py`:

```python
import numpy as np
import pytest
from app.config import N_INPUT_HOURS, N_FORECAST_HOURS, FEATURE_COLS, SSA_WINDOW
from app.models.bilstm import build_bilstm
from app.models.ssa import apply_ssa_to_dataframe

N_SSA_FEATURES = len(FEATURE_COLS) * 2


def test_bilstm_output_shape():
    model = build_bilstm()
    X_dummy = np.random.randn(2, N_INPUT_HOURS, N_SSA_FEATURES)
    output = model.predict(X_dummy, verbose=0)
    assert output.shape == (2, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_bilstm_model_has_correct_input_shape():
    model = build_bilstm()
    assert model.input_shape == (None, N_INPUT_HOURS, N_SSA_FEATURES)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_bilstm.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `app/models/bilstm.py`**

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from app.config import (
    BILSTM_UNITS, N_INPUT_HOURS, N_FORECAST_HOURS,
    N_FEATURES, MODELS_DIR,
)
from app.models.ssa import apply_ssa_to_dataframe

N_SSA_FEATURES = N_FEATURES * 2


def build_bilstm() -> tf.keras.Model:
    model = Sequential([
        Bidirectional(
            LSTM(BILSTM_UNITS, return_sequences=True),
            input_shape=(N_INPUT_HOURS, N_SSA_FEATURES),
        ),
        Dropout(0.2),
        Bidirectional(LSTM(BILSTM_UNITS)),
        Dropout(0.2),
        Dense(N_FORECAST_HOURS * N_FEATURES),
        Reshape((N_FORECAST_HOURS, N_FEATURES)),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_bilstm(X: np.ndarray, y: np.ndarray, uid: str) -> tf.keras.Model:
    X_ssa = np.array([apply_ssa_to_dataframe(x) for x in X])
    model = build_bilstm()
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_ssa, y,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model


def predict_bilstm(X: np.ndarray, uid: str) -> np.ndarray:
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    model = tf.keras.models.load_model(model_path)
    X_ssa = np.array([apply_ssa_to_dataframe(x) for x in X])
    return model.predict(X_ssa, verbose=0)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_bilstm.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/bilstm.py tests/test_bilstm.py
git commit -m "feat: add BiLSTM model with SSA preprocessing and multi-output"
```

---

## Task 8: LightGBM Post-Processor

**Files:**
- Create: `app/models/lgbm.py`
- Create: `tests/test_lgbm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_lgbm.py`:

```python
import numpy as np
import pandas as pd
import pytest
from app.config import N_FORECAST_HOURS, FEATURE_COLS
from app.models.lgbm import build_lgbm_features

N_FEATURES = len(FEATURE_COLS)


def test_build_lgbm_features_output_shape():
    bilstm_out = np.random.randn(N_FORECAST_HOURS, N_FEATURES)
    base_time = pd.Timestamp("2024-06-15 14:00:00")
    result = build_lgbm_features(bilstm_out, base_time)
    assert result.shape == (N_FORECAST_HOURS, N_FEATURES + 3)


def test_build_lgbm_features_time_values_are_correct():
    bilstm_out = np.zeros((N_FORECAST_HOURS, N_FEATURES))
    base_time = pd.Timestamp("2024-06-15 14:00:00")
    result = build_lgbm_features(bilstm_out, base_time)
    # step 0 → t+1 = 15:00
    assert result[0, N_FEATURES] == 15     # hour
    assert result[0, N_FEATURES + 1] == 5  # Saturday = dayofweek 5
    assert result[0, N_FEATURES + 2] == 6  # June = month 6
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lgbm.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `app/models/lgbm.py`**

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
) -> None:
    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        feats = build_lgbm_features(bilstm_preds[i], bt)
        X_all.append(feats)
        y_all.append(y_true[i])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    base_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
    model = MultiOutputRegressor(base_model)
    model.fit(X_all, y_all)
    model_path = os.path.join(MODELS_DIR, uid, "lgbm.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def predict_lgbm(bilstm_output: np.ndarray, base_time: pd.Timestamp, uid: str) -> np.ndarray:
    model_path = os.path.join(MODELS_DIR, uid, "lgbm.pkl")
    model = joblib.load(model_path)
    X = build_lgbm_features(bilstm_output, base_time)
    return model.predict(X)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lgbm.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/lgbm.py tests/test_lgbm.py
git commit -m "feat: add LightGBM post-processor with time features"
```

---

## Task 9: Pipeline Orchestration

**Files:**
- Create: `app/models/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from app.config import N_FORECAST_HOURS, FEATURE_COLS


def _make_mock_X():
    return np.random.randn(1, 24, len(FEATURE_COLS))


def _make_mock_timestamps():
    return pd.date_range("2024-01-01", periods=24, freq="h")


def test_run_prediction_returns_6_steps():
    from app.models.pipeline import run_prediction
    with patch("app.models.pipeline.preprocess_for_predict") as mock_pre, \
         patch("app.models.pipeline.predict_bilstm") as mock_bilstm, \
         patch("app.models.pipeline.predict_lgbm") as mock_lgbm, \
         patch("app.models.pipeline.denormalize") as mock_denorm:
        mock_pre.return_value = (_make_mock_X(), _make_mock_timestamps())
        mock_bilstm.return_value = np.random.randn(1, N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_lgbm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_denorm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        result = run_prediction("test_uid")
    assert len(result) == N_FORECAST_HOURS
    assert result[0]["step"] == 1
    assert result[5]["step"] == 6
    assert "pm_25" in result[0]
    assert "aqi_index" in result[0]


def test_run_prediction_step_numbers_are_sequential():
    from app.models.pipeline import run_prediction
    with patch("app.models.pipeline.preprocess_for_predict") as mock_pre, \
         patch("app.models.pipeline.predict_bilstm") as mock_bilstm, \
         patch("app.models.pipeline.predict_lgbm") as mock_lgbm, \
         patch("app.models.pipeline.denormalize") as mock_denorm:
        mock_pre.return_value = (_make_mock_X(), _make_mock_timestamps())
        mock_bilstm.return_value = np.random.randn(1, N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_lgbm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_denorm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        result = run_prediction("test_uid")
    steps = [r["step"] for r in result]
    assert steps == list(range(1, N_FORECAST_HOURS + 1))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `app/models/pipeline.py`**

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

    bilstm_out = predict_bilstm(X, uid)          # (1, 6, 14)
    lgbm_out = predict_lgbm(bilstm_out[0], last_time, uid)  # (6, 14)
    final = denormalize(lgbm_out, uid)            # (6, 14)

    results = []
    for step in range(N_FORECAST_HOURS):
        target_time = last_time + pd.Timedelta(hours=step + 1)
        row = {"step": step + 1, "target_time": target_time.isoformat()}
        for i, col in enumerate(FEATURE_COLS):
            row[col] = float(final[step, i])
        results.append(row)
    return results


def run_training(uid: str) -> dict:
    X, y = preprocess_for_training(uid)

    if len(X) < 10:
        raise ValueError(f"Insufficient data for {uid}: only {len(X)} sequences")

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_bilstm(X_train, y_train, uid)

    bilstm_preds_all = np.array([predict_bilstm(X[i : i + 1], uid)[0] for i in range(len(X))])
    base_times = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * N_INPUT_HOURS)
        for i in range(len(X))
    ]
    train_lgbm(bilstm_preds_all, y, base_times, uid)

    val_bilstm = np.array([predict_bilstm(X_val[i : i + 1], uid)[0] for i in range(len(X_val))])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i], base_times[split + i], uid)
        for i in range(len(X_val))
    ])
    mae = float(np.mean(np.abs(val_lgbm - y_val)))

    return {"training_samples": len(X), "mae_score": round(mae, 6)}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/pipeline.py tests/test_pipeline.py
git commit -m "feat: add prediction and training pipeline orchestration"
```

---

## Task 10: Prediction & Retrain Services

**Files:**
- Create: `app/services/predict.py`
- Create: `app/services/retrain.py`

- [ ] **Step 1: Create `app/services/predict.py`**

```python
import logging
from datetime import datetime, timezone
from app.db import save_predictions, upsert_metadata
from app.models.pipeline import run_prediction

logger = logging.getLogger(__name__)


def predict_sensor(uid: str) -> dict:
    try:
        predictions = run_prediction(uid)
        predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        save_predictions(uid, predicted_at, predictions)
        upsert_metadata(uid, status="ready", last_predicted_at=predicted_at)
        return {"uid": uid, "status": "success", "predictions": predictions}
    except FileNotFoundError:
        msg = "Model not trained yet — run /retrain first"
        logger.warning(f"[{uid}] {msg}")
        upsert_metadata(uid, status="untrained", error_message=msg)
        return {"uid": uid, "status": "error", "message": msg}
    except Exception as e:
        logger.error(f"[{uid}] Prediction failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
```

- [ ] **Step 2: Create `app/services/retrain.py`**

```python
import logging
from datetime import datetime, timezone
from app.db import upsert_metadata
from app.models.pipeline import run_training

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str) -> dict:
    upsert_metadata(uid, status="training")
    try:
        result = run_training(uid)
        trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_metadata(
            uid,
            status="ready",
            last_trained_at=trained_at,
            training_samples=result["training_samples"],
            mae_score=result["mae_score"],
            error_message=None,
        )
        return {"uid": uid, "status": "success", **result}
    except Exception as e:
        logger.error(f"[{uid}] Training failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
```

- [ ] **Step 3: Commit**

```bash
git add app/services/predict.py app/services/retrain.py
git commit -m "feat: add prediction and retrain services with DB persistence"
```

---

## Task 11: FastAPI Routers

**Files:**
- Create: `app/routers/prediction.py`
- Create: `app/routers/training.py`

- [ ] **Step 1: Create `app/routers/prediction.py`**

```python
from fastapi import APIRouter
from app.db import get_all_uids, get_latest_predictions
from app.services.predict import predict_sensor

router = APIRouter(tags=["prediction"])


@router.post("/predict/all")
def predict_all():
    uids = get_all_uids()
    return {"results": [predict_sensor(uid) for uid in uids]}


@router.post("/predict/{uid}")
def predict_one(uid: str):
    return predict_sensor(uid)


@router.get("/predictions/{uid}")
def get_predictions(uid: str):
    return {"uid": uid, "predictions": get_latest_predictions(uid)}
```

> Note: `/predict/all` must be defined **before** `/predict/{uid}` so FastAPI does not treat "all" as a uid.

- [ ] **Step 2: Create `app/routers/training.py`**

```python
from fastapi import APIRouter
from app.db import get_all_uids
from app.services.retrain import retrain_sensor

router = APIRouter(tags=["training"])


@router.post("/retrain/all")
def retrain_all():
    uids = get_all_uids()
    return {"results": [retrain_sensor(uid) for uid in uids]}


@router.post("/retrain/{uid}")
def retrain_one(uid: str):
    return retrain_sensor(uid)
```

- [ ] **Step 3: Commit**

```bash
git add app/routers/prediction.py app/routers/training.py
git commit -m "feat: add prediction and training API routers"
```

---

## Task 12: FastAPI Main App + API Tests

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_returns_list(client):
    with patch("app.main.get_model_status", return_value=[]):
        response = client.get("/status")
    assert response.status_code == 200
    assert "sensors" in response.json()


def test_predict_one_calls_service(client):
    with patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "success", "predictions": []}
        response = client.post("/predict/uid_001")
    assert response.status_code == 200
    mock.assert_called_once_with("uid_001")


def test_predict_all_calls_service_for_each_uid(client):
    with patch("app.routers.prediction.get_all_uids", return_value=["u1", "u2"]), \
         patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "u1", "status": "success", "predictions": []}
        response = client.post("/predict/all")
    assert response.status_code == 200
    assert mock.call_count == 2


def test_retrain_one_calls_service(client):
    with patch("app.routers.training.retrain_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "success"}
        response = client.post("/retrain/uid_001")
    assert response.status_code == 200
    mock.assert_called_once_with("uid_001")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError` — `app.main` not found.

- [ ] **Step 3: Create `app/main.py`**

```python
import logging
import os
from fastapi import FastAPI
from app.db import get_model_status
from app.routers.prediction import router as prediction_router
from app.routers.training import router as training_router
from app.config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Air Quality Forecast API", version="1.0.0")
app.include_router(prediction_router)
app.include_router(training_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {"sensors": get_model_status()}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI app entry point with health and status endpoints"
```

---

## Task 13: Verify Service Runs Locally

- [ ] **Step 1: Start the FastAPI server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

- [ ] **Step 2: Verify health endpoint**

Open browser or run:
```bash
curl http://localhost:8001/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 3: Check interactive docs**

Open `http://localhost:8001/docs` in browser. Verify all 7 endpoints appear:
- GET `/health`
- GET `/status`
- POST `/predict/all`
- POST `/predict/{uid}`
- GET `/predictions/{uid}`
- POST `/retrain/all`
- POST `/retrain/{uid}`

- [ ] **Step 4: Stop server and commit**

```bash
git add -A
git commit -m "chore: verify FastAPI service runs and all endpoints are accessible"
```

---

## Task 14: Laravel — ForecastService

**Files:**
- Create: `app/Services/ForecastService.php` (in Laravel project)

- [ ] **Step 1: Add `FORECAST_API_URL` to Laravel `.env`**

Add to the Laravel `.env` file:
```env
FORECAST_API_URL=http://localhost:8001
```

- [ ] **Step 2: Add to `config/services.php`**

```php
'forecast' => [
    'url' => env('FORECAST_API_URL', 'http://localhost:8001'),
],
```

- [ ] **Step 3: Create `app/Services/ForecastService.php`**

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class ForecastService
{
    private string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = config('services.forecast.url');
    }

    public function getModelStatus(): array
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/status");
            return $response->successful() ? $response->json('sensors', []) : [];
        } catch (\Exception $e) {
            Log::warning("ForecastService::getModelStatus failed: {$e->getMessage()}");
            return [];
        }
    }

    public function getLatestPredictions(string $uid): array
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/predictions/{$uid}");
            return $response->successful() ? $response->json('predictions', []) : [];
        } catch (\Exception $e) {
            Log::warning("ForecastService::getLatestPredictions failed for {$uid}: {$e->getMessage()}");
            return [];
        }
    }

    public function triggerPredict(string $uid): bool
    {
        try {
            $response = Http::timeout(30)->post("{$this->baseUrl}/predict/{$uid}");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerPredict failed for {$uid}: {$e->getMessage()}");
            return false;
        }
    }

    public function triggerPredictAll(): bool
    {
        try {
            $response = Http::timeout(120)->post("{$this->baseUrl}/predict/all");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerPredictAll failed: {$e->getMessage()}");
            return false;
        }
    }

    public function triggerRetrainAll(): bool
    {
        try {
            $response = Http::timeout(3600)->post("{$this->baseUrl}/retrain/all");
            return $response->successful();
        } catch (\Exception $e) {
            Log::error("ForecastService::triggerRetrainAll failed: {$e->getMessage()}");
            return false;
        }
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add app/Services/ForecastService.php config/services.php .env
git commit -m "feat: add ForecastService for FastAPI integration"
```

---

## Task 15: Laravel — Controller, Routes, Views

**Files:**
- Create: `app/Http/Controllers/ForecastController.php`
- Modify: `routes/web.php`
- Create: `resources/views/forecast/index.blade.php`
- Create: `resources/views/forecast/show.blade.php`

- [ ] **Step 1: Create `app/Http/Controllers/ForecastController.php`**

```php
<?php

namespace App\Http\Controllers;

use App\Services\ForecastService;
use Illuminate\Http\Request;

class ForecastController extends Controller
{
    public function __construct(private ForecastService $forecast) {}

    public function index()
    {
        $statuses = $this->forecast->getModelStatus();
        return view('forecast.index', compact('statuses'));
    }

    public function show(string $uid)
    {
        $predictions = $this->forecast->getLatestPredictions($uid);
        return view('forecast.show', compact('uid', 'predictions'));
    }

    public function predict(string $uid)
    {
        $this->forecast->triggerPredict($uid);
        return redirect()->route('forecast.show', $uid)
                         ->with('success', 'Prediksi berhasil diperbarui');
    }
}
```

- [ ] **Step 2: Add routes to `routes/web.php`**

```php
use App\Http\Controllers\ForecastController;

Route::prefix('forecast')->name('forecast.')->group(function () {
    Route::get('/', [ForecastController::class, 'index'])->name('index');
    Route::get('/{uid}', [ForecastController::class, 'show'])->name('show');
    Route::post('/{uid}/predict', [ForecastController::class, 'predict'])->name('predict');
});
```

- [ ] **Step 3: Create `resources/views/forecast/index.blade.php`**

```blade
@extends('layouts.app')

@section('content')
<div class="container-fluid">
    <h2 class="mb-4">Forecast Status — Semua Sensor</h2>

    @if(session('success'))
        <div class="alert alert-success">{{ session('success') }}</div>
    @endif

    <div class="row">
        @forelse($statuses as $sensor)
        <div class="col-md-4 mb-3">
            <div class="card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title font-monospace">{{ $sensor['uid'] }}</h5>
                    <span class="badge
                        @if($sensor['status'] === 'ready') bg-success
                        @elseif($sensor['status'] === 'training') bg-warning text-dark
                        @elseif($sensor['status'] === 'error') bg-danger
                        @else bg-secondary @endif">
                        {{ strtoupper($sensor['status']) }}
                    </span>
                    @if($sensor['mae_score'])
                        <p class="mt-2 mb-1 small text-muted">MAE: {{ number_format($sensor['mae_score'], 4) }}</p>
                    @endif
                    @if($sensor['last_predicted_at'])
                        <p class="mb-1 small text-muted">
                            Last predicted: {{ $sensor['last_predicted_at'] }}
                        </p>
                    @endif
                    <a href="{{ route('forecast.show', $sensor['uid']) }}"
                       class="btn btn-sm btn-primary mt-2">Lihat Prediksi</a>
                </div>
            </div>
        </div>
        @empty
            <div class="col-12">
                <p class="text-muted">Belum ada data. Jalankan training terlebih dahulu.</p>
            </div>
        @endforelse
    </div>
</div>
@endsection
```

- [ ] **Step 4: Create `resources/views/forecast/show.blade.php`**

```blade
@extends('layouts.app')

@section('content')
<div class="container-fluid">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h2>Prediksi 6 Jam — <span class="font-monospace text-primary">{{ $uid }}</span></h2>
        <div>
            <form action="{{ route('forecast.predict', $uid) }}" method="POST" class="d-inline">
                @csrf
                <button class="btn btn-outline-primary btn-sm">Perbarui Prediksi</button>
            </form>
            <a href="{{ route('forecast.index') }}" class="btn btn-outline-secondary btn-sm ms-2">
                &larr; Kembali
            </a>
        </div>
    </div>

    @if(session('success'))
        <div class="alert alert-success">{{ session('success') }}</div>
    @endif

    @if(empty($predictions))
        <div class="alert alert-warning">
            Belum ada prediksi untuk sensor ini. Klik "Perbarui Prediksi" atau tunggu cron berikutnya.
        </div>
    @else

    {{-- Parameter Selector --}}
    <div class="mb-3">
        <label class="form-label fw-bold">Tampilkan Parameter:</label>
        <select id="paramSelector" class="form-select w-auto d-inline-block ms-2" onchange="updateChart()">
            <option value="aqi_index">AQI Index</option>
            <option value="pm_25">PM2.5</option>
            <option value="pm_10">PM10</option>
            <option value="tsp">TSP</option>
            <option value="temp">Suhu (°C)</option>
            <option value="humidity">Kelembaban (%)</option>
            <option value="noise">Kebisingan (dB)</option>
            <option value="mmhg">Tekanan (mmHg)</option>
        </select>
    </div>

    {{-- Chart --}}
    <div class="card shadow-sm mb-4">
        <div class="card-body">
            <div id="forecastChart" style="height: 320px;"></div>
        </div>
    </div>

    {{-- Prediction Table --}}
    <div class="card shadow-sm">
        <div class="card-header fw-bold">Tabel Prediksi</div>
        <div class="table-responsive">
            <table class="table table-sm table-striped table-hover mb-0">
                <thead class="table-dark">
                    <tr>
                        <th>Waktu</th>
                        <th>PM2.5</th>
                        <th>PM10</th>
                        <th>TSP</th>
                        <th>Suhu</th>
                        <th>Kelembaban</th>
                        <th>Kebisingan</th>
                        <th>Tekanan</th>
                        <th>AQI</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach($predictions as $pred)
                    <tr>
                        <td class="font-monospace">
                            {{ \Carbon\Carbon::parse($pred['target_time'])->format('d M H:i') }}
                        </td>
                        <td>{{ number_format($pred['pm_25'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['pm_10'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['tsp'] ?? 0, 2) }}</td>
                        <td>{{ number_format($pred['temp'] ?? 0, 1) }}</td>
                        <td>{{ number_format($pred['humidity'] ?? 0, 1) }}%</td>
                        <td>{{ number_format($pred['noise'] ?? 0, 1) }}</td>
                        <td>{{ number_format($pred['mmhg'] ?? 0, 1) }}</td>
                        <td><strong>{{ number_format($pred['aqi_index'] ?? 0, 1) }}</strong></td>
                    </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    </div>

    @endif
</div>
@endsection

@push('scripts')
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<script>
const predictions = @json($predictions);
let chart = null;

function buildSeries(param) {
    return predictions.map(p => ({
        x: new Date(p.target_time).getTime(),
        y: parseFloat(p[param] ?? 0).toFixed(2)
    }));
}

function updateChart() {
    const param = document.getElementById('paramSelector').value;
    const label = document.getElementById('paramSelector').selectedOptions[0].text;

    if (chart) { chart.destroy(); }

    chart = new ApexCharts(document.getElementById('forecastChart'), {
        chart: { type: 'line', height: 320, toolbar: { show: false } },
        series: [{ name: 'Prediksi ' + label, data: buildSeries(param) }],
        stroke: { curve: 'smooth', dashArray: 5, width: 2 },
        xaxis: { type: 'datetime', labels: { format: 'HH:mm' } },
        yaxis: { title: { text: label }, decimalsInFloat: 2 },
        colors: ['#3b82f6'],
        tooltip: { x: { format: 'dd MMM HH:mm' } },
        markers: { size: 5 },
        annotations: {
            xaxis: [{
                x: predictions[0] ? new Date(predictions[0].target_time).getTime() : 0,
                borderColor: '#ef4444',
                label: { text: 'Sekarang +1h', style: { color: '#fff', background: '#ef4444' } }
            }]
        }
    });
    chart.render();
}

document.addEventListener('DOMContentLoaded', updateChart);
</script>
@endpush
```

- [ ] **Step 5: Commit**

```bash
git add app/Http/Controllers/ForecastController.php \
        routes/web.php \
        resources/views/forecast/
git commit -m "feat: add forecast controller, routes, and dashboard views"
```

---

## Task 16: Laravel Cron Scheduler

**Files:**
- Modify: `app/Console/Kernel.php`

- [ ] **Step 1: Add scheduled tasks to `app/Console/Kernel.php`**

Find the `schedule` method and add:

```php
protected function schedule(Schedule $schedule): void
{
    // existing schedules...

    // Trigger prediction for all sensors every hour
    $schedule->call(function () {
        app(\App\Services\ForecastService::class)->triggerPredictAll();
    })->hourly()->name('forecast:predict-all')->withoutOverlapping();

    // Retrain all models every night at 02:00
    $schedule->call(function () {
        app(\App\Services\ForecastService::class)->triggerRetrainAll();
    })->dailyAt('02:00')->name('forecast:retrain-all')->withoutOverlapping();
}
```

- [ ] **Step 2: Verify scheduler is registered**

```bash
php artisan schedule:list
```

Expected: Two new entries `forecast:predict-all` (hourly) and `forecast:retrain-all` (daily 02:00) appear.

- [ ] **Step 3: Commit**

```bash
git add app/Console/Kernel.php
git commit -m "feat: add hourly predict and nightly retrain cron jobs"
```

---

## Task 17: End-to-End Smoke Test

- [ ] **Step 1: Start FastAPI service**

```bash
cd air_quality_forecast
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

- [ ] **Step 2: Run DB init script**

```bash
python3 scripts/init_db.py
```

Expected: `Tables created: predictions, model_metadata`

- [ ] **Step 3: Trigger retrain for one sensor via API**

```bash
curl -X POST http://localhost:8001/retrain/<real_uid>
```

Expected JSON: `{"uid": "...", "status": "success", "training_samples": ..., "mae_score": ...}`

- [ ] **Step 4: Trigger prediction for the same sensor**

```bash
curl -X POST http://localhost:8001/predict/<real_uid>
```

Expected JSON: `{"uid": "...", "status": "success", "predictions": [6 items]}`

- [ ] **Step 5: Verify predictions are stored in DB**

```sql
SELECT uid, step, target_time, aqi_index FROM predictions LIMIT 6;
```

Expected: 6 rows with valid values.

- [ ] **Step 6: Open Laravel dashboard**

Navigate to `/forecast` in the browser. Verify sensor cards appear with status "READY".

- [ ] **Step 7: Open single sensor view**

Click a sensor → verify chart renders with 6 prediction points and table shows 6 rows.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end smoke test verified"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project scaffold | — |
| 2 | Config module | manual |
| 3 | DB init script | manual |
| 4 | DB utilities | — |
| 5 | Preprocessor | 6 unit tests |
| 6 | SSA decomposition | 3 unit tests |
| 7 | BiLSTM model | 2 unit tests |
| 8 | LightGBM post-processor | 2 unit tests |
| 9 | Pipeline orchestration | 2 unit tests |
| 10 | Prediction & retrain services | — |
| 11 | API routers | — |
| 12 | FastAPI main app | 5 integration tests |
| 13 | Service verification | manual |
| 14 | Laravel ForecastService | — |
| 15 | Laravel controller + views | — |
| 16 | Laravel cron scheduler | manual |
| 17 | End-to-end smoke test | manual |
