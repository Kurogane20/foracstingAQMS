# Accuracy Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match past predictions to actual sensor readings, compute per-step MAE, detect model drift, and auto-retrain degraded models.

**Architecture:** Add `actual_values JSON NULL` to the `predictions` table. An accuracy service resolves unmatched predictions by querying `t_loggers` for the closest reading within ±30 min. Rolling MAE over the last 24 resolved predictions is compared to training-time MAE; if the ratio exceeds 1.5 the sensor is marked "drifted" and a retrain thread is spawned. The hourly worker runs the resolution + drift check after each predict cycle.

**Tech Stack:** Python 3.11, FastAPI, MySQL (SENSOR_DB for `t_loggers`, RESULT_DB for `predictions`/`model_metadata`)

---

## Files

| Action | File |
|---|---|
| Migrate | RESULT_DB `predictions`, `model_metadata` |
| Modify | `app/db.py` |
| Create | `app/services/accuracy.py` |
| Create | `app/routers/accuracy.py` |
| Modify | `app/main.py` |
| Modify | `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx` |
| Test | `tests/test_db.py` (append) |
| Create | `tests/test_accuracy.py` |
| Create | `tests/test_accuracy_router.py` |

---

### Task 1: DB Migration — actual_values + drift columns

**Files:**
- Migrate: RESULT_DB `predictions` table
- Migrate: RESULT_DB `model_metadata` table

- [ ] **Step 1: Write the failing migration test**

```python
# tests/test_migration_accuracy.py
import mysql.connector
import pytest
from app.config import RESULT_DB_CONFIG


@pytest.fixture
def db():
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    yield conn
    conn.close()


def test_predictions_has_actual_values_column(db):
    cursor = db.cursor()
    cursor.execute("DESCRIBE predictions")
    cols = {row[0] for row in cursor.fetchall()}
    assert "actual_values" in cols


def test_model_metadata_has_drift_columns(db):
    cursor = db.cursor()
    cursor.execute("DESCRIBE model_metadata")
    cols = {row[0] for row in cursor.fetchall()}
    assert "drift_score" in cols
    assert "last_accuracy_check_at" in cols
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\nurch\OneDrive\Documents\project\air_quality_forecast
python -m pytest tests/test_migration_accuracy.py -v
```
Expected: FAIL — columns do not exist yet.

- [ ] **Step 3: Run the migration SQL against RESULT_DB**

Connect to RESULT_DB (host=103.150.194.228 port=3306 database=aqms_db user=admin password=Ubt@2026) and run:

```sql
ALTER TABLE predictions
    ADD COLUMN actual_values JSON NULL;

ALTER TABLE model_metadata
    ADD COLUMN drift_score FLOAT NULL,
    ADD COLUMN last_accuracy_check_at DATETIME NULL;
```

You can run this via MySQL CLI or any DB client:
```
mysql -h 103.150.194.228 -P 3306 -u admin -pUbt@2026 aqms_db
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_migration_accuracy.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_migration_accuracy.py
git commit -m "test: verify accuracy tracking DB migration"
```

---

### Task 2: Update db.py — four new functions

**Files:**
- Modify: `app/db.py`
- Modify: `tests/test_db.py` (append new tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_db.py`)

```python
# ---------------------------------------------------------------------------
# get_unresolved_predictions
# ---------------------------------------------------------------------------

def test_get_unresolved_predictions_returns_rows():
    from app.db import get_unresolved_predictions
    raw_row = {"uid": "s1", "target_time": datetime(2024, 1, 1, 10), "step": 1, "predicted_at": datetime(2024, 1, 1, 9)}
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_unresolved_predictions("s1")
    assert len(result) == 1
    assert result[0]["step"] == 1


# ---------------------------------------------------------------------------
# update_prediction_actuals
# ---------------------------------------------------------------------------

def test_update_prediction_actuals_executes_update():
    import json as _json
    from app.db import update_prediction_actuals
    conn_mock, cursor_mock = _make_conn_mock()
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        update_prediction_actuals("s1", datetime(2024, 1, 1, 10), {"aqi_index": 55.0})
    assert cursor_mock.execute.called
    call_args = cursor_mock.execute.call_args[0]
    assert "UPDATE" in call_args[0]
    stored = call_args[1][0]
    assert _json.loads(stored) == {"aqi_index": 55.0}


# ---------------------------------------------------------------------------
# get_resolved_predictions
# ---------------------------------------------------------------------------

def test_get_resolved_predictions_parses_actual_values():
    from app.db import get_resolved_predictions
    from app.config import FEATURE_COLS
    import json as _json
    raw_row = {"step": 1, "actual_values": _json.dumps({"aqi_index": 60.0})}
    raw_row.update({c: 50.0 for c in FEATURE_COLS})
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_resolved_predictions("s1", n=10)
    assert result[0]["actual_values"] == {"aqi_index": 60.0}


# ---------------------------------------------------------------------------
# update_drift_metadata
# ---------------------------------------------------------------------------

def test_update_drift_metadata_executes_update():
    from app.db import update_drift_metadata
    conn_mock, cursor_mock = _make_conn_mock()
    ts = datetime(2024, 1, 2, 0, 0, 0)
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        update_drift_metadata("s1", 1.8, ts)
    assert cursor_mock.execute.called
    call_sql = cursor_mock.execute.call_args[0][0]
    assert "UPDATE" in call_sql
    assert "drift_score" in call_sql
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_db.py -k "unresolved or update_prediction or resolved_pred or drift_metadata" -v
```
Expected: FAIL — functions not defined.

- [ ] **Step 3: Add the four functions to app/db.py**

Add `FEATURE_COLS` to the top-level import in `app/db.py`:
```python
# change existing line:
from app.config import SENSOR_DB_CONFIG, RESULT_DB_CONFIG
# to:
from app.config import SENSOR_DB_CONFIG, RESULT_DB_CONFIG, FEATURE_COLS
```

Then add at the bottom of `app/db.py`:

```python
def get_unresolved_predictions(uid: str) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT uid, target_time, step, predicted_at
            FROM predictions
            WHERE uid = %s AND target_time <= NOW() AND actual_values IS NULL
            ORDER BY target_time ASC
            """,
            (uid,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def update_prediction_actuals(uid: str, target_time, actual_values: dict) -> None:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE predictions
            SET actual_values = %s
            WHERE uid = %s AND target_time = %s AND actual_values IS NULL
            """,
            (json.dumps(actual_values), uid, target_time),
        )
        conn.commit()
    finally:
        conn.close()


def get_resolved_predictions(uid: str, n: int = 100) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cols_sql = ", ".join(f"`{c}`" for c in FEATURE_COLS)
        cursor.execute(
            f"""
            SELECT step, {cols_sql}, actual_values
            FROM predictions
            WHERE uid = %s AND actual_values IS NOT NULL
            ORDER BY target_time DESC
            LIMIT %s
            """,
            (uid, n),
        )
        rows = []
        for r in cursor.fetchall():
            row = dict(r)
            val = row.get("actual_values")
            if isinstance(val, str):
                try:
                    row["actual_values"] = json.loads(val)
                except json.JSONDecodeError:
                    row["actual_values"] = None
            rows.append(row)
        return rows
    finally:
        conn.close()


def update_drift_metadata(
    uid: str,
    drift_score: float | None,
    last_accuracy_check_at,
) -> None:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE model_metadata
            SET drift_score = %s, last_accuracy_check_at = %s
            WHERE uid = %s
            """,
            (drift_score, last_accuracy_check_at, uid),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_db.py -v
```
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db.py
git commit -m "feat: add accuracy DB functions to db.py"
```

---

### Task 3: Create app/services/accuracy.py

**Files:**
- Create: `app/services/accuracy.py`
- Create: `tests/test_accuracy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_accuracy.py`:

```python
import json
import threading
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, call

from app.config import FEATURE_COLS

UID = "sensor_01"
TARGET_TIME = datetime(2024, 1, 1, 12, 0, 0)


def _make_resolved_row(step: int, pred_val: float = 50.0, act_val: float = 55.0) -> dict:
    row = {col: pred_val for col in FEATURE_COLS}
    row["step"] = step
    row["actual_values"] = {col: act_val for col in FEATURE_COLS}
    return row


def _make_conn_mock():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# -------------------------------------------------------------------
# fetch_actual_for_step
# -------------------------------------------------------------------

def test_fetch_actual_for_step_returns_dict_when_found():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = {col: 50.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert result is not None
    assert "aqi_index" in result
    assert isinstance(result["aqi_index"], float)


def test_fetch_actual_for_step_returns_none_when_not_found():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = None
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert result is None


# -------------------------------------------------------------------
# resolve_prediction_actuals
# -------------------------------------------------------------------

def test_resolve_prediction_actuals_stores_actuals():
    from app.services.accuracy import resolve_prediction_actuals
    unresolved = [{"uid": UID, "target_time": TARGET_TIME, "step": 1, "predicted_at": TARGET_TIME}]
    actual = {col: 55.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.get_unresolved_predictions", return_value=unresolved), \
         patch("app.services.accuracy.fetch_actual_for_step", return_value=actual) as mock_fetch, \
         patch("app.services.accuracy.update_prediction_actuals") as mock_update:
        count = resolve_prediction_actuals(UID)
    assert count == 1
    mock_fetch.assert_called_once_with(UID, TARGET_TIME)
    mock_update.assert_called_once_with(UID, TARGET_TIME, actual)


def test_resolve_prediction_actuals_deduplicates_target_times():
    from app.services.accuracy import resolve_prediction_actuals
    same_time = TARGET_TIME
    unresolved = [
        {"uid": UID, "target_time": same_time, "step": 1, "predicted_at": datetime(2024, 1, 1, 10)},
        {"uid": UID, "target_time": same_time, "step": 1, "predicted_at": datetime(2024, 1, 1, 11)},
    ]
    actual = {col: 55.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.get_unresolved_predictions", return_value=unresolved), \
         patch("app.services.accuracy.fetch_actual_for_step", return_value=actual) as mock_fetch, \
         patch("app.services.accuracy.update_prediction_actuals"):
        count = resolve_prediction_actuals(UID)
    assert count == 1
    mock_fetch.assert_called_once()


# -------------------------------------------------------------------
# compute_accuracy
# -------------------------------------------------------------------

def test_compute_accuracy_returns_per_step_mae():
    from app.services.accuracy import compute_accuracy, DRIFT_WINDOW
    rows = [_make_resolved_row(step=i % 6 + 1) for i in range(DRIFT_WINDOW * 6)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        result = compute_accuracy(UID)
    assert "step_1" in result
    assert "overall_mae" in result
    assert result["overall_mae"] == pytest.approx(5.0, abs=0.01)


def test_compute_accuracy_returns_empty_when_no_resolved():
    from app.services.accuracy import compute_accuracy
    with patch("app.services.accuracy.get_resolved_predictions", return_value=[]):
        result = compute_accuracy(UID)
    assert result == {}


# -------------------------------------------------------------------
# detect_drift
# -------------------------------------------------------------------

def test_detect_drift_returns_ratio():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1, pred_val=50.0, act_val=60.0) for _ in range(DRIFT_WINDOW)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID, baseline_mae=5.0)
    assert ratio is not None
    assert ratio > 1.0


def test_detect_drift_returns_none_when_no_baseline():
    from app.services.accuracy import detect_drift
    ratio = detect_drift(UID, baseline_mae=None)
    assert ratio is None


def test_detect_drift_returns_none_when_insufficient_data():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1) for _ in range(DRIFT_WINDOW - 1)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID, baseline_mae=5.0)
    assert ratio is None


# -------------------------------------------------------------------
# check_and_auto_retrain
# -------------------------------------------------------------------

def test_check_and_auto_retrain_triggers_thread_on_drift():
    from app.services.accuracy import check_and_auto_retrain, DRIFT_THRESHOLD
    high_ratio = DRIFT_THRESHOLD + 0.5
    started_threads = []
    original_start = threading.Thread.start
    def capture_start(self):
        started_threads.append(self)
    with patch("app.services.accuracy.detect_drift", return_value=high_ratio), \
         patch("app.services.accuracy.update_drift_metadata"), \
         patch("app.services.accuracy.upsert_metadata") as mock_upsert, \
         patch.object(threading.Thread, "start", capture_start):
        result = check_and_auto_retrain(UID, baseline_mae=5.0)
    assert result is True
    assert len(started_threads) == 1
    mock_upsert.assert_called_once_with(UID, status="drifted")


def test_check_and_auto_retrain_no_trigger_below_threshold():
    from app.services.accuracy import check_and_auto_retrain, DRIFT_THRESHOLD
    low_ratio = DRIFT_THRESHOLD - 0.1
    with patch("app.services.accuracy.detect_drift", return_value=low_ratio), \
         patch("app.services.accuracy.update_drift_metadata"), \
         patch("app.services.accuracy.upsert_metadata") as mock_upsert:
        result = check_and_auto_retrain(UID, baseline_mae=5.0)
    assert result is False
    mock_upsert.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_accuracy.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Create app/services/accuracy.py**

```python
import json
import logging
import threading
import numpy as np
import mysql.connector
from datetime import datetime, timezone

from app.config import SENSOR_DB_CONFIG, FEATURE_COLS
from app.db import (
    get_unresolved_predictions,
    update_prediction_actuals,
    get_resolved_predictions,
    upsert_metadata,
    update_drift_metadata,
)

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 1.5
DRIFT_WINDOW = 24


def fetch_actual_for_step(uid: str, target_time: datetime) -> dict | None:
    target_unix = int(target_time.timestamp())
    window_secs = 1800
    conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cols_sql = ", ".join(FEATURE_COLS)
        cursor.execute(
            f"SELECT {cols_sql} FROM t_loggers "
            "WHERE uid = %s AND datetime_unix BETWEEN %s AND %s "
            "AND deleted_at IS NULL "
            "ORDER BY ABS(datetime_unix - %s) LIMIT 1",
            (uid, target_unix - window_secs, target_unix + window_secs, target_unix),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {k: float(v) for k, v in row.items() if v is not None}
    finally:
        conn.close()


def resolve_prediction_actuals(uid: str) -> int:
    unresolved = get_unresolved_predictions(uid)
    seen: set = set()
    resolved = 0
    for pred in unresolved:
        tt = pred["target_time"]
        if tt in seen:
            continue
        seen.add(tt)
        actual = fetch_actual_for_step(uid, tt)
        if actual:
            update_prediction_actuals(uid, tt, actual)
            resolved += 1
    return resolved


def compute_accuracy(uid: str) -> dict:
    rows = get_resolved_predictions(uid, n=DRIFT_WINDOW * 6)
    if not rows:
        return {}
    errors_by_step: dict[int, dict[str, list]] = {}
    for row in rows:
        step = row["step"]
        actual = row.get("actual_values") or {}
        step_errors = errors_by_step.setdefault(step, {col: [] for col in FEATURE_COLS})
        for col in FEATURE_COLS:
            pred_val = row.get(col)
            act_val = actual.get(col)
            if pred_val is not None and act_val is not None:
                step_errors[col].append(abs(float(pred_val) - float(act_val)))
    result: dict = {}
    all_errors: list[float] = []
    for step in sorted(errors_by_step):
        col_errors = errors_by_step[step]
        result[f"step_{step}"] = {
            col: round(float(np.mean(errs)), 4) if errs else None
            for col, errs in col_errors.items()
        }
        all_errors.extend(e for errs in col_errors.values() for e in errs)
    if all_errors:
        result["overall_mae"] = round(float(np.mean(all_errors)), 4)
    return result


def detect_drift(uid: str, baseline_mae: float | None) -> float | None:
    if not baseline_mae:
        return None
    rows = get_resolved_predictions(uid, n=DRIFT_WINDOW)
    if len(rows) < DRIFT_WINDOW:
        return None
    errors: list[float] = []
    for row in rows:
        actual = row.get("actual_values") or {}
        for col in FEATURE_COLS:
            pred_val = row.get(col)
            act_val = actual.get(col)
            if pred_val is not None and act_val is not None:
                errors.append(abs(float(pred_val) - float(act_val)))
    if not errors:
        return None
    return round(float(np.mean(errors)) / baseline_mae, 4)


def check_and_auto_retrain(uid: str, baseline_mae: float | None) -> bool:
    drift_score = detect_drift(uid, baseline_mae)
    update_drift_metadata(uid, drift_score, datetime.now(timezone.utc))
    if drift_score is not None and drift_score > DRIFT_THRESHOLD:
        upsert_metadata(uid, status="drifted")
        from app.services.retrain import retrain_sensor
        threading.Thread(target=retrain_sensor, args=(uid,), daemon=True).start()
        logger.warning(f"[drift] Auto-retrain triggered for {uid} (drift_score={drift_score})")
        return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_accuracy.py -v
```
Expected: All 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/accuracy.py tests/test_accuracy.py
git commit -m "feat: accuracy service — resolve actuals, compute MAE, detect drift, auto-retrain"
```

---

### Task 4: Create app/routers/accuracy.py

**Files:**
- Create: `app/routers/accuracy.py`
- Create: `tests/test_accuracy_router.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_accuracy_router.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.config import API_KEY


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def test_get_accuracy_uid_returns_200(client, auth_headers):
    with patch("app.routers.accuracy.resolve_prediction_actuals", return_value=3), \
         patch("app.routers.accuracy.compute_accuracy", return_value={"overall_mae": 4.5}):
        response = client.get("/accuracy/sensor_01", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "sensor_01"
    assert data["resolved"] == 3
    assert data["accuracy"]["overall_mae"] == 4.5


def test_get_accuracy_all_returns_200(client, auth_headers):
    with patch("app.routers.accuracy.get_all_uids", return_value=["s1", "s2"]), \
         patch("app.routers.accuracy.resolve_prediction_actuals", return_value=0), \
         patch("app.routers.accuracy.compute_accuracy", return_value={"overall_mae": 3.1}):
        response = client.get("/accuracy/all", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "s1" in data
    assert "s2" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_accuracy_router.py -v
```
Expected: FAIL — router not registered.

- [ ] **Step 3: Create app/routers/accuracy.py**

```python
from fastapi import APIRouter, Depends
from app.auth import require_api_key
from app.db import get_all_uids
from app.services.accuracy import resolve_prediction_actuals, compute_accuracy

router = APIRouter(tags=["accuracy"])


@router.get("/accuracy/all", dependencies=[Depends(require_api_key)])
def get_accuracy_all():
    uids = get_all_uids()
    result = {}
    for uid in uids:
        resolve_prediction_actuals(uid)
        result[uid] = compute_accuracy(uid)
    return result


@router.get("/accuracy/{uid}", dependencies=[Depends(require_api_key)])
def get_accuracy(uid: str):
    resolved = resolve_prediction_actuals(uid)
    metrics = compute_accuracy(uid)
    return {"uid": uid, "resolved": resolved, "accuracy": metrics}
```

Note: `/accuracy/all` is defined first so FastAPI does not match "all" as a uid parameter.

- [ ] **Step 4: Register the router in app/main.py**

Add to `app/main.py`:

```python
# Add import after existing router imports:
from app.routers.accuracy import router as accuracy_router

# Add include after existing includes:
app.include_router(accuracy_router)
```

- [ ] **Step 5: Run tests to verify they pass**

```
python -m pytest tests/test_accuracy_router.py -v
```
Expected: Both tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/routers/accuracy.py tests/test_accuracy_router.py app/main.py
git commit -m "feat: add GET /accuracy/all and GET /accuracy/{uid} endpoints"
```

---

### Task 5: Integrate accuracy into hourly worker (app/main.py)

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Read and understand the current hourly worker**

Current `_hourly_predict_worker` in `app/main.py`:
```python
def _hourly_predict_worker() -> None:
    logger = logging.getLogger(__name__)
    while True:
        _time.sleep(3600)
        try:
            from app.db import get_all_uids
            from app.services.predict import predict_sensor
            uids = get_all_uids()
            for uid in uids:
                try:
                    predict_sensor(uid)
                except Exception as e:
                    logger.error(f"[scheduler] predict failed for {uid}: {e}")
            logger.info(f"[scheduler] Hourly predict complete for {len(uids)} sensors")
        except Exception as e:
            logger.error(f"[scheduler] Predict cycle failed: {e}")
```

- [ ] **Step 2: Add accuracy check loop after predict loop**

Replace the worker body with:

```python
def _hourly_predict_worker() -> None:
    logger = logging.getLogger(__name__)
    while True:
        _time.sleep(3600)
        try:
            from app.db import get_all_uids, get_model_status
            from app.services.predict import predict_sensor
            uids = get_all_uids()
            for uid in uids:
                try:
                    predict_sensor(uid)
                except Exception as e:
                    logger.error(f"[scheduler] predict failed for {uid}: {e}")
            logger.info(f"[scheduler] Hourly predict complete for {len(uids)} sensors")
        except Exception as e:
            logger.error(f"[scheduler] Predict cycle failed: {e}")

        try:
            from app.db import get_all_uids, get_model_status
            from app.services.accuracy import resolve_prediction_actuals, check_and_auto_retrain
            uids = get_all_uids()
            status_map = {s["uid"]: s for s in get_model_status()}
            for uid in uids:
                try:
                    resolve_prediction_actuals(uid)
                    baseline = (status_map.get(uid) or {}).get("mae_score")
                    check_and_auto_retrain(uid, baseline)
                except Exception as e:
                    logger.error(f"[scheduler] accuracy check failed for {uid}: {e}")
            logger.info(f"[scheduler] Accuracy check complete for {len(uids)} sensors")
        except Exception as e:
            logger.error(f"[scheduler] Accuracy check cycle failed: {e}")
```

- [ ] **Step 3: Run existing tests to check no regressions**

```
python -m pytest tests/ -v --ignore=tests/test_migration_accuracy.py
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: run accuracy resolution and drift check in hourly worker"
```

---

### Task 6: Frontend — drift_score + drifted badge

**Files:**
- Modify: `bc-enviro-web/resources/js/main/be-aqms/ml-forecast/index.tsx`

Read the file before editing. The `GET /status` response already returns `drift_score` and `last_accuracy_check_at` from `model_metadata` (since it does `SELECT *`). These will appear in the sensor status object automatically.

- [ ] **Step 1: Add drift_score to the sensor status TypeScript interface**

Find the interface that describes a sensor's model status (has fields like `model_status`, `mae_score`, `last_trained_at`). Add:

```ts
drift_score: number | null
last_accuracy_check_at: string | null
```

- [ ] **Step 2: Add "drifted" to the statusBadge map**

Find the `statusBadge` map or switch (has cases for `"ready"`, `"training"`, `"tuning"`, `"error"`). Add:

```ts
drifted: `<span class="inline-flex items-center gap-1 rounded-full bg-orange-100 text-orange-700 px-2 py-0.5 text-xs font-medium">
  <i class="fa fa-exclamation-triangle"></i>Drifted
</span>`,
```

Match the exact class and structure of the adjacent "error" badge.

- [ ] **Step 3: Show drift score in the sensor row**

Find where `mae_score` is displayed in the sensor row (the training MAE). Immediately after it, add:

```ts
${sensor.drift_score != null
  ? `<div style="font-size:0.75em;color:#f97316" title="Drift ratio (recent MAE / baseline MAE)">drift ×${sensor.drift_score.toFixed(2)}</div>`
  : ''}
```

Where `sensor` is the sensor status object in the row rendering loop. Match the variable name used in that scope.

- [ ] **Step 4: TypeScript check**

```
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
npx tsc --noEmit 2>&1 | grep "ml-forecast/index.tsx"
```
Expected: no output (no errors).

- [ ] **Step 5: Commit**

```bash
cd C:\Users\nurch\OneDrive\Documents\project\bc-enviro-web
git add resources/js/main/be-aqms/ml-forecast/index.tsx
git commit -m "feat: show drift_score indicator and drifted status badge"
```

---

## Self-Review

**Spec coverage:**
- ✅ actual_values stored per prediction step (Task 1 + Task 2)
- ✅ Actuals fetched from t_loggers within ±30 min (Task 3)
- ✅ Deduplication of target_times in resolve loop (Task 3)
- ✅ Per-step MAE computed from resolved predictions (Task 3)
- ✅ Drift detected when recent_mae / baseline > 1.5 (Task 3)
- ✅ Auto-retrain spawns daemon thread on drift (Task 3)
- ✅ GET /accuracy/all + GET /accuracy/{uid} (Task 4)
- ✅ Hourly worker resolves actuals + checks drift (Task 5)
- ✅ Frontend drift badge + score display (Task 6)

**Type consistency check:**
- `update_prediction_actuals(uid, target_time, actual_values)` — used in Task 2 + Task 3 ✅
- `get_resolved_predictions(uid, n=100)` — used in Task 2 + Task 3 ✅
- `update_drift_metadata(uid, drift_score, last_accuracy_check_at)` — used in Task 2 + Task 3 ✅
- `resolve_prediction_actuals` / `compute_accuracy` — used in Tasks 3, 4, 5 ✅
