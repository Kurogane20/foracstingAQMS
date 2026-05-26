# Accuracy Tracking — Sub-project B Design

## Goal

Track forecast accuracy by matching predictions against actual sensor readings; detect model drift and auto-retrain when performance degrades.

## Architecture

When `target_time` is in the past, fetch the actual sensor reading from `t_loggers` and store it in the `predictions` row as `actual_values JSON`. Compute rolling MAE per step from resolved rows. If recent MAE exceeds 1.5× training-time MAE over the last 24 resolved predictions, mark the sensor as "drifted" and launch an auto-retrain thread.

## Tech Stack

Python 3.11, FastAPI, MySQL (SENSOR_DB `t_loggers` for actuals, RESULT_DB `predictions`/`model_metadata` for storage), TypeScript/Highcharts

---

## Section 1: DB Changes

### predictions table
```sql
ALTER TABLE predictions ADD COLUMN actual_values JSON NULL;
```
One row per forecast step. `actual_values` stores the matching sensor reading as `{"pm_25": 42.1, "aqi_index": 67.3, ...}` once resolved, or NULL until resolved.

### model_metadata table
```sql
ALTER TABLE model_metadata ADD COLUMN drift_score FLOAT NULL;
ALTER TABLE model_metadata ADD COLUMN last_accuracy_check_at DATETIME NULL;
```
`drift_score` = recent_mae / training_mae ratio (NULL until first check). Returned automatically via existing `GET /status` since that does `SELECT *`.

---

## Section 2: DB Layer Additions (app/db.py)

Four new functions in `app/db.py`:

| Function | Purpose |
|---|---|
| `get_unresolved_predictions(uid)` | SELECT uid, target_time, step where target_time ≤ NOW() AND actual_values IS NULL |
| `update_prediction_actuals(uid, target_time, actual_values)` | UPDATE predictions SET actual_values = JSON where uid + target_time |
| `get_resolved_predictions(uid, n=100)` | SELECT step + FEATURE_COLS + actual_values where actual_values IS NOT NULL, last n rows |
| `update_drift_metadata(uid, drift_score, last_accuracy_check_at)` | UPDATE model_metadata SET drift_score, last_accuracy_check_at (no status change) |

---

## Section 3: Accuracy Service (app/services/accuracy.py)

New file. Five functions:

| Function | Purpose |
|---|---|
| `fetch_actual_for_step(uid, target_time)` | Query `t_loggers` for reading closest to target_time within ±30 min. Returns `dict\|None`. |
| `resolve_prediction_actuals(uid)` | For each unresolved prediction, call `fetch_actual_for_step` and store. Returns count resolved. |
| `compute_accuracy(uid)` | Compute per-step MAE across FEATURE_COLS from last `DRIFT_WINDOW*6` resolved rows. Returns `{"step_1": {"aqi_index": 3.2, ...}, ..., "overall_mae": 4.5}`. |
| `detect_drift(uid, baseline_mae)` | Compare MAE over last `DRIFT_WINDOW=24` resolved predictions to baseline. Returns float ratio or None. |
| `check_and_auto_retrain(uid, baseline_mae)` | If drift > `DRIFT_THRESHOLD=1.5`, set status "drifted", launch retrain thread. Returns bool. |

Constants: `DRIFT_THRESHOLD = 1.5`, `DRIFT_WINDOW = 24`

---

## Section 4: Accuracy API (app/routers/accuracy.py)

Two endpoints (auth-protected):

- `GET /accuracy/all` — resolves actuals + computes accuracy for all sensors
- `GET /accuracy/{uid}` — resolves actuals + computes accuracy for one sensor; returns `{"uid": ..., "resolved": N, "accuracy": {...}}`

Note: `/accuracy/all` must be registered **before** `/accuracy/{uid}` in the router to prevent `all` being parsed as a UID.

---

## Section 5: Hourly Worker Integration (app/main.py)

After the predict loop in `_hourly_predict_worker`, for each uid:
1. `resolve_prediction_actuals(uid)` — store any newly available actuals
2. `check_and_auto_retrain(uid, baseline_mae)` — detect drift; trigger retrain if exceeded

The hourly worker already has `uids` and `get_model_status()` available, so `baseline_mae` is fetched from the status response.

---

## Section 6: Frontend (index.tsx)

Minimal additions:
- `drift_score: number | null` added to sensor status interface (already returned by `GET /status` after DB migration)
- "drifted" status badge — orange/red, same structure as existing badges
- Drift score indicator shown in the sensor row when drift_score is not null

---

## Files to Create / Modify

| Action | File |
|---|---|
| Migrate | `predictions` table — add `actual_values JSON NULL` |
| Migrate | `model_metadata` table — add `drift_score FLOAT NULL`, `last_accuracy_check_at DATETIME NULL` |
| Modify | `app/db.py` — 4 new functions |
| Create | `app/services/accuracy.py` |
| Create | `app/routers/accuracy.py` |
| Modify | `app/main.py` — register accuracy router + integrate in hourly worker |
| Modify | `bc-enviro-web/.../ml-forecast/index.tsx` — drift_score display + drifted badge |
