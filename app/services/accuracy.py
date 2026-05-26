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
    cursor = None
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
        if cursor is not None:
            cursor.close()
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
    if not all_errors:
        return {}
    result["overall_mae"] = round(float(np.mean(all_errors)), 4)
    return result


def detect_drift(uid: str, baseline_mae: float | None) -> float | None:
    if baseline_mae is None:
        return None
    if baseline_mae == 0.0:
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
