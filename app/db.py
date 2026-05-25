import json
from datetime import datetime
import mysql.connector
from app.config import SENSOR_DB_CONFIG, RESULT_DB_CONFIG


def _parse_bounds(row: dict) -> dict:
    row = dict(row)
    for key in ("lower_bounds", "upper_bounds"):
        val = row.get(key)
        if isinstance(val, str):
            try:
                row[key] = json.loads(val)
            except json.JSONDecodeError:
                row[key] = None
    return row


def get_all_uids() -> list[str]:
    conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT uid FROM t_loggers WHERE deleted_at IS NULL ORDER BY uid"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_latest_predictions(uid: str) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
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
        return [_parse_bounds(dict(r)) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_all_latest_predictions() -> dict[str, list[dict]]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT p.*
            FROM predictions p
            INNER JOIN (
                SELECT uid, MAX(predicted_at) AS max_pa
                FROM predictions
                GROUP BY uid
            ) latest ON p.uid = latest.uid AND p.predicted_at = latest.max_pa
            ORDER BY p.uid, p.step ASC
            """
        )
        rows = cursor.fetchall()
        result: dict[str, list] = {}
        for row in rows:
            uid = row["uid"]
            if uid not in result:
                result[uid] = []
            result[uid].append(_parse_bounds(dict(row)))
        return result
    finally:
        conn.close()


def get_model_status() -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM model_metadata ORDER BY uid")
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def upsert_metadata(
    uid: str,
    status: str,
    last_trained_at: datetime = None,
    last_predicted_at: datetime = None,
    training_samples: int = None,
    mae_score: float = None,
    error_message: str = None,
) -> None:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
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
    finally:
        conn.close()


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
            lower = json.dumps(pred["lower_bounds"]) if pred.get("lower_bounds") is not None else None
            upper = json.dumps(pred["upper_bounds"]) if pred.get("upper_bounds") is not None else None
            cursor.execute(
                f"""
                INSERT INTO predictions
                    (uid, predicted_at, target_time, step, {cols_sql}, lower_bounds, upper_bounds)
                VALUES (%s, %s, %s, %s, {placeholders}, %s, %s)
                """,
                (uid, predicted_at, pred["target_time"], pred["step"], *values, lower, upper),
            )
        conn.commit()
    finally:
        conn.close()
