import json
from datetime import datetime, timezone
import mysql.connector
from app.config import SENSOR_DB_CONFIG, RESULT_DB_CONFIG, FEATURE_COLS


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


def upsert_lat_lng(uid: str, lat: float, lng: float) -> None:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_metadata (uid, status, lat, lng)
            VALUES (%s, 'untrained', %s, %s)
            ON DUPLICATE KEY UPDATE lat = VALUES(lat), lng = VALUES(lng)
            """,
            (uid, lat, lng),
        )
        conn.commit()
    finally:
        conn.close()


def get_sensor_lat_lng(uid: str) -> dict | None:
    """Return {"lat": float, "lng": float} or None if not stored yet."""
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT lat, lng FROM model_metadata WHERE uid = %s", (uid,))
        row = cursor.fetchone()
        if not row or row.get("lat") is None or row.get("lng") is None:
            return None
        return {"lat": float(row["lat"]), "lng": float(row["lng"])}
    finally:
        conn.close()


def save_predictions(uid: str, predicted_at: datetime, predictions: list[dict]) -> None:
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


def update_prediction_actuals(uid: str, target_time, actual_values: dict) -> int:
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
        return cursor.rowcount
    finally:
        conn.close()


def get_resolved_predictions(uid: str, n: int = 100) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cols_sql = ", ".join(f"`{c}`" for c in FEATURE_COLS)
        cursor.execute(
            f"""
            SELECT step, target_time, {cols_sql}, actual_values
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


def get_prediction_history(uid: str, days: int = 7) -> list[dict]:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cols_sql = ", ".join(f"`{c}`" for c in FEATURE_COLS)
        cursor.execute(
            f"""
            SELECT step, target_time, predicted_at, {cols_sql}
            FROM predictions
            WHERE uid = %s
              AND predicted_at >= NOW() - INTERVAL %s DAY
            ORDER BY predicted_at ASC, step ASC
            """,
            (uid, days),
        )
        rows = []
        for r in cursor.fetchall():
            row = dict(r)
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.isoformat()
            rows.append(row)
        return rows
    finally:
        conn.close()


def get_hourly_tsp(uid: str, hours: int = 24) -> dict[int, float]:
    """
    Hourly-mean TSP for the past `hours` hours from the sensor DB.
    Returns {unix_hour_start: mean_tsp}.
    """
    import time as _time
    since = int(_time.time()) - hours * 3600
    conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT FLOOR(datetime_unix / 3600) * 3600 AS hour_start, AVG(tsp)
            FROM t_loggers
            WHERE uid = %s AND tsp > 0 AND deleted_at IS NULL
              AND datetime_unix >= %s
            GROUP BY hour_start
            """,
            (uid, since),
        )
        return {int(h): float(v) for h, v in cursor.fetchall() if v is not None}
    finally:
        conn.close()


def _ensure_emission_sources_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS emission_sources (
            id INT AUTO_INCREMENT PRIMARY KEY,
            lat DOUBLE NOT NULL,
            lng DOUBLE NOT NULL,
            strength DOUBLE NOT NULL,
            cell_deg DOUBLE NOT NULL,
            method VARCHAR(32) NOT NULL DEFAULT 'inversion',
            run_at DATETIME NOT NULL,
            active TINYINT(1) NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def replace_emission_sources(rows: list[dict], method: str, run_at: datetime) -> None:
    """
    Replace the active emission-source set produced by an inversion run.
    Each row: {lat, lng, strength, cell_deg}.
    """
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        _ensure_emission_sources_table(cursor)
        cursor.execute(
            "UPDATE emission_sources SET active = 0 WHERE method = %s", (method,)
        )
        for r in rows:
            cursor.execute(
                """
                INSERT INTO emission_sources (lat, lng, strength, cell_deg, method, run_at, active)
                VALUES (%s, %s, %s, %s, %s, %s, 1)
                """,
                (r["lat"], r["lng"], r["strength"], r["cell_deg"], method, run_at),
            )
        conn.commit()
    finally:
        conn.close()


def get_emission_sources() -> list[dict]:
    """Active emission sources (empty list when none / table missing)."""
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT lat, lng, strength, cell_deg, method, run_at "
                "FROM emission_sources WHERE active = 1 ORDER BY strength DESC"
            )
            return [dict(r) for r in cursor.fetchall()]
        except mysql.connector.Error:
            return []
    finally:
        conn.close()


def _ensure_dispersion_validation_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dispersion_validation (
            id INT AUTO_INCREMENT PRIMARY KEY,
            target_uid VARCHAR(32) NOT NULL,
            forecast_at DATETIME NOT NULL,
            target_time DATETIME NOT NULL,
            step TINYINT NOT NULL,
            pred_conc FLOAT NOT NULL,
            actual_tsp FLOAT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_dv (target_uid, forecast_at, step),
            KEY idx_dv_pending (actual_tsp, target_time)
        )
        """
    )


def save_dispersion_validation(rows: list[dict]) -> None:
    """
    Log cross-sensor dispersal predictions for later validation.
    Each row: {target_uid, forecast_at, target_time, step, pred_conc}.
    Duplicate (target_uid, forecast_at, step) rows are ignored.
    """
    if not rows:
        return
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        _ensure_dispersion_validation_table(cursor)
        for r in rows:
            cursor.execute(
                """
                INSERT IGNORE INTO dispersion_validation
                    (target_uid, forecast_at, target_time, step, pred_conc)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (r["target_uid"], r["forecast_at"], r["target_time"],
                 r["step"], r["pred_conc"]),
            )
        conn.commit()
    finally:
        conn.close()


def resolve_dispersion_actuals(limit: int = 200) -> int:
    """
    Fill actual_tsp for validation rows whose target_time has passed, using the
    sensor DB's mean TSP within ±30 min of target_time (UTC). Returns the
    number of rows resolved.
    """
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, target_uid, target_time FROM dispersion_validation
                WHERE actual_tsp IS NULL AND target_time <= UTC_TIMESTAMP()
                ORDER BY target_time ASC LIMIT %s
                """,
                (limit,),
            )
            pending = cursor.fetchall()
        except mysql.connector.Error:
            return 0  # table not created yet
        if not pending:
            return 0

        sconn = mysql.connector.connect(**SENSOR_DB_CONFIG)
        try:
            scur = sconn.cursor()
            ucur = conn.cursor()
            resolved = 0
            for row in pending:
                ts = int(row["target_time"].replace(tzinfo=timezone.utc).timestamp())
                scur.execute(
                    """
                    SELECT AVG(tsp) FROM t_loggers
                    WHERE uid = %s AND tsp > 0 AND deleted_at IS NULL
                      AND datetime_unix BETWEEN %s AND %s
                    """,
                    (row["target_uid"], ts - 1800, ts + 1800),
                )
                val = scur.fetchone()[0]
                if val is not None:
                    ucur.execute(
                        "UPDATE dispersion_validation SET actual_tsp = %s WHERE id = %s",
                        (float(val), row["id"]),
                    )
                    resolved += 1
            conn.commit()
            return resolved
        finally:
            sconn.close()
    finally:
        conn.close()


def update_drift_metadata(
    uid: str,
    drift_score: float | None,
    last_accuracy_check_at,
) -> int:
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
        return cursor.rowcount
    finally:
        conn.close()
