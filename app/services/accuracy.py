import logging
import threading
import numpy as np
import mysql.connector
from datetime import datetime, timezone

from app.config import SENSOR_DB_CONFIG, FEATURE_COLS, PHYSICAL_LIMITS
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

# Sensor melapor tiap menit (~60 bacaan/jam). Rata-rata dari segelintir bacaan
# bukan rata-rata jam yang bermakna, dan akan dicatat sebagai "aktual" seolah
# setara dengan yang dilatihkan ke model.
MIN_READINGS_PER_HOUR = 10

# Parameter resmi yang dipantau & dilaporkan. Dipakai sebagai dasar metrik.
PRIMARY_PARAM = "tsp"


def fetch_actual_for_step(uid: str, target_time: datetime) -> dict | None:
    """Nilai aktual untuk satu jam target: RATA-RATA JAM, sama seperti yang
    dilatihkan ke model.

    Sebelumnya fungsi ini mengambil SATU bacaan mentah terdekat (±30 menit),
    padahal model dilatih pada rata-rata per jam (`resample("h").mean()`).
    Selisih antara dua definisi itu saja terukur 29,67 µg/m³ pada TSP —
    sebesar hampir seluruh galat model. Akibatnya angka akurasi produksi
    menghukum model atas derau menit-ke-menit yang memang tidak diramalkannya.

    Basis waktu: `target_time` naif dan berasal dari indeks data yang dibangun
    dengan `pd.to_datetime(datetime_unix, unit="s")`, jadi maknanya UTC. Memakai
    `.timestamp()` polos akan menafsirkannya sebagai waktu lokal server dan
    menggeser jendela sejauh offset zona waktu.
    """
    target_unix = int(target_time.replace(tzinfo=timezone.utc).timestamp())
    conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cols_sql = ", ".join(f"AVG(`{c}`) AS `{c}`" for c in FEATURE_COLS)
        cursor.execute(
            f"SELECT {cols_sql}, COUNT(*) AS n_readings FROM t_loggers "
            "WHERE uid = %s AND datetime_unix >= %s AND datetime_unix < %s "
            "AND deleted_at IS NULL",
            (uid, target_unix, target_unix + 3600),
        )
        row = cursor.fetchone()
        if row is None or (row.get("n_readings") or 0) < MIN_READINGS_PER_HOUR:
            return None

        # Terapkan batas fisik yang sama seperti pada data latih. Tanpa ini,
        # sensor tanpa modul cuaca melaporkan mmhg/temp/humidity = 0 sementara
        # model memprediksi nilai cadangan (760/28/82) — galat konstan ratusan
        # satuan yang menenggelamkan seluruh metrik.
        actual: dict[str, float] = {}
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is None:
                continue
            val = float(val)
            limits = PHYSICAL_LIMITS.get(col)
            if limits is not None and not (limits[0] <= val <= limits[1]):
                continue
            actual[col] = val
        return actual or None
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
    # `overall_mae` mencampur parameter berskala sangat berbeda (mmhg ~760 vs tsp
    # ~50), jadi angkanya didominasi yang skalanya terbesar dan sulit ditafsirkan.
    # Dipertahankan demi kompatibilitas, tapi `tsp_mae` yang seharusnya dibaca.
    result["overall_mae"] = round(float(np.mean(all_errors)), 4)
    tsp_errors = [
        e for step in errors_by_step.values() for e in step.get(PRIMARY_PARAM, [])
    ]
    result["tsp_mae"] = round(float(np.mean(tsp_errors)), 4) if tsp_errors else None
    return result


def _primary_errors(rows: list[dict]) -> list[float]:
    """Galat absolut TSP per baris, dalam µg/m³."""
    errors: list[float] = []
    for row in rows:
        actual = row.get("actual_values") or {}
        pred_val = row.get(PRIMARY_PARAM)
        act_val = actual.get(PRIMARY_PARAM)
        if pred_val is not None and act_val is not None:
            errors.append(abs(float(pred_val) - float(act_val)))
    return errors


def detect_drift(uid: str) -> float | None:
    """Rasio galat TSP terkini terhadap galat TSP historisnya sendiri.

    Sebelumnya fungsi ini membandingkan galat produksi (satuan fisik, µg/m³)
    dengan `mae_score` hasil pelatihan (ruang ternormalisasi 0–1). Dua besaran
    itu berbeda skala sekitar tiga orde, sehingga rasionya selalu jauh di atas
    DRIFT_THRESHOLD=1.5 dan auto-retrain akan terpicu pada hampir setiap siklus.

    Sekarang keduanya diambil dari sumber yang sama — tabel prediksi, parameter
    TSP, satuan µg/m³ — sehingga rasionya bermakna: "apakah model belakangan ini
    lebih meleset dibanding biasanya?"
    """
    rows = get_resolved_predictions(uid, n=DRIFT_WINDOW * 8)
    if len(rows) < DRIFT_WINDOW * 2:
        return None

    recent = _primary_errors(rows[:DRIFT_WINDOW])
    reference = _primary_errors(rows[DRIFT_WINDOW:])
    if len(recent) < DRIFT_WINDOW // 2 or len(reference) < DRIFT_WINDOW:
        return None

    ref_mae = float(np.mean(reference))
    if ref_mae <= 0:
        return None
    return round(float(np.mean(recent)) / ref_mae, 4)


def check_and_auto_retrain(uid: str) -> bool:
    drift_score = detect_drift(uid)
    update_drift_metadata(uid, drift_score, datetime.now(timezone.utc))
    if drift_score is not None and drift_score > DRIFT_THRESHOLD:
        upsert_metadata(uid, status="drifted")
        from app.services.retrain import retrain_sensor
        threading.Thread(target=retrain_sensor, args=(uid,), daemon=True).start()
        logger.warning(f"[drift] Auto-retrain triggered for {uid} (drift_score={drift_score})")
        return True
    return False
