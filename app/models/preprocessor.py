"""Data sensor → deret per jam yang bersih, TANPA nilai buatan.

Perubahan mendasar dari versi sebelumnya: lubang data tidak lagi diisi. Dulu
celah berjam-jam (bahkan berhari-hari) diinterpolasi garis lurus lalu dilatihkan
seolah-olah pengukuran. Model pohon menerima NaN apa adanya, jadi jam yang tidak
terukur cukup dibiarkan kosong: ia tidak menjadi target latih, tidak menjadi
jangkar, dan tidak dihitung dalam skor.
"""
import logging
import time

import mysql.connector
import numpy as np
import pandas as pd

from app.config import (
    SENSOR_DB_CONFIG, FEATURE_COLS, PHYSICAL_LIMITS,
    MIN_READINGS_PER_HOUR, PREDICT_HISTORY_HOURS, TRAIN_HISTORY_HOURS,
)

logger = logging.getLogger(__name__)

COUNT_COL = "n_readings"


def fetch_hourly_rows(uid: str, hours: int) -> pd.DataFrame:
    """Rata-rata per jam + jumlah bacaan, dihitung di MySQL.

    Agregasi di basis data menggantikan penarikan ~525 ribu baris per menit per
    sensor untuk setahun data. Definisinya sama persis dengan nilai aktual di
    `services.accuracy.fetch_actual_for_step`: AVG atas [awal jam, awal jam + 1
    jam), sehingga yang dilatihkan dan yang dipakai menilai adalah besaran yang
    sama.

    Kolom: `hour_start` (detik UNIX, UTC), `n_readings`, lalu FEATURE_COLS.
    """
    # time.time() selalu UTC. pd.Timestamp.now() naif berisi jam dinding lokal
    # server, dan .timestamp() menafsirkannya sebagai UTC — di server WITA
    # jendelanya bergeser 8 jam.
    cutoff_unix = (int(time.time()) // 3600 - hours) * 3600
    cols_sql = ", ".join(f"AVG(`{c}`) AS `{c}`" for c in FEATURE_COLS)
    conn = None
    try:
        conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT FLOOR(datetime_unix / 3600) * 3600 AS hour_start,
                   COUNT(*) AS {COUNT_COL}, {cols_sql}
            FROM t_loggers
            WHERE uid = %s AND datetime_unix >= %s AND deleted_at IS NULL
            GROUP BY hour_start
            ORDER BY hour_start ASC
            """,
            (uid, cutoff_unix),
        )
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        if conn is not None:
            conn.close()


def to_hourly_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Baris agregat → DataFrame per jam berindeks waktu (UTC naif, label = awal
    jam) yang KONTINU: jam tanpa data hadir sebagai baris NaN, sehingga `shift(k)`
    selalu berarti "k jam yang lalu"."""
    if rows.empty:
        return pd.DataFrame(columns=FEATURE_COLS + [COUNT_COL], dtype="float64",
                            index=pd.DatetimeIndex([], freq="h"))
    df = rows.copy()
    df.index = pd.to_datetime(df.pop("hour_start").astype("int64"), unit="s")
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.asfreq("h")


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Kosongkan (NaN) nilai yang tidak layak dipercaya; kembalikan FEATURE_COLS.

    1. Jam dengan bacaan terlalu sedikit bukan rata-rata jam yang bermakna.
    2. Nilai di luar batas fisik berasal dari sensor rusak, bukan dari udara.
       Batasnya sengaja longgar — lihat catatan PHYSICAL_LIMITS: penyaring
       statistik (3-sigma, pagar Tukey) menghapus persis kejadian debu yang
       ingin diramalkan.

    Tidak ada interpolasi di sini maupun di tempat lain.
    """
    out = df[FEATURE_COLS].astype("float64").copy()
    if COUNT_COL in df.columns:
        out[df[COUNT_COL].fillna(0) < MIN_READINGS_PER_HOUR] = np.nan
    for col in FEATURE_COLS:
        limits = PHYSICAL_LIMITS.get(col)
        if limits is None:
            continue
        lo, hi, _ = limits
        out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    return out


def drop_incomplete_hour(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Buang bin jam yang belum selesai (indeks naif UTC, label = awal jam).

    Bin jam berjalan hanya berisi bacaan beberapa menit pertama. Bila ikut masuk,
    ia menjadi JANGKAR yang selisihnya diramal model — padahal saat latih setiap
    jangkar adalah rata-rata satu jam penuh.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    return df[df.index < now.floor("h")]


def extend_to(df: pd.DataFrame, last_hour: pd.Timestamp) -> pd.DataFrame:
    """Perpanjang indeks sampai `last_hour` dengan baris NaN.

    Bila sensor berhenti mengirim, baris terakhir yang ada adalah jam lampau.
    Tanpa perpanjangan ini prakiraan akan berangkat dari jam itu dan diberi label
    waktu yang sudah lewat. Dengan baris kosong, usia jangkar terhitung benar dan
    prakiraan ditolak bila datanya sudah basi.
    """
    if df.empty or df.index[-1] >= last_hour:
        return df
    return df.reindex(pd.date_range(df.index[0], last_hour, freq="h"))


def load_hourly(uid: str, hours: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Deret per jam siap pakai: jam penuh saja, tersaring, kontinu sampai jam
    penuh terakhir."""
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is not None:
        now_utc = now_utc.tz_convert("UTC").tz_localize(None)
    frame = drop_incomplete_hour(to_hourly_frame(fetch_hourly_rows(uid, hours)), now_utc)
    clean = apply_quality_filters(frame)
    return extend_to(clean, now_utc.floor("h") - pd.Timedelta(hours=1))


def load_for_training(uid: str) -> pd.DataFrame:
    return load_hourly(uid, TRAIN_HISTORY_HOURS)


def load_for_predict(uid: str) -> pd.DataFrame:
    return load_hourly(uid, PREDICT_HISTORY_HOURS)
