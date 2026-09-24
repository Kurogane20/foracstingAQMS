import os
import logging
import numpy as np
import pandas as pd
import joblib
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from app.config import (
    SENSOR_DB_CONFIG, FEATURE_COLS, TIME_COLS, METEO_COLS,
    N_FEATURES, N_INPUT_HOURS, N_FORECAST_HOURS,
    TRAIN_HISTORY_HOURS, MODELS_DIR,
    PHYSICAL_LIMITS, LOG_SCALE_COLS, FEATURE_SCHEMA_VERSION,
    VAL_FRACTION,
)

logger = logging.getLogger(__name__)


def fetch_sensor_data(uid: str, hours: int = N_INPUT_HOURS) -> pd.DataFrame:
    conn = None
    try:
        conn = mysql.connector.connect(**SENSOR_DB_CONFIG)
        cutoff_unix = int(pd.Timestamp.now().timestamp()) - hours * 3600
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT datetime_unix, pm_25, pm_25_correction, pm_10, pm_10_correction,
                   tsp, tsp_correction, noise, temp, mmhg, humidity,
                   aqi_index_pm25, aqi_index_pm10, aqi_index_tsp, aqi_index
            FROM t_loggers
            WHERE uid = %s AND datetime_unix >= %s AND deleted_at IS NULL
            ORDER BY datetime_unix ASC
            """,
            (uid, cutoff_unix),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
    finally:
        if conn is not None:
            conn.close()


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["datetime_unix"], unit="s")
    df = df.set_index("timestamp")
    return df[FEATURE_COLS].resample("h").mean()


def detect_and_remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Buang hanya nilai yang mustahil secara fisik, lalu interpolasi lubangnya.

    Sebelumnya fungsi ini memakai ambang 3-sigma terhadap median bergulir 24 jam.
    Masalahnya, kejadian debu nyata yang berlangsung beberapa jam PASTI melewati
    ambang itu — jadi penyaringnya justru menghapus persis kejadian yang ingin
    diprediksi. Penyaring statistik tidak bisa membedakan sensor rusak dari hari
    berdebu; batas fisik bisa.
    """
    df = df.copy()
    for col in df.columns:
        limits = PHYSICAL_LIMITS.get(col)
        if limits is None:
            continue
        lo, hi, _ = limits
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Isi sisa lubang. TIDAK ada pemotongan outlier di sini — lihat catatan
    PHYSICAL_LIMITS di config: pemotongan Tukey yang lama menghapus 59% jam
    pelampauan baku mutu sebelum model sempat melihatnya."""
    df = df.copy()
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()

    # Kolom yang kosong total (mis. sensor tanpa modul cuaca) tidak bisa
    # diinterpolasi; pakai nilai cadangan agar retrain tidak gagal total.
    for col in df.columns:
        limits = PHYSICAL_LIMITS.get(col)
        if limits is not None and df[col].isnull().all():
            df[col] = limits[2]
            logger.warning(f"Kolom '{col}' kosong seluruhnya — diisi cadangan {limits[2]}")

    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Masih ada NaN setelah pengisian pada kolom: {bad}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append cyclical hour-of-day and day-of-week features using the DatetimeIndex."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df


_LOG_IDX = [FEATURE_COLS.index(c) for c in LOG_SCALE_COLS]


def _to_log_space(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    out[:, _LOG_IDX] = np.log1p(np.clip(out[:, _LOG_IDX], 0.0, None))
    return out


def _from_log_space(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    out[:, _LOG_IDX] = np.expm1(np.clip(out[:, _LOG_IDX], 0.0, None))
    return out


def _load_scaler(uid: str):
    """Muat scaler beserta versi skemanya.

    joblib.load memakai pickle; berkas ini ditulis sendiri oleh proses retrain ke
    MODELS_DIR milik aplikasi dan tidak pernah berasal dari input pengguna, jadi
    aman. Pola ini sudah dipakai sejak awal, tidak berubah di sini.

    Scaler lama (objek MinMaxScaler polos)
    dipasang pada data linier terpotong, jadi menerapkan expm1 padanya akan
    menghasilkan angka ngawur — lebih baik gagal dengan pesan jelas."""
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    blob = joblib.load(scaler_path)
    if not isinstance(blob, dict) or blob.get("schema") != FEATURE_SCHEMA_VERSION:
        found = blob.get("schema") if isinstance(blob, dict) else "pra-versi"
        raise ValueError(
            f"[{uid}] Scaler tersimpan memakai skema {found}, versi saat ini "
            f"{FEATURE_SCHEMA_VERSION}. Skema fitur/target sudah berubah — "
            f"jalankan retrain untuk uid ini."
        )
    return blob["scaler"]


def normalize(df: pd.DataFrame, uid: str, fit: bool = False) -> tuple:
    """Skalakan FEATURE_COLS ke [0,1]; TIME_COLS lewat tanpa diubah.
    Kolom konsentrasi dilewatkan log1p lebih dulu (lihat LOG_SCALE_COLS)."""
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    sensor_values = _to_log_space(df[FEATURE_COLS].values)
    extra_cols = [c for c in df.columns if c not in FEATURE_COLS]

    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(sensor_values)
        joblib.dump(
            {"scaler": scaler, "schema": FEATURE_SCHEMA_VERSION, "log_cols": LOG_SCALE_COLS},
            scaler_path,
        )
    else:
        scaler = _load_scaler(uid)
        scaled = scaler.transform(sensor_values)

    result = pd.DataFrame(scaled, columns=FEATURE_COLS, index=df.index)
    if extra_cols:
        result = pd.concat([result, df[extra_cols]], axis=1)
    return result, scaler


def denormalize(arr: np.ndarray, uid: str) -> np.ndarray:
    scaler = _load_scaler(uid)
    return _from_log_space(scaler.inverse_transform(arr))


# ── Prakiraan selisih (delta) ────────────────────────────────────────────────
#
# Model sebelumnya meramal LEVEL absolut dan tidak punya jalur langsung ke nilai
# terakhir teramati: tahap LightGBM hanya menerima keluaran BiLSTM + fitur
# kalender. Akibatnya terukur — model kalah dari tebakan naif "nilai H+h = nilai
# sekarang" sebesar 67% di H+1, justru di horizon tempat nilai terakhir paling
# informatif.
#
# Dengan meramal selisih terhadap nilai terakhir, persistence menjadi perilaku
# BAWAAN model (delta = 0), dan ia hanya perlu mempelajari simpangannya. Secara
# struktural model tidak bisa lagi kalah telak dari baseline.
#
# Catatan: ruang ternormalisasi ini log1p→MinMax, jadi selisih di sini setara
# rasio dalam satuan fisik — bentuk yang memang tepat untuk konsentrasi.


def anchor_from_X(X: np.ndarray) -> np.ndarray:
    """Nilai sensor terakhir yang teramati pada tiap jendela. (N, N_FEATURES)"""
    return X[:, -1, :N_FEATURES]


def to_delta(y_abs: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Level absolut → selisih terhadap jangkar. y: (N, n_out, N_FEATURES)"""
    return y_abs - anchor[:, np.newaxis, :]


def from_delta(y_delta: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Selisih → level absolut (kebalikan to_delta)."""
    return y_delta + anchor[:, np.newaxis, :]


def create_sequences(data: np.ndarray, n_in: int, n_out: int,
                     n_target_cols: int = None) -> tuple:
    """
    X  — shape (N, n_in,  data.shape[1])   — all feature columns
    y  — shape (N, n_out, n_target_cols)   — first n_target_cols only (sensor targets)
    """
    n_target = n_target_cols if n_target_cols is not None else data.shape[1]
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i : i + n_in])
        y.append(data[i + n_in : i + n_in + n_out, :n_target])
    return np.array(X), np.array(y)


def sequence_base_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Jam input terakhir tiap sekuens hasil create_sequences — padanan persis
    `last_time` di jalur prediksi. LightGBM menurunkan fitur kalendernya dari
    sini, jadi saat latih dan saat produksi ia harus menerima jam yang sama."""
    return index[N_INPUT_HOURS - 1 : len(index) - N_FORECAST_HOURS]


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
    # Append meteo features for prediction window
    from app.db import get_sensor_lat_lng
    from app.services.meteo import fetch_meteo_predict
    loc = get_sensor_lat_lng(uid)
    if loc:
        try:
            meteo_df = fetch_meteo_predict(loc["lat"], loc["lng"], df_norm.index)
            df_norm = pd.concat([df_norm, meteo_df], axis=1)
            logger.info(f"[{uid}] Meteo features merged: {len(meteo_df)} rows")
        except Exception as e:
            logger.warning(f"[{uid}] Meteo predict fetch failed ({e}); using neutral defaults")
            for col in METEO_COLS:
                df_norm[col] = 0.0
    else:
        for col in METEO_COLS:
            df_norm[col] = 0.0
    X = df_norm.values[-N_INPUT_HOURS:][np.newaxis, :, :]  # (1, 24, 22)
    return X, df_norm.index[-N_INPUT_HOURS:]


def train_split_index(n_sequences: int) -> int:
    """Indeks sekuens pertama yang masuk validasi (pembagian kronologis)."""
    return int(n_sequences * (1.0 - VAL_FRACTION))


def scaler_fit_rows(n_rows: int) -> int:
    """Jumlah baris awal yang boleh dilihat scaler saat di-fit.

    Sekuens latih ke-i memakai baris [i, i + N_INPUT + N_FORECAST). Sekuens latih
    terakhir berindeks split-1, jadi baris terjauh yang disentuh data latih adalah
    split - 1 + N_INPUT + N_FORECAST - 1. Scaler tidak boleh melihat lebih jauh
    dari itu — kalau tidak, rentang min–maks data validasi ikut bocor ke
    transformasi dan skor evaluasi menjadi optimistis.
    """
    window = N_INPUT_HOURS + N_FORECAST_HOURS
    n_seq = max(n_rows - window + 1, 0)
    if n_seq == 0:
        return n_rows
    return min(train_split_index(n_seq) + window - 1, n_rows)


def preprocess_for_training(uid: str) -> tuple:
    df_raw    = fetch_sensor_data(uid, hours=TRAIN_HISTORY_HOURS + 2)
    df_hourly = resample_hourly(df_raw)
    df_clean  = detect_and_remove_anomalies(df_hourly)
    df_clean  = clean(df_clean)
    df_timed  = add_time_features(df_clean)
    # Fit scaler HANYA pada baris yang disentuh sekuens latih, lalu transform
    # seluruh deret dengan scaler itu. Sebelumnya scaler di-fit pada seluruh
    # deret termasuk porsi validasi — kebocoran informasi yang membuat skor
    # validasi sedikit optimistis. Nilai validasi di luar rentang latih boleh
    # melewati [0, 1]; itu perilaku yang benar, bukan galat.
    fit_end = scaler_fit_rows(len(df_timed))
    normalize(df_timed.iloc[:fit_end], uid, fit=True)
    df_norm, _ = normalize(df_timed, uid, fit=False)
    # Append meteo features (fetched per-uid lat/lng stored at retrain time)
    from app.db import get_sensor_lat_lng
    from app.services.meteo import fetch_meteo_training
    loc = get_sensor_lat_lng(uid)
    if loc:
        try:
            meteo_df = fetch_meteo_training(loc["lat"], loc["lng"], df_norm.index)
            df_norm = pd.concat([df_norm, meteo_df], axis=1)
            logger.info(f"[{uid}] Meteo features merged: {len(meteo_df)} rows")
        except Exception as e:
            logger.warning(f"[{uid}] Meteo fetch failed ({e}); using neutral defaults")
            for col in METEO_COLS:
                df_norm[col] = 0.0
    else:
        logger.warning(f"[{uid}] No lat/lng stored; meteo features set to neutral")
        for col in METEO_COLS:
            df_norm[col] = 0.0
    X, y = create_sequences(df_norm.values, N_INPUT_HOURS, N_FORECAST_HOURS,
                            n_target_cols=N_FEATURES)
    return X, y, sequence_base_times(df_norm.index)
