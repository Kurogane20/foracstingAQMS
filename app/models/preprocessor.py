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
    try:
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
    finally:
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
