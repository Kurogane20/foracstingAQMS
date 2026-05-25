import os
import numpy as np
import pandas as pd
import joblib
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from app.config import (
    SENSOR_DB_CONFIG, FEATURE_COLS, TIME_COLS,
    N_FEATURES, N_INPUT_HOURS, N_FORECAST_HOURS,
    TRAIN_HISTORY_HOURS, MODELS_DIR,
)


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
    df = df.copy()
    for col in df.columns:
        rolling_median = df[col].rolling(24, min_periods=1, center=True).median()
        rolling_std = df[col].rolling(24, min_periods=1, center=True).std().fillna(1.0)
        anomaly_mask = (df[col] - rolling_median).abs() > 3 * rolling_std
        df.loc[anomaly_mask, col] = np.nan
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()
    if df.isnull().any().any():
        raise ValueError("DataFrame still contains NaN after fill — check for all-null sensor columns")
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append cyclical hour-of-day and day-of-week features using the DatetimeIndex."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df


def normalize(df: pd.DataFrame, uid: str, fit: bool = False) -> tuple:
    """Normalize only FEATURE_COLS with MinMaxScaler; TIME_COLS pass through unchanged."""
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    sensor_df = df[FEATURE_COLS]
    extra_cols = [c for c in df.columns if c not in FEATURE_COLS]

    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(sensor_df.values)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(sensor_df.values)

    result = pd.DataFrame(scaled, columns=FEATURE_COLS, index=df.index)
    if extra_cols:
        result = pd.concat([result, df[extra_cols]], axis=1)
    return result, scaler


def denormalize(arr: np.ndarray, uid: str) -> np.ndarray:
    scaler_path = os.path.join(MODELS_DIR, uid, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(arr)


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
    X = df_norm.values[-N_INPUT_HOURS:][np.newaxis, :, :]  # (1, 24, 18)
    return X, df_norm.index[-N_INPUT_HOURS:]


def preprocess_for_training(uid: str) -> tuple:
    df_raw    = fetch_sensor_data(uid, hours=TRAIN_HISTORY_HOURS + 2)
    df_hourly = resample_hourly(df_raw)
    df_clean  = detect_and_remove_anomalies(df_hourly)
    df_clean  = clean(df_clean)
    df_timed  = add_time_features(df_clean)
    df_norm, _ = normalize(df_timed, uid, fit=True)
    return create_sequences(df_norm.values, N_INPUT_HOURS, N_FORECAST_HOURS,
                            n_target_cols=N_FEATURES)
