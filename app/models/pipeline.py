import numpy as np
import pandas as pd
from app.config import FEATURE_COLS, N_FORECAST_HOURS, N_INPUT_HOURS
from app.models.preprocessor import (
    preprocess_for_predict,
    preprocess_for_training,
    denormalize,
)
from app.models.bilstm import predict_bilstm, train_bilstm
from app.models.lgbm import predict_lgbm, train_lgbm


def run_prediction(uid: str) -> list:
    X, timestamps = preprocess_for_predict(uid)
    last_time = timestamps[-1]

    bilstm_out = predict_bilstm(X, uid)                         # (1, 6, 14)
    lgbm_result = predict_lgbm(bilstm_out[0], last_time, uid)  # dict

    final_point = denormalize(lgbm_result["point"], uid)        # (6, 14)
    final_lower = denormalize(lgbm_result["lower"], uid) if lgbm_result.get("lower") is not None else None
    final_upper = denormalize(lgbm_result["upper"], uid) if lgbm_result.get("upper") is not None else None

    results = []
    for step in range(N_FORECAST_HOURS):
        target_time = last_time + pd.Timedelta(hours=step + 1)
        row = {"step": step + 1, "target_time": target_time.isoformat()}
        for i, col in enumerate(FEATURE_COLS):
            row[col] = float(final_point[step, i])
        row["lower_bounds"] = (
            {col: float(final_lower[step, i]) for i, col in enumerate(FEATURE_COLS)}
            if final_lower is not None else None
        )
        row["upper_bounds"] = (
            {col: float(final_upper[step, i]) for i, col in enumerate(FEATURE_COLS)}
            if final_upper is not None else None
        )
        results.append(row)
    return results


def run_training(uid: str, bilstm_params: dict = None, lgbm_params: dict = None,
                 X: np.ndarray = None, y: np.ndarray = None) -> dict:
    if X is None or y is None:
        X, y = preprocess_for_training(uid)

    if len(X) < 10:
        raise ValueError(f"Insufficient data for {uid}: only {len(X)} sequences")

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    bilstm_kw = bilstm_params or {}
    train_bilstm(X_train, y_train, uid, **bilstm_kw)

    bilstm_preds_train = np.array([
        predict_bilstm(X_train[i : i + 1], uid)[0]
        for i in range(len(X_train))
    ])
    base_times_train = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        for i in range(len(X_train))
    ]
    train_lgbm(bilstm_preds_train, y_train, base_times_train, uid, lgbm_params=lgbm_params)

    val_bilstm = np.array([
        predict_bilstm(X_val[i : i + 1], uid)[0]
        for i in range(len(X_val))
    ])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i], pd.Timestamp("2024-01-01") + pd.Timedelta(hours=split + i), uid)["point"]
        for i in range(len(X_val))
    ])
    mae = float(np.mean(np.abs(val_lgbm - y_val)))

    return {"training_samples": len(X), "mae_score": round(mae, 6)}
