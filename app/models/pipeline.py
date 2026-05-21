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

    bilstm_out = predict_bilstm(X, uid)          # (1, 6, 14)
    lgbm_out = predict_lgbm(bilstm_out[0], last_time, uid)  # (6, 14)
    final = denormalize(lgbm_out, uid)            # (6, 14)

    results = []
    for step in range(N_FORECAST_HOURS):
        target_time = last_time + pd.Timedelta(hours=step + 1)
        row = {"step": step + 1, "target_time": target_time.isoformat()}
        for i, col in enumerate(FEATURE_COLS):
            row[col] = float(final[step, i])
        results.append(row)
    return results


def run_training(uid: str) -> dict:
    X, y = preprocess_for_training(uid)

    if len(X) < 10:
        raise ValueError(f"Insufficient data for {uid}: only {len(X)} sequences")

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_bilstm(X_train, y_train, uid)

    bilstm_preds_all = np.array([predict_bilstm(X[i : i + 1], uid)[0] for i in range(len(X))])
    base_times = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * N_INPUT_HOURS)
        for i in range(len(X))
    ]
    train_lgbm(bilstm_preds_all, y, base_times, uid)

    val_bilstm = np.array([predict_bilstm(X_val[i : i + 1], uid)[0] for i in range(len(X_val))])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i], base_times[split + i], uid)
        for i in range(len(X_val))
    ])
    mae = float(np.mean(np.abs(val_lgbm - y_val)))

    return {"training_samples": len(X), "mae_score": round(mae, 6)}
