import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from app.config import N_FORECAST_HOURS, N_FEATURES, MODELS_DIR


def build_lgbm_features(bilstm_output: np.ndarray, base_time: pd.Timestamp) -> np.ndarray:
    rows = []
    for step in range(N_FORECAST_HOURS):
        t = base_time + pd.Timedelta(hours=step + 1)
        time_feats = [t.hour, t.dayofweek, t.month]
        rows.append(np.concatenate([bilstm_output[step], time_feats]))
    return np.array(rows)


def train_lgbm(
    bilstm_preds: np.ndarray,
    y_true: np.ndarray,
    base_times: list,
    uid: str,
) -> None:
    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        feats = build_lgbm_features(bilstm_preds[i], bt)
        X_all.append(feats)
        y_all.append(y_true[i])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    base_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
    model = MultiOutputRegressor(base_model)
    model.fit(X_all, y_all)
    model_path = os.path.join(MODELS_DIR, uid, "lgbm.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def predict_lgbm(bilstm_output: np.ndarray, base_time: pd.Timestamp, uid: str) -> np.ndarray:
    model_path = os.path.join(MODELS_DIR, uid, "lgbm.pkl")
    model = joblib.load(model_path)
    X = build_lgbm_features(bilstm_output, base_time)
    return model.predict(X)
