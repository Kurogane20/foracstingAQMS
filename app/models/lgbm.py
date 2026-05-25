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
    lgbm_params: dict = None,
) -> None:
    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        feats = build_lgbm_features(bilstm_preds[i], bt)
        X_all.append(feats)
        y_all.append(y_true[i])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    base = dict(
        n_estimators=400, learning_rate=0.03, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    if lgbm_params:
        base.update(lgbm_params)

    uid_dir = os.path.join(MODELS_DIR, uid)
    os.makedirs(uid_dir, exist_ok=True)

    for suffix, extra in [
        ("",       {}),
        ("_lower", {"objective": "quantile", "alpha": 0.1}),
        ("_upper", {"objective": "quantile", "alpha": 0.9}),
    ]:
        params = {**base, **extra}
        model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        model.fit(X_all, y_all)
        joblib.dump(model, os.path.join(uid_dir, f"lgbm{suffix}.pkl"))


def predict_lgbm(
    bilstm_output: np.ndarray,
    base_time: pd.Timestamp,
    uid: str,
) -> dict:
    X = build_lgbm_features(bilstm_output, base_time)
    result = {}
    for key, suffix in [("point", ""), ("lower", "_lower"), ("upper", "_upper")]:
        path = os.path.join(MODELS_DIR, uid, f"lgbm{suffix}.pkl")
        result[key] = joblib.load(path).predict(X) if os.path.exists(path) else None
    return result
