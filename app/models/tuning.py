import json
import os
import numpy as np
import optuna
import joblib
import pandas as pd
from app.config import MODELS_DIR, N_FORECAST_HOURS
from app.models.bilstm import train_bilstm, predict_bilstm
from app.models.lgbm import train_lgbm, build_lgbm_features


BILSTM_TRIALS = 15
LGBM_TRIALS = 30
PARAMS_FILENAME = "best_params.json"


def save_best_params(uid: str, bilstm: dict, lgbm: dict) -> None:
    uid_dir = os.path.join(MODELS_DIR, uid)
    os.makedirs(uid_dir, exist_ok=True)
    path = os.path.join(uid_dir, PARAMS_FILENAME)
    with open(path, "w") as f:
        json.dump({"bilstm": bilstm, "lgbm": lgbm}, f)


def load_best_params(uid: str) -> dict | None:
    path = os.path.join(MODELS_DIR, uid, PARAMS_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tune_bilstm(X: np.ndarray, y: np.ndarray, uid: str) -> dict:
    """Run Optuna tuning for BiLSTM. Returns best params dict."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        units = trial.suggest_int("units", 64, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_bilstm(X_train, y_train, uid, units=units, dropout=dropout,
                     batch_size=batch_size, epochs=30, patience=5)

        preds = np.array([predict_bilstm(X_val[i:i+1], uid)[0] for i in range(len(X_val))])
        return float(np.mean(np.abs(preds - y_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=BILSTM_TRIALS)
    return study.best_params


def tune_lgbm(bilstm_preds: np.ndarray, y_true: np.ndarray,
              base_times: list, uid: str) -> dict:
    """Run Optuna tuning for LightGBM. Returns best params dict."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        }
        train_lgbm(bilstm_preds, y_true, base_times, uid, lgbm_params=params)

        preds = np.array([
            joblib.load(os.path.join(MODELS_DIR, uid, "lgbm.pkl"))
            .predict(build_lgbm_features(bilstm_preds[i], base_times[i]))[0]
            for i in range(len(bilstm_preds))
        ])
        return float(np.mean(np.abs(preds - y_true[:, 0, :])))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=LGBM_TRIALS)
    return study.best_params
