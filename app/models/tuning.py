"""Penyetelan hiperparameter LightGBM dengan Optuna.

Disetel pada parameter utama (TSP) lalu dipakai untuk semua parameter. Hanya
data SEBELUM lipatan backtest pertama yang boleh dilihat: lipatan backtest
dipakai melaporkan skor, jadi tidak boleh ikut memilih hiperparameter.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from app.config import FEATURE_SCHEMA_VERSION, MODELS_DIR, N_FORECAST_HOURS, PRIMARY_PARAM
from app.models.backtest import plan_folds
from app.models.features import build_rows, to_model_space
from app.models.forecaster import BASE_PARAMS

LGBM_TRIALS = 30
PARAMS_FILENAME = "best_params.json"

# Porsi awal data tuning untuk melatih kandidat; sisanya untuk menilainya.
TUNE_FIT_FRACTION = 0.8
MIN_EVAL_ROWS = 24 * 3 * N_FORECAST_HOURS


def save_best_params(uid: str, lgbm: dict) -> None:
    uid_dir = os.path.join(MODELS_DIR, uid)
    os.makedirs(uid_dir, exist_ok=True)
    with open(os.path.join(uid_dir, PARAMS_FILENAME), "w") as f:
        json.dump({"schema": FEATURE_SCHEMA_VERSION, "lgbm": lgbm}, f)


def load_best_params(uid: str) -> dict | None:
    """Parameter hasil tuning, atau None.

    Berkas dari skema lain diabaikan: parameter lama disetel untuk tahap koreksi
    LightGBM di atas BiLSTM (sampai 800 pohon x 127 daun) — masalah yang berbeda,
    dan akan overfit bila dipakai di sini.
    """
    path = os.path.join(MODELS_DIR, uid, PARAMS_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or blob.get("schema") != FEATURE_SCHEMA_VERSION:
        return None
    return blob.get("lgbm")


def tuning_rows(meta: pd.DataFrame, tune_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """(mask latih, mask evaluasi) untuk objektif tuning.

    Dibagi menurut JAM TARGET sehingga target baris latih terakhir tidak pernah
    tumpang-tindih dengan jam asal baris evaluasi pertama.
    """
    target_time = meta["t"] + pd.to_timedelta(meta["h"], unit="h")
    usable = meta["delta"].notna().values & (target_time < tune_end).values
    origins = meta.loc[usable, "t"]
    if origins.empty:
        raise ValueError("Data tuning kosong.")
    cut = origins.min() + (origins.max() - origins.min()) * TUNE_FIT_FRACTION
    fit = usable & (target_time < cut).values
    evaluate = usable & (meta["t"] >= cut).values
    if int(evaluate.sum()) < MIN_EVAL_ROWS or not fit.any():
        raise ValueError(
            f"Data tuning terlalu sedikit ({int(evaluate.sum())} baris evaluasi, "
            f"butuh {MIN_EVAL_ROWS})."
        )
    return fit, evaluate


def tune_lgbm(hourly: pd.DataFrame, n_trials: int = LGBM_TRIALS) -> dict:
    """Jalankan Optuna pada parameter utama. Mengembalikan dict parameter terbaik."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune_end = plan_folds(hourly.index)[0][0]
    X, meta = build_rows(to_model_space(hourly), PRIMARY_PARAM)
    fit, evaluate = tuning_rows(meta, tune_end)
    X_fit, y_fit = X[fit], meta.loc[fit, "delta"].values
    X_eval, y_eval = X[evaluate], meta.loc[evaluate, "delta"].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        }
        model = lgb.LGBMRegressor(**{**BASE_PARAMS, **params, "objective": "regression_l1"})
        model.fit(X_fit, y_fit)
        return float(np.mean(np.abs(model.predict(X_eval) - y_eval)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
