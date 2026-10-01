import json
import os
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from app.config import MODELS_DIR
from app.models.bilstm import train_bilstm, predict_bilstm, _model_cache
from app.models.lgbm import LGBM_BASE_PARAMS, build_lgbm_dataset
from app.models.pipeline import PURGE_SEQUENCES, MIN_EVAL_SEQUENCES


BILSTM_TRIALS = 15
LGBM_TRIALS = 30
PARAMS_FILENAME = "best_params.json"

# Porsi awal data tuning untuk melatih kandidat; sisanya (setelah celah purge)
# untuk menilainya.
TUNE_FIT_FRACTION = 0.8


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


def tuning_split(n_sequences: int) -> tuple[int, int]:
    """(akhir potongan latih, awal potongan evaluasi) untuk objektif tuning.

    Celah PURGE_SEQUENCES mencegah target sekuens latih terakhir tumpang-tindih
    dengan target sekuens evaluasi pertama — sama seperti pembagian run_training.
    """
    fit_end = int(n_sequences * TUNE_FIT_FRACTION)
    eval_start = fit_end + PURGE_SEQUENCES
    if n_sequences - eval_start < MIN_EVAL_SEQUENCES:
        raise ValueError(
            f"Data tuning terlalu sedikit ({n_sequences} sekuens) untuk menyisakan "
            f"{MIN_EVAL_SEQUENCES} sekuens evaluasi setelah celah purge."
        )
    return fit_end, eval_start


def tune_bilstm(X: np.ndarray, y: np.ndarray, uid: str) -> dict:
    """Run Optuna tuning for BiLSTM. Returns best params dict.

    `X`/`y` harus porsi LATIH saja: porsi validasi run_training dipakai mencari α
    dan melaporkan skor, jadi tidak boleh ikut memilih hiperparameter.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_end, eval_start = tuning_split(len(X))
    X_train, X_val = X[:fit_end], X[eval_start:]
    y_train, y_val = y[:fit_end], y[eval_start:]

    def objective(trial):
        units = trial.suggest_int("units", 64, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_bilstm(X_train, y_train, uid, units=units, dropout=dropout,
                     batch_size=batch_size, epochs=30, patience=5)
        _model_cache.pop(uid, None)  # force reload of just-saved model

        preds = np.array([predict_bilstm(X_val[i:i+1], uid)[0] for i in range(len(X_val))])
        return float(np.mean(np.abs(preds - y_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=BILSTM_TRIALS)
    return study.best_params


def tune_lgbm(bilstm_preds: np.ndarray, y_true: np.ndarray,
              base_times: list, uid: str, anchors: np.ndarray) -> dict:
    """Run Optuna tuning for LightGBM. Returns best params dict.

    `y_true` dan `bilstm_preds` di sini berada dalam ruang SELISIH, dan `anchors`
    adalah nilai terakhir teramati per jendela — sama seperti jalur pelatihan.

    Tiap kandidat dilatih pada potongan awal dan dinilai pada potongan akhir yang
    tidak dilihatnya. Sebelumnya kandidat dinilai pada data latihnya sendiri,
    sehingga Optuna selalu memenangkan konfigurasi paling overfit (pohon dan daun
    terbanyak). Kandidat dilatih di memori: trial tidak lagi menimpa lgbm*.pkl
    yang sedang dipakai produksi.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_end, eval_start = tuning_split(len(bilstm_preds))
    X_fit, y_fit = build_lgbm_dataset(
        bilstm_preds[:fit_end], y_true[:fit_end], base_times[:fit_end], anchors[:fit_end],
    )
    X_eval, y_eval = build_lgbm_dataset(
        bilstm_preds[eval_start:], y_true[eval_start:],
        base_times[eval_start:], anchors[eval_start:],
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        }
        model = MultiOutputRegressor(lgb.LGBMRegressor(**{**LGBM_BASE_PARAMS, **params}))
        model.fit(X_fit, y_fit)
        return float(np.mean(np.abs(model.predict(X_eval) - y_eval)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=LGBM_TRIALS)
    return study.best_params
