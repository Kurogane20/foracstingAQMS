import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from app.config import N_FORECAST_HOURS, N_FEATURES, MODELS_DIR


def build_lgbm_features(
    bilstm_output: np.ndarray,
    base_time: pd.Timestamp,
    anchor: np.ndarray,
) -> np.ndarray:
    """Fitur tahap koreksi: prakiraan selisih BiLSTM + jangkar + kalender.

    `anchor` (nilai sensor terakhir teramati, ternormalisasi) wajib ada di sini.
    Sebelumnya tahap ini hanya melihat keluaran BiLSTM dan fitur kalender, jadi
    ia tidak tahu sedang berada di level konsentrasi berapa — koreksi yang tepat
    untuk hari 30 µg/m³ jelas berbeda dari hari 400 µg/m³.
    """
    rows = []
    for step in range(N_FORECAST_HOURS):
        t = base_time + pd.Timedelta(hours=step + 1)
        time_feats = [t.hour, t.dayofweek, t.month]
        rows.append(np.concatenate([bilstm_output[step], anchor, time_feats]))
    return np.array(rows)


# 14 selisih BiLSTM + 14 jangkar + 3 kalender
N_LGBM_FEATURES = N_FEATURES * 2 + 3


def train_lgbm(
    bilstm_preds: np.ndarray,
    y_true: np.ndarray,
    base_times: list,
    uid: str,
    anchors: np.ndarray,
    lgbm_params: dict = None,
) -> None:
    X_all, y_all = [], []
    for i, bt in enumerate(base_times):
        feats = build_lgbm_features(bilstm_preds[i], bt, anchors[i])
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
    anchor: np.ndarray,
) -> dict:
    X = build_lgbm_features(bilstm_output, base_time, anchor)
    result = {}
    for key, suffix in [("point", ""), ("lower", "_lower"), ("upper", "_upper")]:
        path = os.path.join(MODELS_DIR, uid, f"lgbm{suffix}.pkl")
        if not os.path.exists(path):
            result[key] = None
            continue
        # joblib.load memakai pickle; berkas ini ditulis sendiri oleh proses
        # retrain ke MODELS_DIR milik aplikasi, tidak pernah dari input pengguna.
        model = joblib.load(path)

        # Model lama dilatih pada 17 fitur (tanpa jangkar) dan meramal level
        # absolut, bukan selisih. Memuatnya di sini akan gagal dengan galat
        # bentuk LightGBM yang tidak menjelaskan apa-apa.
        n_in = getattr(model.estimators_[0], "n_features_in_", N_LGBM_FEATURES)
        if n_in != N_LGBM_FEATURES:
            raise ValueError(
                f"[{uid}] Model LightGBM tersimpan mengharapkan {n_in} fitur, "
                f"skema saat ini {N_LGBM_FEATURES}. Model kini meramal SELISIH "
                f"terhadap nilai terakhir — jalankan retrain untuk uid ini."
            )
        result[key] = model.predict(X)
    return result
