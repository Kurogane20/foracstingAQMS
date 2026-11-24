"""Model prakiraan: LightGBM langsung multi-horizon, satu set per parameter.

Menggantikan SSA-BiLSTM-LightGBM. Alasan penggantian terukur pada data nyata
(lihat README, bagian "Dasar pemilihan metode"): pada baris uji yang sama
pendekatan ini lebih akurat di setiap sensor dan setiap horizon, intervalnya
mencapai cakupan nominal, dan waktu latihnya jauh lebih singkat — tanpa
TensorFlow.

Per parameter ada tiga model yang berbagi fitur yang sama (horizon ikut sebagai
fitur, jadi keenam horizon saling menguatkan):

    titik  — regresi L1 (median bersyarat): galat absolut adalah metrik yang
             dilaporkan, dan median tahan terhadap lonjakan sesaat;
    bawah  — kuantil 0,1;
    atas   — kuantil 0,9.

Artefak tersimpan sebagai satu berkas `forecaster.pkl` per sensor, memuat model,
kalibrasi (lihat calibration.py), dan versi skema.
"""
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from app.config import FEATURE_SCHEMA_VERSION, MODELS_DIR

FILENAME = "forecaster.pkl"

QUANTILE_LOWER = 0.1
QUANTILE_UPPER = 0.9

# Jumlah thread sengaja dibatasi. Datanya kecil (±50 ribu baris), jadi lebih dari
# beberapa thread nyaris tidak mempercepat, sedangkan LightGBM melambat drastis
# bila thread-nya melebihi inti yang benar-benar bebas — dan pelatihan berjalan
# di server yang sama dengan API, penjadwal per jam, dan layanan lain.
N_JOBS = int(os.getenv("LGBM_N_JOBS", min(4, os.cpu_count() or 1)))

BASE_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=0, n_jobs=N_JOBS,
)

_OBJECTIVES = {
    "point": {"objective": "regression_l1"},
    "lower": {"objective": "quantile", "alpha": QUANTILE_LOWER},
    "upper": {"objective": "quantile", "alpha": QUANTILE_UPPER},
}

# uid → (sidik berkas, bundel). Sidik (mtime_ns, ukuran) memastikan model baru
# hasil retrain langsung dipakai, juga oleh worker uvicorn lain yang tidak ikut
# melatih.
_cache: dict[str, tuple[tuple[int, int], dict]] = {}


def fit_models(X: pd.DataFrame, y: np.ndarray, params: dict | None = None) -> dict:
    """Latih model titik/bawah/atas pada baris yang targetnya ada."""
    base = {**BASE_PARAMS, **(params or {})}
    models = {}
    for name, extra in _OBJECTIVES.items():
        model = lgb.LGBMRegressor(**{**base, **extra})
        model.fit(X, y)
        models[name] = model
    return models


def predict_deltas(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Selisih mentah (belum dikalibrasi) dari ketiga model."""
    return {name: np.asarray(model.predict(X), dtype=np.float64) for name, model in models.items()}


def bundle_path(uid: str) -> str:
    return os.path.join(MODELS_DIR, uid, FILENAME)


def save_bundle(uid: str, bundle: dict) -> None:
    path = bundle_path(uid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Tulis ke berkas sementara lalu ganti nama: prediksi yang berjalan bersamaan
    # tidak pernah membaca berkas setengah jadi.
    tmp = path + ".tmp"
    joblib.dump({**bundle, "schema": FEATURE_SCHEMA_VERSION}, tmp, compress=3)
    os.replace(tmp, path)


def load_bundle(uid: str) -> dict:
    """Bundel model tersimpan. FileNotFoundError = sensor belum dilatih.

    joblib.load memakai pickle; berkas ini ditulis sendiri oleh proses retrain ke
    MODELS_DIR milik aplikasi dan tidak pernah berasal dari input pengguna.
    """
    path = bundle_path(uid)
    st = os.stat(path)
    fingerprint = (st.st_mtime_ns, st.st_size)
    cached = _cache.get(uid)
    if cached is None or cached[0] != fingerprint:
        bundle = joblib.load(path)
        found = bundle.get("schema") if isinstance(bundle, dict) else None
        if found != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"[{uid}] Model tersimpan memakai skema {found}, versi saat ini "
                f"{FEATURE_SCHEMA_VERSION}. Jalankan retrain untuk uid ini."
            )
        _cache[uid] = (fingerprint, bundle)
    return _cache[uid][1]
