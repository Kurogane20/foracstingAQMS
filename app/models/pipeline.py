import numpy as np
import pandas as pd
from app.config import FEATURE_COLS, N_FORECAST_HOURS, N_INPUT_HOURS
from app.models.preprocessor import (
    preprocess_for_predict,
    preprocess_for_training,
    denormalize,
    anchor_from_X,
    to_delta,
    from_delta,
)
from app.models.bilstm import predict_bilstm, train_bilstm
from app.models.shrinkage import (
    fit_alpha, save_alpha, load_alpha, apply_alpha, recenter_band,
)
from app.models.lgbm import predict_lgbm, train_lgbm
from app.training_progress import set_phase


def run_prediction(uid: str) -> list:
    X, timestamps = preprocess_for_predict(uid)
    last_time = timestamps[-1]

    # Model meramal SELISIH terhadap nilai terakhir teramati, jadi jangkarnya
    # harus ditambahkan kembali sebelum denormalisasi.
    anchor = anchor_from_X(X)                                   # (1, 14)

    bilstm_out = predict_bilstm(X, uid)                         # (1, 6, 14) selisih
    lgbm_result = predict_lgbm(bilstm_out[0], last_time, uid, anchor[0])

    # Susutkan selisih menuju persistence memakai bobot hasil validasi.
    alpha = load_alpha(uid)
    point_raw = lgbm_result["point"]
    point_shrunk = apply_alpha(point_raw, alpha)

    def _to_abs(delta):
        if delta is None:
            return None
        return denormalize(from_delta(delta[np.newaxis, ...], anchor)[0], uid)

    final_point = _to_abs(point_shrunk)                         # (6, 14)
    # Pita kuantil digeser mengikuti titik yang disusutkan, lebarnya dipertahankan
    # — ketidakpastian tidak berkurang hanya karena titiknya ditarik ke persistence.
    final_lower = _to_abs(recenter_band(point_shrunk, point_raw, lgbm_result["lower"])
                          if lgbm_result.get("lower") is not None else None)
    final_upper = _to_abs(recenter_band(point_shrunk, point_raw, lgbm_result["upper"])
                          if lgbm_result.get("upper") is not None else None)

    # Kuantil bisa saling menyilang setelah koreksi LightGBM per-keluaran;
    # pita yang terbalik akan tampil aneh di grafik dan merusak ambang p90.
    if final_lower is not None and final_upper is not None:
        final_lower, final_upper = (
            np.minimum(final_lower, final_upper),
            np.maximum(final_lower, final_upper),
        )

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

    # Latih pada selisih terhadap nilai terakhir teramati, bukan level absolut.
    anchor_train = anchor_from_X(X_train)
    anchor_val   = anchor_from_X(X_val)
    dy_train = to_delta(y_train, anchor_train)

    bilstm_kw = bilstm_params or {}
    train_bilstm(X_train, dy_train, uid, **bilstm_kw)

    set_phase(uid, phase="lgbm", percent=88, eta_seconds=None)
    bilstm_preds_train = np.array([
        predict_bilstm(X_train[i : i + 1], uid)[0]
        for i in range(len(X_train))
    ])
    base_times_train = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        for i in range(len(X_train))
    ]
    train_lgbm(bilstm_preds_train, dy_train, base_times_train, uid,
               anchors=anchor_train, lgbm_params=lgbm_params)

    set_phase(uid, phase="evaluating", percent=95, eta_seconds=None)
    val_bilstm = np.array([
        predict_bilstm(X_val[i : i + 1], uid)[0]
        for i in range(len(X_val))
    ])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i],
                     pd.Timestamp("2024-01-01") + pd.Timedelta(hours=split + i),
                     uid, anchor_val[i])["point"]
        for i in range(len(X_val))
    ])
    # Cari bobot penyusutan pada potongan validasi yang sama, lalu simpan.
    dy_val = to_delta(y_val, anchor_val)
    alpha = fit_alpha(val_lgbm, dy_val)
    save_alpha(uid, alpha)

    # Dievaluasi pada level absolut agar mae_score tetap sebanding antar versi
    # dan bisa dipakai deteksi drift; selisih akan tampak menipu bagusnya.
    # Memakai selisih yang SUDAH disusutkan — inilah yang nanti dipakai produksi.
    val_abs = from_delta(apply_alpha(val_lgbm, alpha), anchor_val)
    mae = float(np.mean(np.abs(val_abs - y_val)))

    # Baseline persistence pada potongan validasi yang sama. Tanpa ini tidak ada
    # yang tahu apakah 0,08 itu bagus — dan uji 1 Agu 2026 menunjukkan model
    # sebelumnya justru KALAH dari tebakan naif ini.
    naive_abs = from_delta(np.zeros_like(val_lgbm), anchor_val)
    mae_naive = float(np.mean(np.abs(naive_abs - y_val)))
    skill = round(1.0 - mae / mae_naive, 4) if mae_naive > 0 else None

    mae_raw = float(np.mean(np.abs(from_delta(val_lgbm, anchor_val) - y_val)))

    return {
        "training_samples": len(X),
        "mae_score":        round(mae, 6),
        "mae_persistence":  round(mae_naive, 6),
        "skill_vs_naive":   skill,
        # Diagnostik: MAE tanpa penyusutan + rata-rata bobot. alpha_mean mendekati
        # 0 berarti model tidak menambahkan informasi apa pun di sensor ini.
        "mae_unshrunk":     round(mae_raw, 6),
        "alpha_mean":       round(float(alpha.mean()), 4),
    }
