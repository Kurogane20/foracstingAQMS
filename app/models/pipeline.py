import numpy as np
import pandas as pd
from app.config import FEATURE_COLS, N_FORECAST_HOURS, N_INPUT_HOURS

# Sekuens yang dibuang di tiap batas (latih|validasi, kalibrasi|pelaporan) agar
# jendela target tidak tumpang-tindih.
PURGE_SEQUENCES = N_FORECAST_HOURS
MIN_EVAL_SEQUENCES = 20
from app.models.preprocessor import (
    preprocess_for_predict,
    preprocess_for_training,
    denormalize,
    anchor_from_X,
    to_delta,
    from_delta,
    train_split_index,
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
                 X: np.ndarray = None, y: np.ndarray = None,
                 base_times: pd.DatetimeIndex = None) -> dict:
    if X is None or y is None:
        X, y, base_times = preprocess_for_training(uid)
    elif base_times is None or len(base_times) != len(X):
        # Tanpa jam nyata, fitur kalender LightGBM terpaksa dikarang — itulah
        # train/serve skew yang diperbaiki di sini. Lebih baik gagal jelas.
        raise ValueError(
            f"run_training({uid}): X/y dari luar wajib disertai base_times "
            f"sepanjang X (jam input terakhir tiap sekuens)."
        )

    if len(X) < 10:
        raise ValueError(f"Insufficient data for {uid}: only {len(X)} sequences")

    split = train_split_index(len(X))
    # Jendela geser membuat target sekuens latih terakhir tumpang-tindih dengan
    # target sekuens validasi pertama (N_FORECAST_HOURS baris). Buang sekuens di
    # celah itu ("purging") agar validasi benar-benar tak pernah terlihat.
    val_start = min(split + PURGE_SEQUENCES, len(X))
    X_train, X_val = X[:split], X[val_start:]
    y_train, y_val = y[:split], y[val_start:]
    if len(X_val) < 2 * MIN_EVAL_SEQUENCES + PURGE_SEQUENCES:
        raise ValueError(
            f"Data validasi {uid} terlalu sedikit ({len(X_val)} sekuens) untuk "
            f"memisahkan set kalibrasi α dari set pelaporan."
        )

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
    # Jam nyata, bukan 2024-01-01 + i: produksi memberi LightGBM jam sungguhan,
    # jadi fitur jam/hari/bulan saat latih harus berasal dari jam yang sama.
    train_lgbm(bilstm_preds_train, dy_train, base_times[:split], uid,
               anchors=anchor_train, lgbm_params=lgbm_params)

    set_phase(uid, phase="evaluating", percent=95, eta_seconds=None)
    val_bilstm = np.array([
        predict_bilstm(X_val[i : i + 1], uid)[0]
        for i in range(len(X_val))
    ])
    val_lgbm = np.array([
        predict_lgbm(val_bilstm[i], base_times[val_start + i], uid, anchor_val[i])["point"]
        for i in range(len(X_val))
    ])
    dy_val = to_delta(y_val, anchor_val)

    # Validasi dibagi dua secara kronologis: paruh awal HANYA untuk mencari α,
    # paruh akhir HANYA untuk melaporkan skor. Sebelumnya α dicari dan skor
    # dilaporkan pada data yang sama, sehingga sebagian keunggulan yang terlihat
    # adalah hasil penyetelan pada set itu sendiri (bias optimistis). Celah
    # PURGE_SEQUENCES di antaranya mencegah target kedua paruh tumpang-tindih.
    n_val = len(X_val)
    cal_end = (n_val - PURGE_SEQUENCES) // 2
    rep = slice(cal_end + PURGE_SEQUENCES, None)

    alpha = fit_alpha(val_lgbm[:cal_end], dy_val[:cal_end])
    save_alpha(uid, alpha)

    # Dievaluasi pada level absolut agar mae_score tetap sebanding antar versi
    # dan bisa dipakai deteksi drift. Memakai selisih yang SUDAH disusutkan —
    # inilah yang nanti dipakai produksi — pada data yang tidak dipakai mencari α.
    y_rep, a_rep, p_rep = y_val[rep], anchor_val[rep], val_lgbm[rep]
    val_abs = from_delta(apply_alpha(p_rep, alpha), a_rep)
    mae = float(np.mean(np.abs(val_abs - y_rep)))

    # Baseline persistence pada potongan pelaporan yang sama.
    naive_abs = from_delta(np.zeros_like(p_rep), a_rep)
    mae_naive = float(np.mean(np.abs(naive_abs - y_rep)))
    skill = round(1.0 - mae / mae_naive, 4) if mae_naive > 0 else None

    mae_raw = float(np.mean(np.abs(from_delta(p_rep, a_rep) - y_rep)))

    return {
        "training_samples": len(X),
        "mae_score":        round(mae, 6),
        "mae_persistence":  round(mae_naive, 6),
        "skill_vs_naive":   skill,
        # Diagnostik: MAE tanpa penyusutan + rata-rata bobot. alpha_mean mendekati
        # 0 berarti model tidak menambahkan informasi apa pun di sensor ini.
        "mae_unshrunk":     round(mae_raw, 6),
        "alpha_mean":       round(float(alpha.mean()), 4),
        # Ukuran tiap potongan, agar skor dapat ditafsirkan dengan benar.
        "alpha_calibration_samples": cal_end,
        "report_samples":            len(y_rep),
    }
