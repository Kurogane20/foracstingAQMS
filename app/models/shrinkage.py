"""Penyusutan prakiraan selisih menuju persistence, per horizon & per parameter.

LATAR BELAKANG TERUKUR (retrain 1 Agu 2026, evaluasi luar-sampel 10% terakhir):
meski model sudah meramal selisih terhadap nilai terakhir, hanya 4 dari 9 sensor
yang mengalahkan tebakan naif; skill median −3,9% dan AQI02L −85,7%. Penyebabnya:
meramal selisih membuat persistence TERJANGKAU (selisih = 0), bukan TERJAMIN.
Di sensor yang perubahan antar jamnya kecil (AQI02L: MAE naif 0,0182, terendah),
yang dipelajari model praktis hanya derau, dan ia menggandakan galat.

Solusinya menskalakan selisih dengan bobot α yang dicari pada potongan validasi:

    prediksi = nilai_terakhir + α × selisih_model

α dicari per (horizon, parameter) dengan meminimalkan MAE — metrik yang sama
yang dilaporkan — pada rentang [0, 1]. Karena α = 0 selalu ada di dalam rentang
pencarian dan menghasilkan persistence PERSIS, hasilnya tidak bisa lebih buruk
daripada persistence pada data validasi itu. Sensor seperti AQI02L akan otomatis
mendapat α ≈ 0 dan berhenti merusak; AQI01G (+24,8%) akan mendapat α ≈ 1 dan
tetap memberikan nilainya.

Disimpan sebagai JSON, bukan pickle: isinya hanya array angka kecil, dan tidak
ada alasan menambah berkas yang dieksekusi saat dimuat.
"""
import json
import logging
import os

import numpy as np

from app.config import MODELS_DIR, N_FORECAST_HOURS, N_FEATURES, FEATURE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

FILENAME = "shrinkage.json"

# Hanya menyusutkan, tidak pernah memperkuat: α > 1 akan melipatgandakan derau
# pada sensor yang modelnya sudah terlalu percaya diri.
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def fit_alpha(delta_pred: np.ndarray, delta_true: np.ndarray) -> np.ndarray:
    """Cari α optimal per (horizon, parameter). Bentuk masukan (N, 6, 14)."""
    alpha = np.zeros((N_FORECAST_HOURS, N_FEATURES), dtype=np.float64)
    for h in range(N_FORECAST_HOURS):
        for f in range(N_FEATURES):
            p = delta_pred[:, h, f]
            t = delta_true[:, h, f]
            if p.size == 0:
                alpha[h, f] = 0.0
                continue
            # MAE untuk tiap kandidat α sekaligus: (n_alpha, N)
            errs = np.abs(ALPHA_GRID[:, None] * p[None, :] - t[None, :]).mean(axis=1)
            alpha[h, f] = float(ALPHA_GRID[int(np.argmin(errs))])
    return alpha


def save_alpha(uid: str, alpha: np.ndarray) -> None:
    path = os.path.join(MODELS_DIR, uid, FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": FEATURE_SCHEMA_VERSION, "alpha": alpha.tolist()}, fh)


def load_alpha(uid: str) -> np.ndarray:
    """Bobot tersimpan, atau 1.0 (tanpa penyusutan) bila belum ada.

    Sengaja TIDAK melempar galat: model yang dilatih sebelum penyusutan ada tetap
    bisa memprediksi seperti sebelumnya, hanya tanpa perlindungan ini.
    """
    path = os.path.join(MODELS_DIR, uid, FILENAME)
    if not os.path.exists(path):
        logger.warning(f"[{uid}] Bobot penyusutan tidak ditemukan — memakai α=1 "
                       f"(tanpa perlindungan terhadap persistence). Jalankan retrain.")
        return np.ones((N_FORECAST_HOURS, N_FEATURES), dtype=np.float64)
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return np.asarray(blob["alpha"], dtype=np.float64)


def apply_alpha(delta_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Susutkan prakiraan selisih. Bentuk (6, 14) atau (N, 6, 14)."""
    return delta_pred * (alpha if delta_pred.ndim == 2 else alpha[None, ...])


def recenter_band(
    point_shrunk: np.ndarray,
    point_raw: np.ndarray,
    bound_raw: np.ndarray,
) -> np.ndarray:
    """Pindahkan batas kuantil agar mengikuti titik yang sudah disusutkan, dengan
    LEBAR PITA DIPERTAHANKAN.

    Menyusutkan batasnya sendiri akan mempersempit pita ketidakpastian dan
    melemahkan ambang peringatan p90 — padahal ketidakpastiannya tidak berkurang
    hanya karena prakiraan titiknya ditarik ke persistence. Yang berubah
    seharusnya pusat pita, bukan lebarnya.
    """
    return point_shrunk + (bound_raw - point_raw)
