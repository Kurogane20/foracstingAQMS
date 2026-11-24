"""Kalibrasi pasca-model, per horizon, dari prakiraan LUAR-SAMPEL.

Dua koreksi kecil yang dicari pada prakiraan yang dibuat model untuk data yang
belum pernah dilihatnya (lipatan backtest), tidak pernah pada data latihnya:

1. Penyusutan menuju persistence (α).  prakiraan = jangkar + α × selisih_model.
   α dicari pada [0, 1] dengan meminimalkan MAE. α = 0 menghasilkan persistence
   persis, jadi pada data kalibrasinya hasil ini tidak bisa lebih buruk daripada
   persistence. Parameter yang selisihnya praktis derau otomatis mendapat α ≈ 0.

2. Pelebaran interval (conformalized quantile regression, Romano dkk. 2019).
   Kuantil P10/P90 keluaran model cenderung terlalu sempit di luar sampel.
   Satu tetapan q per horizon ditambahkan ke kedua sisi sehingga cakupan empiris
   pada data kalibrasi mencapai nominalnya.
"""
import numpy as np

from app.config import INTERVAL_COVERAGE, N_FORECAST_HOURS

# Hanya menyusutkan, tidak pernah memperkuat: α > 1 melipatgandakan derau.
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)

MIN_ALPHA_SAMPLES = 20
MIN_CONFORMAL_SAMPLES = 30


def fit_alpha(delta_pred: np.ndarray, delta_true: np.ndarray, h: np.ndarray) -> np.ndarray:
    """α per horizon, bentuk (N_FORECAST_HOURS,). Masukan vektor sejajar (N,).

    Horizon dengan sampel terlalu sedikit mendapat α = 1 (model apa adanya):
    tanpa bukti, tidak ada dasar untuk mengubah keluaran model.
    """
    alpha = np.ones(N_FORECAST_HOURS, dtype=np.float64)
    for step in range(1, N_FORECAST_HOURS + 1):
        m = h == step
        if int(m.sum()) < MIN_ALPHA_SAMPLES:
            continue
        p, t = delta_pred[m], delta_true[m]
        errs = np.abs(ALPHA_GRID[:, None] * p[None, :] - t[None, :]).mean(axis=1)
        alpha[step - 1] = float(ALPHA_GRID[int(np.argmin(errs))])
    return alpha


def recenter_band(point_shrunk: np.ndarray, point_raw: np.ndarray,
                  bound_raw: np.ndarray) -> np.ndarray:
    """Geser batas kuantil mengikuti titik yang disusutkan, LEBAR DIPERTAHANKAN.

    Ketidakpastian tidak berkurang hanya karena prakiraan titiknya ditarik ke
    persistence; yang berubah pusat pita, bukan lebarnya.
    """
    return point_shrunk + (bound_raw - point_raw)


def fit_conformal(lower: np.ndarray, upper: np.ndarray, true: np.ndarray, h: np.ndarray,
                  coverage: float = INTERVAL_COVERAGE) -> np.ndarray:
    """Pelebaran q per horizon sehingga [lower − q, upper + q] mencakup `coverage`.

    q boleh negatif (pita menyempit) bila kuantil model ternyata terlalu lebar.
    Horizon dengan sampel terlalu sedikit mendapat q = 0.
    """
    q = np.zeros(N_FORECAST_HOURS, dtype=np.float64)
    for step in range(1, N_FORECAST_HOURS + 1):
        m = h == step
        n = int(m.sum())
        if n < MIN_CONFORMAL_SAMPLES:
            continue
        score = np.maximum(lower[m] - true[m], true[m] - upper[m])
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        q[step - 1] = float(np.quantile(score, level))
    return q


def calibrated_deltas(point: np.ndarray, lower: np.ndarray, upper: np.ndarray, h: np.ndarray,
                      alpha: np.ndarray, conformal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Terapkan α dan q pada selisih mentah model. Mengembalikan (titik, bawah, atas)
    dengan jaminan bawah ≤ titik ≤ atas: kuantil dari model terpisah bisa saling
    menyilang, dan pita terbalik merusak ambang peringatan P90."""
    a = alpha[h - 1]
    q = conformal[h - 1]
    p = a * point
    lo = recenter_band(p, point, lower) - q
    hi = recenter_band(p, point, upper) + q
    return p, np.minimum(lo, p), np.maximum(hi, p)
