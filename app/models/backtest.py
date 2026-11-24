"""Evaluasi rolling-origin dan metrik dalam satuan fisik.

Skor model dilaporkan dari prakiraan yang benar-benar di luar sampel:

  * deret dibagi menjadi beberapa lipatan berurutan di ujung data;
  * model untuk lipatan k hanya dilatih pada baris yang JAM TARGETNYA sebelum
    awal lipatan k (jendela target tidak tumpang-tindih dengan data uji);
  * kalibrasi (α dan pelebaran interval) untuk lipatan k hanya berasal dari
    prakiraan luar-sampel lipatan-lipatan SEBELUMNYA — persis seperti yang
    tersedia bila model benar-benar dijalankan saat itu;
  * lipatan pertama tidak punya pendahulu, jadi hanya menyumbang kalibrasi.

Dibanding satu potongan uji 2–3 minggu, beberapa lipatan 4-mingguan mencakup
berbagai musim, sehingga keunggulan model tidak bisa kebetulan satu periode.
"""
import numpy as np
import pandas as pd
from scipy import stats

from app.config import (
    BACKTEST_FOLDS, BACKTEST_FOLD_HOURS, MIN_FOLD_HOURS, MIN_TRAIN_HOURS,
    N_FORECAST_HOURS, EXCEEDANCE_THRESHOLDS,
)
from app.models.calibration import calibrated_deltas, fit_alpha, fit_conformal
from app.models.features import from_model_space
from app.models.forecaster import fit_models, predict_deltas

# Baris latih minimum agar satu lipatan layak dilatih.
MIN_TRAIN_ROWS = 24 * 7 * N_FORECAST_HOURS


def plan_folds(index: pd.DatetimeIndex, n_folds: int | None = None,
               fold_hours: int | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Lipatan uji [awal, akhir) berurutan yang berakhir di ujung data.

    Untuk sensor baru, panjang lalu jumlah lipatan diperkecil agar tetap tersisa
    MIN_TRAIN_HOURS data latih di depan. Minimal dua lipatan (satu kalibrasi,
    satu pelaporan); kurang dari itu data belum cukup untuk dilatih secara jujur.
    """
    n_folds = BACKTEST_FOLDS if n_folds is None else n_folds
    fold_hours = BACKTEST_FOLD_HOURS if fold_hours is None else fold_hours
    available = len(index) - MIN_TRAIN_HOURS
    while n_folds > 2 and available // n_folds < MIN_FOLD_HOURS:
        n_folds -= 1
    size = min(fold_hours, available // n_folds) if n_folds > 0 else 0
    if size < MIN_FOLD_HOURS:
        need = MIN_TRAIN_HOURS + 2 * MIN_FOLD_HOURS
        raise ValueError(
            f"Data belum cukup: {len(index)} jam, butuh minimal {need} jam "
            f"({need // 24} hari) untuk melatih dan mengevaluasi model."
        )
    end = index[-1] + pd.Timedelta(hours=1)
    return [
        (end - pd.Timedelta(hours=size * k), end - pd.Timedelta(hours=size * (k - 1)))
        for k in range(n_folds, 0, -1)
    ]


def fit_calibration(oos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(α, q) dari sekumpulan prakiraan luar-sampel mentah."""
    h = oos["h"].values
    point, true = oos["point"].values, oos["delta"].values
    alpha = fit_alpha(point, true, h)
    # q dicari pada pita yang SUDAH digeser mengikuti titik tersusut — pita
    # itulah yang nanti dikeluarkan, jadi cakupannya yang harus dijamin.
    zero = np.zeros(N_FORECAST_HOURS)
    _, lo, hi = calibrated_deltas(point, oos["lower"].values, oos["upper"].values, h, alpha, zero)
    return alpha, fit_conformal(lo, hi, true, h)


def backtest_target(X: pd.DataFrame, meta: pd.DataFrame, series: pd.Series,
                    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
                    params: dict | None = None) -> pd.DataFrame:
    """Prakiraan luar-sampel satu parameter di seluruh lipatan.

    Kolom hasil: META_COLS + `fold`, selisih mentah (`point`, `lower`, `upper`),
    selisih terkalibrasi (`cal_point`, `cal_lower`, `cal_upper`; NaN di lipatan
    pertama), dan `z_seasonal` (nilai jam-target yang sama kemarin, pembanding).
    `series` adalah deret parameter di ruang model, berindeks per jam.
    """
    target_time = meta["t"] + pd.to_timedelta(meta["h"], unit="h")
    has_y = meta["delta"].notna().values
    done: list[pd.DataFrame] = []

    for k, (start, end) in enumerate(folds):
        train = has_y & (target_time < start).values
        test = has_y & ((meta["t"] >= start) & (meta["t"] < end)).values
        if int(train.sum()) < MIN_TRAIN_ROWS or not test.any():
            continue
        models = fit_models(X[train], meta.loc[train, "delta"].values, params)
        raw = predict_deltas(models, X[test])

        out = meta[test].copy()
        out["fold"] = k
        for name in ("point", "lower", "upper"):
            out[name] = raw[name]

        h = out["h"].values
        if done:
            alpha, conformal = fit_calibration(pd.concat(done))
            p, lo, hi = calibrated_deltas(raw["point"], raw["lower"], raw["upper"], h, alpha, conformal)
        else:
            p = lo = hi = np.full(len(out), np.nan)
        out["cal_point"], out["cal_lower"], out["cal_upper"] = p, lo, hi
        done.append(out)

    if not done:
        return pd.DataFrame()
    oos = pd.concat(done, ignore_index=True)
    tt = oos["t"] + pd.to_timedelta(oos["h"], unit="h")
    oos["z_seasonal"] = series.shift(24).reindex(tt).values
    return oos


# ── Metrik ───────────────────────────────────────────────────────────────────

def diebold_mariano(err_model: np.ndarray, err_ref: np.ndarray, h: int) -> float | None:
    """Nilai-p dua sisi uji Diebold–Mariano pada galat absolut, horizon h.

    Varians jangka panjang memakai autokovarians sampai lag h−1 (galat prakiraan
    h-langkah berkorelasi sampai lag itu), dengan koreksi sampel kecil
    Harvey–Leybourne–Newbold (1997). None bila tidak bisa dihitung.
    """
    d = np.abs(err_model) - np.abs(err_ref)
    n = len(d)
    if n < 30 + h:
        return None
    dc = d - d.mean()
    var = float(np.dot(dc, dc)) / n
    for k in range(1, h):
        var += 2.0 * float(np.dot(dc[k:], dc[:-k])) / n
    if var <= 0:
        return None
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    return float(2.0 * stats.t.sf(abs(dm * hln), df=n - 1))


def _detection(event: np.ndarray, warn: np.ndarray) -> dict:
    hits = int((event & warn).sum())
    misses = int((event & ~warn).sum())
    false = int((~event & warn).sum())
    return {
        "pod": round(hits / (hits + misses), 4) if hits + misses else None,
        "far": round(false / (hits + false), 4) if hits + false else None,
        "csi": round(hits / (hits + misses + false), 4) if hits + misses + false else None,
    }


def _block(y, pred, pers, seas, lo, hi) -> dict:
    err, err_p = pred - y, pers - y
    mae, mae_p = float(np.abs(err).mean()), float(np.abs(err_p).mean())
    s = ~np.isnan(seas)
    mae_s = float(np.abs(seas[s] - y[s]).mean()) if s.any() else None
    mae_on_s = float(np.abs(err[s]).mean()) if s.any() else None
    return {
        "n": int(len(y)),
        "mae": round(mae, 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "bias": round(float(err.mean()), 4),
        "mae_persistence": round(mae_p, 4),
        "rmse_persistence": round(float(np.sqrt((err_p ** 2).mean())), 4),
        "mae_seasonal_naive": round(mae_s, 4) if mae_s is not None else None,
        "skill_vs_persistence": round(1.0 - mae / mae_p, 4) if mae_p > 0 else None,
        "skill_vs_seasonal_naive": round(1.0 - mae_on_s / mae_s, 4) if mae_s else None,
        "coverage": round(float(((y >= lo) & (y <= hi)).mean()), 4),
        "interval_width_median": round(float(np.median(hi - lo)), 4),
    }


def score_target(oos: pd.DataFrame, col: str) -> dict | None:
    """Ringkasan kinerja satu parameter dalam SATUAN FISIKNYA (µg/m³, °C, …).

    Hanya baris yang kalibrasinya berasal dari lipatan sebelumnya yang dinilai.
    """
    if oos.empty:
        return None
    r = oos[oos["cal_point"].notna()].sort_values(["h", "t"])
    if r.empty:
        return None

    def phys(delta):
        return from_model_space(r["anchor"].values + np.asarray(delta), col)

    y = from_model_space(r["z_true"].values, col)
    pred, lo, hi = phys(r["cal_point"].values), phys(r["cal_lower"].values), phys(r["cal_upper"].values)
    pers = from_model_space(r["anchor"].values, col)
    seas = from_model_space(r["z_seasonal"].values, col)
    h = r["h"].values

    report = {"overall": _block(y, pred, pers, seas, lo, hi), "by_horizon": {}}
    report["overall"]["mae_uncalibrated"] = round(float(np.abs(phys(r["point"].values) - y).mean()), 4)
    report["folds_scored"] = int(r["fold"].nunique())
    report["period"] = {
        "start": r["t"].min().isoformat(),
        "end": (r["t"].max() + pd.Timedelta(hours=N_FORECAST_HOURS)).isoformat(),
    }
    for step in range(1, N_FORECAST_HOURS + 1):
        m = h == step
        if not m.any():
            continue
        block = _block(y[m], pred[m], pers[m], seas[m], lo[m], hi[m])
        block["dm_pvalue_vs_persistence"] = diebold_mariano(pred[m] - y[m], pers[m] - y[m], step)
        report["by_horizon"][str(step)] = block

    threshold = EXCEEDANCE_THRESHOLDS.get(col)
    if threshold is not None:
        event = y > threshold
        report["exceedance"] = {
            "threshold": threshold,
            # Jumlah baris (jam asal × horizon) yang nilai sebenarnya melampaui ambang.
            "events": int(event.sum()),
            "point": _detection(event, pred > threshold),
            "upper_bound": _detection(event, hi > threshold),
            "persistence": _detection(event, pers > threshold),
        }
    return report


def normalized_mae(oos: pd.DataFrame, series: pd.Series) -> float | None:
    """MAE di ruang model dibagi rentang deret — besaran tanpa satuan di [0, 1]-an.

    Hanya untuk kolom `model_metadata.mae_score`, yang oleh dasbor ditampilkan
    sebagai "(1 − mae) × 100 % acc". Angka ini TIDAK untuk dilaporkan; yang
    bermakna adalah MAE per parameter dalam satuan fisik pada laporan backtest.
    """
    r = oos[oos["cal_point"].notna()] if not oos.empty else oos
    span = float(series.max() - series.min()) if series.notna().any() else 0.0
    if r.empty or span <= 0:
        return None
    return float(np.abs(r["cal_point"] - r["delta"]).mean() / span)
