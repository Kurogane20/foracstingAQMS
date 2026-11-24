"""Alur latih dan prediksi per sensor."""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLS, MODELS_DIR, N_FORECAST_HOURS, PHYSICAL_LIMITS, PRIMARY_PARAM,
    MAX_ANCHOR_AGE_HOURS,
)
from app.models.backtest import (
    backtest_target, fit_calibration, normalized_mae, plan_folds, score_target,
)
from app.models.calibration import calibrated_deltas
from app.models.features import build_rows, from_model_space, to_model_space
from app.models.forecaster import fit_models, load_bundle, predict_deltas, save_bundle
from app.models.preprocessor import load_for_predict, load_for_training
from app.training_progress import set_phase

REPORT_FILENAME = "backtest.json"

# Parameter dengan jam teramati lebih sedikit dari ini tidak dimodelkan (mis.
# suhu/tekanan/kelembapan pada sensor tanpa modul cuaca, yang melapor nol terus).
MIN_TARGET_HOURS = 24 * 14


def report_path(uid: str) -> str:
    return os.path.join(MODELS_DIR, uid, REPORT_FILENAME)


def load_report(uid: str) -> dict | None:
    path = report_path(uid)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fit_sensor(df: pd.DataFrame, lgbm_params: dict | None = None,
               on_progress=None) -> tuple[dict, dict, dict]:
    """Latih dan evaluasi semua parameter dari satu deret per jam. Tanpa I/O.

    Per parameter: (1) backtest rolling-origin → prakiraan luar-sampel,
    (2) kalibrasi α dan interval dari SELURUH prakiraan luar-sampel itu,
    (3) latih ulang model akhir pada seluruh data. Skor yang dilaporkan berasal
    dari langkah 1, jadi sedikit konservatif: model akhir melihat lebih banyak
    data daripada model mana pun yang dinilai.

    Mengembalikan (bundel model, laporan backtest, ringkasan).
    """
    folds = plan_folds(df.index)
    z = to_model_space(df)

    targets: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    norm_maes: list[float] = []
    skipped: list[str] = []

    for i, col in enumerate(FEATURE_COLS):
        if on_progress is not None:
            on_progress(i / len(FEATURE_COLS))
        if int(z[col].notna().sum()) < MIN_TARGET_HOURS:
            skipped.append(col)
            continue
        X, meta = build_rows(z, col)
        oos = backtest_target(X, meta, z[col], folds, lgbm_params)
        col_scores = score_target(oos, col)
        if col_scores is None:
            skipped.append(col)
            continue

        alpha, conformal = fit_calibration(oos)
        fit = meta["delta"].notna().values
        targets[col] = {
            "models": fit_models(X[fit], meta.loc[fit, "delta"].values, lgbm_params),
            "alpha": alpha,
            "conformal": conformal,
            "features": list(X.columns),
        }
        scores[col] = col_scores
        nm = normalized_mae(oos, z[col])
        if nm is not None:
            norm_maes.append(nm)

    if not targets:
        raise ValueError("Insufficient data: tidak ada parameter yang cukup datanya")

    trained_until = df.index[-1]
    bundle = {
        "targets": targets,
        "trained_until": trained_until.isoformat(),
        "params": lgbm_params or {},
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "LightGBM langsung multi-horizon (selisih log, fitur lag + musiman harian)",
        "data": {
            "start": df.index[0].isoformat(), "end": trained_until.isoformat(),
            "hours": int(len(df)),
            "hours_observed": {c: int(df[c].notna().sum()) for c in FEATURE_COLS},
        },
        "folds": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in folds],
        "skipped_targets": skipped,
        "targets": scores,
    }
    primary = scores.get(PRIMARY_PARAM, {}).get("overall", {})
    summary = {
        "training_samples": int(df[PRIMARY_PARAM].notna().sum()),
        # Tanpa satuan, hanya untuk lencana "% acc" di dasbor (lihat normalized_mae).
        "mae_score": round(float(np.mean(norm_maes)), 6),
        # Yang bermakna: TSP dalam µg/m³, dari lipatan luar-sampel.
        "mae_tsp": primary.get("mae"),
        "mae_persistence_tsp": primary.get("mae_persistence"),
        "skill_vs_naive": primary.get("skill_vs_persistence"),
        "skill_vs_seasonal_naive": primary.get("skill_vs_seasonal_naive"),
        "coverage_tsp": primary.get("coverage"),
        "trained_targets": list(targets),
        "skipped_targets": skipped,
        "folds": len(folds),
    }
    return bundle, report, summary


def run_training(uid: str, lgbm_params: dict | None = None,
                 hourly: pd.DataFrame | None = None) -> dict:
    """Latih satu sensor, simpan model dan laporan backtest, kembalikan ringkasan."""
    df = hourly if hourly is not None else load_for_training(uid)
    bundle, report, summary = fit_sensor(
        df, lgbm_params,
        on_progress=lambda frac: set_phase(uid, phase="training", percent=5 + int(90 * frac)),
    )
    save_bundle(uid, bundle)
    with open(report_path(uid), "w", encoding="utf-8") as fh:
        json.dump({"uid": uid, **report}, fh, ensure_ascii=False, indent=1)
    return summary


def _clip(values: np.ndarray, col: str) -> np.ndarray:
    limits = PHYSICAL_LIMITS.get(col)
    return values if limits is None else np.clip(values, limits[0], limits[1])


def run_prediction(uid: str) -> list:
    """Prakiraan 6 jam ke depan dari jam penuh terakhir.

    Langkah 1 adalah jam yang sedang berjalan — jam pertama yang belum teramati
    penuh. Parameter yang tidak dimodelkan, atau yang bacaan terakhirnya lebih
    tua dari MAX_ANCHOR_AGE_HOURS, bernilai None: tidak ada angka karangan.
    """
    bundle = load_bundle(uid)                      # FileNotFoundError = belum dilatih
    df = load_for_predict(uid)
    if df.empty:
        raise ValueError(f"Insufficient hourly data for uid={uid}: tidak ada data")
    z = to_model_space(df)
    last_time = z.index[-1]
    origin = pd.DatetimeIndex([last_time])

    point = {c: [None] * N_FORECAST_HOURS for c in FEATURE_COLS}
    lower = {c: [None] * N_FORECAST_HOURS for c in FEATURE_COLS}
    upper = {c: [None] * N_FORECAST_HOURS for c in FEATURE_COLS}
    n_forecast = 0

    for col in FEATURE_COLS:
        spec = bundle["targets"].get(col)
        if spec is None:
            continue
        X, meta = build_rows(z, col, origins=origin)
        if X.empty:                                # jangkar basi / tidak ada
            continue
        # Kolom konteks bisa berbeda dari saat latih bila sebuah parameter kosong
        # total di jendela prediksi; kolom yang hilang masuk sebagai NaN.
        raw = predict_deltas(spec["models"], X.reindex(columns=spec["features"]))
        h = meta["h"].values
        p, lo, hi = calibrated_deltas(raw["point"], raw["lower"], raw["upper"], h,
                                      spec["alpha"], spec["conformal"])
        anchor = meta["anchor"].values
        for values, store in ((p, point), (lo, lower), (hi, upper)):
            phys = _clip(from_model_space(anchor + values, col), col)
            for step, v in zip(h, phys):
                store[col][step - 1] = float(v)
        n_forecast += 1

    if n_forecast == 0:
        raise ValueError(
            f"Data sensor {uid} basi: tidak ada bacaan sah dalam "
            f"{MAX_ANCHOR_AGE_HOURS + 1} jam terakhir, prakiraan tidak dibuat."
        )

    results = []
    for step in range(1, N_FORECAST_HOURS + 1):
        row = {"step": step, "target_time": (last_time + pd.Timedelta(hours=step)).isoformat()}
        for col in FEATURE_COLS:
            row[col] = point[col][step - 1]
        row["lower_bounds"] = {c: lower[c][step - 1] for c in FEATURE_COLS}
        row["upper_bounds"] = {c: upper[c][step - 1] for c in FEATURE_COLS}
        results.append(row)
    return results
