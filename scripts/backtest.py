"""Evaluasi rolling-origin model prakiraan, tanpa menyentuh model produksi.

Menjalankan alur yang sama dengan /retrain (app.models.pipeline.fit_sensor)
tetapi hanya menyimpan LAPORAN: satu JSON per sensor dan satu CSV ringkasan
(per sensor × parameter × horizon, satuan fisik) yang siap dipakai untuk tabel
naskah.

    python scripts/backtest.py                          # semua sensor, dari basis data
    python scripts/backtest.py --uid AQI01B AQI02S
    python scripts/backtest.py --csv-dir /path/ekspor   # dari CSV per menit (<uid>.csv)
    python scripts/backtest.py --out laporan/

CSV per menit: wajib punya kolom `datetime_unix` dan kolom parameter sensor
(`mmhg` boleh bernama `tekanan_mmhg`, seperti pada berkas ekspor AQMS).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.config import FEATURE_COLS, LOGS_DIR, TRAIN_HISTORY_HOURS
from app.models import preprocessor as pp
from app.models.pipeline import fit_sensor
from app.models.tuning import load_best_params

SUMMARY_FIELDS = [
    "uid", "parameter", "horizon", "n", "mae", "rmse", "bias",
    "mae_persistence", "rmse_persistence", "mae_seasonal_naive",
    "skill_vs_persistence", "skill_vs_seasonal_naive",
    "coverage", "interval_width_median", "dm_pvalue_vs_persistence",
]


def hourly_from_csv(path: str) -> pd.DataFrame:
    """CSV per menit → deret per jam, dengan aturan yang sama seperti jalur DB."""
    raw = pd.read_csv(path, low_memory=False).rename(columns={"tekanan_mmhg": "mmhg"})
    missing = [c for c in ["datetime_unix"] + FEATURE_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: kolom tidak ditemukan: {missing}")
    for col in FEATURE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["hour_start"] = (raw["datetime_unix"] // 3600) * 3600
    grouped = raw.groupby("hour_start")
    rows = grouped[FEATURE_COLS].mean()
    rows.insert(0, pp.COUNT_COL, grouped.size())
    frame = pp.to_hourly_frame(rows.reset_index())
    # Jam terakhir pada berkas ekspor biasanya terpotong; perlakukan seperti jam
    # yang sedang berjalan.
    return pp.apply_quality_filters(frame.iloc[:-1])


def summary_rows(uid: str, report: dict):
    for col, scores in report["targets"].items():
        blocks = [("semua", scores["overall"])] + sorted(scores["by_horizon"].items())
        for horizon, block in blocks:
            yield {"uid": uid, "parameter": col, "horizon": horizon,
                   **{k: block.get(k) for k in SUMMARY_FIELDS[3:]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", nargs="*", help="sensor yang dievaluasi (bawaan: semua)")
    ap.add_argument("--csv-dir", help="folder CSV per menit; tanpa ini data diambil dari basis data")
    ap.add_argument("--out", default=os.path.join(LOGS_DIR, "backtest"), help="folder keluaran")
    args = ap.parse_args()

    if args.csv_dir:
        available = sorted(f[:-4] for f in os.listdir(args.csv_dir)
                           if f.endswith(".csv") and not f.startswith("_"))
        load = lambda uid: hourly_from_csv(os.path.join(args.csv_dir, f"{uid}.csv"))
    else:
        from app.db import get_all_uids
        available = get_all_uids()
        load = lambda uid: pp.load_hourly(uid, TRAIN_HISTORY_HOURS)
    uids = args.uid or available

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "ringkasan.csv")
    print(f"{'sensor':<12}{'jam':>7}{'MAE TSP':>10}{'persist.':>10}{'musiman':>10}"
          f"{'skill':>8}{'cakupan':>9}{'detik':>7}")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for uid in uids:
            started = time.time()
            try:
                df = load(uid)
                _, report, _ = fit_sensor(df, load_best_params(uid))
            except Exception as exc:                       # satu sensor gagal tidak menghentikan sisanya
                print(f"{uid:<12}dilewati: {exc}")
                continue
            with open(os.path.join(args.out, f"{uid}.json"), "w", encoding="utf-8") as jf:
                json.dump({"uid": uid, **report}, jf, ensure_ascii=False, indent=1)
            writer.writerows(summary_rows(uid, report))
            fh.flush()
            tsp = report["targets"].get("tsp", {}).get("overall")
            if tsp is None:
                print(f"{uid:<12}{len(df):>7}  (TSP tidak dimodelkan)")
                continue
            print(f"{uid:<12}{len(df):>7}{tsp['mae']:>10.2f}{tsp['mae_persistence']:>10.2f}"
                  f"{(tsp['mae_seasonal_naive'] or float('nan')):>10.2f}"
                  f"{tsp['skill_vs_persistence']:>8.1%}{tsp['coverage']:>9.1%}"
                  f"{time.time() - started:>7.0f}")
    print(f"\nRingkasan: {summary_path}\nLaporan per sensor: {args.out}/<uid>.json")


if __name__ == "__main__":
    main()
