"""Fitur tabular untuk prakiraan LANGSUNG multi-horizon.

Satu baris = satu pasangan (jam asal t, horizon h). Targetnya SELISIH di ruang
model, z[t+h] − jangkar[t], dengan z = log1p(nilai) untuk konsentrasi/indeks.
Meramal selisih membuat persistence (selisih = 0) menjadi titik berangkat model.

Mengapa tabular, bukan jendela mentah 24 jam ke jaringan rekuren? Pada data
Berau Coal (9 sensor, 13 bulan) sinyal terkuat ternyata sederhana:

  * nilai sekarang (persistence) — dominan di H+1;
  * nilai pada JAM YANG SAMA kemarin dan rata-ratanya sepekan — siklus harian
    TSP di sini besar (median malam ±4× siang), sehingga di H+6 "jam yang sama
    kemarin" saja sudah mengalahkan persistence di semua sensor;
  * arah perubahan beberapa jam terakhir.

Fitur di bawah menyajikan ketiganya secara eksplisit, semuanya RELATIF terhadap
jangkar, sehingga pohon keputusan tidak perlu menemukannya sendiri dari 24×38
angka mentah.

Seluruh fitur baris (t, h) hanya memakai data sampai jam t. NaN dibiarkan:
LightGBM menanganinya secara bawaan.
"""
import numpy as np
import pandas as pd

from app.config import (
    LOG_SCALE_COLS, LOCAL_UTC_OFFSET_HOURS, MAX_ANCHOR_AGE_HOURS, N_FORECAST_HOURS,
)

# Parameter lain yang ikut dilihat setiap model sebagai konteks: tiga fraksi
# partikulat saling menjelaskan, dan kelembapan/suhu membawa informasi hujan dan
# stabilitas udara dari sensor itu sendiri.
CONTEXT_COLS = ["tsp", "pm_10", "pm_25", "humidity", "temp", "noise"]

_DIFF_LAGS = (1, 2, 3, 6, 12, 24)
_MEAN_WINDOWS = (3, 6, 12, 24, 72, 168)
_STD_WINDOWS = (6, 24, 168)
_SEASONAL_DAYS = 7

META_COLS = ["t", "h", "anchor", "z_true", "delta"]


def to_model_space(df: pd.DataFrame) -> pd.DataFrame:
    """Nilai fisik → ruang model (log1p untuk kolom condong-kanan)."""
    z = df.astype("float64").copy()
    cols = [c for c in LOG_SCALE_COLS if c in z.columns]
    z[cols] = np.log1p(z[cols].clip(lower=0.0))
    return z


def from_model_space(values, col: str):
    """Kebalikan `to_model_space` untuk satu kolom."""
    values = np.asarray(values, dtype="float64")
    return np.expm1(values) if col in LOG_SCALE_COLS else values


def anchor_series(z_col: pd.Series) -> pd.Series:
    """Nilai teramati terakhir pada tiap jam, paling tua MAX_ANCHOR_AGE_HOURS."""
    return z_col.ffill(limit=MAX_ANCHOR_AGE_HOURS)


def origin_features(z: pd.DataFrame, target: str) -> pd.DataFrame:
    """Fitur yang diketahui pada jam asal t (indeks = t), sama untuk semua horizon."""
    s = z[target]
    a = anchor_series(s)
    f: dict[str, pd.Series] = {"lvl": a}

    for k in _DIFF_LAGS:
        f[f"d{k}"] = a - s.shift(k)
    for w in _MEAN_WINDOWS:
        roll = s.rolling(w, min_periods=max(2, w // 3))
        f[f"m{w}"] = roll.mean() - a
        if w in _STD_WINDOWS:
            f[f"s{w}"] = roll.std()
    r24 = s.rolling(24, min_periods=8)
    f["mx24"] = r24.max() - a
    f["mn24"] = r24.min() - a
    # Seberapa lengkap sehari terakhir: model boleh kurang percaya pada fitur
    # bergulir yang dihitung dari sedikit data.
    f["obs24"] = s.notna().astype("float64").rolling(24, min_periods=1).mean()

    for c in CONTEXT_COLS:
        if c == target or c not in z.columns or not z[c].notna().any():
            continue
        cz = anchor_series(z[c])
        f[f"{c}_lvl"] = cz
        f[f"{c}_d1"] = cz - z[c].shift(1)
        f[f"{c}_d3"] = cz - z[c].shift(3)
        f[f"{c}_m24"] = z[c].rolling(24, min_periods=8).mean() - cz
    return pd.DataFrame(f, index=z.index)


def _horizon_features(s: pd.Series, a: pd.Series, h: int) -> dict[str, pd.Series]:
    """Fitur musiman: apa yang terjadi pada jam-target yang sama di hari-hari lalu.

    `s.shift(24*d − h)` pada jam t adalah nilai di t + h − 24·d: jam dinding yang
    sama dengan target, d hari sebelumnya. Untuk h ≤ 24 nilainya selalu sudah
    teramati pada jam t.
    """
    same_hour = [s.shift(24 * d - h) for d in range(1, _SEASONAL_DAYS + 1)]
    change = [s.shift(24 * d - h) - s.shift(24 * d) for d in range(1, _SEASONAL_DAYS + 1)]
    return {
        "sn24": same_hour[0] - a,                      # jam target kemarin vs sekarang
        "sn48": same_hour[1] - a,
        "sn168": same_hour[6] - a,
        "snm7": pd.concat(same_hour, axis=1).mean(axis=1) - a,
        "dy24": change[0],                             # perubahan kemarin pada rentang jam yang sama
        "dym7": pd.concat(change, axis=1).mean(axis=1),
    }


def build_rows(z: pd.DataFrame, target: str,
               origins: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Susun baris (t, h) untuk satu parameter.

    `z` harus berindeks per jam yang KONTINU (lihat `to_hourly_frame`).
    Mengembalikan `(X, meta)`:
      X    — fitur, satu baris per (t, h), hanya jam asal yang jangkarnya ada;
      meta — kolom META_COLS: jam asal, horizon, jangkar, nilai sebenarnya di
             ruang model (NaN bila belum/tidak teramati), dan selisih target.
    `origins` membatasi jam asal (dipakai saat prediksi: hanya jam terakhir).
    """
    base = origin_features(z, target)
    s = z[target]
    a = base["lvl"]
    idx = z.index
    keep = a.notna().values
    if origins is not None:
        keep &= idx.isin(origins)

    local = idx + pd.Timedelta(hours=LOCAL_UTC_OFFSET_HOURS)
    parts_x, parts_m = [], []
    for h in range(1, N_FORECAST_HOURS + 1):
        x = base.copy()
        for name, val in _horizon_features(s, a, h).items():
            x[name] = val
        target_local = local + pd.Timedelta(hours=h)
        x["h"] = h
        x["hour_t"] = target_local.hour
        x["hsin"] = np.sin(2 * np.pi * target_local.hour / 24)
        x["hcos"] = np.cos(2 * np.pi * target_local.hour / 24)
        x["hour_o"] = local.hour
        x["dow"] = target_local.dayofweek
        z_true = s.shift(-h)
        m = pd.DataFrame({"t": idx, "h": h, "anchor": a.values,
                          "z_true": z_true.values, "delta": (z_true - a).values})
        parts_x.append(x[keep])
        parts_m.append(m[keep])

    X = pd.concat(parts_x, axis=0, ignore_index=True)
    meta = pd.concat(parts_m, axis=0, ignore_index=True)
    return X, meta
