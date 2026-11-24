import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, PHYSICAL_LIMITS

# Model kecil agar tes cepat; perilaku yang diuji tidak bergantung ukuran model.
FAST_PARAMS = {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 7,
               "min_child_samples": 10, "n_jobs": 1}


def make_hourly(days: int = 40, seed: int = 0, start: str = "2026-01-01") -> pd.DataFrame:
    """Deret per jam sintetis yang menyerupai data nyata.

    TSP punya siklus harian kuat (malam tinggi, siang rendah, jam lokal UTC+8)
    dikalikan derau lognormal berautokorelasi — struktur yang sama dengan data
    Berau, sehingga model yang benar HARUS mengalahkan persistence di sini.
    Semua nilai berada di dalam batas fisik.
    """
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h")
    local_hour = (idx.hour + 8) % 24
    diurnal = 1.0 + 0.8 * np.cos(2 * np.pi * (local_hour - 3) / 24)

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.2)
    tsp = 60.0 * diurnal * np.exp(noise)

    df = pd.DataFrame(index=idx)
    df["tsp"] = tsp
    df["tsp_correction"] = tsp
    df["pm_10"] = 0.6 * tsp * np.exp(rng.normal(0, 0.05, n))
    df["pm_10_correction"] = 0.4 * df["pm_10"]
    df["pm_25"] = 0.3 * tsp * np.exp(rng.normal(0, 0.05, n))
    df["pm_25_correction"] = 0.5 * df["pm_25"]
    df["noise"] = 52.0 + rng.normal(0, 0.5, n)
    df["temp"] = 27.0 - 3.0 * np.cos(2 * np.pi * (local_hour - 14) / 24) + rng.normal(0, 0.3, n)
    df["mmhg"] = 757.0 + rng.normal(0, 0.5, n)
    df["humidity"] = np.clip(85.0 + 10.0 * np.cos(2 * np.pi * (local_hour - 4) / 24)
                             + rng.normal(0, 2, n), 1, 100)
    df["aqi_index_pm25"] = np.clip(df["pm_25"] * 4.0, 0, 500)
    df["aqi_index_pm10"] = np.clip(df["pm_10"] * 0.9, 0, 500)
    df["aqi_index_tsp"] = np.clip(tsp * 0.24, 0, 500)
    df["aqi_index"] = df[["aqi_index_pm25", "aqi_index_pm10", "aqi_index_tsp"]].max(axis=1)
    df = df[FEATURE_COLS].astype("float64")
    for col, (lo, hi, _) in PHYSICAL_LIMITS.items():
        assert df[col].between(lo, hi).all(), col
    return df


@pytest.fixture
def hourly_df():
    return make_hourly()


@pytest.fixture
def fast_params():
    return dict(FAST_PARAMS)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Arahkan semua modul yang menulis artefak ke folder sementara."""
    for mod in ("app.models.forecaster", "app.models.pipeline", "app.models.tuning"):
        monkeypatch.setattr(f"{mod}.MODELS_DIR", str(tmp_path))
    from app.models import forecaster
    monkeypatch.setattr(forecaster, "_cache", {})
    return tmp_path
