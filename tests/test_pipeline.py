import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURE_COLS, MAX_ANCHOR_AGE_HOURS, N_FORECAST_HOURS, PHYSICAL_LIMITS
from app.models import pipeline
from tests.conftest import FAST_PARAMS, make_hourly

UID = "AQI_TEST"


@pytest.fixture(scope="module")
def fitted():
    """Satu pelatihan penuh pada data sintetis; sensor ini tanpa modul cuaca
    (suhu/tekanan/kelembapan kosong total), seperti sebagian sensor nyata."""
    df = make_hourly(days=40)
    df[["temp", "mmhg", "humidity"]] = np.nan
    with patch("app.models.backtest.BACKTEST_FOLDS", 3):
        bundle, report, summary = pipeline.fit_sensor(df, FAST_PARAMS)
    return df, bundle, report, summary


# ── fit_sensor ───────────────────────────────────────────────────────────────

def test_model_beats_persistence_on_series_with_daily_cycle(fitted):
    """Deret sintetis punya siklus harian yang bisa dipelajari. Model yang benar
    HARUS mengalahkan persistence di luar sampel; kalau tidak, ada yang rusak."""
    _, _, report, summary = fitted
    assert summary["skill_vs_naive"] > 0.10
    tsp = report["targets"]["tsp"]
    assert tsp["overall"]["mae"] < tsp["overall"]["mae_persistence"]
    # Keunggulan terbesar di horizon jauh, tempat siklus harian paling berperan.
    assert tsp["by_horizon"]["6"]["skill_vs_persistence"] > tsp["by_horizon"]["1"]["skill_vs_persistence"]


def test_parameters_without_data_are_skipped_not_fabricated(fitted):
    _, bundle, report, summary = fitted
    assert set(summary["skipped_targets"]) == {"temp", "mmhg", "humidity"}
    assert set(bundle["targets"]) == set(FEATURE_COLS) - {"temp", "mmhg", "humidity"}
    assert report["skipped_targets"] == summary["skipped_targets"]
    assert summary["trained_targets"] == list(bundle["targets"])


def test_bundle_holds_models_and_calibration_per_target(fitted):
    _, bundle, _, _ = fitted
    spec = bundle["targets"]["tsp"]
    assert set(spec["models"]) == {"point", "lower", "upper"}
    assert spec["alpha"].shape == (N_FORECAST_HOURS,) and spec["conformal"].shape == (N_FORECAST_HOURS,)
    assert ((spec["alpha"] >= 0) & (spec["alpha"] <= 1)).all()
    assert "sn24" in spec["features"] and "h" in spec["features"]
    assert bundle["trained_until"] == fitted[0].index[-1].isoformat()


def test_report_structure(fitted):
    df, _, report, summary = fitted
    assert report["data"]["hours"] == len(df)
    assert report["data"]["hours_observed"]["temp"] == 0
    assert len(report["folds"]) == summary["folds"] == 3
    tsp = report["targets"]["tsp"]
    assert tsp["folds_scored"] == 2                      # lipatan pertama hanya kalibrasi
    assert set(tsp["by_horizon"]) == {str(h) for h in range(1, N_FORECAST_HOURS + 1)}
    assert "exceedance" in tsp and "exceedance" not in report["targets"]["noise"]
    json.dumps(report)                                   # harus bisa diserialisasi


def test_summary_keeps_dashboard_score_dimensionless(fitted):
    """`mae_score` ditampilkan dasbor sebagai (1 − mae)·100 %; harus tetap di
    kisaran 0–1, sedangkan angka bermakna (µg/m³) ada di kunci tersendiri."""
    _, _, report, summary = fitted
    assert 0 < summary["mae_score"] < 0.5
    assert summary["mae_tsp"] == report["targets"]["tsp"]["overall"]["mae"]
    assert summary["mae_tsp"] > 1.0                       # µg/m³, bukan pecahan
    assert summary["training_samples"] == int(fitted[0]["tsp"].notna().sum())
    assert 0.5 < summary["coverage_tsp"] <= 1.0


def test_fit_sensor_reports_progress(hourly_df):
    seen = []
    with patch("app.models.backtest.BACKTEST_FOLDS", 2), \
         patch("app.models.pipeline.backtest_target", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="Insufficient data"):
            pipeline.fit_sensor(hourly_df, FAST_PARAMS, on_progress=seen.append)
    assert len(seen) == len(FEATURE_COLS) and seen[0] == 0.0 and max(seen) < 1.0


def test_fit_sensor_rejects_too_short_history():
    with pytest.raises(ValueError, match="Data belum cukup"):
        pipeline.fit_sensor(make_hourly(days=10), FAST_PARAMS)


# ── run_training ─────────────────────────────────────────────────────────────

def test_run_training_saves_bundle_and_report(fitted, models_dir):
    df, bundle, report, summary = fitted
    with patch("app.models.pipeline.fit_sensor", return_value=(bundle, report, summary)) as fit, \
         patch("app.models.pipeline.set_phase"):
        result = pipeline.run_training(UID, lgbm_params={"num_leaves": 7}, hourly=df)
    assert result == summary
    assert fit.call_args[0][0] is df and fit.call_args[0][1] == {"num_leaves": 7}
    saved = pipeline.load_report(UID)
    assert saved["uid"] == UID and saved["targets"]["tsp"] == report["targets"]["tsp"]
    from app.models.forecaster import load_bundle
    assert set(load_bundle(UID)["targets"]) == set(bundle["targets"])


def test_run_training_loads_from_db_when_no_frame_given(fitted, models_dir):
    df, bundle, report, summary = fitted
    with patch("app.models.pipeline.load_for_training", return_value=df) as load, \
         patch("app.models.pipeline.fit_sensor", return_value=(bundle, report, summary)), \
         patch("app.models.pipeline.set_phase"):
        pipeline.run_training(UID)
    load.assert_called_once_with(UID)


def test_load_report_missing_returns_none(models_dir):
    assert pipeline.load_report("tidak_ada") is None


# ── run_prediction ───────────────────────────────────────────────────────────

@pytest.fixture
def trained(fitted, models_dir):
    from app.models.forecaster import save_bundle
    save_bundle(UID, fitted[1])
    return fitted[0]


def _predict(df):
    with patch("app.models.pipeline.load_for_predict", return_value=df):
        return pipeline.run_prediction(UID)


def test_prediction_has_six_sequential_steps_from_last_full_hour(trained):
    recent = trained.iloc[-24 * 9:]
    result = _predict(recent)
    assert [r["step"] for r in result] == list(range(1, N_FORECAST_HOURS + 1))
    last = recent.index[-1]
    for r in result:
        assert r["target_time"] == (last + pd.Timedelta(hours=r["step"])).isoformat()
        assert set(FEATURE_COLS) <= set(r)


def test_prediction_values_are_physical_and_band_is_ordered(trained):
    result = _predict(trained.iloc[-24 * 9:])
    for r in result:
        for col in set(FEATURE_COLS) - {"temp", "mmhg", "humidity"}:
            lo, hi, _ = PHYSICAL_LIMITS[col]
            assert isinstance(r[col], float) and lo <= r[col] <= hi
            assert r["lower_bounds"][col] <= r[col] <= r["upper_bounds"][col]
            assert r["lower_bounds"][col] >= lo


def test_prediction_is_close_to_recent_level(trained):
    """Keluaran harus kembali ke satuan fisik: TSP sekitar puluhan–ratusan µg/m³,
    bukan nilai ruang-log (~4) dan bukan angka liar."""
    recent = trained.iloc[-24 * 9:]
    result = _predict(recent)
    last = recent["tsp"].iloc[-1]
    assert 0.3 * last < result[0]["tsp"] < 3.0 * last


def test_untrained_parameters_are_none_everywhere(trained):
    for r in _predict(trained.iloc[-24 * 9:]):
        for col in ("temp", "mmhg", "humidity"):
            assert r[col] is None
            assert r["lower_bounds"][col] is None and r["upper_bounds"][col] is None


def test_stale_parameter_is_none_while_others_are_forecast(trained):
    recent = trained.iloc[-24 * 9:].copy()
    recent.iloc[-(MAX_ANCHOR_AGE_HOURS + 1):, recent.columns.get_loc("pm_25")] = np.nan
    result = _predict(recent)
    assert all(r["pm_25"] is None for r in result)
    assert all(isinstance(r["tsp"], float) for r in result)


def test_short_gap_still_forecasts_from_last_reading(trained):
    recent = trained.iloc[-24 * 9:].copy()
    recent.iloc[-MAX_ANCHOR_AGE_HOURS:, recent.columns.get_loc("tsp")] = np.nan
    assert all(isinstance(r["tsp"], float) for r in _predict(recent))


def test_offline_sensor_raises_instead_of_forecasting_from_stale_data(trained):
    """Regresi: versi lama meramal dari 24 baris terakhir berapa pun usianya,
    lalu melabelinya dengan jam yang sudah lewat."""
    recent = trained.iloc[-24 * 9:].copy()
    recent.iloc[-12:] = np.nan
    with pytest.raises(ValueError, match="basi"):
        _predict(recent)


def test_prediction_without_model_raises_file_not_found(models_dir):
    with patch("app.models.pipeline.load_for_predict", return_value=make_hourly(days=9)):
        with pytest.raises(FileNotFoundError):
            pipeline.run_prediction("belum_dilatih")


def test_prediction_with_no_data_raises(trained):
    with pytest.raises(ValueError, match="Insufficient"):
        _predict(pd.DataFrame(columns=FEATURE_COLS, index=pd.DatetimeIndex([])))


def test_prediction_tolerates_context_column_missing_at_predict_time(trained):
    """Kebisingan ada saat latih tapi kosong di jendela prediksi: fitur konteksnya
    masuk sebagai NaN, prakiraan TSP tetap dibuat."""
    recent = trained.iloc[-24 * 9:].copy()
    recent["noise"] = np.nan
    result = _predict(recent)
    assert all(isinstance(r["tsp"], float) for r in result)
    assert all(r["noise"] is None for r in result)
