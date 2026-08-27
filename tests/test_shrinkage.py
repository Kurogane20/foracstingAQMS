import json

import numpy as np
import pytest

from app.config import N_FORECAST_HOURS, N_FEATURES
from app.models.shrinkage import (
    ALPHA_GRID, fit_alpha, save_alpha, load_alpha, apply_alpha, recenter_band,
)


def test_alpha_is_zero_when_prediction_is_pure_noise():
    """Sensor seperti AQI02L: perubahan antar jam praktis derau, model kalah
    −85,7% dari persistence. Bobotnya harus runtuh ke nol."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(400, N_FORECAST_HOURS, N_FEATURES))
    true = rng.normal(size=(400, N_FORECAST_HOURS, N_FEATURES)) * 0.05
    alpha = fit_alpha(pred, true)
    assert alpha.mean() < 0.15


def test_alpha_is_high_when_prediction_is_accurate():
    """Sensor seperti AQI01G (+24,8%): model benar-benar informatif, jangan
    disusutkan tanpa alasan."""
    rng = np.random.default_rng(1)
    true = rng.normal(size=(400, N_FORECAST_HOURS, N_FEATURES))
    pred = true + rng.normal(size=true.shape) * 0.05
    alpha = fit_alpha(pred, true)
    assert alpha.mean() > 0.85


def test_shrunk_error_never_worse_than_persistence():
    """Sifat inti: karena α=0 ada di dalam rentang pencarian dan menghasilkan
    persistence persis, MAE hasil penyusutan tidak boleh melebihi MAE naif pada
    data yang sama — apa pun keluaran modelnya."""
    rng = np.random.default_rng(2)
    for scale in (0.01, 1.0, 50.0):          # model buruk, sedang, ngawur
        true = rng.normal(size=(300, N_FORECAST_HOURS, N_FEATURES))
        pred = rng.normal(size=true.shape) * scale
        alpha = fit_alpha(pred, true)
        mae_shrunk = np.abs(apply_alpha(pred, alpha) - true).mean()
        mae_naive = np.abs(true).mean()      # persistence = selisih nol
        assert mae_shrunk <= mae_naive + 1e-12


def test_alpha_stays_within_grid_bounds():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(50, N_FORECAST_HOURS, N_FEATURES)) * 10
    true = rng.normal(size=pred.shape)
    alpha = fit_alpha(pred, true)
    assert alpha.min() >= ALPHA_GRID.min()
    assert alpha.max() <= ALPHA_GRID.max()


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.shrinkage.MODELS_DIR", str(tmp_path))
    alpha = np.random.default_rng(4).random((N_FORECAST_HOURS, N_FEATURES)).round(2)
    save_alpha("uid_x", alpha)
    np.testing.assert_allclose(load_alpha("uid_x"), alpha)


def test_saved_file_is_plain_json(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.shrinkage.MODELS_DIR", str(tmp_path))
    save_alpha("uid_x", np.zeros((N_FORECAST_HOURS, N_FEATURES)))
    with open(tmp_path / "uid_x" / "shrinkage.json", encoding="utf-8") as fh:
        blob = json.load(fh)
    assert "alpha" in blob and "schema" in blob


def test_missing_alpha_falls_back_to_no_shrinkage(tmp_path, monkeypatch):
    """Model lama tanpa berkas bobot harus tetap bisa memprediksi, bukan gagal."""
    monkeypatch.setattr("app.models.shrinkage.MODELS_DIR", str(tmp_path))
    alpha = load_alpha("uid_never_trained")
    assert alpha.shape == (N_FORECAST_HOURS, N_FEATURES)
    assert np.all(alpha == 1.0)


def test_recenter_band_preserves_width():
    """Menyusutkan pita ikut-ikutan akan melemahkan ambang peringatan p90 —
    yang berubah seharusnya pusatnya, bukan lebarnya."""
    rng = np.random.default_rng(5)
    point_raw = rng.normal(size=(N_FORECAST_HOURS, N_FEATURES))
    upper_raw = point_raw + np.abs(rng.normal(size=point_raw.shape)) + 0.1
    point_shrunk = point_raw * 0.3

    upper_new = recenter_band(point_shrunk, point_raw, upper_raw)

    np.testing.assert_allclose(upper_new - point_shrunk, upper_raw - point_raw)
    assert np.all(upper_new > point_shrunk)


def test_apply_alpha_handles_batched_and_single():
    alpha = np.full((N_FORECAST_HOURS, N_FEATURES), 0.5)
    single = np.ones((N_FORECAST_HOURS, N_FEATURES))
    batched = np.ones((4, N_FORECAST_HOURS, N_FEATURES))
    assert apply_alpha(single, alpha).shape == single.shape
    assert apply_alpha(batched, alpha).shape == batched.shape
    assert apply_alpha(single, alpha)[0, 0] == pytest.approx(0.5)
