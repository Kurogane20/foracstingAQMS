import numpy as np
import pytest

from app.config import INTERVAL_COVERAGE, N_FORECAST_HOURS
from app.models.calibration import (
    ALPHA_GRID, calibrated_deltas, fit_alpha, fit_conformal, recenter_band,
)

H = N_FORECAST_HOURS


def _h(n_per_step):
    return np.repeat(np.arange(1, H + 1), n_per_step)


# ── fit_alpha ────────────────────────────────────────────────────────────────

def test_alpha_is_one_when_model_is_perfect():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 1, 600)
    assert np.allclose(fit_alpha(true.copy(), true, _h(100)), 1.0)


def test_alpha_is_zero_when_model_is_pure_noise():
    """Sensor yang selisihnya hanya derau harus jatuh ke persistence."""
    rng = np.random.default_rng(1)
    true = rng.normal(0, 0.1, 6000)
    pred = rng.normal(0, 1.0, 6000)           # tak berkorelasi, jauh lebih besar
    assert np.all(fit_alpha(pred, true, _h(1000)) <= 0.05)


def test_alpha_shrinks_overconfident_model():
    rng = np.random.default_rng(2)
    true = rng.normal(0, 1, 6000)
    alpha = fit_alpha(2.0 * true, true, _h(1000))
    assert np.allclose(alpha, 0.5)


def test_alpha_never_worse_than_persistence_on_calibration_data():
    rng = np.random.default_rng(3)
    true = rng.normal(0, 1, 1200)
    pred = 0.3 * true + rng.normal(0, 1.5, 1200)
    h = _h(200)
    alpha = fit_alpha(pred, true, h)
    for step in range(1, H + 1):
        m = h == step
        assert np.abs(alpha[step - 1] * pred[m] - true[m]).mean() <= np.abs(true[m]).mean() + 1e-12


def test_alpha_is_fitted_per_horizon():
    rng = np.random.default_rng(4)
    true = rng.normal(0, 1, 1200)
    h = _h(200)
    pred = np.where(h == 1, true, rng.normal(0, 3, 1200))     # hanya H+1 yang bagus
    alpha = fit_alpha(pred, true, h)
    assert alpha[0] == 1.0 and np.all(alpha[1:] <= 0.1)


def test_alpha_defaults_to_one_without_evidence():
    alpha = fit_alpha(np.array([5.0, -5.0]), np.array([0.0, 0.0]), np.array([1, 1]))
    assert np.allclose(alpha, 1.0)


def test_alpha_stays_within_grid():
    rng = np.random.default_rng(5)
    alpha = fit_alpha(rng.normal(0, 1, 600) * 0.1, rng.normal(0, 1, 600), _h(100))
    assert alpha.min() >= ALPHA_GRID.min() and alpha.max() <= ALPHA_GRID.max()


# ── fit_conformal ────────────────────────────────────────────────────────────

def test_conformal_widens_too_narrow_band_to_nominal_coverage():
    rng = np.random.default_rng(6)
    n = 6000
    true = rng.normal(0, 1, n)
    lower, upper = np.full(n, -0.5), np.full(n, 0.5)          # mencakup hanya ±38%
    h = _h(n // H)
    q = fit_conformal(lower, upper, true, h)
    assert np.all(q > 0)
    covered = (true >= lower - q[h - 1]) & (true <= upper + q[h - 1])
    assert covered.mean() == pytest.approx(INTERVAL_COVERAGE, abs=0.01)


def test_conformal_narrows_too_wide_band():
    rng = np.random.default_rng(7)
    n = 6000
    true = rng.normal(0, 1, n)
    h = _h(n // H)
    q = fit_conformal(np.full(n, -10.0), np.full(n, 10.0), true, h)
    assert np.all(q < 0)
    covered = (true >= -10.0 - q[h - 1]) & (true <= 10.0 + q[h - 1])
    assert covered.mean() == pytest.approx(INTERVAL_COVERAGE, abs=0.01)


def test_conformal_is_zero_without_enough_samples():
    q = fit_conformal(np.zeros(12), np.ones(12), np.full(12, 5.0), _h(2))
    assert np.allclose(q, 0.0)


# ── calibrated_deltas ────────────────────────────────────────────────────────

def test_recentering_preserves_band_width():
    point, lo, hi = np.array([1.0]), np.array([0.2]), np.array([2.5])
    shrunk = 0.25 * point
    assert (recenter_band(shrunk, point, hi) - recenter_band(shrunk, point, lo))[0] == pytest.approx(2.3)


def test_calibrated_deltas_apply_per_horizon_values():
    h = np.array([1, 6])
    alpha = np.linspace(1.0, 0.0, H)           # H+1: α=1, H+6: α=0
    conformal = np.linspace(0.0, 0.5, H)
    p, lo, hi = calibrated_deltas(np.array([2.0, 2.0]), np.array([1.0, 1.0]),
                                  np.array([4.0, 4.0]), h, alpha, conformal)
    assert p.tolist() == [2.0, 0.0]
    assert lo.tolist() == pytest.approx([1.0, -1.5])     # digeser −2, dilebarkan 0,5
    assert hi.tolist() == pytest.approx([4.0, 2.5])


def test_calibrated_band_always_contains_point():
    """Kuantil dari model terpisah bisa menyilang; keluaran tidak boleh terbalik."""
    h = np.array([1, 1, 1])
    one, zero = np.ones(H), np.zeros(H)
    p, lo, hi = calibrated_deltas(np.array([0.0, 1.0, -1.0]), np.array([0.5, 1.2, -3.0]),
                                  np.array([-0.5, 0.8, -2.0]), h, one, zero)
    assert np.all(lo <= p) and np.all(p <= hi)
