import numpy as np
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, SSA_WINDOW
from app.models.ssa import ssa_decompose, apply_ssa_to_dataframe


def test_ssa_decompose_returns_two_arrays():
    series = np.sin(np.linspace(0, 4 * np.pi, N_INPUT_HOURS)) + np.random.randn(N_INPUT_HOURS) * 0.1
    trend, oscillation = ssa_decompose(series, SSA_WINDOW)
    assert trend.shape == (N_INPUT_HOURS,)
    assert oscillation.shape == (N_INPUT_HOURS,)


def test_ssa_decompose_trend_plus_oscillation_approximates_series():
    series = np.sin(np.linspace(0, 4 * np.pi, N_INPUT_HOURS))
    trend, oscillation = ssa_decompose(series, SSA_WINDOW)
    np.testing.assert_allclose(trend + oscillation, series, atol=1e-10)


def test_apply_ssa_doubles_feature_count():
    n_features = len(FEATURE_COLS)
    data = np.random.randn(N_INPUT_HOURS, n_features)
    result = apply_ssa_to_dataframe(data, SSA_WINDOW)
    assert result.shape == (N_INPUT_HOURS, n_features * 2)
