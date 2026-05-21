import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from app.config import N_FORECAST_HOURS, FEATURE_COLS


def _make_mock_X():
    return np.random.randn(1, 24, len(FEATURE_COLS))


def _make_mock_timestamps():
    return pd.date_range("2024-01-01", periods=24, freq="h")


def test_run_prediction_returns_6_steps():
    from app.models.pipeline import run_prediction
    with patch("app.models.pipeline.preprocess_for_predict") as mock_pre, \
         patch("app.models.pipeline.predict_bilstm") as mock_bilstm, \
         patch("app.models.pipeline.predict_lgbm") as mock_lgbm, \
         patch("app.models.pipeline.denormalize") as mock_denorm:
        mock_pre.return_value = (_make_mock_X(), _make_mock_timestamps())
        mock_bilstm.return_value = np.random.randn(1, N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_lgbm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_denorm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        result = run_prediction("test_uid")
    assert len(result) == N_FORECAST_HOURS
    assert result[0]["step"] == 1
    assert result[N_FORECAST_HOURS - 1]["step"] == N_FORECAST_HOURS
    assert "pm_25" in result[0]
    assert "aqi_index" in result[0]


def test_run_prediction_step_numbers_are_sequential():
    from app.models.pipeline import run_prediction
    with patch("app.models.pipeline.preprocess_for_predict") as mock_pre, \
         patch("app.models.pipeline.predict_bilstm") as mock_bilstm, \
         patch("app.models.pipeline.predict_lgbm") as mock_lgbm, \
         patch("app.models.pipeline.denormalize") as mock_denorm:
        mock_pre.return_value = (_make_mock_X(), _make_mock_timestamps())
        mock_bilstm.return_value = np.random.randn(1, N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_lgbm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        mock_denorm.return_value = np.random.randn(N_FORECAST_HOURS, len(FEATURE_COLS))
        result = run_prediction("test_uid")
    steps = [r["step"] for r in result]
    assert steps == list(range(1, N_FORECAST_HOURS + 1))
