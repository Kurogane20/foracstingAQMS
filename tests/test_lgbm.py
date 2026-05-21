import numpy as np
import pandas as pd
import pytest
from app.config import N_FORECAST_HOURS, FEATURE_COLS
from app.models.lgbm import build_lgbm_features

N_FEATURES = len(FEATURE_COLS)


def test_build_lgbm_features_output_shape():
    bilstm_out = np.random.randn(N_FORECAST_HOURS, N_FEATURES)
    base_time = pd.Timestamp("2024-06-15 14:00:00")
    result = build_lgbm_features(bilstm_out, base_time)
    assert result.shape == (N_FORECAST_HOURS, N_FEATURES + 3)


def test_build_lgbm_features_time_values_are_correct():
    bilstm_out = np.zeros((N_FORECAST_HOURS, N_FEATURES))
    base_time = pd.Timestamp("2024-06-15 14:00:00")
    result = build_lgbm_features(bilstm_out, base_time)
    # step 0 → t+1 = 15:00
    assert result[0, N_FEATURES] == 15     # hour
    assert result[0, N_FEATURES + 1] == 5  # Saturday = dayofweek 5
    assert result[0, N_FEATURES + 2] == 6  # June = month 6
