import numpy as np
import pytest
from app.config import N_FEATURES, N_TIME_FEATURES, N_METEO_FEATURES, N_INPUT_HOURS
from app.models.bilstm import _prepare_input, N_SSA_FEATURES


class TestPrepareInput:
    def test_output_shape(self):
        total_in = N_FEATURES + N_TIME_FEATURES + N_METEO_FEATURES  # 22
        X = np.zeros((5, N_INPUT_HOURS, total_in))
        result = _prepare_input(X)
        assert result.shape == (5, N_INPUT_HOURS, N_SSA_FEATURES)

    def test_n_ssa_features_value(self):
        # 28 SSA sensor + 4 time + 4 meteo = 36
        assert N_SSA_FEATURES == N_FEATURES * 2 + N_TIME_FEATURES + N_METEO_FEATURES

    def test_single_sample(self):
        total_in = N_FEATURES + N_TIME_FEATURES + N_METEO_FEATURES
        X = np.random.rand(1, N_INPUT_HOURS, total_in)
        result = _prepare_input(X)
        assert result.shape == (1, N_INPUT_HOURS, N_SSA_FEATURES)
        assert not np.isnan(result).any()
