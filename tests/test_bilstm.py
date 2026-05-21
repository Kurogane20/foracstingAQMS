import numpy as np
import pytest
from app.config import N_INPUT_HOURS, N_FORECAST_HOURS, FEATURE_COLS, SSA_WINDOW
from app.models.bilstm import build_bilstm
from app.models.ssa import apply_ssa_to_dataframe

N_SSA_FEATURES = len(FEATURE_COLS) * 2


def test_bilstm_output_shape():
    model = build_bilstm()
    X_dummy = np.random.randn(2, N_INPUT_HOURS, N_SSA_FEATURES)
    output = model.predict(X_dummy, verbose=0)
    assert output.shape == (2, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_bilstm_model_has_correct_input_shape():
    model = build_bilstm()
    assert model.input_shape == (None, N_INPUT_HOURS, N_SSA_FEATURES)
