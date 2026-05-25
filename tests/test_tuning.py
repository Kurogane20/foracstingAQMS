import json
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Tests for save_best_params / load_best_params
# ---------------------------------------------------------------------------

def test_save_best_params_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    from app.models.tuning import save_best_params

    bilstm = {"units": 128, "dropout": 0.2, "batch_size": 32}
    lgbm = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63}
    save_best_params("uid_test", bilstm, lgbm)

    expected_path = tmp_path / "uid_test" / "best_params.json"
    assert expected_path.exists()
    with open(expected_path) as f:
        data = json.load(f)
    assert data["bilstm"] == bilstm
    assert data["lgbm"] == lgbm


def test_load_best_params_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    from app.models.tuning import load_best_params

    result = load_best_params("nonexistent_uid")
    assert result is None


def test_load_best_params_returns_dict_when_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    from app.models.tuning import save_best_params, load_best_params

    bilstm = {"units": 64, "dropout": 0.3, "batch_size": 16}
    lgbm = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31}
    save_best_params("uid_abc", bilstm, lgbm)

    result = load_best_params("uid_abc")
    assert result is not None
    assert result["bilstm"] == bilstm
    assert result["lgbm"] == lgbm


# ---------------------------------------------------------------------------
# Tests for tune_bilstm
# ---------------------------------------------------------------------------

def test_tune_bilstm_returns_correct_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))

    with patch("app.models.tuning.train_bilstm") as mock_train, \
         patch("app.models.tuning.predict_bilstm") as mock_predict:

        # predict_bilstm returns array of shape (1, N_FORECAST_HOURS, N_FEATURES)
        # The objective does: preds = np.array([predict_bilstm(X_val[i:i+1], uid)[0] for i in range(len(X_val))])
        # predict_bilstm(...)[0] picks element 0 from the returned array
        mock_predict.return_value = np.zeros((1, 6, 14))

        from app.models.tuning import tune_bilstm, BILSTM_TRIALS

        N = 20
        X = np.random.randn(N, 24, 18)
        y = np.random.randn(N, 6, 14)

        result = tune_bilstm(X, y, "uid_bilstm")

    assert isinstance(result, dict)
    assert "units" in result
    assert "dropout" in result
    assert "batch_size" in result


# ---------------------------------------------------------------------------
# Tests for tune_lgbm
# ---------------------------------------------------------------------------

def test_tune_lgbm_returns_correct_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))

    N = 10
    from app.config import N_FORECAST_HOURS, N_FEATURES
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]

    mock_model = MagicMock()
    # predict returns shape (N_FORECAST_HOURS, N_FEATURES); [0] picks first row
    mock_model.predict.return_value = np.zeros((N_FORECAST_HOURS, N_FEATURES))

    with patch("app.models.tuning.train_lgbm") as mock_train, \
         patch("joblib.load", return_value=mock_model):

        from app.models.tuning import tune_lgbm, LGBM_TRIALS

        result = tune_lgbm(bilstm_preds, y_true, base_times, "uid_lgbm")

    assert isinstance(result, dict)
    assert "n_estimators" in result
    assert "learning_rate" in result
    assert "num_leaves" in result
