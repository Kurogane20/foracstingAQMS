import numpy as np
import pandas as pd
import pytest
from app.config import N_FORECAST_HOURS, FEATURE_COLS
from app.models.lgbm import build_lgbm_features, train_lgbm, predict_lgbm

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


def test_predict_lgbm_returns_dict_with_three_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    N = 30
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]
    train_lgbm(bilstm_preds, y_true, base_times, "uid_test")

    bilstm_out = np.random.randn(N_FORECAST_HOURS, N_FEATURES)
    result = predict_lgbm(bilstm_out, pd.Timestamp("2024-06-15 14:00:00"), "uid_test")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"point", "lower", "upper"}
    assert result["point"].shape == (N_FORECAST_HOURS, N_FEATURES)
    assert result["lower"].shape == (N_FORECAST_HOURS, N_FEATURES)
    assert result["upper"].shape == (N_FORECAST_HOURS, N_FEATURES)


def test_train_lgbm_creates_three_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    N = 20
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]
    train_lgbm(bilstm_preds, y_true, base_times, "uid_test")

    uid_dir = tmp_path / "uid_test"
    assert (uid_dir / "lgbm.pkl").exists()
    assert (uid_dir / "lgbm_lower.pkl").exists()
    assert (uid_dir / "lgbm_upper.pkl").exists()


def test_train_lgbm_accepts_custom_params(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    N = 20
    bilstm_preds = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    y_true = np.random.randn(N, N_FORECAST_HOURS, N_FEATURES)
    base_times = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(N)]
    # Should complete without error with custom params
    train_lgbm(bilstm_preds, y_true, base_times, "uid_custom",
               lgbm_params={"n_estimators": 10, "learning_rate": 0.1})
    assert (tmp_path / "uid_custom" / "lgbm.pkl").exists()


def test_predict_lgbm_returns_none_for_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.lgbm.MODELS_DIR", str(tmp_path))
    bilstm_out = np.random.randn(N_FORECAST_HOURS, N_FEATURES)
    result = predict_lgbm(bilstm_out, pd.Timestamp("2024-06-15 14:00:00"), "nonexistent_uid")
    assert result["point"] is None
    assert result["lower"] is None
    assert result["upper"] is None
