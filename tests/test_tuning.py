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

        N = 200
        X = np.random.randn(N, 24, 18)
        y = np.random.randn(N, 6, 14)

        result = tune_bilstm(X, y, "uid_bilstm")

    assert isinstance(result, dict)
    assert "units" in result
    assert "dropout" in result
    assert "batch_size" in result


def test_tune_bilstm_scores_only_unseen_purged_sequences(tmp_path, monkeypatch):
    """Sekuens yang dinilai tidak boleh ikut dilatih, dan targetnya tidak boleh
    tumpang-tindih dengan target sekuens latih terakhir (celah purge)."""
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("app.models.tuning.BILSTM_TRIALS", 1)
    from app.models.tuning import tune_bilstm, tuning_split

    N = 200
    X = np.zeros((N, 24, 18))
    X[:, 0, 0] = np.arange(N)            # penanda indeks sekuens
    y = np.zeros((N, 6, 14))
    trained, scored = [], []

    with patch("app.models.tuning.train_bilstm",
               side_effect=lambda Xt, yt, uid, **kw: trained.extend(Xt[:, 0, 0])), \
         patch("app.models.tuning.predict_bilstm",
               side_effect=lambda Xv, uid: scored.append(Xv[0, 0, 0]) or np.zeros((1, 6, 14))):
        tune_bilstm(X, y, "uid_bilstm")

    fit_end, eval_start = tuning_split(N)
    assert max(trained) == fit_end - 1
    assert min(scored) == eval_start
    assert eval_start - fit_end >= 6     # N_FORECAST_HOURS


def test_tuning_split_rejects_too_little_data():
    from app.models.tuning import tuning_split
    with pytest.raises(ValueError):
        tuning_split(20)


# ---------------------------------------------------------------------------
# Tests for tune_lgbm
# ---------------------------------------------------------------------------

def _lgbm_inputs(N):
    from app.config import N_FORECAST_HOURS, N_FEATURES
    rng = np.random.default_rng(0)
    bilstm_preds = rng.normal(size=(N, N_FORECAST_HOURS, N_FEATURES))
    y_true = rng.normal(size=(N, N_FORECAST_HOURS, N_FEATURES))
    base_times = pd.date_range("2024-01-01", periods=N, freq="h")
    anchors = rng.normal(size=(N, N_FEATURES))
    return bilstm_preds, y_true, base_times, anchors


def test_tune_lgbm_returns_correct_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.tuning.MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("app.models.tuning.LGBM_TRIALS", 1)
    from app.models.tuning import tune_lgbm

    bilstm_preds, y_true, base_times, anchors = _lgbm_inputs(150)
    result = tune_lgbm(bilstm_preds, y_true, base_times, "uid_lgbm", anchors=anchors)

    assert isinstance(result, dict)
    assert "n_estimators" in result
    assert "learning_rate" in result
    assert "num_leaves" in result
    # Kandidat dilatih di memori — tidak ada lgbm*.pkl produksi yang tertimpa.
    assert list(tmp_path.iterdir()) == []


def test_tune_lgbm_scores_on_rows_it_never_fit(monkeypatch):
    """Regresi: dulu objektif menilai model pada data latihnya sendiri, sehingga
    Optuna selalu memilih konfigurasi paling overfit."""
    monkeypatch.setattr("app.models.tuning.LGBM_TRIALS", 1)
    from app.models.tuning import tune_lgbm, tuning_split
    from app.config import N_FORECAST_HOURS, N_FEATURES

    seen = {}

    class SpyRegressor:
        def __init__(self, estimator):
            pass

        def fit(self, X, y):
            seen["fit"] = X.copy()
            return self

        def predict(self, X):
            seen["predict"] = X.copy()
            return np.zeros((len(X), N_FEATURES))

    N = 150
    bilstm_preds, y_true, base_times, anchors = _lgbm_inputs(N)
    with patch("app.models.tuning.MultiOutputRegressor", SpyRegressor):
        tune_lgbm(bilstm_preds, y_true, base_times, "uid_lgbm", anchors=anchors)

    fit_end, eval_start = tuning_split(N)
    assert len(seen["fit"]) == fit_end * N_FORECAST_HOURS
    assert len(seen["predict"]) == (N - eval_start) * N_FORECAST_HOURS
    fit_rows = {r.tobytes() for r in seen["fit"]}
    assert not any(r.tobytes() in fit_rows for r in seen["predict"])
