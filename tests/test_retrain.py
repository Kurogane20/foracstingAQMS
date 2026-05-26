import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


UID = "sensor_001"

_MOCK_RESULT = {"training_samples": 100, "mae_score": 0.123456}

_BILSTM_PARAMS = {"units": 128, "dropout": 0.2, "batch_size": 32}
_LGBM_PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63}
_BEST_PARAMS = {"bilstm": _BILSTM_PARAMS, "lgbm": _LGBM_PARAMS}

_MOCK_X = np.random.randn(20, 24, 14)
_MOCK_Y = np.random.randn(20, 6, 14)


def _make_bilstm_pred():
    return np.random.randn(1, 6, 14)


# ---------------------------------------------------------------------------
# Test 1: no best_params, run_tuning=False (default) → trains with defaults
# ---------------------------------------------------------------------------

def test_retrain_sensor_no_params_default_behavior():
    """retrain_sensor(uid) with no best_params trains with default (None) params."""
    with patch("app.services.retrain.load_best_params", return_value=None) as mock_load, \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT) as mock_train, \
         patch("app.services.retrain.upsert_metadata") as mock_upsert, \
         patch("app.services.retrain.tune_bilstm") as mock_tune_bilstm, \
         patch("app.services.retrain.tune_lgbm") as mock_tune_lgbm, \
         patch("app.services.retrain.save_best_params") as mock_save:
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID)

    mock_load.assert_called_once_with(UID)
    mock_train.assert_called_once_with(UID, bilstm_params=None, lgbm_params=None)
    mock_tune_bilstm.assert_not_called()
    mock_tune_lgbm.assert_not_called()
    mock_save.assert_not_called()
    assert result["status"] == "ready"
    assert result["uid"] == UID


def test_retrain_sensor_run_tuning_false_no_params_uses_defaults():
    """retrain_sensor(uid, run_tuning=False) with no best_params → defaults, no tuning."""
    with patch("app.services.retrain.load_best_params", return_value=None) as mock_load, \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT) as mock_train, \
         patch("app.services.retrain.upsert_metadata"), \
         patch("app.services.retrain.tune_bilstm") as mock_tune_bilstm, \
         patch("app.services.retrain.tune_lgbm") as mock_tune_lgbm, \
         patch("app.services.retrain.save_best_params") as mock_save:
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID, run_tuning=False)

    mock_train.assert_called_once_with(UID, bilstm_params=None, lgbm_params=None)
    mock_tune_bilstm.assert_not_called()
    mock_tune_lgbm.assert_not_called()
    mock_save.assert_not_called()
    assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# Test 2: best_params present, run_tuning=False → uses saved params, no tuning
# ---------------------------------------------------------------------------

def test_retrain_sensor_uses_saved_params_when_present():
    """When best_params.json exists, use saved params and skip tuning."""
    with patch("app.services.retrain.load_best_params", return_value=_BEST_PARAMS) as mock_load, \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT) as mock_train, \
         patch("app.services.retrain.upsert_metadata"), \
         patch("app.services.retrain.tune_bilstm") as mock_tune_bilstm, \
         patch("app.services.retrain.tune_lgbm") as mock_tune_lgbm, \
         patch("app.services.retrain.save_best_params") as mock_save:
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID, run_tuning=False)

    mock_load.assert_called_once_with(UID)
    mock_train.assert_called_once_with(UID, bilstm_params=_BILSTM_PARAMS, lgbm_params=_LGBM_PARAMS)
    mock_tune_bilstm.assert_not_called()
    mock_tune_lgbm.assert_not_called()
    mock_save.assert_not_called()
    assert result["status"] == "ready"


def test_retrain_sensor_uses_saved_params_when_present_run_tuning_true():
    """When best_params.json exists, saved params are used even if run_tuning=True."""
    with patch("app.services.retrain.load_best_params", return_value=_BEST_PARAMS), \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT) as mock_train, \
         patch("app.services.retrain.upsert_metadata"), \
         patch("app.services.retrain.tune_bilstm") as mock_tune_bilstm, \
         patch("app.services.retrain.tune_lgbm") as mock_tune_lgbm, \
         patch("app.services.retrain.save_best_params") as mock_save:
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID, run_tuning=True)

    mock_train.assert_called_once_with(UID, bilstm_params=_BILSTM_PARAMS, lgbm_params=_LGBM_PARAMS)
    mock_tune_bilstm.assert_not_called()
    mock_tune_lgbm.assert_not_called()
    mock_save.assert_not_called()
    assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# Test 3: no best_params + run_tuning=True → tuning runs, params saved, training uses tuned
# ---------------------------------------------------------------------------

def test_retrain_sensor_runs_tuning_when_no_params_and_flag_set():
    """run_tuning=True with no saved params triggers tuning and saves results."""
    bilstm_pred = np.random.randn(6, 14)

    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.preprocess_for_training", return_value=(_MOCK_X, _MOCK_Y)) as mock_pre, \
         patch("app.services.retrain.tune_bilstm", return_value=_BILSTM_PARAMS) as mock_tune_b, \
         patch("app.services.retrain.train_bilstm") as mock_train_bilstm, \
         patch("app.services.retrain.predict_bilstm", return_value=np.random.randn(1, 6, 14)) as mock_pred_bilstm, \
         patch("app.services.retrain.tune_lgbm", return_value=_LGBM_PARAMS) as mock_tune_l, \
         patch("app.services.retrain.save_best_params") as mock_save, \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT) as mock_train, \
         patch("app.services.retrain.upsert_metadata"):
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID, run_tuning=True)

    mock_pre.assert_called_once_with(UID)
    mock_tune_b.assert_called_once()
    mock_train_bilstm.assert_called_once()
    mock_tune_l.assert_called_once()
    mock_save.assert_called_once_with(UID, _BILSTM_PARAMS, _LGBM_PARAMS)
    # Tuning path must pass pre-fetched X/y to avoid scaler refit
    mock_train.assert_called_once_with(
        UID,
        bilstm_params=_BILSTM_PARAMS,
        lgbm_params=_LGBM_PARAMS,
        X=_MOCK_X,
        y=_MOCK_Y,
    )
    assert result["status"] == "ready"


def test_retrain_sensor_tuning_evicts_model_cache():
    """After intermediate train_bilstm, _model_cache entry is evicted before predict_bilstm."""
    eviction_order = []

    def fake_train_bilstm(X, y, uid, **kw):
        # Inject a stale entry so we can verify it gets popped
        from app.models.bilstm import _model_cache
        _model_cache[uid] = "stale_model"

    def fake_predict_bilstm(x, uid):
        from app.models.bilstm import _model_cache
        eviction_order.append(uid in _model_cache)
        return np.random.randn(1, 6, 14)

    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.preprocess_for_training", return_value=(_MOCK_X, _MOCK_Y)), \
         patch("app.services.retrain.tune_bilstm", return_value=_BILSTM_PARAMS), \
         patch("app.services.retrain.train_bilstm", side_effect=fake_train_bilstm), \
         patch("app.services.retrain.predict_bilstm", side_effect=fake_predict_bilstm), \
         patch("app.services.retrain.tune_lgbm", return_value=_LGBM_PARAMS), \
         patch("app.services.retrain.save_best_params"), \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT), \
         patch("app.services.retrain.upsert_metadata"):
        from app.services.retrain import retrain_sensor
        retrain_sensor(UID, run_tuning=True)

    # All predict_bilstm calls should have seen the cache as empty (evicted)
    assert all(not present for present in eviction_order), (
        "Expected _model_cache to be evicted before predict_bilstm calls, "
        f"but cache presence was: {eviction_order}"
    )


# ---------------------------------------------------------------------------
# Test 4: Status transitions
# ---------------------------------------------------------------------------

def test_status_transitions_with_tuning():
    """When run_tuning=True and no saved params: status goes 'tuning' then 'training'."""
    status_calls = []

    def fake_upsert(uid, status=None, **kwargs):
        if status is not None:
            status_calls.append(status)

    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.preprocess_for_training", return_value=(_MOCK_X, _MOCK_Y)), \
         patch("app.services.retrain.tune_bilstm", return_value=_BILSTM_PARAMS), \
         patch("app.services.retrain.train_bilstm"), \
         patch("app.services.retrain.predict_bilstm", return_value=np.random.randn(1, 6, 14)), \
         patch("app.services.retrain.tune_lgbm", return_value=_LGBM_PARAMS), \
         patch("app.services.retrain.save_best_params"), \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT), \
         patch("app.services.retrain.upsert_metadata", side_effect=fake_upsert):
        from app.services.retrain import retrain_sensor
        retrain_sensor(UID, run_tuning=True)

    assert "tuning" in status_calls, f"Expected 'tuning' in status calls: {status_calls}"
    assert "training" in status_calls, f"Expected 'training' in status calls: {status_calls}"
    tuning_idx = status_calls.index("tuning")
    training_idx = status_calls.index("training")
    assert tuning_idx < training_idx, "Expected 'tuning' status before 'training' status"


def test_status_transitions_without_tuning():
    """When run_tuning=False: no 'tuning' status, only 'training' then 'ready'."""
    status_calls = []

    def fake_upsert(uid, status=None, **kwargs):
        if status is not None:
            status_calls.append(status)

    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT), \
         patch("app.services.retrain.upsert_metadata", side_effect=fake_upsert):
        from app.services.retrain import retrain_sensor
        retrain_sensor(UID, run_tuning=False)

    assert "tuning" not in status_calls, f"Unexpected 'tuning' in status calls: {status_calls}"
    assert "training" in status_calls
    assert "ready" in status_calls


# ---------------------------------------------------------------------------
# Test 5: Return value structure
# ---------------------------------------------------------------------------

def test_retrain_sensor_return_value_structure():
    """Return dict includes uid, status, training_samples, mae_score."""
    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.run_training", return_value=_MOCK_RESULT), \
         patch("app.services.retrain.upsert_metadata"):
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID)

    assert result["uid"] == UID
    assert result["status"] == "ready"
    assert result["training_samples"] == 100
    assert result["mae_score"] == 0.123456


# ---------------------------------------------------------------------------
# Test 6: Error handling still works
# ---------------------------------------------------------------------------

def test_retrain_sensor_error_handling():
    """Exception during training returns error status dict."""
    with patch("app.services.retrain.load_best_params", return_value=None), \
         patch("app.services.retrain.run_training", side_effect=RuntimeError("DB gone")), \
         patch("app.services.retrain.upsert_metadata"):
        from app.services.retrain import retrain_sensor
        result = retrain_sensor(UID)

    assert result["status"] == "error"
    assert result["uid"] == UID
    assert "DB gone" in result["message"]
