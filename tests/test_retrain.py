from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest

UID = "sensor_001"

_RESULT = {
    "training_samples": 9000, "mae_score": 0.031, "mae_tsp": 30.5,
    "mae_persistence_tsp": 40.2, "skill_vs_naive": 0.24, "coverage_tsp": 0.81,
}
_PARAMS = {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 20, "min_child_samples": 30}
_HOURLY = pd.DataFrame({"tsp": [1.0]}, index=pd.date_range("2026-01-01", periods=1, freq="h"))


@pytest.fixture
def mocks():
    names = ["load_for_training", "load_best_params", "save_best_params", "tune_lgbm",
             "run_training", "upsert_metadata", "set_phase", "clear_progress"]
    with ExitStack() as stack:
        m = {n: stack.enter_context(patch(f"app.services.retrain.{n}")) for n in names}
        m["load_for_training"].return_value = _HOURLY
        m["load_best_params"].return_value = None
        m["tune_lgbm"].return_value = _PARAMS
        m["run_training"].return_value = dict(_RESULT)
        yield m


def _retrain(*args, **kwargs):
    from app.services.retrain import retrain_sensor
    return retrain_sensor(*args, **kwargs)


def test_default_retrain_uses_default_params_and_no_tuning(mocks):
    result = _retrain(UID)
    mocks["load_for_training"].assert_called_once_with(UID)
    mocks["run_training"].assert_called_once_with(UID, lgbm_params=None, hourly=_HOURLY)
    mocks["tune_lgbm"].assert_not_called()
    mocks["save_best_params"].assert_not_called()
    assert result["status"] == "ready" and result["uid"] == UID
    assert result["mae_tsp"] == 30.5


def test_saved_params_are_reused_without_tuning(mocks):
    mocks["load_best_params"].return_value = _PARAMS
    _retrain(UID, run_tuning=True)
    mocks["tune_lgbm"].assert_not_called()
    mocks["run_training"].assert_called_once_with(UID, lgbm_params=_PARAMS, hourly=_HOURLY)


def test_tuning_runs_once_on_the_same_data_then_trains(mocks):
    _retrain(UID, run_tuning=True)
    mocks["tune_lgbm"].assert_called_once_with(_HOURLY)
    mocks["save_best_params"].assert_called_once_with(UID, _PARAMS)
    mocks["run_training"].assert_called_once_with(UID, lgbm_params=_PARAMS, hourly=_HOURLY)
    statuses = [c.kwargs.get("status") for c in mocks["upsert_metadata"].call_args_list]
    assert statuses == ["tuning", "training", "ready"]


def test_success_updates_metadata_with_dashboard_score(mocks):
    _retrain(UID)
    final = mocks["upsert_metadata"].call_args_list[-1]
    assert final.args == (UID,)
    assert final.kwargs["status"] == "ready"
    assert final.kwargs["training_samples"] == 9000
    assert final.kwargs["mae_score"] == 0.031
    assert final.kwargs["error_message"] is None
    assert final.kwargs["last_trained_at"].tzinfo is None        # UTC naif, seperti kolomnya
    mocks["clear_progress"].assert_called_once_with(UID)


def test_training_failure_is_recorded_not_raised(mocks):
    mocks["run_training"].side_effect = ValueError("Data belum cukup: 100 jam")
    result = _retrain(UID)
    assert result == {"uid": UID, "status": "error", "message": "Data belum cukup: 100 jam"}
    final = mocks["upsert_metadata"].call_args_list[-1]
    assert final.kwargs == {"status": "error", "error_message": "Data belum cukup: 100 jam"}
    mocks["clear_progress"].assert_called_once_with(UID)


def test_data_loading_failure_is_recorded_not_raised(mocks):
    mocks["load_for_training"].side_effect = RuntimeError("DB down")
    result = _retrain(UID)
    assert result["status"] == "error" and "DB down" in result["message"]
    mocks["run_training"].assert_not_called()
