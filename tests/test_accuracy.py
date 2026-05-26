import threading
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.config import FEATURE_COLS

UID = "sensor_01"
TARGET_TIME = datetime(2024, 1, 1, 12, 0, 0)


def _make_resolved_row(step: int, pred_val: float = 50.0, act_val: float = 55.0) -> dict:
    row = {col: pred_val for col in FEATURE_COLS}
    row["step"] = step
    row["target_time"] = TARGET_TIME
    row["actual_values"] = {col: act_val for col in FEATURE_COLS}
    return row


def _make_conn_mock():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# fetch_actual_for_step

def test_fetch_actual_for_step_returns_dict_when_found():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = {col: 50.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert result is not None
    assert "aqi_index" in result
    assert isinstance(result["aqi_index"], float)


def test_fetch_actual_for_step_returns_none_when_not_found():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = None
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert result is None


# resolve_prediction_actuals

def test_resolve_prediction_actuals_stores_actuals():
    from app.services.accuracy import resolve_prediction_actuals
    unresolved = [{"uid": UID, "target_time": TARGET_TIME, "step": 1, "predicted_at": TARGET_TIME}]
    actual = {col: 55.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.get_unresolved_predictions", return_value=unresolved), \
         patch("app.services.accuracy.fetch_actual_for_step", return_value=actual) as mock_fetch, \
         patch("app.services.accuracy.update_prediction_actuals") as mock_update:
        count = resolve_prediction_actuals(UID)
    assert count == 1
    mock_fetch.assert_called_once_with(UID, TARGET_TIME)
    mock_update.assert_called_once_with(UID, TARGET_TIME, actual)


def test_resolve_prediction_actuals_deduplicates_target_times():
    from app.services.accuracy import resolve_prediction_actuals
    unresolved = [
        {"uid": UID, "target_time": TARGET_TIME, "step": 1, "predicted_at": datetime(2024, 1, 1, 10)},
        {"uid": UID, "target_time": TARGET_TIME, "step": 1, "predicted_at": datetime(2024, 1, 1, 11)},
    ]
    actual = {col: 55.0 for col in FEATURE_COLS}
    with patch("app.services.accuracy.get_unresolved_predictions", return_value=unresolved), \
         patch("app.services.accuracy.fetch_actual_for_step", return_value=actual) as mock_fetch, \
         patch("app.services.accuracy.update_prediction_actuals"):
        count = resolve_prediction_actuals(UID)
    assert count == 1
    mock_fetch.assert_called_once()


# compute_accuracy

def test_compute_accuracy_returns_per_step_mae():
    from app.services.accuracy import compute_accuracy, DRIFT_WINDOW
    rows = [_make_resolved_row(step=i % 6 + 1) for i in range(DRIFT_WINDOW * 6)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        result = compute_accuracy(UID)
    assert "step_1" in result
    assert "overall_mae" in result
    assert result["overall_mae"] == pytest.approx(5.0, abs=0.01)


def test_compute_accuracy_returns_empty_when_no_resolved():
    from app.services.accuracy import compute_accuracy
    with patch("app.services.accuracy.get_resolved_predictions", return_value=[]):
        result = compute_accuracy(UID)
    assert result == {}


# detect_drift

def test_detect_drift_returns_ratio():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1, pred_val=50.0, act_val=60.0) for _ in range(DRIFT_WINDOW)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID, baseline_mae=5.0)
    assert ratio is not None
    assert ratio > 1.0


def test_detect_drift_returns_none_when_no_baseline():
    from app.services.accuracy import detect_drift
    ratio = detect_drift(UID, baseline_mae=None)
    assert ratio is None


def test_detect_drift_returns_none_when_baseline_zero():
    from app.services.accuracy import detect_drift
    ratio = detect_drift(UID, baseline_mae=0.0)
    assert ratio is None


def test_detect_drift_returns_none_when_insufficient_data():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1) for _ in range(DRIFT_WINDOW - 1)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID, baseline_mae=5.0)
    assert ratio is None


# check_and_auto_retrain

def test_check_and_auto_retrain_triggers_thread_on_drift():
    from app.services.accuracy import check_and_auto_retrain, DRIFT_THRESHOLD
    high_ratio = DRIFT_THRESHOLD + 0.5
    started_threads = []
    def capture_start(self):
        started_threads.append(self)
    with patch("app.services.accuracy.detect_drift", return_value=high_ratio), \
         patch("app.services.accuracy.update_drift_metadata"), \
         patch("app.services.accuracy.upsert_metadata") as mock_upsert, \
         patch.object(threading.Thread, "start", capture_start):
        result = check_and_auto_retrain(UID, baseline_mae=5.0)
    assert result is True
    assert len(started_threads) == 1
    mock_upsert.assert_called_once_with(UID, status="drifted")


def test_check_and_auto_retrain_no_trigger_below_threshold():
    from app.services.accuracy import check_and_auto_retrain, DRIFT_THRESHOLD
    low_ratio = DRIFT_THRESHOLD - 0.1
    with patch("app.services.accuracy.detect_drift", return_value=low_ratio), \
         patch("app.services.accuracy.update_drift_metadata"), \
         patch("app.services.accuracy.upsert_metadata") as mock_upsert:
        result = check_and_auto_retrain(UID, baseline_mae=5.0)
    assert result is False
    mock_upsert.assert_not_called()
