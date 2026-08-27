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

def _hourly_row(**overrides) -> dict:
    """Baris hasil agregasi AVG per jam, nilainya masuk akal secara fisik."""
    row = {
        "pm_25": 30.0, "pm_25_correction": 30.0,
        "pm_10": 60.0, "pm_10_correction": 60.0,
        "tsp": 90.0, "tsp_correction": 90.0,
        "noise": 55.0, "temp": 28.0, "mmhg": 760.0, "humidity": 82.0,
        "aqi_index_pm25": 40.0, "aqi_index_pm10": 45.0,
        "aqi_index_tsp": 50.0, "aqi_index": 50.0,
        "n_readings": 58,
    }
    row.update(overrides)
    return row


def test_fetch_actual_for_step_returns_dict_when_found():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = _hourly_row()
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert result is not None
    assert result["tsp"] == pytest.approx(90.0)
    assert "n_readings" not in result       # metadata kueri, bukan parameter


def test_fetch_actual_for_step_averages_the_target_hour():
    """Model dilatih pada rata-rata per jam, jadi 'aktual' harus rata-rata jam
    yang sama — bukan satu bacaan sesaat (selisih dua definisi itu 29,67 µg/m³)."""
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = _hourly_row()
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        fetch_actual_for_step(UID, TARGET_TIME)

    sql, params = cursor_mock.execute.call_args[0]
    assert "AVG(" in sql
    target_unix = int(TARGET_TIME.replace(tzinfo=timezone.utc).timestamp())
    assert params[1] == target_unix              # awal jam
    assert params[2] == target_unix + 3600       # akhir jam, eksklusif


def test_fetch_actual_for_step_rejects_hour_with_too_few_readings():
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = _hourly_row(n_readings=3)
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        assert fetch_actual_for_step(UID, TARGET_TIME) is None


def test_fetch_actual_for_step_drops_physically_impossible_columns():
    """5 dari 9 sensor tidak punya modul cuaca dan melaporkan 0. Model memprediksi
    nilai cadangan (760 hPa), jadi galat konstan itu akan menenggelamkan metrik."""
    from app.services.accuracy import fetch_actual_for_step
    conn_mock, cursor_mock = _make_conn_mock()
    cursor_mock.fetchone.return_value = _hourly_row(mmhg=0.0, temp=0.0, humidity=0.0)
    with patch("app.services.accuracy.mysql.connector.connect", return_value=conn_mock):
        result = fetch_actual_for_step(UID, TARGET_TIME)
    assert "mmhg" not in result
    assert "temp" not in result
    assert "humidity" not in result
    assert result["tsp"] == pytest.approx(90.0)


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

def test_detect_drift_flags_recent_errors_larger_than_history():
    """Baris diurutkan target_time DESC, jadi DRIFT_WINDOW pertama = terkini."""
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    recent = [_make_resolved_row(step=1, pred_val=50.0, act_val=110.0)
              for _ in range(DRIFT_WINDOW)]                       # galat 60
    history = [_make_resolved_row(step=1, pred_val=50.0, act_val=70.0)
               for _ in range(DRIFT_WINDOW * 7)]                  # galat 20
    with patch("app.services.accuracy.get_resolved_predictions", return_value=recent + history):
        ratio = detect_drift(UID)
    assert ratio == pytest.approx(3.0, abs=0.01)


def test_detect_drift_stable_model_scores_near_one():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1, pred_val=50.0, act_val=70.0)
            for _ in range(DRIFT_WINDOW * 8)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID)
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_detect_drift_compares_like_with_like():
    """Regresi: sebelumnya galat produksi (µg/m³) dibandingkan dengan mae_score
    hasil pelatihan (ruang ternormalisasi 0–1) — beda ~3 orde besaran, sehingga
    rasionya selalu jauh di atas ambang dan auto-retrain terpicu terus-menerus.
    Model yang tidak berubah harus menghasilkan skor jauh di bawah ambang."""
    from app.services.accuracy import detect_drift, DRIFT_WINDOW, DRIFT_THRESHOLD
    rows = [_make_resolved_row(step=1, pred_val=50.0, act_val=80.0)
            for _ in range(DRIFT_WINDOW * 8)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID)
    assert ratio < DRIFT_THRESHOLD


def test_detect_drift_returns_none_when_insufficient_data():
    from app.services.accuracy import detect_drift, DRIFT_WINDOW
    rows = [_make_resolved_row(step=1) for _ in range(DRIFT_WINDOW * 2 - 1)]
    with patch("app.services.accuracy.get_resolved_predictions", return_value=rows):
        ratio = detect_drift(UID)
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
        result = check_and_auto_retrain(UID)
    assert result is True
    assert len(started_threads) == 1
    mock_upsert.assert_called_once_with(UID, status="drifted")


def test_check_and_auto_retrain_no_trigger_below_threshold():
    from app.services.accuracy import check_and_auto_retrain, DRIFT_THRESHOLD
    low_ratio = DRIFT_THRESHOLD - 0.1
    with patch("app.services.accuracy.detect_drift", return_value=low_ratio), \
         patch("app.services.accuracy.update_drift_metadata"), \
         patch("app.services.accuracy.upsert_metadata") as mock_upsert:
        result = check_and_auto_retrain(UID)
    assert result is False
    mock_upsert.assert_not_called()
