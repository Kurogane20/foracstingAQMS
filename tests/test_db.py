import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# _parse_bounds
# ---------------------------------------------------------------------------

def test_parse_bounds_converts_json_string_to_dict():
    from app.db import _parse_bounds
    row = {
        "lower_bounds": '{"pm_25": 10.0, "pm_10": 20.0}',
        "upper_bounds": '{"pm_25": 15.0, "pm_10": 25.0}',
    }
    result = _parse_bounds(row)
    assert result["lower_bounds"] == {"pm_25": 10.0, "pm_10": 20.0}
    assert result["upper_bounds"] == {"pm_25": 15.0, "pm_10": 25.0}


def test_parse_bounds_passes_through_none():
    from app.db import _parse_bounds
    row = {"lower_bounds": None, "upper_bounds": None}
    result = _parse_bounds(row)
    assert result["lower_bounds"] is None
    assert result["upper_bounds"] is None


def test_parse_bounds_passes_through_already_dict():
    from app.db import _parse_bounds
    bounds = {"pm_25": 5.0}
    row = {"lower_bounds": bounds, "upper_bounds": bounds}
    result = _parse_bounds(row)
    assert result["lower_bounds"] == {"pm_25": 5.0}
    assert result["upper_bounds"] == {"pm_25": 5.0}


def test_parse_bounds_handles_corrupt_json():
    from app.db import _parse_bounds
    row = {"lower_bounds": "{corrupt", "upper_bounds": None}
    result = _parse_bounds(row)
    assert result["lower_bounds"] is None


# ---------------------------------------------------------------------------
# save_predictions — bounds serialization
# ---------------------------------------------------------------------------

def _make_conn_mock():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def _make_predictions(lower_bounds, upper_bounds):
    from app.config import FEATURE_COLS, N_FORECAST_HOURS
    from datetime import timedelta
    base = datetime(2024, 1, 1, 12, 0, 0)
    return [
        {
            "step": i + 1,
            "target_time": base + timedelta(hours=i + 1),
            **{c: float(i) for c in FEATURE_COLS},
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        }
        for i in range(N_FORECAST_HOURS)
    ]


def test_save_predictions_serializes_bounds_to_json():
    from app.db import save_predictions
    lower = {"pm_25": 10.0, "pm_10": 20.0}
    upper = {"pm_25": 15.0, "pm_10": 25.0}
    preds = _make_predictions(lower, upper)
    predicted_at = datetime(2024, 1, 1, 12, 0, 0)

    conn_mock, cursor_mock = _make_conn_mock()
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        save_predictions("uid_test", predicted_at, preds)

    # Each INSERT call should have JSON strings for lower/upper bounds
    insert_calls = [c for c in cursor_mock.execute.call_args_list if "INSERT" in str(c)]
    assert len(insert_calls) > 0
    for c in insert_calls:
        args = c[0][1]  # positional tuple passed to execute
        lower_val = args[-2]
        upper_val = args[-1]
        assert lower_val == json.dumps(lower)
        assert upper_val == json.dumps(upper)


def test_save_predictions_writes_null_for_none_bounds():
    from app.db import save_predictions
    preds = _make_predictions(None, None)
    predicted_at = datetime(2024, 1, 1, 12, 0, 0)

    conn_mock, cursor_mock = _make_conn_mock()
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        save_predictions("uid_test", predicted_at, preds)

    insert_calls = [c for c in cursor_mock.execute.call_args_list if "INSERT" in str(c)]
    assert len(insert_calls) > 0
    for c in insert_calls:
        args = c[0][1]
        lower_val = args[-2]
        upper_val = args[-1]
        assert lower_val is None
        assert upper_val is None


# ---------------------------------------------------------------------------
# get_latest_predictions — calls _parse_bounds
# ---------------------------------------------------------------------------

def test_get_latest_predictions_calls_parse_bounds():
    from app.db import get_latest_predictions
    raw_row = {
        "uid": "uid_test",
        "step": 1,
        "lower_bounds": '{"pm_25": 10.0}',
        "upper_bounds": '{"pm_25": 15.0}',
    }
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock

    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_latest_predictions("uid_test")

    assert result[0]["lower_bounds"] == {"pm_25": 10.0}
    assert result[0]["upper_bounds"] == {"pm_25": 15.0}


# ---------------------------------------------------------------------------
# get_all_latest_predictions — calls _parse_bounds
# ---------------------------------------------------------------------------

def test_get_all_latest_predictions_calls_parse_bounds():
    from app.db import get_all_latest_predictions
    raw_row = {
        "uid": "uid_test",
        "step": 1,
        "lower_bounds": '{"pm_25": 10.0}',
        "upper_bounds": None,
    }
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock

    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_all_latest_predictions()

    assert result["uid_test"][0]["lower_bounds"] == {"pm_25": 10.0}
    assert result["uid_test"][0]["upper_bounds"] is None


# ---------------------------------------------------------------------------
# get_unresolved_predictions
# ---------------------------------------------------------------------------

def test_get_unresolved_predictions_returns_rows():
    from app.db import get_unresolved_predictions
    raw_row = {"uid": "s1", "target_time": datetime(2024, 1, 1, 10), "step": 1, "predicted_at": datetime(2024, 1, 1, 9)}
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_unresolved_predictions("s1")
    assert len(result) == 1
    assert result[0]["step"] == 1


# ---------------------------------------------------------------------------
# update_prediction_actuals
# ---------------------------------------------------------------------------

def test_update_prediction_actuals_executes_update():
    import json as _json
    from app.db import update_prediction_actuals
    conn_mock, cursor_mock = _make_conn_mock()
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        update_prediction_actuals("s1", datetime(2024, 1, 1, 10), {"aqi_index": 55.0})
    assert cursor_mock.execute.called
    call_args = cursor_mock.execute.call_args[0]
    assert "UPDATE" in call_args[0]
    stored = call_args[1][0]
    assert _json.loads(stored) == {"aqi_index": 55.0}


# ---------------------------------------------------------------------------
# get_resolved_predictions
# ---------------------------------------------------------------------------

def test_get_resolved_predictions_parses_actual_values():
    from app.db import get_resolved_predictions
    from app.config import FEATURE_COLS
    import json as _json
    raw_row = {"step": 1, "actual_values": _json.dumps({"aqi_index": 60.0})}
    raw_row.update({c: 50.0 for c in FEATURE_COLS})
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [raw_row]
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        result = get_resolved_predictions("s1", n=10)
    assert result[0]["actual_values"] == {"aqi_index": 60.0}


# ---------------------------------------------------------------------------
# update_drift_metadata
# ---------------------------------------------------------------------------

def test_update_drift_metadata_executes_update():
    from app.db import update_drift_metadata
    conn_mock, cursor_mock = _make_conn_mock()
    ts = datetime(2024, 1, 2, 0, 0, 0)
    with patch("app.db.mysql.connector.connect", return_value=conn_mock):
        update_drift_metadata("s1", 1.8, ts)
    assert cursor_mock.execute.called
    call_args = cursor_mock.execute.call_args[0]
    call_sql = call_args[0]
    call_params = call_args[1]
    assert "UPDATE" in call_sql
    assert "drift_score" in call_sql
    assert call_params == (1.8, ts, "s1")
