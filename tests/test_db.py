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
