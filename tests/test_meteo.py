import math
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from app.services.meteo import fetch_meteo_training, fetch_meteo_predict

METEO_COLS = ["meteo_wind_speed", "meteo_wind_dir_sin", "meteo_wind_dir_cos", "meteo_cloudcover"]


def _mock_response(wind_speeds, wind_dirs, cloudcovers, times):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = {
        "hourly": {
            "time": times,
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_dirs,
            "cloudcover": cloudcovers,
        }
    }
    return mock_resp


class TestFetchMeteoTraining:
    def test_returns_dataframe_with_meteo_cols(self):
        times = ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([5.0, 3.0, 0.0], [90.0, 180.0, 270.0], [50.0, 0.0, 100.0], times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_training(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert list(df.columns) == METEO_COLS
        assert len(df) == 3

    def test_wind_speed_normalised_to_1_at_20ms(self):
        times = ["2024-01-01T00:00"]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([20.0], [0.0], [0.0], times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_training(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert df["meteo_wind_speed"].iloc[0] == pytest.approx(1.0)

    def test_wind_speed_capped_at_1_when_over_20(self):
        times = ["2024-01-01T00:00"]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([40.0], [0.0], [0.0], times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_training(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert df["meteo_wind_speed"].iloc[0] == pytest.approx(1.0)

    def test_wind_dir_sin_at_90_degrees(self):
        times = ["2024-01-01T00:00"]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([1.0], [90.0], [50.0], times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_training(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert df["meteo_wind_dir_sin"].iloc[0] == pytest.approx(1.0, abs=1e-5)

    def test_cloudcover_normalised(self):
        times = ["2024-01-01T00:00"]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([1.0], [0.0], [75.0], times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_training(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert df["meteo_cloudcover"].iloc[0] == pytest.approx(0.75)


class TestFetchMeteoPredict:
    def test_returns_dataframe_with_meteo_cols(self):
        N = 24
        times = [f"2024-01-01T{h:02d}:00" for h in range(N)]
        timestamps = pd.DatetimeIndex(pd.to_datetime(times))
        mock_resp = _mock_response([2.0] * N, [180.0] * N, [30.0] * N, times)

        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
            df = fetch_meteo_predict(lat=-1.0, lng=116.0, timestamps=timestamps)

        assert list(df.columns) == METEO_COLS
        assert len(df) == N


def test_error_body_raises_value_error():
    """fetch_meteo_training raises ValueError when API returns {"error": true}"""
    with patch("httpx.Client") as MockClient:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": True, "reason": "No data available"}
        mock_resp.raise_for_status = lambda: None
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
        ts = pd.date_range("2024-01-01", periods=3, freq="h")
        with pytest.raises(ValueError, match="Open-Meteo error"):
            fetch_meteo_training(1.0, 116.0, ts)


def test_nan_in_api_response_filled_with_zero():
    """NaN values in API response are filled with 0.0 (not passed through)"""
    with patch("httpx.Client") as MockClient:
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "wind_speed_10m": [float("nan"), 5.0],
                "wind_direction_10m": [None, 90.0],
                "cloudcover": [50.0, None],
            }
        }
        mock_resp.raise_for_status = lambda: None
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
        result = fetch_meteo_training(1.0, 116.0, times)
        assert result["meteo_wind_speed"].iloc[0] == 0.0   # NaN → 0.0
        assert result["meteo_wind_dir_cos"].iloc[0] != float("nan")  # None → 0.0 direction
        assert result["meteo_cloudcover"].iloc[1] == 0.0   # None → 0.0


def test_tz_aware_timestamps_stripped():
    """tz-aware timestamps from API are handled without crash"""
    with patch("httpx.Client") as MockClient:
        times_utc = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        times_str = [t.strftime("%Y-%m-%dT%H:%M+00:00") for t in times_utc]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": times_str,
                "wind_speed_10m": [3.0, 4.0],
                "wind_direction_10m": [90.0, 180.0],
                "cloudcover": [50.0, 80.0],
            }
        }
        mock_resp.raise_for_status = lambda: None
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
        query_ts = pd.date_range("2024-01-01", periods=2, freq="h")  # tz-naive
        result = fetch_meteo_training(1.0, 116.0, query_ts)
        assert not result.empty  # no TypeError from tz mismatch


def test_missing_timestamps_warning(caplog):
    """Logs warning when >10% of requested timestamps have no meteo match"""
    import logging
    with patch("httpx.Client") as MockClient:
        api_times = pd.date_range("2024-01-01", periods=2, freq="h")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in api_times],
                "wind_speed_10m": [3.0, 4.0],
                "wind_direction_10m": [90.0, 180.0],
                "cloudcover": [50.0, 80.0],
            }
        }
        mock_resp.raise_for_status = lambda: None
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp
        # Request 20 timestamps but API only has 2 → >10% unmatched
        query_ts = pd.date_range("2024-01-01", periods=20, freq="h")
        with caplog.at_level(logging.WARNING, logger="app.services.meteo"):
            fetch_meteo_training(1.0, 116.0, query_ts)
        assert any("unmatched" in r.message.lower() for r in caplog.records)
