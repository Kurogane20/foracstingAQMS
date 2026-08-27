import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.dispersion import compute_dispersion_forecast


def _mock_forecast_response():
    """Build a minimal Open-Meteo hourly forecast response for 7 hours."""
    times   = [f"2024-01-15T{14+h:02d}:00" for h in range(7)]
    speeds  = [3.0] * 7
    dirs    = [90.0] * 7
    clouds  = [50.0] * 7
    mock = MagicMock()
    mock.raise_for_status = lambda: None
    mock.json.return_value = {
        "hourly": {
            "time":                times,
            "wind_speed_10m":      speeds,
            "wind_direction_10m":  dirs,
            "cloudcover":          clouds,
        }
    }
    return mock


def run(coro):
    return asyncio.run(coro)


class TestComputeDispersionForecast:
    def test_empty_sensors_returns_empty_frames(self):
        result = run(compute_dispersion_forecast([]))
        assert result == {"frames": []}

    def test_returns_seven_frames(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        assert len(result["frames"]) == 7

    def test_hour_offsets_0_to_6(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        assert [f["hour_offset"] for f in result["frames"]] == [0, 1, 2, 3, 4, 5, 6]

    def test_frame_zero_label_starts_with_sekarang(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        assert result["frames"][0]["label"].startswith("Sekarang")

    def test_later_frames_label_starts_with_h_plus(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        for h in range(1, 7):
            assert result["frames"][h]["label"].startswith(f"H+{h}")

    def test_each_frame_has_required_keys(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        for frame in result["frames"]:
            assert "hour_offset" in frame
            assert "label" in frame
            assert "grid" in frame
            assert "wind_vectors" in frame

    def test_failed_fetch_uses_neutral_defaults_still_returns_seven_frames(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(side_effect=Exception("network error"))
            result = run(compute_dispersion_forecast(sensors))
        assert len(result["frames"]) == 7

    def test_grid_values_normalised_0_to_1(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 100.0, "tsp": 200.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        for frame in result["frames"]:
            for point in frame["grid"]:
                assert 0.0 <= point[2] <= 1.0

    def test_p90_field_scales_with_upper_bound(self):
        """Kanal peringatan memakai kuantil 0,9 sebagai kekuatan sumber, jadi
        puncaknya harus sebanding dengan rasio tsp_p90/tsp."""
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0,
                    "pm25": 50.0, "tsp": 100.0, "tsp_p90": 300.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))

        for frame in result["frames"]:
            assert frame["max_conc_p90"] == pytest.approx(3.0 * frame["max_conc"], rel=1e-3)

    def test_p90_absent_when_upper_bound_not_supplied(self):
        """Tanpa tsp_p90 tidak boleh ada uplift palsu — klien harus jatuh kembali
        ke max_conc, bukan memakai angka yang dikarang."""
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))

        for frame in result["frames"]:
            assert frame["max_conc_p90"] == 0.0

    def test_p90_never_below_point_forecast(self):
        """Batas atas yang lebih kecil dari prakiraan titik (mis. model belum
        terlatih) tidak boleh menurunkan ambang peringatan."""
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0,
                    "pm25": 50.0, "tsp": 200.0, "tsp_p90": 50.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))

        for frame in result["frames"]:
            assert frame["max_conc_p90"] == 0.0
