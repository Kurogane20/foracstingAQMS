import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.dispersion import (
    compute_dispersion_forecast,
    _fetch_wind_forecast,
    _fetch_wind_past24,
)

WITA = 8 * 3600


def _open_meteo_response(local_midnight: datetime, n_hours: int = 48,
                         offset: int = WITA, direction=None):
    """Tiru respons Open-Meteo timezone=auto: jam dinding LOKAL tanpa offset,
    selalu mulai 00:00 lokal hari pertama, plus `utc_offset_seconds`."""
    times = [(local_midnight + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
             for h in range(n_hours)]
    mock = MagicMock()
    mock.raise_for_status = lambda: None
    mock.json.return_value = {
        "utc_offset_seconds": offset,
        "hourly": {
            "time":               times,
            "wind_speed_10m":     [3.0] * n_hours,
            # Arah = jam lokal × 10 agar tiap jam bisa dikenali di asersi.
            "wind_direction_10m": direction or [float(int(t[11:13]) * 10) for t in times],
            "cloudcover":         [50.0] * n_hours,
        },
    }
    return mock


def _mock_forecast_response():
    """Respons prakiraan untuk hari ini (lokal WITA), seperti API sungguhan."""
    now_local = datetime.now(timezone.utc) + timedelta(seconds=WITA)
    midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return _open_meteo_response(midnight, direction=[90.0] * 48)


def _client_returning(resp):
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    return client


def _unix(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def run(coro):
    return asyncio.run(coro)


class TestWindHourSelection:
    """Regresi: Open-Meteo selalu mulai 00:00 lokal. Dulu indeks 0..6 diambil
    apa adanya, jadi prakiraan jam 11 siang memakai angin 00:00–06:00."""

    def test_forecast_starts_at_requested_hour_not_local_midnight(self):
        resp = _open_meteo_response(datetime(2026, 10, 1))
        start = _unix("2026-10-01T03:00")            # = 11:00 WITA
        rows = run(_fetch_wind_forecast(0.0, 0.0, 6, _client_returning(resp), start))
        assert [r["label_time"] for r in rows] == [f"{h:02d}:00" for h in range(11, 18)]
        assert [r["hour"] for r in rows] == list(range(11, 18))
        assert rows[0]["direction"] == 110.0

    def test_forecast_window_crosses_local_midnight(self):
        resp = _open_meteo_response(datetime(2026, 10, 1))
        start = _unix("2026-10-01T15:00")            # = 23:00 WITA
        rows = run(_fetch_wind_forecast(0.0, 0.0, 6, _client_returning(resp), start))
        assert [r["hour"] for r in rows] == [23, 0, 1, 2, 3, 4, 5]

    def test_forecast_without_current_hour_returns_empty(self):
        resp = _open_meteo_response(datetime(2024, 1, 15))
        start = _unix("2026-10-01T03:00")
        rows = run(_fetch_wind_forecast(0.0, 0.0, 6, _client_returning(resp), start))
        assert rows == []

    def test_past24_window_ends_at_current_local_hour(self):
        """Dulu jendela dipotong memakai jam UTC → tertinggal 8 jam dari TSP."""
        resp = _open_meteo_response(datetime(2026, 9, 30))   # past_days=1
        end = _unix("2026-10-01T03:00")                       # = 11:00 WITA
        rows = run(_fetch_wind_past24(0.0, 0.0, _client_returning(resp), end))
        assert len(rows) == 24
        assert rows[0]["time"] == "2026-09-30T12:00"
        assert rows[-1]["time"] == "2026-10-01T11:00"

    def test_past24_near_local_midnight_never_includes_future_hours(self):
        resp = _open_meteo_response(datetime(2026, 9, 30))
        end = _unix("2026-09-30T20:00")                       # = 04:00 WITA 1 Okt
        rows = run(_fetch_wind_past24(0.0, 0.0, _client_returning(resp), end))
        assert rows[-1]["time"] == "2026-10-01T04:00"
        assert rows[0]["time"] == "2026-09-30T05:00"


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

    def test_frame_labels_follow_current_local_hour(self):
        sensors = [{"uid": "S1", "lat": -1.0, "lng": 116.0, "pm25": 50.0, "tsp": 100.0}]
        with patch("app.services.dispersion.httpx.AsyncClient") as MockClient:
            inst = MockClient.return_value.__aenter__.return_value
            inst.get = AsyncMock(return_value=_mock_forecast_response())
            result = run(compute_dispersion_forecast(sensors))
        now_local = datetime.now(timezone.utc) + timedelta(seconds=WITA)
        assert result["frames"][0]["label"] == f"Sekarang {now_local:%H}:00"
        h1 = now_local + timedelta(hours=1)
        assert result["frames"][1]["label"] == f"H+1 {h1:%H}:00"

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
