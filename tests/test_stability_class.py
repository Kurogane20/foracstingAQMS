import pytest
from app.services.dispersion import _stability_class


class TestStabilityClass:
    def test_clear_day_low_wind_gives_a(self):
        assert _stability_class(wind_speed=1.0, cloudcover=10.0, local_hour=12) == "A"

    def test_clear_day_medium_wind_gives_b(self):
        assert _stability_class(wind_speed=3.0, cloudcover=10.0, local_hour=12) == "B"

    def test_moderate_day_low_wind_gives_b(self):
        assert _stability_class(wind_speed=1.5, cloudcover=50.0, local_hour=10) == "B"

    def test_overcast_day_any_wind_gives_c_or_d(self):
        result = _stability_class(wind_speed=4.0, cloudcover=80.0, local_hour=10)
        assert result in ("C", "D")

    def test_overcast_day_high_wind_gives_d(self):
        assert _stability_class(wind_speed=6.0, cloudcover=80.0, local_hour=10) == "D"

    def test_clear_night_low_wind_gives_f(self):
        assert _stability_class(wind_speed=1.0, cloudcover=10.0, local_hour=2) == "F"

    def test_clear_night_medium_wind_gives_e(self):
        assert _stability_class(wind_speed=4.0, cloudcover=20.0, local_hour=23) == "E"

    def test_overcast_night_gives_d(self):
        assert _stability_class(wind_speed=3.0, cloudcover=80.0, local_hour=22) == "D"

    def test_high_wind_neutral(self):
        result = _stability_class(wind_speed=7.0, cloudcover=0.0, local_hour=14)
        assert result in ("C", "D")

    def test_old_signature_still_works(self):
        result = _stability_class(wind_speed=3.0)
        assert result in ("A", "B", "C", "D", "E", "F")
