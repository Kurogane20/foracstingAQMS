from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURE_COLS, MIN_READINGS_PER_HOUR
from app.models import preprocessor as pp


def _rows(hours, n=60, **overrides):
    """Baris agregat seperti keluaran fetch_hourly_rows untuk jam-jam `hours`."""
    data = {"hour_start": [int(pd.Timestamp(h).timestamp()) for h in hours],
            pp.COUNT_COL: n}
    for col in FEATURE_COLS:
        data[col] = overrides.get(col, {"noise": 55.0, "temp": 28.0, "mmhg": 757.0,
                                        "humidity": 80.0}.get(col, 50.0))
    return pd.DataFrame(data)


# ── to_hourly_frame ──────────────────────────────────────────────────────────

def test_hourly_frame_is_continuous_with_nan_for_missing_hours():
    """Jam tanpa data harus hadir sebagai baris NaN: fitur lag memakai shift(k)
    dan menganggap baris ke-k di belakang adalah k JAM yang lalu."""
    rows = _rows(["2026-03-01 00:00", "2026-03-01 01:00", "2026-03-01 04:00"])
    frame = pp.to_hourly_frame(rows)
    assert len(frame) == 5
    assert (frame.index[1:] - frame.index[:-1] == pd.Timedelta(hours=1)).all()
    assert frame.loc["2026-03-01 02:00":"2026-03-01 03:00", "tsp"].isna().all()
    assert frame.loc["2026-03-01 04:00", "tsp"] == 50.0


def test_hourly_frame_accepts_mysql_decimal_and_null():
    """MySQL mengembalikan Decimal untuk FLOOR()/AVG() dan None untuk AVG kosong."""
    rows = pd.DataFrame({
        "hour_start": [Decimal("1772323200"), Decimal("1772326800")],
        pp.COUNT_COL: [60, 58],
        **{c: [Decimal("12.5"), None] for c in FEATURE_COLS},
    })
    frame = pp.to_hourly_frame(rows)
    assert frame["tsp"].dtype == np.float64
    assert frame["tsp"].iloc[0] == 12.5
    assert np.isnan(frame["tsp"].iloc[1])
    assert frame.index[0] == pd.Timestamp("2026-03-01 00:00")


def test_hourly_frame_empty_input():
    frame = pp.to_hourly_frame(pd.DataFrame(columns=["hour_start", pp.COUNT_COL] + FEATURE_COLS))
    assert frame.empty


# ── apply_quality_filters ────────────────────────────────────────────────────

def test_hours_with_too_few_readings_are_blanked():
    hours = pd.date_range("2026-03-01", periods=3, freq="h")
    rows = _rows(hours)
    rows.loc[1, pp.COUNT_COL] = MIN_READINGS_PER_HOUR - 1
    clean = pp.apply_quality_filters(pp.to_hourly_frame(rows))
    assert clean.iloc[1].isna().all()
    assert clean.iloc[0].notna().all() and clean.iloc[2].notna().all()


def test_physically_impossible_values_are_blanked_not_clipped():
    hours = pd.date_range("2026-03-01", periods=4, freq="h")
    rows = _rows(hours)
    rows.loc[1, "tsp"] = -171.9        # negatif: nyata ada di data ekspor
    rows.loc[2, "tsp"] = 9999.0
    rows.loc[3, "temp"] = 0.0          # sensor tanpa modul cuaca melapor nol
    clean = pp.apply_quality_filters(pp.to_hourly_frame(rows))
    assert np.isnan(clean["tsp"].iloc[1]) and np.isnan(clean["tsp"].iloc[2])
    assert np.isnan(clean["temp"].iloc[3])
    assert clean["pm_10"].notna().all()


def test_real_dust_events_survive_filtering():
    """TSP 1.400 µg/m³ itu buruk, tapi nyata — tidak boleh dibuang."""
    rows = _rows(pd.date_range("2026-03-01", periods=2, freq="h"))
    rows.loc[1, "tsp"] = 1400.0
    clean = pp.apply_quality_filters(pp.to_hourly_frame(rows))
    assert clean["tsp"].iloc[1] == 1400.0


def test_gaps_are_never_interpolated():
    """Regresi: versi lama mengisi celah berhari-hari dengan garis lurus dan
    melatihkannya sebagai pengukuran."""
    rows = _rows(["2026-03-01 00:00", "2026-03-03 00:00"])
    rows.loc[0, "tsp"], rows.loc[1, "tsp"] = 10.0, 500.0
    clean = pp.apply_quality_filters(pp.to_hourly_frame(rows))
    assert len(clean) == 49
    assert clean["tsp"].notna().sum() == 2


def test_filters_return_only_feature_cols():
    clean = pp.apply_quality_filters(pp.to_hourly_frame(_rows(pd.date_range("2026-03-01", periods=2, freq="h"))))
    assert list(clean.columns) == FEATURE_COLS


# ── drop_incomplete_hour / extend_to ─────────────────────────────────────────

def test_drop_incomplete_hour_removes_running_bin():
    idx = pd.date_range("2026-03-01 00:00", periods=4, freq="h")
    df = pd.DataFrame({"tsp": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    out = pp.drop_incomplete_hour(df, now=pd.Timestamp("2026-03-01 03:20", tz="UTC"))
    assert out.index[-1] == pd.Timestamp("2026-03-01 02:00")


def test_drop_incomplete_hour_handles_local_timezone():
    idx = pd.date_range("2026-03-01 00:00", periods=4, freq="h")
    df = pd.DataFrame({"tsp": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    # 11:20 WITA = 03:20 UTC
    out = pp.drop_incomplete_hour(df, now=pd.Timestamp("2026-03-01 11:20", tz="Asia/Makassar"))
    assert out.index[-1] == pd.Timestamp("2026-03-01 02:00")


def test_extend_to_pads_offline_sensor_with_nan():
    """Sensor mati 5 jam: indeks tetap harus sampai jam penuh terakhir, supaya
    prakiraan tidak berangkat dari — dan dilabeli — jam yang sudah lewat."""
    idx = pd.date_range("2026-03-01 00:00", periods=3, freq="h")
    df = pd.DataFrame({"tsp": [1.0, 2.0, 3.0]}, index=idx)
    out = pp.extend_to(df, pd.Timestamp("2026-03-01 07:00"))
    assert out.index[-1] == pd.Timestamp("2026-03-01 07:00")
    assert out["tsp"].iloc[3:].isna().all()
    assert out["tsp"].iloc[:3].tolist() == [1.0, 2.0, 3.0]


def test_extend_to_is_noop_when_current():
    idx = pd.date_range("2026-03-01 00:00", periods=3, freq="h")
    df = pd.DataFrame({"tsp": [1.0, 2.0, 3.0]}, index=idx)
    assert pp.extend_to(df, idx[-1]).equals(df)


# ── fetch_hourly_rows / load_hourly ──────────────────────────────────────────

def test_fetch_hourly_rows_aggregates_in_sql_and_closes_connection():
    cursor = MagicMock()
    cursor.description = [("hour_start",), (pp.COUNT_COL,)] + [(c,) for c in FEATURE_COLS]
    cursor.fetchall.return_value = [(1772323200, 60) + (1.0,) * len(FEATURE_COLS)]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("app.models.preprocessor.mysql.connector.connect", return_value=conn), \
         patch("app.models.preprocessor.time.time", return_value=1772330000.0):
        rows = pp.fetch_hourly_rows("AQI01B", hours=24)

    sql, params = cursor.execute.call_args[0]
    assert "GROUP BY hour_start" in sql and "COUNT(*)" in sql and "AVG(`tsp`)" in sql
    assert "deleted_at IS NULL" in sql
    uid, cutoff = params
    assert uid == "AQI01B"
    # Batas bawah jatuh tepat di awal jam, jadi bin pertama tidak pernah terpotong.
    assert cutoff % 3600 == 0
    assert cutoff == (1772330000 // 3600 - 24) * 3600
    assert list(rows.columns) == ["hour_start", pp.COUNT_COL] + FEATURE_COLS
    conn.close.assert_called_once()


def test_fetch_hourly_rows_closes_connection_on_error():
    conn = MagicMock()
    conn.cursor.side_effect = RuntimeError("boom")
    with patch("app.models.preprocessor.mysql.connector.connect", return_value=conn):
        with pytest.raises(RuntimeError):
            pp.fetch_hourly_rows("AQI01B", hours=24)
    conn.close.assert_called_once()


def test_load_hourly_drops_running_hour_and_extends_to_last_full_hour():
    hours = pd.date_range("2026-03-01 00:00", periods=6, freq="h")   # 00..05
    rows = _rows(hours)
    now = pd.Timestamp("2026-03-01 08:10", tz="UTC")                 # jam penuh terakhir: 07
    with patch("app.models.preprocessor.fetch_hourly_rows", return_value=rows):
        df = pp.load_hourly("AQI01B", hours=48, now=now)
    assert df.index[-1] == pd.Timestamp("2026-03-01 07:00")
    assert df["tsp"].iloc[-2:].isna().all()
    assert df["tsp"].iloc[:6].notna().all()


def test_load_hourly_excludes_the_hour_in_progress():
    hours = pd.date_range("2026-03-01 00:00", periods=6, freq="h")   # 00..05
    now = pd.Timestamp("2026-03-01 05:30", tz="UTC")
    with patch("app.models.preprocessor.fetch_hourly_rows", return_value=_rows(hours)):
        df = pp.load_hourly("AQI01B", hours=48, now=now)
    assert df.index[-1] == pd.Timestamp("2026-03-01 04:00")
