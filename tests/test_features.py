import numpy as np
import pandas as pd
import pytest

from app.config import (
    FEATURE_COLS, LOG_SCALE_COLS, LOCAL_UTC_OFFSET_HOURS, MAX_ANCHOR_AGE_HOURS,
    N_FORECAST_HOURS,
)
from app.models.features import (
    META_COLS, build_rows, from_model_space, origin_features, to_model_space,
)
from tests.conftest import make_hourly


def test_model_space_roundtrip_and_log_only_on_skewed_columns(hourly_df):
    z = to_model_space(hourly_df)
    assert np.allclose(z["tsp"], np.log1p(hourly_df["tsp"]))
    assert np.allclose(z["temp"], hourly_df["temp"])          # linier, tidak di-log
    for col in FEATURE_COLS:
        assert np.allclose(from_model_space(z[col].values, col), hourly_df[col].values)
    assert "tsp" in LOG_SCALE_COLS and "temp" not in LOG_SCALE_COLS


def test_model_space_keeps_nan(hourly_df):
    df = hourly_df.copy()
    df.iloc[5, df.columns.get_loc("tsp")] = np.nan
    assert np.isnan(to_model_space(df)["tsp"].iloc[5])


def test_rows_per_origin_and_meta_columns(hourly_df):
    z = to_model_space(hourly_df)
    X, meta = build_rows(z, "tsp")
    assert list(meta.columns) == META_COLS
    assert len(X) == len(meta) == len(z) * N_FORECAST_HOURS
    assert sorted(meta["h"].unique()) == list(range(1, N_FORECAST_HOURS + 1))
    assert not {"t", "anchor", "z_true", "delta"} & set(X.columns)


def test_target_is_delta_to_anchor(hourly_df):
    z = to_model_space(hourly_df)
    _, meta = build_rows(z, "tsp")
    t = z.index[100]
    for h in (1, 6):
        row = meta[(meta["t"] == t) & (meta["h"] == h)].iloc[0]
        assert row["anchor"] == pytest.approx(z["tsp"].loc[t])
        assert row["z_true"] == pytest.approx(z["tsp"].loc[t + pd.Timedelta(hours=h)])
        assert row["delta"] == pytest.approx(row["z_true"] - row["anchor"])


def test_last_origins_have_no_target_yet(hourly_df):
    z = to_model_space(hourly_df)
    _, meta = build_rows(z, "tsp")
    last = meta[meta["t"] == z.index[-1]]
    assert len(last) == N_FORECAST_HOURS
    assert last["delta"].isna().all()


def test_no_lookahead_features_ignore_everything_after_origin(hourly_df):
    """Inti keabsahan evaluasi: fitur baris (t, h) tidak boleh berubah bila data
    SETELAH jam t diubah. Kalau berubah, model mengintip masa depan."""
    z = to_model_space(hourly_df)
    cut = z.index[500]
    tampered = z.copy()
    tampered.loc[tampered.index > cut] = tampered.loc[tampered.index > cut] + 5.0

    X1, m1 = build_rows(z, "tsp")
    X2, m2 = build_rows(tampered, "tsp")
    before = (m1["t"] <= cut).values
    pd.testing.assert_frame_equal(X1[before], X2[before])
    # …sedangkan target (yang memang masa depan) harus ikut berubah.
    assert not np.allclose(m1.loc[before, "delta"].dropna().tail(6),
                           m2.loc[before, "delta"].dropna().tail(6))


def test_seasonal_feature_is_same_clock_hour_yesterday(hourly_df):
    z = to_model_space(hourly_df)
    X, meta = build_rows(z, "tsp")
    t = z.index[200]
    for h in (1, 4, 6):
        i = meta.index[(meta["t"] == t) & (meta["h"] == h)][0]
        expected = z["tsp"].loc[t + pd.Timedelta(hours=h - 24)] - z["tsp"].loc[t]
        assert X.loc[i, "sn24"] == pytest.approx(expected)
        yesterday_change = z["tsp"].loc[t + pd.Timedelta(hours=h - 24)] - z["tsp"].loc[t - pd.Timedelta(hours=24)]
        assert X.loc[i, "dy24"] == pytest.approx(yesterday_change)


def test_calendar_features_use_local_target_hour(hourly_df):
    z = to_model_space(hourly_df)
    X, meta = build_rows(z, "tsp")
    t = pd.Timestamp("2026-01-10 20:00")           # UTC
    i = meta.index[(meta["t"] == t) & (meta["h"] == 3)][0]
    assert X.loc[i, "hour_o"] == (20 + LOCAL_UTC_OFFSET_HOURS) % 24
    assert X.loc[i, "hour_t"] == (20 + LOCAL_UTC_OFFSET_HOURS + 3) % 24
    local_target = t + pd.Timedelta(hours=LOCAL_UTC_OFFSET_HOURS + 3)
    assert X.loc[i, "dow"] == local_target.dayofweek
    assert X.loc[i, "h"] == 3


def test_anchor_may_be_a_few_hours_old_but_not_stale(hourly_df):
    df = hourly_df.copy()
    gap_start = 300
    df.iloc[gap_start:gap_start + 10, df.columns.get_loc("tsp")] = np.nan
    z = to_model_space(df)
    _, meta = build_rows(z, "tsp")
    origins = set(meta["t"])
    last_obs = z["tsp"].iloc[gap_start - 1]
    for k in range(10):
        t = z.index[gap_start + k]
        if k < MAX_ANCHOR_AGE_HOURS:
            assert t in origins
            assert meta[meta["t"] == t]["anchor"].iloc[0] == pytest.approx(last_obs)
        else:
            assert t not in origins            # data basi: tidak ada baris, tidak ada prakiraan


def test_missing_target_gives_nan_delta_not_a_filled_value(hourly_df):
    df = hourly_df.copy()
    df.iloc[400, df.columns.get_loc("tsp")] = np.nan
    _, meta = build_rows(to_model_space(df), "tsp")
    row = meta[(meta["t"] == df.index[398]) & (meta["h"] == 2)].iloc[0]
    assert np.isnan(row["delta"]) and np.isnan(row["z_true"])


def test_origins_filter_returns_only_requested_hour(hourly_df):
    z = to_model_space(hourly_df)
    last = z.index[-1]
    X, meta = build_rows(z, "tsp", origins=pd.DatetimeIndex([last]))
    assert len(X) == N_FORECAST_HOURS
    assert (meta["t"] == last).all()
    assert meta["h"].tolist() == list(range(1, N_FORECAST_HOURS + 1))
    # Fitur jam terakhir identik dengan yang dihasilkan tanpa filter.
    X_all, meta_all = build_rows(z, "tsp")
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True), X_all[(meta_all["t"] == last).values].reset_index(drop=True))


def test_all_nan_context_column_is_left_out():
    """Sensor tanpa modul cuaca: suhu kosong total, tidak boleh jadi fitur."""
    df = make_hourly(days=20)
    df["temp"] = np.nan
    feats = origin_features(to_model_space(df), "tsp")
    assert not [c for c in feats.columns if c.startswith("temp_")]
    assert "humidity_lvl" in feats.columns


def test_target_is_not_its_own_context():
    feats = origin_features(to_model_space(make_hourly(days=20)), "tsp")
    assert not [c for c in feats.columns if c.startswith("tsp_")]
    assert "pm_10_lvl" in feats.columns


def test_predict_window_reproduces_training_features():
    """Train/serve skew: saat prediksi hanya PREDICT_HISTORY_HOURS terakhir yang
    dimuat, saat latih seluruh riwayat. Fitur jam asal terakhir harus sama persis
    di keduanya — kalau tidak, model diberi masukan yang tak pernah dilatihkan."""
    from app.config import PREDICT_HISTORY_HOURS
    df = make_hourly(days=40)
    last = pd.DatetimeIndex([df.index[-1]])
    for target in ("tsp", "humidity"):
        X_full, _ = build_rows(to_model_space(df), target, origins=last)
        X_win, _ = build_rows(to_model_space(df.iloc[-PREDICT_HISTORY_HOURS:]), target, origins=last)
        pd.testing.assert_frame_equal(X_full.reset_index(drop=True), X_win.reset_index(drop=True))
        assert X_win.notna().all().all()          # tidak ada fitur yang kehabisan riwayat
