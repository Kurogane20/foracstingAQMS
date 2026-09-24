import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS
from app.models.preprocessor import resample_hourly, clean, create_sequences, detect_and_remove_anomalies


def test_resample_hourly_returns_24_rows(sample_minute_df):
    result = resample_hourly(sample_minute_df)
    assert len(result) == 24
    assert list(result.columns) == FEATURE_COLS


def test_resample_hourly_index_is_datetime(sample_minute_df):
    result = resample_hourly(sample_minute_df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_clean_fills_missing_values(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[5, 0] = np.nan
    result = clean(df)
    assert not result.isnull().any().any()


def test_clean_preserves_real_exceedance():
    """Regresi: pemotongan Tukey yang lama membuang 59% jam pelampauan baku mutu
    (pagar per sensor 89–252 µg/m³ vs baku mutu 230), sehingga model tidak pernah
    melihat kejadian yang paling perlu diprediksi. Puncak nyata harus lolos utuh."""
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({col: np.full(n, 40.0) for col in FEATURE_COLS}, index=index)
    df["temp"] = 28.0
    df["mmhg"] = 760.0
    df["humidity"] = 82.0
    df["noise"] = 55.0
    df.loc[df.index[100], "tsp"] = 420.0      # kejadian debu nyata di atas baku mutu

    result = clean(detect_and_remove_anomalies(df))

    assert result.loc[df.index[100], "tsp"] == pytest.approx(420.0)


def test_clean_rejects_physically_impossible_value(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.loc[df.index[0], "tsp"] = 999999.0     # mustahil: sensor rusak
    result = clean(detect_and_remove_anomalies(df))
    assert result.loc[df.index[0], "tsp"] < 999999.0


def test_clean_fills_column_that_is_entirely_invalid(sample_hourly_df):
    """Sebagian sensor tidak punya modul cuaca dan melaporkan nol terus-menerus.
    Kolom seperti itu harus diisi nilai cadangan, bukan menggagalkan retrain."""
    df = sample_hourly_df.copy()
    df["mmhg"] = 0.0                          # di luar batas fisik seluruhnya
    result = clean(detect_and_remove_anomalies(df))
    assert not result.isnull().any().any()
    assert result["mmhg"].iloc[0] == pytest.approx(760.0)


def test_create_sequences_output_shapes(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    n_samples = len(sample_sequence_data) - N_INPUT_HOURS - N_FORECAST_HOURS + 1
    assert X.shape == (n_samples, N_INPUT_HOURS, len(FEATURE_COLS))
    assert y.shape == (n_samples, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_create_sequences_values_are_contiguous(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    np.testing.assert_array_equal(X[0], sample_sequence_data[:N_INPUT_HOURS])
    np.testing.assert_array_equal(y[0], sample_sequence_data[N_INPUT_HOURS:N_INPUT_HOURS + N_FORECAST_HOURS])


def test_detect_and_remove_anomalies_replaces_impossible_spike(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.loc[df.index[12], "pm_25"] = 99999.0   # di luar batas fisik
    result = detect_and_remove_anomalies(df)
    assert result.loc[df.index[12], "pm_25"] < 99999.0


def test_detect_and_remove_anomalies_no_nans(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.loc[df.index[5], "pm_25"] = 99999.0
    result = detect_and_remove_anomalies(df)
    assert not result.isnull().any().any()


def test_detect_and_remove_anomalies_preserves_normal(sample_hourly_df):
    result = detect_and_remove_anomalies(sample_hourly_df)
    # Data yang seluruhnya masuk akal secara fisik tidak boleh diubah sama sekali.
    changed = (~np.isclose(result.values, sample_hourly_df.values, atol=1e-6)).sum()
    assert changed == 0


def test_detect_and_remove_anomalies_keeps_sustained_dust_event():
    """Penyaring 3-sigma yang lama menghapus kejadian debu berjam-jam karena
    kejadian nyata memang melewati ambang statistik. Batas fisik tidak boleh."""
    n = 120
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({col: np.full(n, 30.0) for col in FEATURE_COLS}, index=index)
    df["temp"] = 28.0
    df["mmhg"] = 760.0
    df["humidity"] = 82.0
    df["noise"] = 55.0
    df.loc[index[60:66], "tsp"] = 500.0        # 6 jam berdebu berat, nyata

    result = detect_and_remove_anomalies(df)

    assert result.loc[index[60:66], "tsp"].tolist() == [500.0] * 6


def test_delta_round_trip_is_lossless():
    from app.models.preprocessor import anchor_from_X, to_delta, from_delta
    from app.config import N_FEATURES
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, N_INPUT_HOURS, N_FEATURES + 8))
    y = rng.normal(size=(7, N_FORECAST_HOURS, N_FEATURES))
    anchor = anchor_from_X(X)
    np.testing.assert_allclose(from_delta(to_delta(y, anchor), anchor), y, atol=1e-12)


def test_anchor_is_last_observed_sensor_row():
    from app.models.preprocessor import anchor_from_X
    from app.config import N_FEATURES
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, N_INPUT_HOURS, N_FEATURES + 8))
    np.testing.assert_array_equal(anchor_from_X(X), X[:, -1, :N_FEATURES])


def test_zero_delta_reproduces_persistence():
    """Inti perbaikan: selisih nol HARUS menghasilkan tebakan naif 'nilai H+h =
    nilai sekarang'. Itulah yang membuat persistence jadi perilaku bawaan model
    dan mencegahnya kalah telak dari baseline seperti pada uji 1 Agu 2026."""
    from app.models.preprocessor import anchor_from_X, from_delta
    from app.config import N_FEATURES
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, N_INPUT_HOURS, N_FEATURES + 8))
    anchor = anchor_from_X(X)
    persistence = from_delta(np.zeros((5, N_FORECAST_HOURS, N_FEATURES)), anchor)
    for step in range(N_FORECAST_HOURS):
        np.testing.assert_array_equal(persistence[:, step, :], anchor)


def test_scaler_fit_rows_stops_at_last_training_row():
    """Sekuens latih terakhir menyentuh baris split-1 + jendela - 1; scaler
    tidak boleh melihat baris sesudahnya."""
    from app.models.preprocessor import scaler_fit_rows, train_split_index
    n_rows = 1000
    window = N_INPUT_HOURS + N_FORECAST_HOURS
    n_seq = n_rows - window + 1
    split = train_split_index(n_seq)
    assert scaler_fit_rows(n_rows) == split + window - 1
    assert scaler_fit_rows(n_rows) < n_rows


def test_scaler_does_not_see_validation_extremes(tmp_path, monkeypatch):
    """Regresi kebocoran: sebelumnya scaler di-fit pada seluruh deret, sehingga
    nilai ekstrem yang hanya muncul di porsi validasi ikut menentukan skala."""
    import joblib
    from app.models import preprocessor as pp
    monkeypatch.setattr(pp, "MODELS_DIR", str(tmp_path))

    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({c: np.full(n, 40.0) for c in FEATURE_COLS}, index=idx)
    df["temp"], df["mmhg"], df["humidity"], df["noise"] = 28.0, 760.0, 82.0, 55.0
    df.loc[idx[-5], "tsp"] = 900.0              # ekstrem HANYA di ujung validasi

    fit_end = pp.scaler_fit_rows(n)
    pp.normalize(df.iloc[:fit_end], "uid_x", fit=True)
    scaler = joblib.load(tmp_path / "uid_x" / "scaler.pkl")["scaler"]

    tsp_i = FEATURE_COLS.index("tsp")
    assert scaler.data_max_[tsp_i] < np.log1p(900.0)   # ekstrem validasi tak terlihat


def test_sequence_base_times_is_last_observed_hour():
    """Jam dasar tiap sekuens = jam input terakhir — persis `last_time` di jalur
    prediksi. Sebelumnya LightGBM dilatih dengan jam sintetis 2024-01-01 + i,
    sehingga fitur jam/hari/bulannya bergeser dari yang dilihat saat produksi."""
    from app.models.preprocessor import sequence_base_times
    n = 100
    index = pd.date_range("2025-10-03 07:00", periods=n, freq="h")
    data = np.arange(n, dtype=float)[:, np.newaxis]    # nilai = posisi baris
    X, _ = create_sequences(data, N_INPUT_HOURS, N_FORECAST_HOURS)
    base_times = sequence_base_times(index)
    assert len(base_times) == len(X)
    for i in (0, 17, len(X) - 1):
        assert base_times[i] == index[int(X[i, -1, 0])]


def test_preprocess_for_training_returns_real_base_times(tmp_path, monkeypatch):
    import app.db
    from app.models import preprocessor as pp
    monkeypatch.setattr(pp, "MODELS_DIR", str(tmp_path))
    n = 120
    idx = pd.date_range("2025-10-03 07:00", periods=n, freq="h")
    df = pd.DataFrame({c: np.full(n, 40.0) for c in FEATURE_COLS}, index=idx)
    df["temp"], df["mmhg"], df["humidity"], df["noise"] = 28.0, 760.0, 82.0, 55.0
    # Bentuk baris mentah t_loggers: stempel waktu unix sebagai kolom.
    df["datetime_unix"] = (idx - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    df = df.reset_index(drop=True)
    monkeypatch.setattr(pp, "fetch_sensor_data", lambda uid, hours: df)
    monkeypatch.setattr(app.db, "get_sensor_lat_lng", lambda uid: None)

    X, y, base_times = pp.preprocess_for_training("uid_x")

    assert len(base_times) == len(X) == len(y)
    assert base_times[0] == idx[N_INPUT_HOURS - 1]
    assert base_times[-1] == idx[n - N_FORECAST_HOURS - 1]
