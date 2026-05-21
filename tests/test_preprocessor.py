import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS
from app.models.preprocessor import resample_hourly, clean, create_sequences


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


def test_clean_caps_outliers(sample_hourly_df):
    df = sample_hourly_df.copy()
    df.iloc[0, 0] = 999999.0
    result = clean(df)
    assert result.iloc[0, 0] < 999999.0


def test_create_sequences_output_shapes(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    n_samples = len(sample_sequence_data) - N_INPUT_HOURS - N_FORECAST_HOURS + 1
    assert X.shape == (n_samples, N_INPUT_HOURS, len(FEATURE_COLS))
    assert y.shape == (n_samples, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_create_sequences_values_are_contiguous(sample_sequence_data):
    X, y = create_sequences(sample_sequence_data, N_INPUT_HOURS, N_FORECAST_HOURS)
    np.testing.assert_array_equal(X[0], sample_sequence_data[:N_INPUT_HOURS])
    np.testing.assert_array_equal(y[0], sample_sequence_data[N_INPUT_HOURS:N_INPUT_HOURS + N_FORECAST_HOURS])
