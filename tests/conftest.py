import numpy as np
import pandas as pd
import pytest
from app.config import FEATURE_COLS, N_INPUT_HOURS, N_FORECAST_HOURS


@pytest.fixture
def sample_hourly_df():
    """24 rows of hourly sensor data, all columns present."""
    np.random.seed(42)
    index = pd.date_range("2024-01-01", periods=N_INPUT_HOURS, freq="h")
    data = {col: np.random.uniform(0, 100, N_INPUT_HOURS) for col in FEATURE_COLS}
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_minute_df():
    """1440 rows of per-minute sensor data."""
    np.random.seed(42)
    n = 1440
    timestamps = [int(pd.Timestamp("2024-01-01").timestamp()) + i * 60 for i in range(n)]
    data = {"datetime_unix": timestamps}
    for col in FEATURE_COLS:
        data[col] = np.random.uniform(0, 100, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_sequence_data():
    """Array of shape (100, 14) for sequence creation tests."""
    np.random.seed(42)
    return np.random.uniform(0, 1, (100, len(FEATURE_COLS)))
