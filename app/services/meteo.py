import logging
import math
import httpx
import pandas as pd

logger = logging.getLogger(__name__)


def _build_meteo_df(data: dict) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    raw_times = pd.to_datetime(hourly["time"])
    times = raw_times.tz_convert(None) if raw_times.tz is not None else raw_times
    wind_speeds  = pd.Series(hourly["wind_speed_10m"]).fillna(0.0).values
    wind_dirs    = pd.Series(hourly["wind_direction_10m"]).fillna(0.0).values
    cloudcovers  = pd.Series(hourly["cloudcover"]).fillna(0.0).values
    return pd.DataFrame(
        {
            "meteo_wind_speed":   [min(s / 20.0, 1.0) for s in wind_speeds],
            "meteo_wind_dir_sin": [math.sin(math.radians(d)) for d in wind_dirs],
            "meteo_wind_dir_cos": [math.cos(math.radians(d)) for d in wind_dirs],
            "meteo_cloudcover":   [min(c / 100.0, 1.0) for c in cloudcovers],
        },
        index=times,
    )


def _reindex_and_warn(df: pd.DataFrame, timestamps: pd.DatetimeIndex, context: str) -> pd.DataFrame:
    reindexed = df.reindex(timestamps, method="nearest", tolerance=pd.Timedelta(minutes=61))
    n_missing = int(reindexed.isnull().any(axis=1).sum())
    if n_missing > len(timestamps) * 0.1:
        logger.warning(
            f"[meteo/{context}] {n_missing}/{len(timestamps)} timestamps unmatched — filling with zeros"
        )
    return reindexed.fillna(0.0)


def fetch_meteo_training(lat: float, lng: float, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch historical hourly meteo from Open-Meteo archive API (free, no key).
    Returns DataFrame indexed by timestamps with 4 normalised meteo columns.
    """
    start = timestamps.min().normalize().date().isoformat()
    end   = timestamps.max().normalize().date().isoformat()

    with httpx.Client() as client:
        resp = client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":        lat,
                "longitude":       lng,
                "start_date":      start,
                "end_date":        end,
                "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover",
                "wind_speed_unit": "ms",
                "timezone":        "UTC",
            },
            timeout=60.0,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise ValueError(f"Open-Meteo error: {data.get('reason', data)}")
    df = _build_meteo_df(data)
    return _reindex_and_warn(df, timestamps, "training")


def fetch_meteo_predict(lat: float, lng: float, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch current + recent meteo for the prediction input window.
    Uses forecast API with past_days=2 to cover any 24-hour lookback.
    Returns DataFrame with same 4 meteo columns, reindexed to timestamps.
    """
    with httpx.Client() as client:
        resp = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":        lat,
                "longitude":       lng,
                "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover",
                "wind_speed_unit": "ms",
                "timezone":        "UTC",
                "forecast_days":   1,
                "past_days":       2,
            },
            timeout=15.0,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise ValueError(f"Open-Meteo error: {data.get('reason', data)}")
    df = _build_meteo_df(data)
    return _reindex_and_warn(df, timestamps, "predict")
