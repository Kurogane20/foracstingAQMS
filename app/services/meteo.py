import math
import httpx
import pandas as pd


def _build_meteo_df(data: dict) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly["time"])
    wind_speeds  = [s or 0.0 for s in hourly["wind_speed_10m"]]
    wind_dirs    = [d or 0.0 for d in hourly["wind_direction_10m"]]
    cloudcovers  = [c or 0.0 for c in hourly["cloudcover"]]
    return pd.DataFrame(
        {
            "meteo_wind_speed":   [min(s / 20.0, 1.0) for s in wind_speeds],
            "meteo_wind_dir_sin": [math.sin(math.radians(d)) for d in wind_dirs],
            "meteo_wind_dir_cos": [math.cos(math.radians(d)) for d in wind_dirs],
            "meteo_cloudcover":   [c / 100.0 for c in cloudcovers],
        },
        index=times,
    )


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
                "timezone":        "auto",
            },
            timeout=60.0,
        )
    resp.raise_for_status()
    df = _build_meteo_df(resp.json())
    return df.reindex(timestamps, method="nearest", tolerance=pd.Timedelta("61min")).fillna(0.0)


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
                "timezone":        "auto",
                "forecast_days":   1,
                "past_days":       2,
            },
            timeout=15.0,
        )
    resp.raise_for_status()
    df = _build_meteo_df(resp.json())
    return df.reindex(timestamps, method="nearest", tolerance=pd.Timedelta("61min")).fillna(0.0)
