import asyncio
import math
from datetime import datetime, timezone

import httpx
import numpy as np

M_PER_LAT = 111_000.0  # meters per degree latitude

# Pasquill-Gifford (ay, az) dispersion coefficients by stability class
_PG: dict[str, tuple[float, float]] = {
    "A": (0.22, 0.20),
    "B": (0.16, 0.12),
    "C": (0.11, 0.08),
    "D": (0.08, 0.06),
    "E": (0.06, 0.03),
    "F": (0.04, 0.016),
}


def _stability_class(
    wind_speed: float,
    cloudcover: float = 50.0,
    local_hour: int = 12,
) -> str:
    """
    Pasquill-Gifford atmospheric stability class (A–F).
    Args:
        wind_speed: m/s at 10m
        cloudcover: 0–100 % cloud cover (Open-Meteo)
        local_hour: 0–23 local clock hour at sensor location
    Day = 06:00–17:59; insolation proxy: cloudcover < 30 strong, 30–70 moderate, > 70 slight.
    """
    is_day = 6 <= local_hour < 18

    if is_day:
        if cloudcover < 30:
            # strong insolation
            if wind_speed < 2:  return "A"
            if wind_speed < 5:  return "B"
            return "C"
        elif cloudcover < 70:
            # moderate insolation
            if wind_speed < 3:  return "B"
            if wind_speed < 5:  return "C"
            return "D"
        else:
            # slight insolation / overcast
            if wind_speed < 5:  return "C"
            return "D"
    else:
        # nighttime: overcast threshold is 60% (stricter than daytime 70%) per PG night criteria
        if cloudcover > 60:
            return "D"
        if wind_speed < 3:  return "F"
        if wind_speed < 5:  return "E"
        return "D"


async def _fetch_wind(lat: float, lng: float, client: httpx.AsyncClient) -> dict:
    """Fetch current wind, cloudcover, and local hour from Open-Meteo (free, no key)."""
    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lng,
            "current": "wind_speed_10m,wind_direction_10m,cloudcover",
            "wind_speed_unit": "ms",
            "timezone": "auto",
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})
    time_str = current.get("time", "")       # e.g. "2024-01-15T14:00"
    local_hour = int(time_str[11:13]) if len(time_str) >= 13 else 12
    return {
        "speed":      max(float(current.get("wind_speed_10m", 1.0)), 0.5),
        "direction":  float(current.get("wind_direction_10m", 0.0)),
        "cloudcover": float(current.get("cloudcover", 50.0)),
        "hour":       local_hour,
    }


def _rotate_to_plume(
    dlat: np.ndarray,
    dlng: np.ndarray,
    wind_dir_deg: float,
    lat_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (dlat, dlng) degree offsets to (x_downwind, y_crosswind) metres.

    wind_dir_deg — meteorological FROM-direction (0 = from North, 90 = from East).
    The plume extends in the OPPOSITE direction: (wind_dir + 180) % 360.
    """
    m_per_lng = M_PER_LAT * math.cos(math.radians(lat_ref))
    dx_e = dlng * m_per_lng   # eastward metres
    dy_n = dlat * M_PER_LAT   # northward metres

    theta = math.radians((wind_dir_deg + 180.0) % 360.0)  # plume bearing from North
    x_down  = dx_e * math.sin(theta) + dy_n * math.cos(theta)
    y_cross = dx_e * math.cos(theta) - dy_n * math.sin(theta)
    return x_down, y_cross


def _plume_conc(
    x_down: np.ndarray,
    y_cross: np.ndarray,
    Q: float,
    u: float,
    stab: str,
) -> np.ndarray:
    """
    Gaussian ground-level plume concentration (arbitrary relative units).
    Only computed for downwind points (x > 50 m).
    """
    ay, az = _PG[stab]
    result = np.zeros(x_down.shape, dtype=np.float64)
    mask = x_down > 50.0

    xp = x_down[mask]
    yp = y_cross[mask]

    sigma_y = ay * xp * (1.0 + 1e-4 * xp) ** (-0.5)
    sigma_z = az * xp  # linear form; underestimates σ_z for class A at x > 500 m

    denom = math.pi * max(u, 0.5) * sigma_y * sigma_z + 1e-12
    result[mask] = (2.0 * Q / denom) * np.exp(-0.5 * (yp / (sigma_y + 1e-12)) ** 2)
    return result


async def _fetch_wind_forecast(
    lat: float, lng: float, hours: int, client: httpx.AsyncClient
) -> list[dict]:
    """
    Fetch hourly wind forecast for hour indices 0..hours from Open-Meteo.
    Returns list of {speed, direction, cloudcover, hour, label_time} dicts.
    """
    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":        lat,
            "longitude":       lng,
            "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover",
            "wind_speed_unit": "ms",
            "timezone":        "auto",
            "forecast_days":   1,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    times  = hourly.get("time", [])
    speeds = hourly.get("wind_speed_10m", [])
    dirs   = hourly.get("wind_direction_10m", [])
    clouds = hourly.get("cloudcover", [])

    result = []
    for i in range(min(hours + 1, len(times))):
        t = times[i]                                     # e.g. "2024-01-15T14:00"
        local_hour = int(t[11:13]) if len(t) >= 13 else 12
        label_time = t[11:16]      if len(t) >= 16 else "00:00"
        result.append({
            "speed":      max(float(speeds[i]) if speeds[i] is not None else 0.5, 0.5),
            "direction":  float(dirs[i]   or 0.0),
            "cloudcover": float(clouds[i] or 50.0),
            "hour":       local_hour,
            "label_time": label_time,
        })
    return result


_NEUTRAL_HOUR = {"speed": 2.0, "direction": 0.0, "cloudcover": 50.0, "hour": 12, "label_time": None}


async def compute_dispersion_forecast(sensors: list[dict], hours: int = 6) -> dict:
    """
    Compute Gaussian plume for hours 0..6, returning one frame per hour.

    Each sensor dict: {"uid", "lat", "lng", "pm25", "tsp"}

    Returns:
        {
            "frames": [
                {
                    "hour_offset": 0,
                    "label": "Sekarang  14:00",
                    "grid": [[lat, lng, intensity], ...],
                    "wind_vectors": [{"uid","lat","lng","speed","direction"}, ...]
                },
                ...  # 7 items total (H+0 through H+6)
            ]
        }
    """
    if not sensors:
        return {"frames": []}

    # Fetch hourly forecast for all sensors in parallel
    async with httpx.AsyncClient() as client:
        fetch_results = await asyncio.gather(
            *[_fetch_wind_forecast(s["lat"], s["lng"], hours, client) for s in sensors],
            return_exceptions=True,
        )

    neutral_all = [dict(_NEUTRAL_HOUR) for _ in range(hours + 1)]
    hourly_per_sensor = [
        fw if isinstance(fw, list) and fw else neutral_all
        for fw in fetch_results
    ]

    # Shared grid covering all sensors ± EXTENT degrees
    EXTENT = 0.25
    STEP   = 0.008

    lats = [s["lat"] for s in sensors]
    lngs = [s["lng"] for s in sensors]
    lat_vals = np.arange(min(lats) - EXTENT, max(lats) + EXTENT + STEP, STEP)
    lng_vals = np.arange(min(lngs) - EXTENT, max(lngs) + EXTENT + STEP, STEP)
    lat_grid, lng_grid = np.meshgrid(lat_vals, lng_vals, indexing="ij")

    frames = []
    for h in range(hours + 1):
        total = np.zeros(lat_grid.shape, dtype=np.float64)
        wind_vectors = []

        for s, hw in zip(sensors, hourly_per_sensor):
            w    = hw[h] if h < len(hw) else hw[-1]
            Q    = max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0)
            u    = w["speed"]
            stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))

            dlat = lat_grid - s["lat"]
            dlng = lng_grid - s["lng"]
            x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
            field = _plume_conc(x_down, y_cross, Q, u, stab)
            # Calibrate to physical scale: near-source peak ≈ sensor TSP (µg/m³),
            # so the summed field approximates ground-level TSP concentration.
            fpeak = field.max()
            if fpeak > 0:
                field *= Q / fpeak
            total += field

            wind_vectors.append({
                "uid":       s["uid"],
                "lat":       s["lat"],
                "lng":       s["lng"],
                "speed":     w["speed"],
                "direction": w["direction"],
            })

        peak = total.max()
        if peak > 0:
            total /= peak

        THRESHOLD = 0.02
        rows, cols = np.where(total >= THRESHOLD)
        grid = [
            [round(float(lat_grid[r, c]), 5), round(float(lng_grid[r, c]), 5), round(float(total[r, c]), 3)]
            for r, c in zip(rows, cols)
        ]

        raw_time = hourly_per_sensor[0][h].get("label_time")
        label_time = raw_time if raw_time is not None else datetime.now(timezone.utc).strftime("%H:%M")
        label = f"Sekarang {label_time}" if h == 0 else f"H+{h} {label_time}"

        frames.append({
            "hour_offset":  h,
            "label":        label,
            "grid":         grid,
            "wind_vectors": wind_vectors,
            # Peak physical concentration (µg/m³, TSP-calibrated) — multiply the
            # normalized grid intensity by this to recover absolute values.
            "max_conc":     round(float(peak), 1),
        })

    return {"frames": frames}


async def compute_dispersion(sensors: list[dict]) -> dict:
    """
    Compute Gaussian plume superposition for all sensors.

    Each sensor dict: {"uid", "lat", "lng", "pm25", "tsp"}

    Returns:
        {
            "grid": [[lat, lng, intensity], ...],   # intensity 0–1
            "wind_vectors": [{"uid", "lat", "lng", "speed", "direction"}, ...],
            "updated_at": "ISO-8601 string"
        }
    """
    if not sensors:
        return {
            "grid": [],
            "wind_vectors": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Fetch wind for all sensors in parallel
    async with httpx.AsyncClient() as client:
        wind_results = await asyncio.gather(
            *[_fetch_wind(s["lat"], s["lng"], client) for s in sensors],
            return_exceptions=True,
        )

    # Replace failed fetches with a calm-wind default
    wind_data = [
        w if isinstance(w, dict) else {"speed": 2.0, "direction": 0.0, "cloudcover": 50.0, "hour": 12}
        for w in wind_results
    ]

    wind_vectors = [
        {
            "uid":       s["uid"],
            "lat":       s["lat"],
            "lng":       s["lng"],
            "speed":     w["speed"],
            "direction": w["direction"],
        }
        for s, w in zip(sensors, wind_data)
    ]

    # Build shared grid covering all sensors ± EXTENT degrees
    EXTENT = 0.25   # ~27 km
    STEP   = 0.008  # ~900 m

    lats = [s["lat"] for s in sensors]
    lngs = [s["lng"] for s in sensors]

    lat_vals = np.arange(min(lats) - EXTENT, max(lats) + EXTENT + STEP, STEP)
    lng_vals = np.arange(min(lngs) - EXTENT, max(lngs) + EXTENT + STEP, STEP)
    lat_grid, lng_grid = np.meshgrid(lat_vals, lng_vals, indexing="ij")
    total = np.zeros(lat_grid.shape, dtype=np.float64)

    for s, w in zip(sensors, wind_data):
        Q = max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0)
        u    = w["speed"]
        stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))

        dlat = lat_grid - s["lat"]
        dlng = lng_grid - s["lng"]
        x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
        field = _plume_conc(x_down, y_cross, Q, u, stab)
        # Calibrate to physical scale (see compute_dispersion_forecast)
        fpeak = field.max()
        if fpeak > 0:
            field *= Q / fpeak
        total += field

    # Normalise 0–1
    peak = total.max()
    if peak > 0:
        total /= peak

    # Flatten, filter near-zero, round for smaller payload
    THRESHOLD = 0.02
    rows, cols = np.where(total >= THRESHOLD)
    grid = [
        [
            round(float(lat_grid[r, c]), 5),
            round(float(lng_grid[r, c]), 5),
            round(float(total[r, c]), 3),
        ]
        for r, c in zip(rows, cols)
    ]

    return {
        "grid":         grid,
        "wind_vectors": wind_vectors,
        "max_conc":     round(float(peak), 1),
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }
