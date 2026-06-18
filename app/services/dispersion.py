import asyncio
import math
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np

M_PER_LAT = 111_000.0  # meters per degree latitude

# Pasquill-Gifford (ay, az) dispersion coefficients by stability class
_PG: dict[str, tuple[float, float]] = {
    "B": (0.16, 0.12),
    "C": (0.11, 0.08),
    "D": (0.08, 0.06),
    "E": (0.06, 0.03),
    "F": (0.04, 0.016),
}


def _stability_class(wind_speed: float) -> str:
    if wind_speed < 2:
        return "F"
    if wind_speed < 3:
        return "E"
    if wind_speed < 5:
        return "D"
    if wind_speed < 6:
        return "C"
    return "B"


async def _fetch_wind(lat: float, lng: float, client: httpx.AsyncClient) -> dict:
    """Fetch current wind speed + direction from Open-Meteo (free, no API key)."""
    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lng,
            "current": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "timezone": "auto",
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    current = resp.json().get("current", {})
    return {
        "speed":     max(float(current.get("wind_speed_10m", 1.0)), 0.5),
        "direction": float(current.get("wind_direction_10m", 0.0)),
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
    sigma_z = az * xp

    denom = math.pi * max(u, 0.5) * sigma_y * sigma_z + 1e-12
    result[mask] = (2.0 * Q / denom) * np.exp(-0.5 * (yp / (sigma_y + 1e-12)) ** 2)
    return result


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
        w if isinstance(w, dict) else {"speed": 2.0, "direction": 0.0}
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
        stab = _stability_class(u)

        dlat = lat_grid - s["lat"]
        dlng = lng_grid - s["lng"]
        x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
        total += _plume_conc(x_down, y_cross, Q, u, stab)

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
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }
