import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from app.db import (
    save_dispersion_validation,
    resolve_dispersion_actuals,
    get_hourly_tsp,
    get_emission_sources,
)

logger = logging.getLogger(__name__)

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
            "current": "wind_speed_10m,wind_direction_10m,cloudcover,precipitation",
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
        "precip":     float(current.get("precipitation", 0.0) or 0.0),
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
    sigma0_y: float = 0.0,
    sigma0_z: float = 0.0,
) -> np.ndarray:
    """
    Gaussian ground-level plume concentration (arbitrary relative units).
    Only computed for downwind points (x > 50 m).
    sigma0_y/z: initial dispersion of an AREA source (m) — prevents the
    point-source singularity from producing absurd near-field peaks.
    """
    ay, az = _PG[stab]
    result = np.zeros(x_down.shape, dtype=np.float64)
    mask = x_down > 50.0

    xp = x_down[mask]
    yp = y_cross[mask]

    sigma_y = np.sqrt((ay * xp * (1.0 + 1e-4 * xp) ** (-0.5)) ** 2 + sigma0_y ** 2)
    sigma_z = np.sqrt((az * xp) ** 2 + sigma0_z ** 2)

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
            "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover,precipitation",
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
    precip = hourly.get("precipitation", [])

    result = []
    for i in range(min(hours + 1, len(times))):
        t = times[i]                                     # e.g. "2024-01-15T14:00"
        local_hour = int(t[11:13]) if len(t) >= 13 else 12
        label_time = t[11:16]      if len(t) >= 16 else "00:00"
        result.append({
            "speed":      max(float(speeds[i]) if speeds[i] is not None else 0.5, 0.5),
            "direction":  float(dirs[i]   or 0.0),
            "cloudcover": float(clouds[i] or 50.0),
            "precip":     float(precip[i] or 0.0) if i < len(precip) else 0.0,
            "hour":       local_hour,
            "label_time": label_time,
        })
    return result


_NEUTRAL_HOUR = {"speed": 2.0, "direction": 0.0, "cloudcover": 50.0, "precip": 0.0, "hour": 12, "label_time": None}


def _washout_factor(precip_mm: float) -> float:
    """
    Rain scavenging: wet deposition removes suspended dust roughly
    exponentially with rainfall intensity. 0 mm→1.0, 1 mm→0.61, 3 mm→0.22.
    """
    return math.exp(-0.5 * max(precip_mm, 0.0))


def _plume_conc_point(
    dlat: float, dlng: float, wind_dir: float, lat_ref: float,
    Q: float, u: float, stab: str,
    sigma0_y: float = 0.0, sigma0_z: float = 0.0,
) -> float:
    """Evaluate the (uncalibrated) plume at a single lat/lng offset."""
    x, y = _rotate_to_plume(np.array([dlat]), np.array([dlng]), wind_dir, lat_ref)
    return float(_plume_conc(x, y, Q, u, stab, sigma0_y, sigma0_z)[0])


# Initial dispersion for tomography AREA sources (m): a mine pit is hundreds of
# metres wide, not a mathematical point
AREA_SIGMA0_Y = 250.0
AREA_SIGMA0_Z = 40.0


def _prepare_emitters(sources: list[dict]) -> list[dict]:
    """
    Cluster adjacent inversion cells into AREA sources, each emitted from 5
    sub-points so the near-field core is broad (like a pit complex) instead of
    12 needle plumes from isolated cells.
    Returns [{lat, lng, Q}] emitter points.
    """
    pts = [
        {"lat": float(s["lat"]), "lng": float(s["lng"]), "q": max(float(s["strength"]), 0.01)}
        for s in sources
    ]
    order = sorted(range(len(pts)), key=lambda i: -pts[i]["q"])
    used: set[int] = set()
    emitters: list[dict] = []

    for i in order:
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in order:
            if j in used:
                continue
            if abs(pts[j]["lat"] - pts[i]["lat"]) <= 0.045 and \
               abs(pts[j]["lng"] - pts[i]["lng"]) <= 0.045:
                group.append(j)
                used.add(j)

        w = sum(pts[k]["q"] for k in group)
        clat = sum(pts[k]["q"] * pts[k]["lat"] for k in group) / w
        clng = sum(pts[k]["q"] * pts[k]["lng"] for k in group) / w
        r = max(
            max((abs(pts[k]["lat"] - clat) for k in group), default=0.0),
            max((abs(pts[k]["lng"] - clng) for k in group), default=0.0),
            0.012,
        ) * 0.7

        for dlat, dlng in [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)]:
            emitters.append({"lat": clat + dlat, "lng": clng + dlng, "Q": w / 5.0})
    return emitters


def _log_validation(rows: list[dict]) -> None:
    """Persist cross-sensor validation rows + resolve past actuals (thread)."""
    try:
        save_dispersion_validation(rows)
        resolve_dispersion_actuals()
    except Exception as exc:  # never break the endpoint over logging
        logger.warning("dispersion validation logging failed: %s", exc)


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

    # Emission sources from tomography inversion (preferred). When present,
    # plumes are emitted from the RECOVERED SOURCE CELLS (pit/hauling areas)
    # instead of the sensor positions — sensors act purely as receptors.
    try:
        emission_sources = await asyncio.to_thread(get_emission_sources)
    except Exception:
        emission_sources = []

    frames = []
    validation_rows: list[dict] = []
    forecast_at = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for h in range(hours + 1):
        total = np.zeros(lat_grid.shape, dtype=np.float64)
        # Medan paralel memakai batas atas prakiraan (kuantil 0,9) sebagai kekuatan
        # sumber. Dipakai KHUSUS untuk peringatan dini, bukan untuk peta.
        #
        # Alasannya terukur: prakiraan titik dioptimalkan dengan MSE sehingga
        # menghasilkan rata-rata bersyarat — saat TSP aktual melampaui baku mutu
        # (rata-rata 431 µg/m³), prakiraan titik rata-rata hanya 128, dan hanya 4%
        # pelampauan yang terdeteksi. Untuk keputusan "perlu bertindak atau tidak",
        # kuantil atas adalah besaran yang tepat; prakiraan titik memang dirancang
        # untuk tidak pernah ekstrem.
        total_hi = np.zeros(lat_grid.shape, dtype=np.float64)
        wind_vectors = []
        hour_params = []   # per-sensor params for cross-sensor validation

        met_this_hour = []
        for s, hw in zip(sensors, hourly_per_sensor):
            w = hw[h] if h < len(hw) else hw[-1]
            met_this_hour.append(w)
            wind_vectors.append({
                "uid":       s["uid"],
                "lat":       s["lat"],
                "lng":       s["lng"],
                "speed":     w["speed"],
                "direction": w["direction"],
            })

        if emission_sources:
            # ── Source-based field (tomography mode) ─────────────
            emitters = _prepare_emitters(emission_sources)
            pt_conc = np.zeros(len(sensors))   # raw field value at each sensor
            for src in emitters:
                # Met at the source ≈ met of the nearest sensor
                dists = [
                    (src["lat"] - s["lat"]) ** 2 + (src["lng"] - s["lng"]) ** 2
                    for s in sensors
                ]
                w = met_this_hour[int(np.argmin(dists))]
                u    = w["speed"]
                stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))
                wash = _washout_factor(w.get("precip", 0.0))
                Qrel = src["Q"]

                dlat = lat_grid - src["lat"]
                dlng = lng_grid - src["lng"]
                x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], src["lat"])
                total += _plume_conc(x_down, y_cross, Qrel, u, stab,
                                     AREA_SIGMA0_Y, AREA_SIGMA0_Z) * wash

                for si, s in enumerate(sensors):
                    pt_conc[si] += wash * _plume_conc_point(
                        s["lat"] - src["lat"], s["lng"] - src["lng"],
                        w["direction"], src["lat"], Qrel, u, stab,
                        AREA_SIGMA0_Y, AREA_SIGMA0_Z,
                    )

            # Calibrate the whole field so values at the sensor locations best
            # match the sensors' (predicted) TSP — least-squares scalar fit.
            # NOTE: like an AERMOD report, this map shows the MINE INCREMENT
            # only (no ambient background), so far-field falls in low classes.
            # The fit explodes when the sensors sit outside the plume footprint
            # (denominator → 0), so clamp the resulting PEAK to a physical cap.
            obs = np.array([max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0) for s in sensors])
            denom = float(np.sum(pt_conc ** 2))
            cal = float(np.sum(pt_conc * obs) / denom) if denom > 1e-12 else 0.0
            peak_raw = float(total.max())
            peak_cap = max(float(obs.max()) * 2.0, 150.0)
            if peak_raw > 0:
                cal = min(cal, peak_cap / peak_raw) if cal > 0 else (0.5 * peak_cap) / peak_raw
            total *= cal

            if 1 <= h <= 3 and cal > 0:
                target_time = forecast_at + timedelta(hours=h)
                for si, s in enumerate(sensors):
                    validation_rows.append({
                        "target_uid":  s["uid"],
                        "forecast_at": forecast_at.replace(tzinfo=None),
                        "target_time": target_time.replace(tzinfo=None),
                        "step":        h,
                        "pred_conc":   round(float(pt_conc[si]) * cal, 2),
                    })
        else:
            # ── Sensor-as-source field ───────────────────────────
            # Sensors are INSTALLED AT the emission points (in the mine areas),
            # so emitting from the sensor position with the measured TSP as
            # source strength is physically the right representation. Area
            # sigma0 because the sensor stands in a wide pit, not at a point.
            for s, w in zip(sensors, met_this_hour):
                Q    = max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0)
                u    = w["speed"]
                stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))

                dlat = lat_grid - s["lat"]
                dlng = lng_grid - s["lng"]
                x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
                field = _plume_conc(x_down, y_cross, Q, u, stab,
                                    AREA_SIGMA0_Y, AREA_SIGMA0_Z)
                # Calibrate to physical scale: near-source peak ≈ sensor TSP
                fpeak = field.max()
                scale = (Q / fpeak) if fpeak > 0 else 0.0
                scale *= _washout_factor(w.get("precip", 0.0))
                total += field * scale

                # Bentuk spasialnya identik (plume linear terhadap Q), hanya
                # bobotnya berbeda — jadi medan p90 nyaris gratis.
                Q_hi = max(float(s.get("tsp_p90") or 0.0), Q)
                total_hi += field * scale * (Q_hi / Q)

                hour_params.append({
                    "uid": s["uid"], "lat": s["lat"], "lng": s["lng"],
                    "Q": Q, "u": u, "stab": stab,
                    "dir": w["direction"], "scale": scale,
                })

            # Cross-sensor validation (own plume excluded)
            if 1 <= h <= 3:
                target_time = forecast_at + timedelta(hours=h)
                for tgt in hour_params:
                    conc = 0.0
                    for src in hour_params:
                        if src["uid"] == tgt["uid"] or src["scale"] <= 0:
                            continue
                        conc += src["scale"] * _plume_conc_point(
                            tgt["lat"] - src["lat"], tgt["lng"] - src["lng"],
                            src["dir"], src["lat"], src["Q"], src["u"], src["stab"],
                            AREA_SIGMA0_Y, AREA_SIGMA0_Z,
                        )
                    validation_rows.append({
                        "target_uid":  tgt["uid"],
                        "forecast_at": forecast_at.replace(tzinfo=None),
                        "target_time": target_time.replace(tzinfo=None),
                        "step":        h,
                        "pred_conc":   round(conc, 2),
                    })

        peak = total.max()
        peak_hi = float(total_hi.max())
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
            # Versi kuantil-0,9 dari besaran yang sama, untuk ambang peringatan.
            # Nol bila batas atas prakiraan tidak tersedia (mode inversi, atau
            # `tsp_p90` tidak dikirim) — klien harus kembali ke `max_conc`.
            "max_conc_p90": round(peak_hi, 1) if peak_hi > peak else 0.0,
        })

    # Fire-and-forget: log validation rows + resolve past actuals off-thread
    if validation_rows:
        asyncio.get_running_loop().run_in_executor(None, _log_validation, validation_rows)

    return {
        "frames":      frames,
        "source_mode": "inversion" if emission_sources else "sensor",
        "sources": [
            {"lat": float(s["lat"]), "lng": float(s["lng"]),
             "strength": float(s["strength"]), "cell_deg": float(s["cell_deg"])}
            for s in emission_sources
        ],
    }


async def _fetch_wind_past24(lat: float, lng: float, client: httpx.AsyncClient) -> list[dict]:
    """Past-24-hour hourly wind/cloud/precip from Open-Meteo (past_days=1)."""
    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":        lat,
            "longitude":       lng,
            "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover,precipitation",
            "wind_speed_unit": "ms",
            "timezone":        "auto",
            "past_days":       1,
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
    precip = hourly.get("precipitation", [])

    # Keep only hours that have already passed (past_days includes future too)
    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    for i, t in enumerate(times):
        rows.append({
            "time":       t,
            "speed":      max(float(speeds[i]) if speeds[i] is not None else 0.5, 0.5),
            "direction":  float(dirs[i]   or 0.0),
            "cloudcover": float(clouds[i] or 50.0),
            "precip":     float(precip[i] or 0.0) if i < len(precip) else 0.0,
            "hour":       int(t[11:13]) if len(t) >= 13 else 12,
        })
    # Times are local; the array is chronological — the past 24 h are simply the
    # 24 entries ending at "now" position. past_days=1 → first 24 entries are
    # yesterday; entries beyond local-now are forecast. Use a conservative cut:
    # drop the trailing forecast hours by keeping the first 24 + hours elapsed
    # today, then take the last 24.
    elapsed_today = now_utc_hour.hour + 1  # coarse; local offset differences are tolerable here
    usable = rows[: min(len(rows), 24 + elapsed_today)]
    return usable[-24:] if len(usable) >= 24 else usable


async def compute_dispersion_daily(sensors: list[dict]) -> dict:
    """
    24-hour average Gaussian plume map (AERMOD-style reporting product).
    Uses past-24h hourly wind AND each sensor's actual hourly-mean TSP as the
    emission proxy for that hour.

    Returns {"grid": [[lat,lng,intensity],...], "wind_vectors": [...],
             "max_conc": float, "period": {"start","end"}, "updated_at": iso}
    """
    if not sensors:
        return {"grid": [], "wind_vectors": [], "max_conc": 0.0,
                "period": None, "updated_at": datetime.now(timezone.utc).isoformat()}

    async with httpx.AsyncClient() as client:
        wind_results = await asyncio.gather(
            *[_fetch_wind_past24(s["lat"], s["lng"], client) for s in sensors],
            return_exceptions=True,
        )
    wind_per_sensor = [
        w if isinstance(w, list) and w else [dict(_NEUTRAL_HOUR) for _ in range(24)]
        for w in wind_results
    ]

    # Actual hourly TSP per sensor (blocking DB reads off-thread)
    tsp_results = await asyncio.gather(
        *[asyncio.to_thread(get_hourly_tsp, s["uid"], 26) for s in sensors],
        return_exceptions=True,
    )
    tsp_per_sensor = [t if isinstance(t, dict) else {} for t in tsp_results]

    EXTENT = 0.25
    STEP   = 0.008
    lats = [s["lat"] for s in sensors]
    lngs = [s["lng"] for s in sensors]
    lat_vals = np.arange(min(lats) - EXTENT, max(lats) + EXTENT + STEP, STEP)
    lng_vals = np.arange(min(lngs) - EXTENT, max(lngs) + EXTENT + STEP, STEP)
    lat_grid, lng_grid = np.meshgrid(lat_vals, lng_vals, indexing="ij")

    # Prefer tomography-recovered emission sources (same as the forecast path)
    try:
        emission_sources = await asyncio.to_thread(get_emission_sources)
    except Exception:
        emission_sources = []

    now_hour = int(datetime.now(timezone.utc).timestamp() // 3600 * 3600)
    total_sum = np.zeros(lat_grid.shape, dtype=np.float64)
    n_hours = 0

    for h_back in range(24):
        hour_unix = now_hour - (23 - h_back) * 3600
        hour_field = np.zeros(lat_grid.shape, dtype=np.float64)

        if emission_sources:
            # ── Source-based hourly field, calibrated to that hour's obs ──
            emitters = _prepare_emitters(emission_sources)
            pt_conc = np.zeros(len(sensors))
            for src in emitters:
                dists = [
                    (src["lat"] - s["lat"]) ** 2 + (src["lng"] - s["lng"]) ** 2
                    for s in sensors
                ]
                si_near = int(np.argmin(dists))
                winds = wind_per_sensor[si_near]
                w = winds[h_back] if h_back < len(winds) else winds[-1]
                u    = w["speed"]
                stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))
                wash = _washout_factor(w.get("precip", 0.0))
                Qrel = src["Q"]

                dlat = lat_grid - src["lat"]
                dlng = lng_grid - src["lng"]
                x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], src["lat"])
                hour_field += _plume_conc(x_down, y_cross, Qrel, u, stab,
                                          AREA_SIGMA0_Y, AREA_SIGMA0_Z) * wash

                for si, s in enumerate(sensors):
                    pt_conc[si] += wash * _plume_conc_point(
                        s["lat"] - src["lat"], s["lng"] - src["lng"],
                        w["direction"], src["lat"], Qrel, u, stab,
                        AREA_SIGMA0_Y, AREA_SIGMA0_Z,
                    )

            # Calibrate against this hour's ACTUAL sensor TSP where available
            # (mine increment only — no ambient background, AERMOD style).
            # Peak clamped: the scalar fit explodes when sensors are upwind.
            obs = np.array([
                float(tsp_per_sensor[si].get(hour_unix)
                      or max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0))
                for si, s in enumerate(sensors)
            ])
            denom = float(np.sum(pt_conc ** 2))
            cal = float(np.sum(pt_conc * obs) / denom) if denom > 1e-12 else 0.0
            peak_raw = float(hour_field.max())
            peak_cap = max(float(obs.max()) * 2.0, 150.0)
            if peak_raw > 0:
                cal = min(cal, peak_cap / peak_raw) if cal > 0 else (0.5 * peak_cap) / peak_raw
            hour_field *= cal
        else:
            # ── Sensor-as-source hourly field (sensors AT emission points) ──
            for si, s in enumerate(sensors):
                winds = wind_per_sensor[si]
                w = winds[h_back] if h_back < len(winds) else winds[-1]
                Q = tsp_per_sensor[si].get(hour_unix)
                if Q is None:
                    Q = max(float(s.get("tsp", 0) or s.get("pm25", 50.0)), 1.0)
                Q = max(float(Q), 1.0)
                u    = w["speed"]
                stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))

                dlat = lat_grid - s["lat"]
                dlng = lng_grid - s["lng"]
                x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
                field = _plume_conc(x_down, y_cross, Q, u, stab,
                                    AREA_SIGMA0_Y, AREA_SIGMA0_Z)
                fpeak = field.max()
                scale = (Q / fpeak) if fpeak > 0 else 0.0
                scale *= _washout_factor(w.get("precip", 0.0))
                hour_field += field * scale

        total_sum += hour_field
        n_hours += 1

    total = total_sum / max(n_hours, 1)
    peak = float(total.max())
    if peak > 0:
        total /= peak

    THRESHOLD = 0.02
    rows_i, cols_i = np.where(total >= THRESHOLD)
    grid = [
        [round(float(lat_grid[r, c]), 5), round(float(lng_grid[r, c]), 5), round(float(total[r, c]), 3)]
        for r, c in zip(rows_i, cols_i)
    ]

    # Mean wind per sensor (vector average) for display arrows
    wind_vectors = []
    for si, s in enumerate(sensors):
        winds = wind_per_sensor[si]
        us = np.array([w["speed"] for w in winds])
        ds = np.deg2rad(np.array([w["direction"] for w in winds]))
        mean_e = float(np.mean(us * np.sin(ds)))
        mean_n = float(np.mean(us * np.cos(ds)))
        wind_vectors.append({
            "uid": s["uid"], "lat": s["lat"], "lng": s["lng"],
            "speed": round(float(np.hypot(mean_e, mean_n)), 2),
            "direction": round(float((math.degrees(math.atan2(mean_e, mean_n)) + 360) % 360), 1),
        })

    start_dt = datetime.fromtimestamp(now_hour - 23 * 3600, tz=timezone.utc)
    end_dt   = datetime.fromtimestamp(now_hour + 3600, tz=timezone.utc)
    return {
        "grid":         grid,
        "wind_vectors": wind_vectors,
        "max_conc":     round(peak, 1),
        "period":       {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }


def _fetch_archive_wind_center(lat: float, lng: float, days: int, end_dt: datetime | None = None) -> dict[int, dict]:
    """Hourly archive wind for one representative point, keyed by unix hour.
    (Sync httpx — call via to_thread. ERA5 archive lags ~5 days.)"""
    import httpx as _httpx
    cap = datetime.now(timezone.utc) - timedelta(days=6)
    end_ref = min(end_dt, cap) if end_dt else cap
    end = end_ref.date()
    start = end - timedelta(days=days)
    resp = _httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat, "longitude": lng,
            "hourly": "wind_speed_10m,wind_direction_10m,cloudcover,precipitation",
            "wind_speed_unit": "ms", "timezone": "UTC",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    h = resp.json().get("hourly", {})
    out: dict[int, dict] = {}
    for i, t in enumerate(h.get("time", [])):
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        speed = h["wind_speed_10m"][i]
        wdir = h["wind_direction_10m"][i]
        if speed is None or wdir is None:
            continue
        out[int(dt.timestamp())] = {
            "speed": max(float(speed), 0.5),
            "direction": float(wdir),
            "cloudcover": float(h["cloudcover"][i] or 50.0),
            "precip": float(h["precipitation"][i] or 0.0),
            "hour": (dt.hour + 8) % 24,   # WITA local clock
        }
    return out


# Cache: the high-24h envelope is expensive and changes slowly
_high24_cache: dict = {"key": None, "ts": 0.0, "data": None}
_HIGH24_TTL = 6 * 3600.0


async def compute_dispersion_high24(sensors: list[dict], days: int = 30) -> dict:
    """
    AERMOD-style "HIGH 1ST HIGH 24-HR" map: run the plume model over the past
    `days` of archive wind + actual hourly TSP, take 24-hour rolling averages,
    and keep the MAXIMUM at every grid cell. Because the wind visits many
    directions over weeks, the envelope spreads in all directions — this is
    the product consultant AERMOD maps show.
    """
    import time as _time

    if not sensors:
        return {"grid": [], "wind_vectors": [], "max_conc": 0.0,
                "period": None, "updated_at": datetime.now(timezone.utc).isoformat()}

    cache_key = (days, tuple(sorted(s["uid"] for s in sensors)))
    if _high24_cache["data"] is not None and _high24_cache["key"] == cache_key \
            and _time.time() - _high24_cache["ts"] < _HIGH24_TTL:
        return _high24_cache["data"]

    # Hourly TSP per sensor (generous window; anchor to last available hour)
    tsp_per_sensor = await asyncio.gather(
        *[asyncio.to_thread(get_hourly_tsp, s["uid"], (days + 10) * 24) for s in sensors],
        return_exceptions=True,
    )
    tsp_per_sensor = [t if isinstance(t, dict) else {} for t in tsp_per_sensor]
    latest_ts = max((max(t.keys()) for t in tsp_per_sensor if t), default=None)
    latest_dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc) if latest_ts else None

    center_lat = float(np.mean([s["lat"] for s in sensors]))
    center_lng = float(np.mean([s["lng"] for s in sensors]))
    wind = await asyncio.to_thread(_fetch_archive_wind_center, center_lat, center_lng, days, latest_dt)
    if not wind:
        raise ValueError("Data angin archive kosong")

    # Median TSP fallback per sensor for hours without readings
    med_tsp = [
        float(np.median(list(t.values()))) if t else 50.0
        for t in tsp_per_sensor
    ]

    EXTENT = 0.25
    STEP = 0.008
    lats = [s["lat"] for s in sensors]
    lngs = [s["lng"] for s in sensors]
    lat_vals = np.arange(min(lats) - EXTENT, max(lats) + EXTENT + STEP, STEP)
    lng_vals = np.arange(min(lngs) - EXTENT, max(lngs) + EXTENT + STEP, STEP)
    lat_grid, lng_grid = np.meshgrid(lat_vals, lng_vals, indexing="ij")

    envelope = np.zeros(lat_grid.shape, dtype=np.float64)
    window: list[np.ndarray] = []
    rolling = np.zeros(lat_grid.shape, dtype=np.float64)

    def _hour_field_sync(w: dict, hour_unix: int) -> np.ndarray:
        field_total = np.zeros(lat_grid.shape, dtype=np.float64)
        u = w["speed"]
        stab = _stability_class(u, w.get("cloudcover", 50.0), w.get("hour", 12))
        wash = _washout_factor(w.get("precip", 0.0))
        for si, s in enumerate(sensors):
            Q = float(tsp_per_sensor[si].get(hour_unix, med_tsp[si]))
            Q = max(Q, 1.0)
            dlat = lat_grid - s["lat"]
            dlng = lng_grid - s["lng"]
            x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
            f = _plume_conc(x_down, y_cross, Q, u, stab, AREA_SIGMA0_Y, AREA_SIGMA0_Z)
            fpeak = f.max()
            if fpeak > 0:
                field_total += f * (Q / fpeak) * wash
        return field_total

    def _compute_envelope() -> np.ndarray:
        nonlocal rolling
        env = np.zeros(lat_grid.shape, dtype=np.float64)
        for hour_unix in sorted(wind.keys()):
            f = _hour_field_sync(wind[hour_unix], hour_unix)
            window.append(f)
            rolling += f
            if len(window) > 24:
                rolling -= window.pop(0)
            if len(window) == 24:
                np.maximum(env, rolling / 24.0, out=env)
        return env

    envelope = await asyncio.to_thread(_compute_envelope)

    peak = float(envelope.max())
    if peak > 0:
        envelope /= peak

    THRESHOLD = 0.02
    rows_i, cols_i = np.where(envelope >= THRESHOLD)
    grid = [
        [round(float(lat_grid[r, c]), 5), round(float(lng_grid[r, c]), 5), round(float(envelope[r, c]), 3)]
        for r, c in zip(rows_i, cols_i)
    ]

    hours_sorted = sorted(wind.keys())
    result = {
        "grid": grid,
        "wind_vectors": [
            {"uid": s["uid"], "lat": s["lat"], "lng": s["lng"], "speed": 0.0, "direction": 0.0}
            for s in sensors
        ],
        "max_conc": round(peak, 1),
        "period": {
            "start": datetime.fromtimestamp(hours_sorted[0], tz=timezone.utc).isoformat(),
            "end":   datetime.fromtimestamp(hours_sorted[-1] + 3600, tz=timezone.utc).isoformat(),
        },
        "kind": "high24",
        "days": days,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _high24_cache.update({"key": cache_key, "ts": _time.time(), "data": result})
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
        field = _plume_conc(x_down, y_cross, Q, u, stab,
                            AREA_SIGMA0_Y, AREA_SIGMA0_Z)
        # Calibrate to physical scale (see compute_dispersion_forecast)
        fpeak = field.max()
        scale = (Q / fpeak) if fpeak > 0 else 0.0
        scale *= _washout_factor(w.get("precip", 0.0))
        total += field * scale

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
