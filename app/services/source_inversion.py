"""
Emission-source tomography ("inversi sumber emisi").

Uses months of hourly sensor TSP readings + hourly archive wind: as the wind
direction rotates hour to hour, each sensor "sees" candidate source cells from
different directions. Solving a non-negative least-squares over all hours
recovers the emission-strength map — locating pits/hauling roads mathematically
without drawing or external data.

Model per observation (sensor s, hour t):
    TSP[s,t] ≈ bg[s] + Σ_j  K(cell_j → s | met_t) · q_j
where K is the calibrated Gaussian-plume kernel and q_j ≥ 0 the unknown
per-cell emission strength (assumed constant over the window — a reasonable
approximation for a continuously operating mine).
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from app.db import (
    get_hourly_tsp,
    get_model_status,
    replace_emission_sources,
)
from app.services.dispersion import (
    _plume_conc,
    _rotate_to_plume,
    _stability_class,
    _washout_factor,
)

logger = logging.getLogger(__name__)

CELL_DEG   = 0.02    # candidate-cell size (~2.2 km)
EXPAND_DEG = 0.10    # search margin beyond the sensor bounding box
MAX_SOURCES = 12     # keep at most this many strongest cells
UTC_OFFSET  = 8      # WITA


def _fetch_archive_wind(
    lat: float, lng: float, days: int, end_dt: datetime | None = None
) -> dict[int, dict]:
    """Hourly wind/cloud/precip for the window, keyed by unix hour (UTC).
    Capped 6 days back because the ERA5 archive lags real time."""
    cap = datetime.now(timezone.utc) - timedelta(days=6)
    end_ref = min(end_dt, cap) if end_dt else cap
    end   = end_ref.date()
    start = end - timedelta(days=days)
    resp = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":        lat,
            "longitude":       lng,
            "hourly":          "wind_speed_10m,wind_direction_10m,cloudcover,precipitation",
            "wind_speed_unit": "ms",
            "timezone":        "UTC",
            "start_date":      start.isoformat(),
            "end_date":        end.isoformat(),
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    h = resp.json().get("hourly", {})
    out: dict[int, dict] = {}
    for i, t in enumerate(h.get("time", [])):
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        speed = h["wind_speed_10m"][i]
        wdir  = h["wind_direction_10m"][i]
        if speed is None or wdir is None:
            continue
        out[int(dt.timestamp())] = {
            "speed":      max(float(speed), 0.5),
            "direction":  float(wdir),
            "cloudcover": float(h["cloudcover"][i] or 50.0),
            "precip":     float(h["precipitation"][i] or 0.0),
            "hour":       (dt.hour + UTC_OFFSET) % 24,   # local clock for stability class
        }
    return out


def run_source_inversion(days: int = 45, reg: float = 0.02) -> dict:
    """
    Solve for the emission-strength map and store the strongest cells as the
    active emission sources. Returns a summary dict (also used by the API).
    """
    from scipy.optimize import lsq_linear

    # ── Sensors (uid + lat/lng from model_metadata) ──────────────
    # Only production-ready sensors, and only those inside the site cluster —
    # test UIDs with junk coordinates would explode the candidate grid.
    candidates = [
        {"uid": m["uid"], "lat": float(m["lat"]), "lng": float(m["lng"])}
        for m in get_model_status()
        if m.get("lat") is not None and m.get("lng") is not None
        and m.get("status") == "ready"
    ]
    if len(candidates) < 4:
        raise ValueError(f"Butuh ≥4 sensor ready ber-koordinat, dapat {len(candidates)}")
    med_lat = float(np.median([s["lat"] for s in candidates]))
    med_lng = float(np.median([s["lng"] for s in candidates]))
    sensors = [
        s for s in candidates
        if abs(s["lat"] - med_lat) < 0.5 and abs(s["lng"] - med_lng) < 0.5
    ]
    if len(sensors) < 4:
        raise ValueError(f"Butuh ≥4 sensor dalam cluster situs, dapat {len(sensors)}")
    n_sens = len(sensors)

    # ── Candidate source cells ───────────────────────────────────
    lat_min = min(s["lat"] for s in sensors) - EXPAND_DEG
    lat_max = max(s["lat"] for s in sensors) + EXPAND_DEG
    lng_min = min(s["lng"] for s in sensors) - EXPAND_DEG
    lng_max = max(s["lng"] for s in sensors) + EXPAND_DEG
    cell_lats = np.arange(lat_min + CELL_DEG / 2, lat_max, CELL_DEG)
    cell_lngs = np.arange(lng_min + CELL_DEG / 2, lng_max, CELL_DEG)
    cells = np.array([(la, lo) for la in cell_lats for lo in cell_lngs])
    n_cells = len(cells)
    logger.info("Inversi: %d sensor, %d sel kandidat", n_sens, n_cells)

    # ── Observations & met ───────────────────────────────────────
    # Fetch a generous TSP window first, then anchor the wind window to the
    # LAST hour that actually has data (replicas may lag behind real time).
    tsp: list[dict[int, float]] = []
    for s in sensors:
        tsp.append(get_hourly_tsp(s["uid"], hours=240 * 24))
    latest_ts = max((max(t.keys()) for t in tsp if t), default=None)
    if latest_ts is None:
        raise ValueError("Tidak ada data TSP sama sekali di jendela 240 hari")
    latest_dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)

    center_lat = float(np.mean([s["lat"] for s in sensors]))
    center_lng = float(np.mean([s["lng"] for s in sensors]))
    wind = _fetch_archive_wind(center_lat, center_lng, days, end_dt=latest_dt)

    # ── Build the linear system ──────────────────────────────────
    rows: list[np.ndarray] = []
    obs:  list[float] = []
    hours_used = 0

    for hour_unix, w in sorted(wind.items()):
        # require at least half the network reporting this hour
        have = [i for i in range(n_sens) if hour_unix in tsp[i]]
        if len(have) < max(4, n_sens // 2):
            continue
        stab = _stability_class(w["speed"], w["cloudcover"], w["hour"])
        wash = _washout_factor(w["precip"])
        hours_used += 1

        for i in have:
            s = sensors[i]
            dlat = s["lat"] - cells[:, 0]
            dlng = s["lng"] - cells[:, 1]
            x_down, y_cross = _rotate_to_plume(dlat, dlng, w["direction"], s["lat"])
            kernel = _plume_conc(x_down, y_cross, 1.0, w["speed"], stab) * wash

            row = np.zeros(n_cells + n_sens)
            row[:n_cells] = kernel
            row[n_cells + i] = 1.0            # per-sensor background term
            rows.append(row)
            obs.append(tsp[i][hour_unix])

    if len(obs) < 200:
        raise ValueError(f"Observasi terlalu sedikit untuk inversi: {len(obs)}")

    A = np.vstack(rows)
    b = np.array(obs)

    # Drop cells no observation ever "saw" (all-zero kernel columns)
    col_max = A[:, :n_cells].max(axis=0)
    keep = np.where(col_max > 0)[0]
    A_cells = A[:, keep] / col_max[keep]      # column-normalize for conditioning
    A_bg    = A[:, n_cells:]
    A_full  = np.hstack([A_cells, A_bg])

    # Tikhonov regularization on the cell columns only
    lam = reg * float(np.sqrt(np.mean(b ** 2)))
    reg_block = np.hstack([np.eye(len(keep)) * lam, np.zeros((len(keep), n_sens))])
    A_solve = np.vstack([A_full, reg_block])
    b_solve = np.concatenate([b, np.zeros(len(keep))])

    logger.info("Inversi: sistem %s, %d jam terpakai", A_solve.shape, hours_used)
    res = lsq_linear(A_solve, b_solve, bounds=(0, np.inf),
                     lsmr_tol="auto", max_iter=300, verbose=0)

    x_cells_scaled = res.x[: len(keep)]
    x_bg           = res.x[len(keep):]
    x_cells = x_cells_scaled / col_max[keep]  # back to physical scale

    # Diagnostics: variance explained vs background-only baseline
    pred = A_full @ res.x
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - np.mean(b)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Extract the strongest cells as sources ───────────────────
    strengths = x_cells_scaled  # normalized scale is fine for relative weights
    order = np.argsort(strengths)[::-1]
    total = float(strengths.sum()) or 1.0
    chosen = []
    cum = 0.0
    for idx in order[:MAX_SOURCES]:
        if strengths[idx] <= 0:
            break
        cum += float(strengths[idx])
        chosen.append(idx)
        if cum / total >= 0.95:
            break

    max_strength = float(strengths[chosen[0]]) if chosen else 1.0
    source_rows = [
        {
            "lat":      round(float(cells[keep[idx], 0]), 5),
            "lng":      round(float(cells[keep[idx], 1]), 5),
            "strength": round(float(strengths[idx]) / max_strength * 100.0, 2),
            "cell_deg": CELL_DEG,
        }
        for idx in chosen
    ]

    run_at = datetime.now(timezone.utc).replace(tzinfo=None)
    replace_emission_sources(source_rows, method="inversion", run_at=run_at)

    summary = {
        "status":       "ok",
        "days":         days,
        "hours_used":   hours_used,
        "observations": len(obs),
        "cells_tested": n_cells,
        "r2":           round(r2, 4),
        "background_per_sensor": {
            sensors[i]["uid"]: round(float(x_bg[i]), 1) for i in range(n_sens)
        },
        "sources":      source_rows,
        "run_at":       run_at.isoformat(),
    }
    logger.info("Inversi selesai: R²=%.3f, %d sumber", r2, len(source_rows))
    return summary
