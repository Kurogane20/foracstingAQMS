import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.auth import require_api_key
from app.db import get_emission_sources
from app.services.dispersion import (
    compute_dispersion,
    compute_dispersion_forecast,
    compute_dispersion_daily,
)
from app.services.source_inversion import run_source_inversion

router = APIRouter(tags=["dispersion"])


class SensorInput(BaseModel):
    uid:   str
    lat:   float
    lng:   float
    pm25:  float = Field(default=50.0, ge=0)
    tsp:   float = Field(default=100.0, ge=0)


class DispersionRequest(BaseModel):
    sensors: list[SensorInput]


@router.post("/dispersion/map", dependencies=[Depends(require_api_key)])
async def get_dispersion_map(body: DispersionRequest):
    """
    Compute Gaussian plume dust dispersal heatmap for the given sensors.
    Fetches real-time wind data from Open-Meteo.
    Returns grid of [lat, lng, intensity 0-1] + wind vectors.
    """
    return await compute_dispersion([s.model_dump() for s in body.sensors])


@router.post("/dispersion/forecast", dependencies=[Depends(require_api_key)])
async def get_dispersion_forecast(body: DispersionRequest):
    """
    Compute Gaussian plume for H+0 through H+6 using Open-Meteo hourly forecast.
    Returns {"frames": [{hour_offset, label, grid, wind_vectors}, ...]} with 7 frames.
    """
    return await compute_dispersion_forecast([s.model_dump() for s in body.sensors])


@router.post("/dispersion/daily", dependencies=[Depends(require_api_key)])
async def get_dispersion_daily(body: DispersionRequest):
    """
    24-hour average dispersion map (AERMOD-style reporting product), using the
    past-24h hourly wind and each sensor's actual hourly-mean TSP per hour.
    Returns {"grid", "wind_vectors", "max_conc", "period", "updated_at"}.
    """
    return await compute_dispersion_daily([s.model_dump() for s in body.sensors])


@router.post("/sources/calibrate", dependencies=[Depends(require_api_key)])
async def calibrate_emission_sources(days: int = 45):
    """
    Run emission-source tomography: invert months of sensor TSP + archive wind
    into an emission-strength map, storing the strongest cells as the active
    emission sources used by the dispersion endpoints. Heavy (1–3 minutes).
    """
    try:
        return await asyncio.to_thread(run_source_inversion, days)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/sources", dependencies=[Depends(require_api_key)])
async def list_emission_sources():
    """Active emission sources produced by the latest inversion run."""
    return {"sources": await asyncio.to_thread(get_emission_sources)}
