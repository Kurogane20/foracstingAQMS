from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.auth import require_api_key
from app.services.dispersion import compute_dispersion, compute_dispersion_forecast

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
