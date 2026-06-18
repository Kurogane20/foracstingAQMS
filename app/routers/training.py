from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional
from app.auth import require_api_key
from app.db import get_all_uids, upsert_lat_lng
from app.services.retrain import retrain_sensor
from app.training_progress import training_progress

router = APIRouter(tags=["training"])


class RetrainBody(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None


@router.post("/retrain/all", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_all(background_tasks: BackgroundTasks):
    uids = get_all_uids()
    for uid in uids:
        background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {len(uids)} sensors"}


@router.post("/retrain/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_one(uid: str, background_tasks: BackgroundTasks, body: RetrainBody = None):
    if body and body.lat is not None and body.lng is not None:
        upsert_lat_lng(uid, body.lat, body.lng)
    background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {uid}"}


@router.post("/tune/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
async def trigger_tune(uid: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_sensor, uid, run_tuning=True)
    return {"status": "accepted", "uid": uid}


@router.get("/training/progress/{uid}", dependencies=[Depends(require_api_key)])
def get_training_progress(uid: str):
    prog = training_progress.get(uid)
    if prog is None:
        return {"phase": "idle", "percent": 0, "epoch": None, "total_epochs": None, "eta_seconds": None}
    return prog
