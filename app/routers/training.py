from fastapi import APIRouter, BackgroundTasks, Depends
from app.auth import require_api_key
from app.db import get_all_uids
from app.services.retrain import retrain_sensor

router = APIRouter(tags=["training"])


@router.post("/retrain/all", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_all(background_tasks: BackgroundTasks):
    uids = get_all_uids()
    for uid in uids:
        background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {len(uids)} sensors"}


@router.post("/retrain/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
def retrain_one(uid: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_sensor, uid)
    return {"message": f"Retrain queued for {uid}"}


@router.post("/tune/{uid}", status_code=202, dependencies=[Depends(require_api_key)])
async def trigger_tune(uid: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_sensor, uid, run_tuning=True)
    return {"status": "accepted", "uid": uid}
