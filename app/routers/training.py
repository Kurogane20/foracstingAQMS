from fastapi import APIRouter
from app.db import get_all_uids
from app.services.retrain import retrain_sensor

router = APIRouter(tags=["training"])


@router.post("/retrain/all")
def retrain_all():
    uids = get_all_uids()
    return {"results": [retrain_sensor(uid) for uid in uids]}


@router.post("/retrain/{uid}")
def retrain_one(uid: str):
    return retrain_sensor(uid)
