from fastapi import APIRouter, Depends, Query
from app.auth import require_api_key
from app.db import get_all_uids, get_latest_predictions, get_all_latest_predictions, get_prediction_history
from app.services.predict import predict_sensor

router = APIRouter(tags=["prediction"])


@router.post("/predict/all", dependencies=[Depends(require_api_key)])
def predict_all():
    uids = get_all_uids()
    return {"results": [predict_sensor(uid) for uid in uids]}


@router.post("/predict/{uid}", dependencies=[Depends(require_api_key)])
def predict_one(uid: str):
    return predict_sensor(uid)


@router.get("/predictions/all")
def get_all_predictions():
    return get_all_latest_predictions()


@router.get("/predictions/{uid}/history")
def get_history(uid: str, days: int = Query(default=7, ge=1, le=90)):
    return get_prediction_history(uid, days=days)


@router.get("/predictions/{uid}")
def get_predictions(uid: str):
    return {"uid": uid, "predictions": get_latest_predictions(uid)}
