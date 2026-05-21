from fastapi import APIRouter
from app.db import get_all_uids, get_latest_predictions
from app.services.predict import predict_sensor

router = APIRouter(tags=["prediction"])


@router.post("/predict/all")
def predict_all():
    uids = get_all_uids()
    return {"results": [predict_sensor(uid) for uid in uids]}


@router.post("/predict/{uid}")
def predict_one(uid: str):
    return predict_sensor(uid)


@router.get("/predictions/{uid}")
def get_predictions(uid: str):
    return {"uid": uid, "predictions": get_latest_predictions(uid)}
