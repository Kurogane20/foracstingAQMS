from fastapi import APIRouter, Depends
from app.auth import require_api_key
from app.db import get_all_uids
from app.services.accuracy import resolve_prediction_actuals, compute_accuracy

router = APIRouter(tags=["accuracy"])


@router.get("/accuracy/all", dependencies=[Depends(require_api_key)])
def get_accuracy_all():
    uids = get_all_uids()
    result = {}
    for uid in uids:
        resolve_prediction_actuals(uid)
        result[uid] = compute_accuracy(uid)
    return result


@router.get("/accuracy/{uid}", dependencies=[Depends(require_api_key)])
def get_accuracy(uid: str):
    resolved = resolve_prediction_actuals(uid)
    metrics = compute_accuracy(uid)
    return {"uid": uid, "resolved": resolved, "accuracy": metrics}
