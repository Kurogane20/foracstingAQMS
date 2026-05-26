import logging
import os
import threading
import time as _time
from dotenv import load_dotenv
load_dotenv()
if os.getenv("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = os.getenv("LOKY_MAX_CPU_COUNT")

from fastapi import FastAPI
from app.db import get_model_status
from app.routers.prediction import router as prediction_router
from app.routers.training import router as training_router
from app.routers.accuracy import router as accuracy_router
from app.config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Air Quality Forecast API", version="1.0.0")
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(accuracy_router)


def _hourly_predict_worker() -> None:
    """Background thread: run predict for all sensors every hour."""
    logger = logging.getLogger(__name__)
    while True:
        _time.sleep(3600)
        try:
            from app.db import get_all_uids, get_model_status
            from app.services.predict import predict_sensor
            from app.services.accuracy import resolve_prediction_actuals, check_and_auto_retrain
            uids = get_all_uids()

            # predict loop
            for uid in uids:
                try:
                    predict_sensor(uid)
                except Exception as e:
                    logger.error(f"[scheduler] predict failed for {uid}: {e}")
            logger.info(f"[scheduler] Hourly predict complete for {len(uids)} sensors")

            # accuracy loop
            status_map = {s["uid"]: s for s in get_model_status()}
            for uid in uids:
                try:
                    resolve_prediction_actuals(uid)
                    baseline = (status_map.get(uid) or {}).get("mae_score")
                    check_and_auto_retrain(uid, baseline)
                except Exception as e:
                    logger.error(f"[scheduler] accuracy check failed for {uid}: {e}")
            logger.info(f"[scheduler] Accuracy check complete for {len(uids)} sensors")

        except Exception as e:
            logger.error(f"[scheduler] Worker cycle failed: {e}")


threading.Thread(target=_hourly_predict_worker, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {"sensors": get_model_status()}
