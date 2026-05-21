import logging
import os
from fastapi import FastAPI
from app.db import get_model_status
from app.routers.prediction import router as prediction_router
from app.routers.training import router as training_router
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {"sensors": get_model_status()}
