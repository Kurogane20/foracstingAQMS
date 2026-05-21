import logging
from datetime import datetime, timezone
from app.db import upsert_metadata
from app.models.pipeline import run_training

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str) -> dict:
    upsert_metadata(uid, status="training")
    try:
        result = run_training(uid)
        trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_metadata(
            uid,
            status="ready",
            last_trained_at=trained_at,
            training_samples=result["training_samples"],
            mae_score=result["mae_score"],
            error_message=None,
        )
        return {"uid": uid, "status": "ready", **result}
    except Exception as e:
        logger.error(f"[{uid}] Training failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
