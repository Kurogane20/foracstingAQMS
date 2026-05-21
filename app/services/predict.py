import logging
from datetime import datetime, timezone
from app.db import save_predictions, upsert_metadata
from app.models.pipeline import run_prediction

logger = logging.getLogger(__name__)


def predict_sensor(uid: str) -> dict:
    try:
        predictions = run_prediction(uid)
        predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        save_predictions(uid, predicted_at, predictions)
        upsert_metadata(uid, status="ready", last_predicted_at=predicted_at)
        return {"uid": uid, "status": "success", "predictions": predictions}
    except FileNotFoundError:
        msg = "Model not trained yet — run /retrain first"
        logger.warning(f"[{uid}] {msg}")
        upsert_metadata(uid, status="untrained", error_message=msg)
        return {"uid": uid, "status": "error", "error": True, "message": msg}
    except Exception as e:
        logger.error(f"[{uid}] Prediction failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "error": True, "message": str(e)}
