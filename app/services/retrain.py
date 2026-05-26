import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.db import upsert_metadata
from app.models.pipeline import run_training
from app.models.preprocessor import preprocess_for_training
from app.models.tuning import load_best_params, tune_bilstm, tune_lgbm, save_best_params
from app.models.bilstm import predict_bilstm, train_bilstm

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str, run_tuning: bool = False) -> dict:
    try:
        bilstm_params = None
        lgbm_params = None

        params = load_best_params(uid)

        if params is not None:
            bilstm_params = params["bilstm"]
            lgbm_params = params["lgbm"]
        elif run_tuning:
            upsert_metadata(uid, status="tuning")
            logger.info(f"[{uid}] Starting tuning pass")

            X, y = preprocess_for_training(uid)

            bilstm_params = tune_bilstm(X, y, uid)

            split = int(len(X) * 0.9)
            X_train = X[:split]
            y_train = y[:split]
            train_bilstm(X_train, y_train, uid, **bilstm_params)
            bilstm_preds = np.array([
                predict_bilstm(X_train[i:i + 1], uid)[0]
                for i in range(len(X_train))
            ])
            base_times = [
                pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
                for i in range(len(X_train))
            ]

            lgbm_params = tune_lgbm(bilstm_preds, y_train, base_times, uid)
            save_best_params(uid, bilstm_params, lgbm_params)
            logger.info(f"[{uid}] Tuning complete, params saved")

        upsert_metadata(uid, status="training")
        result = run_training(uid, bilstm_params=bilstm_params, lgbm_params=lgbm_params)
        trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_metadata(
            uid,
            status="ready",
            last_trained_at=trained_at,
            training_samples=result["training_samples"],
            mae_score=result["mae_score"],
            error_message=None,
        )
        logger.info(f"[{uid}] Training complete — samples={result['training_samples']} mae={result['mae_score']}")
        return {"uid": uid, "status": "ready", **result}
    except Exception as e:
        logger.error(f"[{uid}] Training failed: {e}")
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
