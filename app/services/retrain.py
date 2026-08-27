import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.db import upsert_metadata
from app.models.pipeline import run_training
from app.models.preprocessor import preprocess_for_training, anchor_from_X, to_delta
from app.models.tuning import load_best_params, tune_bilstm, tune_lgbm, save_best_params
from app.models.bilstm import predict_bilstm, train_bilstm, _model_cache
from app.training_progress import set_phase, clear_progress

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str, run_tuning: bool = False) -> dict:
    try:
        bilstm_params = None
        lgbm_params = None

        set_phase(uid, phase="preprocessing", percent=3)
        params = load_best_params(uid)

        if params is not None:
            bilstm_params = params["bilstm"]
            lgbm_params = params["lgbm"]
        elif run_tuning:
            upsert_metadata(uid, status="tuning")
            logger.info(f"[{uid}] Starting tuning pass")

            X, y = preprocess_for_training(uid)

            # Jalur tuning harus memakai ruang target yang SAMA dengan jalur
            # pelatihan (selisih terhadap nilai terakhir), kalau tidak parameter
            # yang ditemukan dioptimalkan untuk masalah yang berbeda.
            dy = to_delta(y, anchor_from_X(X))

            bilstm_params = tune_bilstm(X, dy, uid)

            split = int(len(X) * 0.9)
            X_train = X[:split]
            dy_train = dy[:split]
            anchor_train = anchor_from_X(X_train)
            train_bilstm(X_train, dy_train, uid, **bilstm_params)
            _model_cache.pop(uid, None)   # force reload of just-saved model
            bilstm_preds = np.array([
                predict_bilstm(X_train[i:i + 1], uid)[0]
                for i in range(len(X_train))
            ])
            base_times = [
                pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
                for i in range(len(X_train))
            ]

            lgbm_params = tune_lgbm(bilstm_preds, dy_train, base_times, uid,
                                    anchors=anchor_train)
            save_best_params(uid, bilstm_params, lgbm_params)
            logger.info(f"[{uid}] Tuning complete, params saved")

        upsert_metadata(uid, status="training")
        if run_tuning and params is None:
            result = run_training(uid, bilstm_params=bilstm_params, lgbm_params=lgbm_params, X=X, y=y)
        else:
            result = run_training(uid, bilstm_params=bilstm_params, lgbm_params=lgbm_params)
        trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
        clear_progress(uid)
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
        clear_progress(uid)
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
