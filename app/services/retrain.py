import logging
from datetime import datetime, timezone

from app.db import upsert_metadata
from app.models.pipeline import run_training
from app.models.preprocessor import load_for_training
from app.models.tuning import load_best_params, save_best_params, tune_lgbm
from app.training_progress import set_phase, clear_progress

logger = logging.getLogger(__name__)


def retrain_sensor(uid: str, run_tuning: bool = False) -> dict:
    try:
        set_phase(uid, phase="preprocessing", percent=2)
        hourly = load_for_training(uid)
        lgbm_params = load_best_params(uid)

        if lgbm_params is None and run_tuning:
            upsert_metadata(uid, status="tuning")
            set_phase(uid, phase="tuning", percent=3)
            logger.info(f"[{uid}] Starting tuning pass")
            lgbm_params = tune_lgbm(hourly)
            save_best_params(uid, lgbm_params)
            logger.info(f"[{uid}] Tuning complete, params saved: {lgbm_params}")

        upsert_metadata(uid, status="training")
        result = run_training(uid, lgbm_params=lgbm_params, hourly=hourly)
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
        logger.info(
            f"[{uid}] Training complete — jam={result['training_samples']} "
            f"MAE TSP={result['mae_tsp']} µg/m³ (persistence {result['mae_persistence_tsp']}), "
            f"skill={result['skill_vs_naive']}, cakupan={result['coverage_tsp']}"
        )
        return {"uid": uid, "status": "ready", **result}
    except Exception as e:
        logger.error(f"[{uid}] Training failed: {e}")
        clear_progress(uid)
        upsert_metadata(uid, status="error", error_message=str(e))
        return {"uid": uid, "status": "error", "message": str(e)}
