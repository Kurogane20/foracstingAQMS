import json
from unittest.mock import patch

import pandas as pd
import pytest

from app.config import FEATURE_SCHEMA_VERSION
from app.models import tuning
from app.models.backtest import plan_folds
from app.models.features import build_rows, to_model_space
from tests.conftest import make_hourly


# ── save_best_params / load_best_params ──────────────────────────────────────

def test_params_roundtrip(models_dir):
    params = {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 20, "min_child_samples": 30}
    tuning.save_best_params("u1", params)
    with open(models_dir / "u1" / tuning.PARAMS_FILENAME) as f:
        assert json.load(f) == {"schema": FEATURE_SCHEMA_VERSION, "lgbm": params}
    assert tuning.load_best_params("u1") == params


def test_load_returns_none_when_missing(models_dir):
    assert tuning.load_best_params("tidak_ada") is None


def test_params_from_the_old_pipeline_are_ignored(models_dir):
    """best_params.json lama (BiLSTM + tahap koreksi LightGBM, tanpa kunci skema)
    disetel untuk masalah lain dan tidak boleh dipakai model baru."""
    (models_dir / "u1").mkdir()
    with open(models_dir / "u1" / tuning.PARAMS_FILENAME, "w") as f:
        json.dump({"bilstm": {"units": 128}, "lgbm": {"n_estimators": 800, "num_leaves": 127}}, f)
    assert tuning.load_best_params("u1") is None


# ── tuning_rows ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tune_inputs():
    # Cukup panjang agar tersisa ±7 minggu data SEBELUM lipatan backtest pertama.
    df = make_hourly(days=220)
    _, meta = build_rows(to_model_space(df), "tsp")
    return df, meta, plan_folds(df.index)[0][0]


def test_tuning_never_sees_backtest_folds(tune_inputs):
    """Lipatan backtest melaporkan skor, jadi tidak boleh memilih hiperparameter."""
    _, meta, tune_end = tune_inputs
    fit, evaluate = tuning.tuning_rows(meta, tune_end)
    used = meta[fit | evaluate]
    assert ((used["t"] + pd.to_timedelta(used["h"], unit="h")) < tune_end).all()


def test_tuning_fit_and_eval_do_not_overlap(tune_inputs):
    _, meta, tune_end = tune_inputs
    fit, evaluate = tuning.tuning_rows(meta, tune_end)
    assert fit.any() and evaluate.any() and not (fit & evaluate).any()
    fit_targets = meta.loc[fit, "t"] + pd.to_timedelta(meta.loc[fit, "h"], unit="h")
    assert fit_targets.max() < meta.loc[evaluate, "t"].min()
    assert meta.loc[fit | evaluate, "delta"].notna().all()


def test_tuning_rows_rejects_too_little_data(tune_inputs):
    _, meta, _ = tune_inputs
    with pytest.raises(ValueError, match="terlalu sedikit"):
        tuning.tuning_rows(meta, meta["t"].min() + pd.Timedelta(hours=40))


# ── tune_lgbm ────────────────────────────────────────────────────────────────

def test_tune_lgbm_returns_params_within_search_space(tune_inputs):
    df = tune_inputs[0]
    best = tuning.tune_lgbm(df, n_trials=2)
    assert set(best) == {"n_estimators", "learning_rate", "num_leaves", "min_child_samples"}
    assert 150 <= best["n_estimators"] <= 600
    assert 0.02 <= best["learning_rate"] <= 0.1
    assert 15 <= best["num_leaves"] <= 63


def test_tune_lgbm_scores_candidates_out_of_sample(tune_inputs):
    """Kandidat dilatih pada potongan awal dan dinilai pada potongan akhir."""
    df, meta, tune_end = tune_inputs
    fit, evaluate = tuning.tuning_rows(meta, tune_end)
    fitted_on, scored_on = [], []

    class Spy:
        def __init__(self, **kw):
            pass

        def fit(self, X, y):
            fitted_on.append(len(X))

        def predict(self, X):
            scored_on.append(len(X))
            return [0.0] * len(X)

    with patch("app.models.tuning.lgb.LGBMRegressor", Spy):
        tuning.tune_lgbm(df, n_trials=1)
    assert fitted_on == [int(fit.sum())] and scored_on == [int(evaluate.sum())]
