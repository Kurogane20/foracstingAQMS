import os
import time

import joblib
import numpy as np
import pandas as pd
import pytest

from app.config import FEATURE_SCHEMA_VERSION
from app.models import forecaster
from app.models.features import build_rows, to_model_space


def _dataset(hourly_df):
    X, meta = build_rows(to_model_space(hourly_df), "tsp")
    fit = meta["delta"].notna().values
    return X[fit], meta.loc[fit, "delta"].values


def test_fit_models_returns_point_and_both_quantiles(hourly_df, fast_params):
    X, y = _dataset(hourly_df)
    models = forecaster.fit_models(X, y, fast_params)
    assert set(models) == {"point", "lower", "upper"}
    raw = forecaster.predict_deltas(models, X)
    assert all(v.shape == (len(X),) for v in raw.values())
    # Rata-rata, kuantil bawah di bawah titik dan kuantil atas di atasnya.
    assert raw["lower"].mean() < raw["point"].mean() < raw["upper"].mean()


def test_quantile_models_bracket_roughly_80_percent_in_sample(hourly_df, fast_params):
    X, y = _dataset(hourly_df)
    raw = forecaster.predict_deltas(forecaster.fit_models(X, y, fast_params), X)
    inside = ((y >= raw["lower"]) & (y <= raw["upper"])).mean()
    assert 0.7 < inside < 0.95


def test_params_override_defaults(hourly_df):
    X, y = _dataset(hourly_df)
    models = forecaster.fit_models(X, y, {"n_estimators": 7, "n_jobs": 1})
    assert all(m.n_estimators == 7 for m in models.values())


def test_training_is_deterministic(hourly_df, fast_params):
    X, y = _dataset(hourly_df)
    a = forecaster.predict_deltas(forecaster.fit_models(X, y, fast_params), X.head(50))
    b = forecaster.predict_deltas(forecaster.fit_models(X, y, fast_params), X.head(50))
    assert np.allclose(a["point"], b["point"])


def test_bundle_roundtrip(models_dir):
    forecaster.save_bundle("u1", {"targets": {"tsp": {"alpha": np.ones(6)}}, "trained_until": "x"})
    assert (models_dir / "u1" / forecaster.FILENAME).exists()
    assert not (models_dir / "u1" / (forecaster.FILENAME + ".tmp")).exists()
    bundle = forecaster.load_bundle("u1")
    assert bundle["schema"] == FEATURE_SCHEMA_VERSION
    assert np.allclose(bundle["targets"]["tsp"]["alpha"], 1.0)


def test_load_bundle_missing_raises_file_not_found(models_dir):
    """predict_sensor menerjemahkan FileNotFoundError menjadi "belum dilatih"."""
    with pytest.raises(FileNotFoundError):
        forecaster.load_bundle("belum_ada")


def test_load_bundle_rejects_other_schema(models_dir):
    path = models_dir / "u1" / forecaster.FILENAME
    os.makedirs(path.parent)
    joblib.dump({"schema": FEATURE_SCHEMA_VERSION - 1, "targets": {}}, path)
    with pytest.raises(ValueError, match="retrain"):
        forecaster.load_bundle("u1")


def test_cache_is_refreshed_after_retrain(models_dir):
    """Regresi A3: model lama tidak boleh terus dipakai setelah retrain."""
    forecaster.save_bundle("u1", {"targets": {}, "marker": 1})
    assert forecaster.load_bundle("u1")["marker"] == 1
    time.sleep(0.01)
    forecaster.save_bundle("u1", {"targets": {}, "marker": 2, "pad": "x" * 50})
    assert forecaster.load_bundle("u1")["marker"] == 2


def test_cache_avoids_reloading_unchanged_file(models_dir, monkeypatch):
    forecaster.save_bundle("u1", {"targets": {}})
    forecaster.load_bundle("u1")
    calls = []
    monkeypatch.setattr(forecaster.joblib, "load", lambda p: calls.append(p))
    forecaster.load_bundle("u1")
    assert calls == []
