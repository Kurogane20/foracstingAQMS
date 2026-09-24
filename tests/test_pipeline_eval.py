"""Evaluasi run_training: α harus dicari di data yang BERBEDA dari data pelaporan
skor, dan kedua potongan dipisah celah agar jendela target tak tumpang-tindih."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.config import N_FEATURES, N_FORECAST_HOURS, N_INPUT_HOURS
from app.models import pipeline
from app.models.pipeline import PURGE_SEQUENCES, run_training
from app.models.preprocessor import train_split_index

N_SEQ = 1000
N_COLS = N_FEATURES + 8


def _times(n):
    # Awal sengaja bukan tengah malam/Senin/Januari, agar jam sintetis lama
    # (2024-01-01 + i) pasti berbeda dari jam nyata.
    return pd.date_range("2025-10-03 07:00", periods=n, freq="h")


def _data(n=N_SEQ, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_INPUT_HOURS, N_COLS))
    y = rng.random((n, N_FORECAST_HOURS, N_FEATURES))
    return X, y


def _run(captured, train_lgbm=None, predict_lgbm=None):
    def fake_fit_alpha(pred, true):
        captured["cal_len"] = len(pred)
        return np.full((N_FORECAST_HOURS, N_FEATURES), 0.5)

    zeros = np.zeros((N_FORECAST_HOURS, N_FEATURES))
    lgbm_train = train_lgbm or (lambda *a, **k: None)
    lgbm_predict = predict_lgbm or (lambda *a, **k: {"point": zeros})
    with patch.object(pipeline, "train_bilstm"), \
         patch.object(pipeline, "train_lgbm", side_effect=lgbm_train), \
         patch.object(pipeline, "set_phase"), \
         patch.object(pipeline, "save_alpha"), \
         patch.object(pipeline, "predict_bilstm", return_value=zeros[np.newaxis]), \
         patch.object(pipeline, "predict_lgbm", side_effect=lgbm_predict), \
         patch.object(pipeline, "fit_alpha", side_effect=fake_fit_alpha):
        X, y = _data()
        return run_training("uid_t", X=X, y=y, base_times=_times(N_SEQ))


def test_alpha_calibration_and_report_sets_are_disjoint():
    captured = {}
    out = _run(captured)

    split = train_split_index(N_SEQ)
    n_val = N_SEQ - (split + PURGE_SEQUENCES)
    cal = (n_val - PURGE_SEQUENCES) // 2

    assert captured["cal_len"] == cal == out["alpha_calibration_samples"]
    assert out["report_samples"] == n_val - cal - PURGE_SEQUENCES
    # Kalibrasi + celah + pelaporan harus pas menutup seluruh validasi.
    assert cal + PURGE_SEQUENCES + out["report_samples"] == n_val


def test_report_excludes_calibration_samples():
    """Skor tidak boleh dihitung pada sampel yang dipakai mencari α."""
    captured = {}
    out = _run(captured)
    assert out["report_samples"] < N_SEQ - train_split_index(N_SEQ)
    assert out["alpha_calibration_samples"] + out["report_samples"] < \
        N_SEQ - train_split_index(N_SEQ)


def test_too_little_validation_data_is_refused():
    X, y = _data(n=60, seed=1)
    with patch.object(pipeline, "train_bilstm"):
        with pytest.raises(ValueError, match="kalibrasi"):
            run_training("uid_t", X=X, y=y, base_times=_times(60))


def test_lgbm_sees_real_calendar_in_training_and_validation():
    """Regresi train/serve skew: LightGBM harus dilatih DAN divalidasi dengan jam
    nyata, sama seperti yang ia terima saat produksi."""
    seen = {"val": []}
    zeros = np.zeros((N_FORECAST_HOURS, N_FEATURES))

    def fake_train(preds, dy, base_times, uid, **kw):
        seen["train"] = list(base_times)

    def fake_predict(bilstm_out, base_time, uid, anchor):
        seen["val"].append(base_time)
        return {"point": zeros}

    _run({}, train_lgbm=fake_train, predict_lgbm=fake_predict)

    times = _times(N_SEQ)
    split = train_split_index(N_SEQ)
    val_start = split + PURGE_SEQUENCES
    assert seen["train"] == list(times[:split])
    assert seen["val"] == list(times[val_start:])


def test_supplied_sequences_without_times_are_refused():
    """X/y dari luar tanpa jam nyata tidak boleh diam-diam diberi jam palsu."""
    X, y = _data()
    with pytest.raises(ValueError, match="base_times"):
        run_training("uid_t", X=X, y=y)
