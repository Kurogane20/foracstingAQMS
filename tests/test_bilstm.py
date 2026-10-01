import numpy as np
import pytest
from app.config import N_INPUT_HOURS, N_FORECAST_HOURS, FEATURE_COLS, SSA_WINDOW, N_TIME_FEATURES, N_METEO_FEATURES
from app.models.bilstm import build_bilstm
from app.models.ssa import apply_ssa_to_dataframe

N_SSA_FEATURES = len(FEATURE_COLS) * 2 + N_TIME_FEATURES + N_METEO_FEATURES


def test_bilstm_output_shape():
    model = build_bilstm()
    X_dummy = np.random.randn(2, N_INPUT_HOURS, N_SSA_FEATURES)
    output = model.predict(X_dummy, verbose=0)
    assert output.shape == (2, N_FORECAST_HOURS, len(FEATURE_COLS))


def test_bilstm_model_has_correct_input_shape():
    model = build_bilstm()
    assert model.input_shape == (None, N_INPUT_HOURS, N_SSA_FEATURES)


def test_model_cache_reloads_after_retrain(tmp_path, monkeypatch):
    """Regresi: cache dulu menyimpan model per uid tanpa syarat, sehingga setelah
    retrain LightGBM, α, MAE — dan produksi — tetap memakai BiLSTM lama."""
    import os
    from unittest.mock import patch
    from app.models import bilstm

    monkeypatch.setattr(bilstm, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(bilstm, "_model_cache", {})
    path = tmp_path / "uid_c" / "bilstm.keras"
    path.parent.mkdir()
    path.write_bytes(b"lama")

    loaded = []
    with patch.object(bilstm.tf.keras.models, "load_model",
                      side_effect=lambda p: loaded.append(object()) or loaded[-1]):
        first = bilstm._load_model("uid_c")
        assert bilstm._load_model("uid_c") is first          # berkas sama → cache
        path.write_bytes(b"model-baru")                       # retrain menulis ulang
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        second = bilstm._load_model("uid_c")

    assert second is not first
    assert len(loaded) == 2


def test_missing_model_raises_file_not_found(tmp_path, monkeypatch):
    """predict_sensor menerjemahkan FileNotFoundError menjadi 'belum dilatih'."""
    from app.models import bilstm
    monkeypatch.setattr(bilstm, "MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        bilstm._load_model("tidak_ada")
