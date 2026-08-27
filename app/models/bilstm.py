import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from app.config import (
    BILSTM_UNITS, N_INPUT_HOURS, N_FORECAST_HOURS,
    N_FEATURES, N_TIME_FEATURES, N_METEO_FEATURES, MODELS_DIR,
)
from app.models.ssa import apply_ssa_to_dataframe
from app.training_progress import set_phase

# SSA doubles sensor features; time + meteo features pass through raw (no SSA)
N_SSA_FEATURES = N_FEATURES * 2 + N_TIME_FEATURES + N_METEO_FEATURES  # 28 + 4 + 6 = 38

_model_cache: dict[str, tf.keras.Model] = {}


class _ProgressCallback(tf.keras.callbacks.Callback):
    """Updates training_progress for a uid on each epoch end."""

    def __init__(self, uid: str, total_epochs: int) -> None:
        super().__init__()
        self._uid = uid
        self._total = max(total_epochs, 1)
        self._start: float = 0.0

    def on_train_begin(self, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._start
        epochs_done = epoch + 1
        # BiLSTM phase occupies 5%–84% of overall progress
        pct = int(5 + (epochs_done / self._total) * 79)
        eta = int((elapsed / epochs_done) * (self._total - epochs_done)) if epochs_done > 0 else None
        set_phase(
            self._uid,
            phase="bilstm",
            percent=min(pct, 84),
            epoch=epochs_done,
            total_epochs=self._total,
            eta_seconds=eta,
        )


def _prepare_input(X: np.ndarray) -> np.ndarray:
    """
    Apply SSA to sensor columns only, then concat time and meteo features.
    X shape: (N, 24, 24)  — [sensor(14) | time(4) | meteo(6)]
    Output:  (N, 24, 38)  — [ssa_sensor(28) | time(4) | meteo(6)]
    """
    n_time_end  = N_FEATURES + N_TIME_FEATURES            # 18
    n_meteo_end = n_time_end + N_METEO_FEATURES           # 24
    X_sensor = X[:, :, :N_FEATURES]                       # (N, 24, 14)
    X_time   = X[:, :, N_FEATURES:n_time_end]             # (N, 24,  4)
    X_meteo  = X[:, :, n_time_end:n_meteo_end]            # (N, 24,  6)
    X_ssa    = np.array([apply_ssa_to_dataframe(x) for x in X_sensor])  # (N, 24, 28)
    return np.concatenate([X_ssa, X_time, X_meteo], axis=2)             # (N, 24, 38)


def build_bilstm(units: int = BILSTM_UNITS, dropout: float = 0.2) -> tf.keras.Model:
    model = Sequential([
        Bidirectional(
            LSTM(units, return_sequences=True),
            input_shape=(N_INPUT_HOURS, N_SSA_FEATURES),
        ),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2, return_sequences=True)),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2)),
        Dropout(dropout),
        Dense(N_FORECAST_HOURS * N_FEATURES),
        Reshape((N_FORECAST_HOURS, N_FEATURES)),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_bilstm(
    X: np.ndarray,
    y: np.ndarray,
    uid: str,
    units: int = BILSTM_UNITS,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 20,
) -> tf.keras.Model:
    X_prepared = _prepare_input(X)
    model = build_bilstm(units=units, dropout=dropout)
    if len(X_prepared) < 10:
        raise ValueError(
            f"Need at least 10 training samples for validation_split=0.1, got {len(X_prepared)}"
        )
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    progress_cb = _ProgressCallback(uid=uid, total_epochs=epochs)
    model.fit(
        X_prepared, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, progress_cb],
        verbose=1,
    )
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model


def predict_bilstm(X: np.ndarray, uid: str) -> np.ndarray:
    if uid not in _model_cache:
        model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
        _model_cache[uid] = tf.keras.models.load_model(model_path)
    model = _model_cache[uid]

    # Model lama dilatih sebelum fitur curah hujan ditambahkan, jadi bentuk
    # inputnya beda. Tanpa penjagaan ini Keras melempar galat bentuk mentah
    # yang menyesatkan; yang sebenarnya dibutuhkan adalah latih ulang.
    expected = model.input_shape[-1]
    if expected != N_SSA_FEATURES:
        raise ValueError(
            f"[{uid}] Model tersimpan mengharapkan {expected} fitur, skema saat ini "
            f"{N_SSA_FEATURES}. Skema fitur berubah (curah hujan ditambahkan) — "
            f"jalankan retrain untuk uid ini sebelum memprediksi."
        )

    X_prepared = _prepare_input(X)
    return model.predict(X_prepared, verbose=0)
