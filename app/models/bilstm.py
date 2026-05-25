import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from app.config import (
    BILSTM_UNITS, N_INPUT_HOURS, N_FORECAST_HOURS,
    N_FEATURES, N_TIME_FEATURES, MODELS_DIR,
)
from app.models.ssa import apply_ssa_to_dataframe

# SSA doubles sensor features; time features appended raw
N_SSA_FEATURES = N_FEATURES * 2 + N_TIME_FEATURES  # 28 + 4 = 32

_model_cache: dict[str, tf.keras.Model] = {}


def _prepare_input(X: np.ndarray) -> np.ndarray:
    """Apply SSA to sensor columns only, then concat time features."""
    X_sensor = X[:, :, :N_FEATURES]                                     # (N, 24, 14)
    X_time   = X[:, :, N_FEATURES:N_FEATURES + N_TIME_FEATURES]         # (N, 24,  4)
    X_ssa    = np.array([apply_ssa_to_dataframe(x) for x in X_sensor])  # (N, 24, 28)
    return np.concatenate([X_ssa, X_time], axis=2)                      # (N, 24, 32)


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
    model.fit(
        X_prepared, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
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
    X_prepared = _prepare_input(X)
    return _model_cache[uid].predict(X_prepared, verbose=0)
