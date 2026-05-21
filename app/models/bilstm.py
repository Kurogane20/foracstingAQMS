import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from app.config import (
    BILSTM_UNITS, N_INPUT_HOURS, N_FORECAST_HOURS,
    N_FEATURES, MODELS_DIR,
)
from app.models.ssa import apply_ssa_to_dataframe

N_SSA_FEATURES = N_FEATURES * 2


def build_bilstm() -> tf.keras.Model:
    model = Sequential([
        Bidirectional(
            LSTM(BILSTM_UNITS, return_sequences=True),
            input_shape=(N_INPUT_HOURS, N_SSA_FEATURES),
        ),
        Dropout(0.2),
        Bidirectional(LSTM(BILSTM_UNITS)),
        Dropout(0.2),
        Dense(N_FORECAST_HOURS * N_FEATURES),
        Reshape((N_FORECAST_HOURS, N_FEATURES)),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_bilstm(X: np.ndarray, y: np.ndarray, uid: str) -> tf.keras.Model:
    X_ssa = np.array([apply_ssa_to_dataframe(x) for x in X])
    model = build_bilstm()
    if len(X_ssa) < 10:
        raise ValueError(
            f"Need at least 10 training samples for validation_split=0.1, got {len(X_ssa)}"
        )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_ssa, y,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model


def predict_bilstm(X: np.ndarray, uid: str) -> np.ndarray:
    model_path = os.path.join(MODELS_DIR, uid, "bilstm.keras")
    model = tf.keras.models.load_model(model_path)
    X_ssa = np.array([apply_ssa_to_dataframe(x) for x in X])
    return model.predict(X_ssa, verbose=0)
