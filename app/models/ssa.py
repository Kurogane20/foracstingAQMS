import numpy as np
from app.config import SSA_WINDOW


def ssa_decompose(series: np.ndarray, window: int = SSA_WINDOW) -> tuple:
    n = len(series)
    k = n - window + 1
    trajectory = np.array([series[i : i + window] for i in range(k)])
    U, sigma, Vt = np.linalg.svd(trajectory, full_matrices=False)
    trend = _diagonal_average(sigma[0] * np.outer(U[:, 0], Vt[0, :]), window, n)
    oscillation = series - trend
    return trend, oscillation


def _diagonal_average(mat: np.ndarray, window: int, n: int) -> np.ndarray:
    k = n - window + 1
    result = np.zeros(n)
    counts = np.zeros(n)
    for i in range(k):
        for j in range(window):
            result[i + j] += mat[i, j]
            counts[i + j] += 1
    return result / counts


def apply_ssa_to_dataframe(data: np.ndarray, window: int = SSA_WINDOW) -> np.ndarray:
    n_timesteps, n_features = data.shape
    output = np.zeros((n_timesteps, n_features * 2))
    for i in range(n_features):
        trend, oscillation = ssa_decompose(data[:, i], window)
        output[:, i] = trend
        output[:, i + n_features] = oscillation
    return output
