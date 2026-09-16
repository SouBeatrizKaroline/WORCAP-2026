"""Metricas de avaliacao compartilhadas (dono: P1).

Usada por P2 e P3 para que os dois modelos sejam comparados de forma justa,
sempre no split de validacao (nunca no teste).
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lags: np.ndarray | None = None,
) -> dict:
    """Retorna RMSE/MAE globais e, se `lags` for passado, quebrados por horizonte (1-24 meses)."""
    result = {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}

    if lags is not None:
        por_lag = {}
        for lag in np.unique(lags):
            mask = lags == lag
            por_lag[int(lag)] = {
                "rmse": rmse(y_true[mask], y_pred[mask]),
                "mae": mae(y_true[mask], y_pred[mask]),
                "n": int(mask.sum()),
            }
        result["por_lag"] = por_lag

    return result
