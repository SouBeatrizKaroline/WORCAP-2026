"""Metricas de avaliacao compartilhadas (dono: P1).

Usada por P2 e P3 para que os dois modelos sejam comparados de forma justa,
sempre no split de validacao (nunca no teste).
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO(P1)")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO(P1)")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lags: np.ndarray | None = None,
) -> dict:
    """Retorna RMSE/MAE globais e, se `lags` for passado, quebrados por horizonte (1-24 meses)."""
    raise NotImplementedError("TODO(P1)")
