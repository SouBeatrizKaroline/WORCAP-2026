"""Modelo A: PCA/EOF + LSTM hindcast/forecast (dono: P2).

Ideia: reduzir a grade (301x261) para N componentes principais por variavel,
treinar o LSTM sobre as series temporais dos coeficientes, e reconstruir o
campo espacial com a transformacao inversa do PCA. Ver Fase 2 do
PLANO_TRABALHO.md.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SpatialPCA:
    """Ajusta um PCA por variavel sobre a dimensao espacial (lat*lon), usando so o treino."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: np.ndarray) -> "SpatialPCA":
        """`data` com shape (tempo, lat*lon)."""
        raise NotImplementedError("TODO(P2): usar sklearn.decomposition.PCA ou IncrementalPCA")

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO(P2)")

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO(P2)")


class HindcastForecastLSTM(nn.Module):
    """Encoder LSTM sobre a janela historica + decoder condicionado no mes alvo e no lag."""

    def __init__(
        self,
        n_features_atm: int,
        n_components_tp: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError("TODO(P2): hindcast LSTM + decoder (MLP ou segundo LSTM)")

    def forward(
        self,
        hindcast_seq: torch.Tensor,
        target_month_features: torch.Tensor,
        lag: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna os coeficientes PCA previstos de tp_alvo para o mes alvo."""
        raise NotImplementedError("TODO(P2)")
