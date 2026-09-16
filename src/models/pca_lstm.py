"""Modelo A: PCA/EOF + LSTM hindcast/forecast (dono: P2).

Ideia: reduzir a grade (301x261) para N componentes principais por variavel,
treinar o LSTM sobre as series temporais dos coeficientes, e reconstruir o
campo espacial com a transformacao inversa do PCA. Ver Fase 2 do
PLANO_TRABALHO.md e o contrato de exemplos em src/data.py.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn


class SpatialPCA:
    """Ajusta um PCA por variavel sobre a dimensao espacial (lat*lon), usando so o treino."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        self.spatial_shape: tuple[int, int] | None = None

    def fit(self, data: np.ndarray) -> "SpatialPCA":
        """`data` com shape (tempo, lat, lon)."""
        self.spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)
        self._pca.fit(flat)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """`data` com shape (tempo, lat, lon) -> (tempo, n_components)."""
        flat = data.reshape(data.shape[0], -1)
        return self._pca.transform(flat)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """`coeffs` com shape (tempo, n_components) -> (tempo, lat, lon)."""
        flat = self._pca.inverse_transform(coeffs)
        return flat.reshape(coeffs.shape[0], *self.spatial_shape)

    def explained_variance_ratio(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())


class HindcastForecastLSTM(nn.Module):
    """Encoder LSTM sobre a janela historica + decoder condicionado no mes alvo,
    no tp congelado (ultima observacao real) e no lag (meses a frente)."""

    def __init__(
        self,
        n_features_hindcast: int,
        n_features_atm: int,
        n_components_tp: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features_hindcast,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        decoder_input_dim = hidden_size + n_features_atm + n_components_tp + 1  # +1 = lag
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_components_tp),
        )

    def forward(
        self,
        hindcast_seq: torch.Tensor,
        target_month_features: torch.Tensor,
        tp_frozen: torch.Tensor,
        lag: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna os coeficientes PCA previstos de tp para o mes alvo (o+L)."""
        _, (hn, _) = self.encoder(hindcast_seq)
        context = hn[-1]  # estado oculto da ultima camada, (batch, hidden_size)

        if lag.dim() == 1:
            lag = lag.unsqueeze(-1)

        decoder_input = torch.cat([context, target_month_features, tp_frozen, lag], dim=-1)
        return self.decoder(decoder_input)
