"""Modelo B: ConvLSTM (dono: P3).

Mantem a estrutura espacial da grade (sem achatar via PCA). Avaliar se a
resolucao cheia (301x261) e viavel no hardware disponivel; se nao, usar
downsample ou patches. Ver Fase 2 do PLANO_TRABALHO.md.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        raise NotImplementedError("TODO(P3): portas de entrada/esquecimento/saida via Conv2d")

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO(P3)")


class ConvLSTMForecaster(nn.Module):
    """Empilha ConvLSTMCells; recebe a janela historica + variaveis do mes alvo + lag."""

    def __init__(
        self,
        n_features_atm: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        raise NotImplementedError("TODO(P3)")

    def forward(
        self,
        hindcast_seq: torch.Tensor,
        target_month_features: torch.Tensor,
        lag: torch.Tensor,
    ) -> torch.Tensor:
        """Retorna o campo (lat, lon) previsto de tp_alvo para o mes alvo."""
        raise NotImplementedError("TODO(P3)")
