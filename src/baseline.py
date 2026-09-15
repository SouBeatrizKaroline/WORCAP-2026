"""Baselines de comparacao (dono: P1).

Todo modelo (PCA+LSTM, ConvLSTM) precisa superar esses dois baselines para ser
considerado util - ver Fase 1 do PLANO_TRABALHO.md.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def persistence_baseline(tp_ultima_obs: xr.DataArray) -> np.ndarray:
    """Repete o ultimo valor real de precipitacao conhecido para todos os lags."""
    raise NotImplementedError("TODO(P1)")


def climatology_baseline(ds_treino: xr.Dataset, target_months: list[int]) -> np.ndarray:
    """Media historica (por mes do calendario) da variavel alvo no periodo de treino."""
    raise NotImplementedError("TODO(P1)")
