"""Baselines de comparacao (dono: P1).

Todo modelo (PCA+LSTM, ConvLSTM) precisa superar esses dois baselines para ser
considerado util - ver Fase 1 do PLANO_TRABALHO.md.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def persistence_baseline(tp_ultima_obs: np.ndarray) -> np.ndarray:
    """A previsao e o proprio ultimo valor real conhecido, repetido para todo lag."""
    return np.asarray(tp_ultima_obs)


def climatology_baseline(tp_treino: xr.DataArray, target_months: list[int]) -> np.ndarray:
    """Media historica (por mes do calendario) da variavel alvo no periodo de treino.

    `tp_treino`: DataArray (time, lat, lon). `target_months`: lista de inteiros 1-12,
    um por exemplo a avaliar. Retorna array (N, lat, lon).
    """
    climatologia = tp_treino.groupby("time.month").mean(dim="time", skipna=True, dtype="float64")
    return np.stack([climatologia.sel(month=mes).values for mes in target_months])
