"""Pipeline de dados compartilhado (dono: P1).

Contrato combinado na Fase 0 do PLANO_TRABALHO.md: para uma origem `o` e um lag `L`
(1 a 24), o exemplo de treino é
    X = [variaveis atmosfericas do mes o+L, tp do mes o (congelado), L]
    y = tp_alvo do mes o+L
simulando exatamente o cenario do teste real (tp_ultima_obs congelado em dez/2022).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from download_data import download_competition_data, load_dataset_files

FEATURE_VARS = [
    "t2",
    "cloud_cover",
    "shum_850",
    "surface_pressure",
    "u_850",
    "v_850",
    "temperature_850",
    "rel_hum_850",
    "geopotential_850",
]
TARGET_VAR = "tp_alvo"
LAST_OBS_VAR = "tp"

TRAIN_END = "2015-12-01"  # inclusive
VAL_END = "2022-12-01"  # inclusive
LAGS = range(1, 25)  # meses a frente, igual ao teste real


def load_all_datasets() -> dict[str, xr.Dataset]:
    """Baixa (se preciso) e carrega todos os .nc/.csv da competicao."""
    path = download_competition_data()
    _dataframes, datasets = load_dataset_files(path)
    return datasets


def stack_features(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Alinha as variaveis de treino (uma por arquivo .nc) num unico Dataset (tempo, lat, lon, variavel)."""
    raise NotImplementedError("TODO(P1): juntar treino_* num Dataset unico por variavel")


def compute_normalization_stats(ds: xr.Dataset, train_end: str = TRAIN_END) -> dict[str, tuple[float, float]]:
    """Media/desvio por variavel, calculados so no periodo de treino. Retorna {variavel: (media, desvio)}."""
    raise NotImplementedError("TODO(P1): usar dtype=float64 nas reducoes (ver nota em eda.py)")


def normalize(ds: xr.Dataset, stats: dict[str, tuple[float, float]]) -> xr.Dataset:
    """Aplica z-score usando as estatisticas do treino."""
    raise NotImplementedError("TODO(P1)")


def build_examples(
    ds: xr.Dataset,
    origins: list[np.datetime64],
    lags: range = LAGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera (X, y, lag) para cada combinacao (origem, lag) valida, seguindo o contrato do modulo.

    Retorna X com shape (n_exemplos, lat, lon, n_features_atm + 1), y com shape
    (n_exemplos, lat, lon), lag com shape (n_exemplos,).
    """
    raise NotImplementedError("TODO(P1): usado tanto pelo pca_lstm quanto pelo convlstm")


def temporal_split(ds: xr.Dataset, train_end: str = TRAIN_END, val_end: str = VAL_END):
    """Corta o Dataset em treino / validacao, respeitando a ordem temporal (nunca split aleatorio)."""
    raise NotImplementedError("TODO(P1)")
