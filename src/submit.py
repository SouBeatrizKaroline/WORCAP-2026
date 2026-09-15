"""Geracao do arquivo de submissao no formato do Kaggle (dono: P1).

Interface unica para os dois modelos, para nao duplicar a logica de formatacao
do id (`AAAA_MM_lat_lon`) - ver sample_submission.csv.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr


def build_submission(
    predictions: xr.DataArray,
    sample_submission_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Formata `predictions` (tempo, lat, lon) para o formato id,tp_mm_day e salva em `output_path`."""
    raise NotImplementedError("TODO(P1): validar que os ids batem 1-para-1 com sample_submission_path")
