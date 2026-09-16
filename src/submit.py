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
    """Formata `predictions` (time, lat, lon) para o formato id,tp_mm_day e salva em `output_path`.

    Valida que o conjunto de ids bate exatamente com sample_submission_path antes de salvar.
    """
    df = predictions.rename("tp_mm_day").to_dataframe().reset_index()
    df["id"] = (
        df["time"].dt.strftime("%Y_%m")
        + "_"
        + df["lat"].map("{:.2f}".format)
        + "_"
        + df["lon"].map("{:.2f}".format)
    )
    df = df[["id", "tp_mm_day"]]

    sample = pd.read_csv(sample_submission_path)
    ids_faltando = set(sample["id"]) - set(df["id"])
    ids_extras = set(df["id"]) - set(sample["id"])
    if ids_faltando or ids_extras:
        raise ValueError(
            f"ids nao batem com sample_submission: {len(ids_faltando)} faltando, "
            f"{len(ids_extras)} extras. Exemplo faltando: {list(ids_faltando)[:3]}"
        )

    df = df.set_index("id").loc[sample["id"]].reset_index()
    df.to_csv(output_path, index=False)
    return df
