"""Treina varias variacoes do Modelo A (PCA/PLS + LSTM) em sequencia.

Uso (a partir da raiz do repositorio):
    python3 run_experiments.py                                   # treina todas as variacoes
    python3 run_experiments.py --models pca pls_concurrent       # so as escolhidas
    python3 run_experiments.py --models pls_lagged --pls-lag-shift 2

Cada variacao roda `python3 -m src.train_pca_lstm --reduction <metodo>` num processo
separado (memoria isolada entre rodadas) e ja salva sozinha, dentro da sua pasta em
`experiments/<metodo>_lstm_run1/`:
    - checkpoint_selecao_epocas.pt / history.csv     (backup por epoca, fase de selecao)
    - checkpoint_retrain_final.pt / retrain_history.csv (backup por epoca, retreino final)
    - train.log                                       (log completo daquela execucao)
    - metrics.json, sample_grids.npz, model_final.pt, reduction_and_stats.joblib

Este script so orquestra as execucoes e, ao final, imprime e salva um resumo
comparando RMSE/MAE de cada variacao (`experiments/run_all_summary.json`). Se uma
variacao falhar, as outras continuam rodando (a menos que --stop-on-error seja
passado) - o resumo final mostra quais falharam.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.train_pca_lstm import PLS_LAG_SHIFT, REDUCTION_METHODS, RUN_DIRS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=REDUCTION_METHODS,
        default=list(REDUCTION_METHODS),
        help="Quais variacoes treinar (padrao: todas). Ex.: --models pca pls_lagged",
    )
    parser.add_argument(
        "--pls-lag-shift",
        type=int,
        default=PLS_LAG_SHIFT,
        help="Repassado para --pls-lag-shift em cada treino com reducao pls_lagged (padrao: %(default)s).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Para a fila assim que uma variacao falhar, em vez de continuar com as demais.",
    )
    return parser.parse_args()


def run_one(method: str, pls_lag_shift: int) -> dict:
    cmd = [sys.executable, "-m", "src.train_pca_lstm", "--reduction", method, "--pls-lag-shift", str(pls_lag_shift)]
    print(f"\n{'#' * 70}\n# Iniciando '{method}': {' '.join(cmd)}\n{'#' * 70}\n")

    t0 = time.time()
    resultado = subprocess.run(cmd)
    duracao = time.time() - t0

    status = "ok" if resultado.returncode == 0 else "falhou"
    print(f"\n>>> '{method}' {status} em {duracao:.0f}s (returncode={resultado.returncode})")

    resumo = {"method": method, "status": status, "returncode": resultado.returncode, "duracao_s": round(duracao, 1)}

    metrics_path = Path(RUN_DIRS[method]) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        resumo["rmse"] = metrics["modelo"]["rmse"]
        resumo["mae"] = metrics["modelo"]["mae"]

    return resumo


def main():
    args = parse_args()
    resumos = []

    for method in args.models:
        resumo = run_one(method, args.pls_lag_shift)
        resumos.append(resumo)
        if resumo["status"] == "falhou" and args.stop_on_error:
            print(f"\nParando a fila: '{method}' falhou e --stop-on-error foi passado.")
            break

    print(f"\n{'=' * 70}\nResumo da fila de treinos\n{'=' * 70}")
    for r in resumos:
        if "rmse" in r:
            print(f"  {r['method']:<16} {r['status']:<8} {r['duracao_s']:>7.0f}s  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}")
        else:
            print(f"  {r['method']:<16} {r['status']:<8} {r['duracao_s']:>7.0f}s  (sem metrics.json)")

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/run_all_summary.json", "w") as f:
        json.dump(resumos, f, indent=2)
    print("\nResumo salvo em experiments/run_all_summary.json")

    if any(r["status"] == "falhou" for r in resumos):
        sys.exit(1)


if __name__ == "__main__":
    main()
