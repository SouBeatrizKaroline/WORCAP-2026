"""Shim de compatibilidade TEMPORARIO - o Modelo A foi movido para
src/models/pca_lstm/train.py (reorganizacao: cada modelo agora tem sua propria
pasta em src/models/). Prefira importar/rodar a partir do caminho novo.

Este arquivo existe so porque, no momento da reorganizacao, havia um treino em
background (run_all_models.py, o antigo run_experiments.py) que ainda ia invocar
`python3 -m src.train_pca_lstm --reduction pls_lagged` antes de terminar a fila -
o processo ja estava rodando com o codigo antigo carregado em memoria, entao so
precisava que esse caminho de modulo continuasse resolvendo. Pode ser apagado
com seguranca depois que essa fila terminar.
"""

from src.models.pca_lstm.train import *  # noqa: F401,F403
from src.models.pca_lstm.train import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    main(
        method=args.reduction,
        pls_lag_shift=args.pls_lag_shift,
        run_dir=args.run_dir,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        lr=args.lr,
    )
