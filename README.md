# WORCAP 2026 — Previsão Climática de Precipitação sobre a América do Sul

Código para baixar e carregar os dados da competição Kaggle
[previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul).

## Setup (rodar uma vez por máquina)

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Aceite as regras da competição (logado na sua conta Kaggle):
   https://kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/rules

3. Configure suas credenciais da API do Kaggle. Existem duas formas — use a que
   corresponder ao tipo de token que você gerou em kaggle.com → Settings → API:

   - **Token novo (formato `KGAT_...`)**: salve o token em `~/.kaggle/access_token`
     (arquivo de uma linha só, sem aspas nem JSON):

     ```bash
     mkdir -p ~/.kaggle
     echo -n "SEU_TOKEN_KGAT_AQUI" > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```

   - **Token clássico (`kaggle.json` com username + key)**: salve o arquivo baixado
     do Kaggle em `~/.kaggle/kaggle.json` e restrinja a permissão:

     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```

   Nunca commite esses arquivos — eles já estão no `.gitignore`.

## Uso

```bash
python3 download_data.py
```

O script:
- Baixa (ou reaproveita o cache local em `~/.cache/kagglehub/`) os arquivos da competição.
- Carrega cada `.csv` em um `pandas.DataFrame`.
- Carrega cada `.nc` (NetCDF, dados climáticos em grade lat/lon) em um `xarray.Dataset`.
- Imprime um resumo (shape/variáveis) de cada arquivo carregado.

Os dados não ficam no repositório (são grandes e cada pessoa baixa a própria cópia
com o token individual do Kaggle).
