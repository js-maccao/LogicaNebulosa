"""Gera a base SINTÉTICA de passagens aéreas usada na Questão 2.

Uso:  python gerar_dados.py            (grava dados/passagens.csv)
      python gerar_dados.py --n 8000   (outra quantidade de passagens)

Não achei uma base pública de preços de passagens que desse para usar offline e
sem cadastro, então o preço sai de uma regra conhecida mais um ruído aleatório.
A vantagem é saber exatamente o que o modelo deveria aprender. Companhias e
valores são fictícios; as distâncias entre as cidades são aproximadas.

Regra do preço (todos os fatores multiplicam):

    preço = base(distância) x companhia x classe x antecedência x escalas
            x dia da semana x temporada x período do dia x ruído

    base(distância)  = 90 + 0,38 x distância_km           (R$)
    companhia        = VoaBrasil 1,00 | CéuAberto 1,12 | Andorinha (low cost) 0,85
    classe           = econômica 1,00 | executiva 2,60
    antecedência     = 1 + 1,2 x e^(-dias/12)               (comprar em cima da hora custa até 2,2x)
    escalas          = 0 -> 1,00 | 1 -> 0,85 | 2 -> 0,75    (voo direto é mais caro)
    dia da semana    = sex/dom 1,15 | ter/qua 0,92 | demais 1,00
    alta temporada   = dez, jan, jul 1,25 | demais 1,00
    período do dia   = madrugada 0,88 | manhã 1,05 | tarde 1,00 | noite 1,02
    ruído            = lognormal(0; 0,10)                   (~10% de variação imprevisível)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "dados" / "passagens.csv"
SEMENTE = 42

# (origem, destino, distância aproximada em km)
ROTAS = [
    ("São Paulo", "Rio de Janeiro", 360), ("São Paulo", "Belo Horizonte", 490),
    ("São Paulo", "Curitiba", 340), ("São Paulo", "Porto Alegre", 850),
    ("São Paulo", "Brasília", 870), ("São Paulo", "Salvador", 1450),
    ("São Paulo", "Recife", 2130), ("São Paulo", "Fortaleza", 2370),
    ("São Paulo", "Manaus", 2690), ("Rio de Janeiro", "Brasília", 930),
    ("Rio de Janeiro", "Salvador", 1210), ("Rio de Janeiro", "Fortaleza", 2190),
    ("Belo Horizonte", "Salvador", 960), ("Brasília", "Manaus", 1940),
    ("Curitiba", "Recife", 2690), ("Porto Alegre", "Fortaleza", 3210),
]
COMPANHIAS = {"VoaBrasil": 1.00, "CéuAberto": 1.12, "Andorinha": 0.85}
CLASSES = {"econômica": 1.00, "executiva": 2.60}
FATOR_ESCALAS = {0: 1.00, 1: 0.85, 2: 0.75}
DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
FATOR_DIA = {"sex": 1.15, "dom": 1.15, "ter": 0.92, "qua": 0.92}
MESES_ALTA = {12, 1, 7}
PERIODOS = {"madrugada": 0.88, "manhã": 1.05, "tarde": 1.00, "noite": 1.02}
DESVIO_RUIDO = 0.10


def preco_sem_ruido(df: pd.DataFrame) -> np.ndarray:
    """A regra do preço, sem o ruído. Serve também para conferir o que o modelo aprendeu."""
    return ((90 + 0.38 * df["distancia_km"])
            * df["companhia"].map(COMPANHIAS)
            * df["classe"].map(CLASSES)
            * (1 + 1.2 * np.exp(-df["antecedencia_dias"] / 12))
            * df["escalas"].map(FATOR_ESCALAS)
            * df["dia_semana"].map(lambda d: FATOR_DIA.get(d, 1.0))
            * np.where(df["alta_temporada"] == 1, 1.25, 1.0)
            * df["periodo_dia"].map(PERIODOS)).to_numpy(dtype=float)


def gerar(n: int = 5000, semente: int = SEMENTE) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    rotas = [ROTAS[i] for i in rng.integers(len(ROTAS), size=n)]
    mes = rng.integers(1, 13, size=n)
    df = pd.DataFrame({
        "origem": [r[0] for r in rotas],
        "destino": [r[1] for r in rotas],
        "distancia_km": [r[2] for r in rotas],
        "companhia": rng.choice(list(COMPANHIAS), size=n, p=[0.40, 0.35, 0.25]),
        "classe": rng.choice(list(CLASSES), size=n, p=[0.85, 0.15]),
        # muita gente compra perto da viagem: distribuição exponencial truncada em 180 dias
        "antecedencia_dias": np.minimum(rng.exponential(30, size=n).astype(int), 180),
        "escalas": rng.choice([0, 1, 2], size=n, p=[0.55, 0.35, 0.10]),
        "dia_semana": rng.choice(DIAS_SEMANA, size=n),
        "mes": mes,
        "alta_temporada": np.isin(mes, list(MESES_ALTA)).astype(int),
        "periodo_dia": rng.choice(list(PERIODOS), size=n, p=[0.10, 0.35, 0.30, 0.25]),
    })
    ruido = rng.lognormal(0.0, DESVIO_RUIDO, size=n)
    df["preco"] = np.round(preco_sem_ruido(df) * ruido, 2)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="quantidade de passagens")
    a = p.parse_args()
    df = gerar(a.n)
    ARQUIVO.parent.mkdir(exist_ok=True)
    df.to_csv(ARQUIVO, index=False, encoding="utf-8")
    print(f"{len(df)} passagens gravadas em {ARQUIVO}")
    print(df.head().to_string(index=False))
    print("\nPreço (R$):", df["preco"].describe().round(2).to_dict())


if __name__ == "__main__":
    main()
