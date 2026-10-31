"""Métricas, matrizes de confusão e erros por grupo.

Classe positiva = Cachorro (1). Na matriz, linhas = classe real e colunas = prevista:

                 previsto 0   previsto 1
    real 0          VN           FP        (FP: animal tomado por cachorro)
    real 1          FN           VP        (FN: cachorro perdido)
"""
import matplotlib

matplotlib.use("Agg")  # só gera as figuras, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score)

import configuracao as cfg
from modelos import coluna_da_classe

METRICAS = ["Acurácia", "Precisão", "Recall", "F1-score", "F2-score"]


def calcular_metricas(y_real, y_previsto) -> dict:
    """Acurácia, precisão, recall, F1 e F2 (o F2 dá mais peso ao recall).

    zero_division=0: se o modelo nunca prevê cachorro, a precisão vale 0 em vez de dar aviso.
    """
    kw = dict(pos_label=cfg.ROTULO_CACHORRO, zero_division=0)
    return {
        "Acurácia": accuracy_score(y_real, y_previsto),
        "Precisão": precision_score(y_real, y_previsto, **kw),
        "Recall": recall_score(y_real, y_previsto, **kw),
        "F1-score": f1_score(y_real, y_previsto, **kw),
        "F2-score": fbeta_score(y_real, y_previsto, beta=2, **kw),
    }


def tabela_metricas(resultados: dict) -> pd.DataFrame:
    """{nome_do_modelo: (y_real, y_previsto)} -> tabela com uma linha por modelo."""
    linhas = {nome: calcular_metricas(y, yp) for nome, (y, yp) in resultados.items()}
    df = pd.DataFrame.from_dict(linhas, orient="index", columns=METRICAS)
    df.index.name = "Modelo"
    return df


def matriz_confusao(y_real, y_previsto) -> np.ndarray:
    """Matriz 2x2. labels=[0, 1] garante a ordem mesmo se faltar uma classe."""
    return confusion_matrix(y_real, y_previsto, labels=[0, 1])


def vp_vn_fp_fn(matriz: np.ndarray) -> dict:
    vn, fp, fn, vp = matriz.ravel()
    return {"VP": int(vp), "VN": int(vn), "FP": int(fp), "FN": int(fn)}


def desenhar_matriz(matriz: np.ndarray, titulo: str):
    """Figura da matriz de confusão (eixo vertical = real, horizontal = previsto)."""
    nomes = [cfg.NOMES_CLASSES[0], cfg.NOMES_CLASSES[1]]
    rotulos_celula = [["VN", "FP"], ["FN", "VP"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(matriz, cmap="Blues")
    limite = matriz.max() / 2
    for i in range(2):
        for j in range(2):
            cor = "white" if matriz[i, j] > limite else "black"
            ax.text(j, i, f"{matriz[i, j]}\n{rotulos_celula[i][j]}", ha="center", va="center",
                    color=cor, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], labels=[f"0 - {nomes[0]}", f"1 - {nomes[1]}"])
    ax.set_yticks([0, 1], labels=[f"0 - {nomes[0]}", f"1 - {nomes[1]}"])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def desenhar_comparacao(df_metricas: pd.DataFrame, titulo: str):
    """Barras agrupadas: uma cor por modelo, uma posição por métrica."""
    fig, ax = plt.subplots(figsize=(8, 4))
    largura = 0.8 / len(df_metricas)
    x = np.arange(len(METRICAS))
    for i, (modelo, linha) in enumerate(df_metricas.iterrows()):
        barras = ax.bar(x + i * largura, linha[METRICAS].to_numpy(dtype=float), largura, label=modelo)
        ax.bar_label(barras, fmt="%.3f", fontsize=8)
    ax.set_xticks(x + largura * (len(df_metricas) - 1) / 2, labels=METRICAS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    return fig


def probabilidade_de_cachorro(modelo, X) -> np.ndarray:
    """P(cachorro) para cada linha de X."""
    return modelo.predict_proba(X)[:, coluna_da_classe(modelo, cfg.ROTULO_CACHORRO)]


def erros_por_grupo(meta_teste: pd.DataFrame, previsoes: dict) -> pd.DataFrame:
    """Quantos erros cada modelo cometeu em cada grupo (cachorro, cat, chicken, ...).

    previsoes = {nome_do_modelo: array com as previsões no teste}
    """
    linhas = []
    for grupo in cfg.GRUPOS:
        mascara = (meta_teste["grupo"] == grupo).to_numpy()
        linha = {"Grupo": grupo, "Imagens no teste": int(mascara.sum())}
        for nome, prev in previsoes.items():
            erros = prev[mascara] != meta_teste["rotulo"].to_numpy()[mascara]
            linha[f"Erros {nome}"] = int(erros.sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)
