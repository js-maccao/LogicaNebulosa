"""Métricas, matrizes de confusão e análise de erros.

Convenção: classe positiva = Cachorro (rótulo 1); negativa = Não cachorro (0).

Matriz de confusão (linhas = classe REAL, colunas = classe PREVISTA):

                     previsto 0 (Não cachorro)   previsto 1 (Cachorro)
  real 0 (Não cach.)   VN  verdadeiro negativo     FP  falso positivo
  real 1 (Cachorro)    FN  falso negativo          VP  verdadeiro positivo

  VP: era cachorro e o modelo disse cachorro.
  VN: não era cachorro e o modelo disse não cachorro.
  FP: NÃO era cachorro, mas o modelo disse cachorro (alarme falso).
  FN: ERA cachorro, mas o modelo disse não cachorro (cachorro perdido).
"""
import matplotlib

matplotlib.use("Agg")  # gera figuras sem abrir janela (necessário no Streamlit e em scripts)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score)

import configuracao as cfg
from modelos import coluna_da_classe

METRICAS = ["Acurácia", "Precisão", "Recall", "F1-score", "F2-score"]


def calcular_metricas(y_real, y_previsto) -> dict:
    """As cinco métricas pedidas, com as funções oficiais do scikit-learn.

    - Acurácia: % de acertos no total.
    - Precisão: dos que o modelo chamou de cachorro, quantos eram cachorro.
    - Recall: dos cachorros reais, quantos o modelo encontrou.
    - F1: média harmônica de precisão e recall (pesos iguais).
    - F2: como o F1, mas o recall pesa mais (beta=2): perder um cachorro custa mais.

    pos_label=1 diz que "positivo" = cachorro. zero_division=0 evita erro/aviso
    quando precisão ou recall teriam divisão por zero (ex.: o modelo nunca prevê
    cachorro): nesse caso a métrica vale 0.
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
    """resultados = {nome_do_modelo: (y_real, y_previsto)} -> DataFrame (modelo x métrica).

    Os valores guardam a precisão total; só a EXIBIÇÃO usa 4 casas decimais.
    """
    linhas = {nome: calcular_metricas(y, yp) for nome, (y, yp) in resultados.items()}
    df = pd.DataFrame.from_dict(linhas, orient="index", columns=METRICAS)
    df.index.name = "Modelo"
    return df


def formatar_4_casas(df: pd.DataFrame):
    return df.style.format("{:.4f}")


def matriz_confusao(y_real, y_previsto) -> np.ndarray:
    """Matriz 2x2; labels=[0, 1] fixa a ordem: linha/coluna 0 = Não cachorro, 1 = Cachorro."""
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
    """Probabilidade da classe 1 para cada linha de X, achando a coluna por `classes_`."""
    return modelo.predict_proba(X)[:, coluna_da_classe(modelo, cfg.ROTULO_CACHORRO)]


def erros_por_grupo(meta_teste: pd.DataFrame, previsoes: dict) -> pd.DataFrame:
    """Quantos erros cada modelo cometeu em cada grupo (cachorro, cat, chicken, ...).

    meta_teste: DataFrame com 'grupo' e 'rotulo'. previsoes: {nome_modelo: array de previsões}.
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
