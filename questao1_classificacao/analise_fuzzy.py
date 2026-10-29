"""Questão 1 (c): probabilidade de cada previsão e o aspecto FUZZY do problema.

Uso:  python analise_fuzzy.py        (antes, rode: python treinar.py)

Ideia central
-------------
A saída "crisp" (clássica) de um classificador é 0 ou 1: cachorro ou não. Mas,
antes de decidir, o modelo calcula P(cachorro) entre 0 e 1. Na lógica nebulosa
esse número é lido como GRAU DE PERTINÊNCIA ao conjunto "cachorro":

    μ_cachorro(imagem)     = P(cachorro)
    μ_não_cachorro(imagem) = 1 - P(cachorro)

Uma imagem com P = 0,97 é "muito cachorro"; com P = 0,52 é "um pouco mais
cachorro do que não", ou seja, quase totalmente ambígua. O corte em 0,5
(predict) é uma DEFUZZIFICAÇÃO: transforma o grau em uma resposta sim/não e
joga fora a informação de incerteza.

Para deixar isso explícito, definimos três termos linguísticos sobre P(cachorro)
(uma partição fuzzy: em qualquer ponto as três pertinências somam 1):

    "não cachorro": 1 até 0,25, cai linearmente até 0 em 0,50
    "incerto":      triângulo 0,25 -> 1 em 0,50 -> 0,75
    "cachorro":     0 até 0,50, sobe linearmente até 1 em 0,75

Gera em resultados/:
  pertinencia_fuzzy.png            funções de pertinência + onde caem as imagens de teste
  distribuicao_probabilidades.png  histograma de P(cachorro) por classe real, por modelo
  metricas_vs_limiar.png           métricas variando o limiar de corte (não só 0,5)
  exemplos_pertinencia.png         imagens reais com seus graus de pertinência
  curvas_roc.png, auc_teste.csv    curva ROC e ROC-AUC (métrica que usa a probabilidade, sem limiar)
  zona_incerta.csv                 acertos/erros por termo linguístico, por modelo
  pertinencia_teste.csv            μ de cada imagem de teste, por modelo
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

import avaliar
import configuracao as cfg
import modelos
from preparar_dados import caminho_absoluto

R = cfg.PASTA_RESULTADOS
ARQ_PREVISOES = R / "previsoes_teste.csv"

# Pontos que definem as funções de pertinência (sobre o eixo P(cachorro))
INICIO_INCERTO, CENTRO, FIM_INCERTO = 0.25, 0.50, 0.75
TERMOS = ["não cachorro", "incerto", "cachorro"]
CORES_TERMOS = {"não cachorro": "#3478d2", "incerto": "#f2a93b", "cachorro": "#d64541"}
LIMIARES = np.round(np.arange(0.05, 0.951, 0.05), 2)


# --------------------------------------------------------------------------
# Funções de pertinência
# --------------------------------------------------------------------------
def pertinencias(p) -> dict:
    """Grau de pertinência de cada termo linguístico para P(cachorro) = p (escalar ou array)."""
    p = np.asarray(p, dtype=float)
    largura = CENTRO - INICIO_INCERTO
    nao = np.clip((CENTRO - p) / largura, 0, 1)
    cachorro = np.clip((p - CENTRO) / (FIM_INCERTO - CENTRO), 0, 1)
    incerto = 1 - nao - cachorro                          # o que sobra: partição fuzzy (soma = 1)
    return {"não cachorro": nao, "incerto": incerto, "cachorro": cachorro}


def termo_dominante(p) -> np.ndarray:
    """Termo linguístico com maior pertinência (empate fica com o termo de menor índice)."""
    graus = pertinencias(p)
    matriz = np.vstack([graus[t] for t in TERMOS])
    return np.array(TERMOS)[matriz.argmax(axis=0)]


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------
def tabela_zonas(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Para cada modelo e termo dominante: quantas imagens e quantas a decisão crisp acertou."""
    linhas = []
    for chave, nome in modelos.NOMES_MODELOS.items():
        termo = termo_dominante(previsoes[f"proba_cachorro_{chave}"])
        acerto = (previsoes[f"previsto_{chave}"] == previsoes["rotulo"]).to_numpy()
        for t in TERMOS:
            m = termo == t
            linhas.append({"Modelo": nome, "Termo": t, "Imagens": int(m.sum()),
                           "Acertos": int(acerto[m].sum()), "Erros": int((~acerto[m]).sum()),
                           "Acurácia": float(acerto[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(linhas)


def tabela_pertinencias(previsoes: pd.DataFrame) -> pd.DataFrame:
    saida = previsoes[["caminho", "grupo", "rotulo"]].copy()
    for chave in modelos.NOMES_MODELOS:
        p = previsoes[f"proba_cachorro_{chave}"]
        saida[f"P_cachorro_{chave}"] = p
        for t, graus in pertinencias(p).items():
            saida[f"mu_{t.replace(' ', '_')}_{chave}"] = np.round(graus, 4)
        saida[f"termo_{chave}"] = termo_dominante(p)
    return saida


def metricas_por_limiar(y, p, limiares=LIMIARES) -> pd.DataFrame:
    """Métricas quando a decisão é 'cachorro se P(cachorro) >= limiar'."""
    linhas = {lim: avaliar.calcular_metricas(y, (np.asarray(p) >= lim).astype(int)) for lim in limiares}
    df = pd.DataFrame.from_dict(linhas, orient="index")
    df.index.name = "Limiar"
    return df


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def figura_pertinencia(previsoes, chave):
    eixo = np.linspace(0, 1, 501)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for t, graus in pertinencias(eixo).items():
        ax1.plot(eixo, graus, lw=2.5, color=CORES_TERMOS[t], label=f'μ "{t}"')
    ax1.plot(eixo, eixo, ls="--", color="gray", lw=1, label="μ cachorro = P(cachorro)")
    ax1.axvline(0.5, color="black", lw=1, ls=":")
    ax1.text(0.505, 1.03, "corte crisp 0,5", fontsize=8)
    ax1.set_ylabel("grau de pertinência μ")
    ax1.set_ylim(-0.03, 1.1)
    ax1.legend(loc="center left", fontsize=8)
    ax1.set_title(f"Termos linguísticos sobre P(cachorro) - imagens de teste do {modelos.NOMES_MODELOS[chave]}")
    ax1.grid(alpha=0.3)

    p = previsoes[f"proba_cachorro_{chave}"].to_numpy()
    y = previsoes["rotulo"].to_numpy()
    rng = np.random.default_rng(0)
    for rotulo, cor in ((0, CORES_TERMOS["não cachorro"]), (1, CORES_TERMOS["cachorro"])):
        m = y == rotulo
        ax2.scatter(p[m], rotulo + rng.uniform(-0.25, 0.25, m.sum()), s=12, alpha=0.6, color=cor)
    ax2.axvspan(0.375, 0.625, color=CORES_TERMOS["incerto"], alpha=0.15)
    ax2.set_yticks([0, 1], labels=["real: não cachorro", "real: cachorro"])
    ax2.set_xlabel("P(cachorro) estimada pelo modelo")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figura_distribuicao(previsoes):
    chaves = list(modelos.NOMES_MODELOS)
    fig, axs = plt.subplots(1, len(chaves), figsize=(4.2 * len(chaves), 3.6), sharey=True)
    caixas = np.linspace(0, 1, 21)
    for ax, chave in zip(np.atleast_1d(axs), chaves):
        p = previsoes[f"proba_cachorro_{chave}"]
        for rotulo, nome, cor in ((0, "real: não cachorro", "#3478d2"), (1, "real: cachorro", "#d64541")):
            ax.hist(p[previsoes["rotulo"] == rotulo], bins=caixas, alpha=0.55, color=cor, label=nome)
        ax.axvline(0.5, color="black", ls=":", lw=1)
        ax.set_title(modelos.NOMES_MODELOS[chave])
        ax.set_xlabel("P(cachorro)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("imagens de teste")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.suptitle("Distribuição das probabilidades no teste (quanto mais misturadas no meio, mais incerteza)")
    fig.tight_layout()
    return fig


def figura_limiares(previsoes):
    chaves = list(modelos.NOMES_MODELOS)
    fig, axs = plt.subplots(1, len(chaves), figsize=(4.2 * len(chaves), 3.8), sharey=True)
    for ax, chave in zip(np.atleast_1d(axs), chaves):
        df = metricas_por_limiar(previsoes["rotulo"], previsoes[f"proba_cachorro_{chave}"])
        for metrica in avaliar.METRICAS:
            ax.plot(df.index, df[metrica], marker="o", ms=3, label=metrica)
        ax.axvline(0.5, color="black", ls=":", lw=1)
        ax.set_title(modelos.NOMES_MODELOS[chave])
        ax.set_xlabel("limiar: cachorro se P(cachorro) ≥ limiar")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("O corte em 0,5 é uma escolha: mover o limiar troca precisão por recall")
    fig.tight_layout()
    return fig


def tabela_auc(previsoes) -> pd.DataFrame:
    """ROC-AUC: chance de uma imagem de cachorro qualquer receber P(cachorro) maior que uma de
    não cachorro qualquer. 0,5 = acaso; 1 = separa perfeitamente. Não depende do limiar."""
    linhas = {n: roc_auc_score(previsoes["rotulo"], previsoes[f"proba_cachorro_{c}"])
              for c, n in modelos.NOMES_MODELOS.items()}
    return pd.Series(linhas, name="ROC-AUC").rename_axis("Modelo").to_frame()


def figura_roc(previsoes, auc):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for chave, nome in modelos.NOMES_MODELOS.items():
        fpr, tpr, _ = roc_curve(previsoes["rotulo"], previsoes[f"proba_cachorro_{chave}"])
        ax.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {auc.loc[nome, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="acaso (AUC = 0,5)")
    ax.set_xlabel("taxa de falsos positivos (1 - especificidade)")
    ax.set_ylabel("taxa de verdadeiros positivos (recall)")
    ax.set_title("Curva ROC no teste: todos os limiares de uma vez")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def escolher_exemplos(previsoes, chave, n_por_tipo=3):
    """Exemplos: decisões seguras e certas, casos incertos e erros 'confiantes'."""
    p = previsoes[f"proba_cachorro_{chave}"]
    certeza = (p - 0.5).abs()
    acerto = previsoes[f"previsto_{chave}"] == previsoes["rotulo"]
    seguros = previsoes[acerto].assign(c=certeza).sort_values("c", ascending=False)
    seguros = pd.concat([seguros[seguros["rotulo"] == 1].head(2), seguros[seguros["rotulo"] == 0].head(1)])
    incertos = previsoes.assign(c=certeza).sort_values("c").head(n_por_tipo)
    errados = previsoes[~acerto].assign(c=certeza).sort_values("c", ascending=False).head(n_por_tipo)
    return [("seguro e certo", seguros), ("incerto", incertos), ("errou com confiança", errados)]


def figura_exemplos(previsoes, chave):
    grupos = escolher_exemplos(previsoes, chave)
    n_col = max(len(g) for _, g in grupos)
    fig, axs = plt.subplots(len(grupos), n_col, figsize=(3.1 * n_col, 3.5 * len(grupos)))
    for i, (tipo, linhas) in enumerate(grupos):
        for j in range(n_col):
            ax = axs[i, j]
            ax.axis("off")
            if j >= len(linhas):
                continue
            linha = linhas.iloc[j]
            arq = caminho_absoluto(linha["caminho"])
            if arq.exists():
                with Image.open(arq) as img:
                    ax.imshow(img.convert("RGB"))
            p = linha[f"proba_cachorro_{chave}"]
            graus = {t: float(v) for t, v in pertinencias(p).items()}
            real = cfg.NOMES_CLASSES[int(linha["rotulo"])]
            ax.set_title(f"{tipo}\nreal: {real} ({cfg.NOMES_GRUPOS_PT.get(linha['grupo'], linha['grupo'])})\n"
                         f"P(cachorro) = {p:.2f}\n"
                         f"μ não={graus['não cachorro']:.2f} | inc={graus['incerto']:.2f} | "
                         f"cach={graus['cachorro']:.2f}", fontsize=8)
    fig.suptitle(f"Graus de pertinência em imagens reais de teste - {modelos.NOMES_MODELOS[chave]}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def modelo_de_referencia() -> str:
    """Modelo usado nas figuras de um só modelo: o de maior acurácia no teste."""
    metricas = pd.read_csv(R / "metricas_teste.csv", index_col="Modelo", encoding="utf-8")
    melhor = metricas["Acurácia"].idxmax()
    return next(c for c, n in modelos.NOMES_MODELOS.items() if n == melhor)


def main():
    if not ARQ_PREVISOES.exists():
        raise FileNotFoundError("Previsões não encontradas. Rode antes: python treinar.py")
    previsoes = pd.read_csv(ARQ_PREVISOES, encoding="utf-8")
    chave = modelo_de_referencia()

    zonas = tabela_zonas(previsoes)
    zonas.to_csv(R / "zona_incerta.csv", index=False, encoding="utf-8")
    tabela_pertinencias(previsoes).to_csv(R / "pertinencia_teste.csv", index=False, encoding="utf-8")
    auc = tabela_auc(previsoes)
    auc.to_csv(R / "auc_teste.csv", encoding="utf-8")

    for nome, fig in [("pertinencia_fuzzy", figura_pertinencia(previsoes, chave)),
                      ("distribuicao_probabilidades", figura_distribuicao(previsoes)),
                      ("metricas_vs_limiar", figura_limiares(previsoes)),
                      ("exemplos_pertinencia", figura_exemplos(previsoes, chave)),
                      ("curvas_roc", figura_roc(previsoes, auc))]:
        fig.savefig(R / f"{nome}.png", dpi=130)
        plt.close(fig)

    print("Acertos da decisão crisp (corte 0,5) por termo linguístico dominante:")
    print(zonas.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nROC-AUC no teste (usa a probabilidade inteira, não só o corte em 0,5):")
    print(auc.round(4).to_string())
    print("\nValores distintos de P(cachorro) no teste (KNN só tem frações de 5 vizinhos):")
    for c, n in modelos.NOMES_MODELOS.items():
        print(f"  {n:<20} {previsoes[f'proba_cachorro_{c}'].nunique():>4} valores distintos")
    print(f"\nFiguras de um modelo só usam o de maior acurácia no teste: {modelos.NOMES_MODELOS[chave]}")
    print("Resultados em", R)


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\nERRO na análise fuzzy: {erro!r}", file=sys.stderr)
        raise SystemExit(1)
