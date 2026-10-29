"""Questão 2: regressão do preço de passagens aéreas.

Uso:  python regressao.py        (antes, rode: python gerar_dados.py)

Fluxo:
  1. Lê dados/passagens.csv e separa X (características do voo) e y (preço em R$).
  2. Divide em treino (80%) e teste (20%) com semente fixa.
  3. Treina dois modelos: Regressão Linear (referência simples) e Gradient Boosting.
  4. Mede MAE, MSE, RMSE e R² no TESTE e gera os gráficos.
  5. (c) Contínuo x discreto: corta o preço em faixas crisp e mostra o que se perde.
  6. (d) Lógica fuzzy: graus de pertinência de "barata", "média" e "cara".

Gera em resultados/:
  metricas.csv, previsoes_teste.csv, exemplos_pertinencia.csv, faixas_matriz.csv
  previsto_vs_real.png, residuos.png, importancia_variaveis.png,
  continuo_vs_discreto.png, pertinencia_preco.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False   # "R$" nos textos não vira fórmula matemática
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PASTA = Path(__file__).resolve().parent
ARQ_DADOS = PASTA / "dados" / "passagens.csv"
R = PASTA / "resultados"
SEMENTE = 42
FRACAO_TESTE = 0.20

NUMERICAS = ["distancia_km", "antecedencia_dias", "escalas", "alta_temporada"]
CATEGORICAS = ["companhia", "classe", "dia_semana", "periodo_dia"]
ALVO = "preco"
NOMES_MODELOS = {"linear": "Regressão Linear", "boosting": "Gradient Boosting"}

# --- Faixas de preço (para comparar contínuo x discreto) -----------------------
FAIXAS = ["barata", "média", "cara"]
LIMITES_CRISP = (600.0, 1200.0)              # barata < 600 <= média < 1200 <= cara
# Funções de pertinência fuzzy: "barata" vale 1 até R$ 300 e cai até 0 em R$ 900;
# "média" é um triângulo 300 -> 900 -> 1500; "cara" sobe de 0 em R$ 900 até 1 em R$ 1500.
# Elas se cruzam (μ = 0,5) exatamente nos limites crisp 600 e 1200.
PONTOS_FUZZY = (300.0, 900.0, 1500.0)
CORES_FAIXAS = {"barata": "#3a9a5b", "média": "#f2a93b", "cara": "#d64541"}


# --------------------------------------------------------------------------
# Dados e modelos
# --------------------------------------------------------------------------
def carregar_dados():
    if not ARQ_DADOS.exists():
        raise FileNotFoundError("Dados não encontrados. Rode antes: python gerar_dados.py")
    df = pd.read_csv(ARQ_DADOS, encoding="utf-8")
    return df[NUMERICAS + CATEGORICAS], df[ALVO].to_numpy(dtype=float)


def preprocessador():
    """Números são padronizados (média 0, desvio 1); categorias viram colunas 0/1 (one-hot).

    Fica dentro do Pipeline: é ajustado só com o treino e reaplicado igual no teste.
    """
    return ColumnTransformer([
        ("numericas", StandardScaler(), NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def criar_modelos() -> dict:
    """Regressão Linear: preço = soma de pesos x atributos (uma reta/plano).
    Gradient Boosting: soma de muitas árvores pequenas, cada uma corrigindo o erro das anteriores;
    capta efeitos não lineares (ex.: antecedência) e combinações (ex.: executiva x distância).
    """
    return {
        "linear": Pipeline([("prep", preprocessador()), ("modelo", LinearRegression())]),
        "boosting": Pipeline([("prep", preprocessador()),
                              ("modelo", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                                       random_state=SEMENTE))]),
    }


def calcular_metricas(y_real, y_previsto) -> dict:
    """MAE: erro médio em R$. MSE: média dos erros ao quadrado (pune erros grandes).
    RMSE: raiz do MSE, de volta em R$. R²: fração da variação do preço explicada (1 = perfeito)."""
    mse = mean_squared_error(y_real, y_previsto)
    return {"MAE": mean_absolute_error(y_real, y_previsto), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_real, y_previsto)}


# --------------------------------------------------------------------------
# Contínuo x discreto e lógica fuzzy
# --------------------------------------------------------------------------
def faixa_crisp(preco) -> np.ndarray:
    """Discretização clássica: cada preço pertence a UMA faixa, com pertinência 0 ou 1."""
    preco = np.asarray(preco, dtype=float)
    return np.array(FAIXAS)[np.digitize(preco, LIMITES_CRISP)]


def pertinencias(preco) -> dict:
    """Graus de pertinência (0 a 1) em cada faixa; somam 1 para qualquer preço."""
    preco = np.asarray(preco, dtype=float)
    a, b, c = PONTOS_FUZZY
    barata = np.clip((b - preco) / (b - a), 0, 1)
    cara = np.clip((preco - b) / (c - b), 0, 1)
    return {"barata": barata, "média": 1 - barata - cara, "cara": cara}


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def figura_previsto_vs_real(y, previsoes, metricas):
    fig, axs = plt.subplots(1, len(previsoes), figsize=(5.5 * len(previsoes), 5), sharey=True)
    limite = max(y.max(), max(p.max() for p in previsoes.values())) * 1.03
    for ax, (chave, prev) in zip(np.atleast_1d(axs), previsoes.items()):
        ax.scatter(y, prev, s=8, alpha=0.4)
        ax.plot([0, limite], [0, limite], color="black", ls="--", lw=1, label="previsto = real")
        m = metricas.loc[NOMES_MODELOS[chave]]
        ax.set_title(f"{NOMES_MODELOS[chave]}\nR² = {m['R2']:.3f} | MAE = R$ {m['MAE']:.0f} | RMSE = R$ {m['RMSE']:.0f}")
        ax.set_xlabel("preço real (R$)")
        ax.set_xlim(0, limite)
        ax.set_ylim(0, limite)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    np.atleast_1d(axs)[0].set_ylabel("preço previsto (R$)")
    fig.suptitle("Valores previstos x valores reais (conjunto de teste)")
    fig.tight_layout()
    return fig


def figura_residuos(y, previsoes):
    fig, axs = plt.subplots(1, len(previsoes), figsize=(5.5 * len(previsoes), 4), sharey=True)
    for ax, (chave, prev) in zip(np.atleast_1d(axs), previsoes.items()):
        ax.scatter(prev, y - prev, s=8, alpha=0.4)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(NOMES_MODELOS[chave])
        ax.set_xlabel("preço previsto (R$)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("resíduo = real - previsto (R$)")
    fig.suptitle("Resíduos: um bom modelo tem erros espalhados em torno de zero, sem padrão")
    fig.tight_layout()
    return fig


def figura_importancia(modelo, X_teste, y_teste):
    """Permutation importance: quanto o R² cai quando embaralhamos uma coluna."""
    imp = permutation_importance(modelo, X_teste, y_teste, n_repeats=10, random_state=SEMENTE, scoring="r2")
    ordem = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(X_teste.columns)[ordem], imp.importances_mean[ordem], xerr=imp.importances_std[ordem])
    ax.set_xlabel("queda do R² ao embaralhar a variável")
    ax.set_title("Importância das variáveis (Gradient Boosting, teste)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def figura_continuo_vs_discreto(y, prev):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    # (a) 40 passagens de teste ordenadas pelo preço previsto: valor contínuo x faixa crisp
    rng = np.random.default_rng(SEMENTE)
    idx = np.sort(rng.choice(len(prev), 40, replace=False))
    idx = idx[np.argsort(prev[idx])]
    p = prev[idx]
    x = np.arange(len(p))
    for faixa, (inf, sup) in zip(FAIXAS, [(0, LIMITES_CRISP[0]), LIMITES_CRISP, (LIMITES_CRISP[1], p.max() * 1.1)]):
        ax1.axhspan(inf, sup, color=CORES_FAIXAS[faixa], alpha=0.12)
        ax1.text(len(p) - 0.5, (inf + min(sup, p.max() * 1.05)) / 2, faixa, ha="right", va="center",
                 color=CORES_FAIXAS[faixa], fontweight="bold")
    ax1.plot(x, p, "o-", color="#333", ms=4, label="previsão contínua (R$)")
    centros = {"barata": 300, "média": 900, "cara": 1800}
    ax1.step(x, [centros[f] for f in faixa_crisp(p)], where="mid", color="purple", lw=2,
             label="faixa crisp (só 3 valores possíveis)")
    ax1.set_xticks([])
    ax1.set_xlabel("40 passagens do teste, ordenadas pelo preço previsto")
    ax1.set_ylabel("preço (R$)")
    ax1.set_title("Contínuo: infinitos valores | Discreto: 3 caixas")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0, p.max() * 1.1)
    ax1.grid(alpha=0.3)

    # (b) a regressão vira classificação quando cortamos em faixas: matriz de confusão
    m = confusion_matrix(faixa_crisp(y), faixa_crisp(prev), labels=FAIXAS)
    ax2.imshow(m, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, m[i, j], ha="center", va="center", fontsize=13,
                     color="white" if m[i, j] > m.max() / 2 else "black")
    ax2.set_xticks(range(3), labels=FAIXAS)
    ax2.set_yticks(range(3), labels=FAIXAS)
    ax2.set_xlabel("faixa do preço previsto")
    ax2.set_ylabel("faixa do preço real")
    ax2.set_title(f"Discretizando a previsão: acerto de faixa = {np.trace(m) / m.sum():.1%}")
    fig.tight_layout()
    return fig, m


def figura_pertinencia(exemplos):
    eixo = np.linspace(0, 2500, 1001)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for faixa, graus in pertinencias(eixo).items():
        ax.plot(eixo, graus, lw=2.5, color=CORES_FAIXAS[faixa], label=f'μ "{faixa}"')
    for lim in LIMITES_CRISP:
        ax.axvline(lim, color="black", ls=":", lw=1)
    ax.text(LIMITES_CRISP[0] + 10, 1.05, "limites crisp", fontsize=8)
    for i, preco in enumerate(exemplos):
        graus = {f: float(g) for f, g in pertinencias(preco).items()}
        ax.axvline(preco, color="gray", lw=1)
        texto = "\n".join(f"{f}: {g:.2f}" for f, g in graus.items() if g > 0)
        ax.annotate(f"R$ {preco:.0f}\n{texto}", (preco, 0.82 - 0.3 * (i % 2)), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.set_xlabel("preço previsto (R$)")
    ax.set_ylabel("grau de pertinência μ")
    ax.set_ylim(-0.03, 1.15)
    ax.set_title("Lógica fuzzy sobre o preço: uma passagem pode ser 'um pouco média e um pouco cara'")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def main():
    R.mkdir(exist_ok=True)
    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=FRACAO_TESTE, random_state=SEMENTE)
    print(f"Treino: {len(X_treino)} passagens | Teste: {len(X_teste)} passagens")

    modelos, previsoes, linhas = {}, {}, {}
    for chave, modelo in criar_modelos().items():
        modelo.fit(X_treino, y_treino)                      # aprende só com o treino
        previsoes[chave] = modelo.predict(X_teste)          # um número real por passagem
        linhas[NOMES_MODELOS[chave]] = calcular_metricas(y_teste, previsoes[chave])
        modelos[chave] = modelo
    metricas = pd.DataFrame.from_dict(linhas, orient="index")
    metricas.index.name = "Modelo"
    print("\nMétricas no TESTE:")
    print(metricas.round(3).to_string())

    melhor = max(previsoes, key=lambda c: linhas[NOMES_MODELOS[c]]["R2"])
    prev = previsoes[melhor]

    saida = X_teste.reset_index(drop=True).assign(preco_real=y_teste)
    for chave, p in previsoes.items():
        saida[f"previsto_{chave}"] = np.round(p, 2)
    saida["faixa_real"] = faixa_crisp(y_teste)
    saida["faixa_prevista"] = faixa_crisp(prev)
    for faixa, graus in pertinencias(prev).items():
        saida[f"mu_{faixa}"] = np.round(graus, 3)

    # exemplos para a discussão: passagens perto e longe dos limites das faixas
    alvos = [450, 590, 610, 900, 1150, 1250, 2000]
    exemplos = saida.iloc[[int(np.argmin(np.abs(prev - a))) for a in alvos]]
    colunas_ex = ["preco_real", f"previsto_{melhor}", "faixa_prevista", "mu_barata", "mu_média", "mu_cara"]

    figuras = {
        "previsto_vs_real": figura_previsto_vs_real(y_teste, previsoes, metricas),
        "residuos": figura_residuos(y_teste, previsoes),
        "importancia_variaveis": figura_importancia(modelos[melhor], X_teste, y_teste),
        "pertinencia_preco": figura_pertinencia([prev[int(np.argmin(np.abs(prev - a)))] for a in (450, 610, 1150, 1900)]),
    }
    figuras["continuo_vs_discreto"], matriz_faixas = figura_continuo_vs_discreto(y_teste, prev)
    for nome, fig in figuras.items():
        fig.savefig(R / f"{nome}.png", dpi=130)
        plt.close(fig)

    metricas.to_csv(R / "metricas.csv", encoding="utf-8")
    saida.to_csv(R / "previsoes_teste.csv", index=False, encoding="utf-8")
    exemplos[colunas_ex].to_csv(R / "exemplos_pertinencia.csv", index=False, encoding="utf-8")
    pd.DataFrame(matriz_faixas, index=[f"real {f}" for f in FAIXAS],
                 columns=[f"previsto {f}" for f in FAIXAS]).to_csv(R / "faixas_matriz.csv", encoding="utf-8")

    print(f"\nMelhor modelo (maior R²): {NOMES_MODELOS[melhor]}")
    print(f"Valores distintos previstos no teste: {len(np.unique(np.round(prev, 2)))} de {len(prev)} "
          f"(contínuo) | faixas crisp possíveis: {len(FAIXAS)} (discreto)")
    print(f"Acerto de faixa ao discretizar a previsão: {np.trace(matriz_faixas) / matriz_faixas.sum():.1%}")
    print("\nExemplos de pertinência fuzzy (previsão do melhor modelo):")
    print(exemplos[colunas_ex].to_string(index=False))
    print("\nResultados em", R)


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\nERRO na regressão: {erro!r}", file=sys.stderr)
        raise SystemExit(1)
