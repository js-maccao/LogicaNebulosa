"""Interface Streamlit da Questão 2:  streamlit run app.py

Os modelos são treinados na primeira abertura (cerca de 10 s) com a mesma divisão
80/20 e a mesma semente de regressao.py, e ficam em cache. As tabelas e figuras
de resultados/ são as geradas por `python regressao.py`.
"""
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

import gerar_dados
import regressao as rg

R = rg.R
AVISO_SINTETICO = (
    "**Atenção: os dados são SINTÉTICOS.** Não existe base pública de preços de passagens que rode offline "
    "e sem cadastro, então `gerar_dados.py` cria o preço a partir de uma regra conhecida mais ruído de ~10%. "
    "As companhias são fictícias e as distâncias são aproximadas. O modelo aprende essa regra, não o mercado real."
)
ROTAS = {f"{o} → {d} ({km} km)": km for o, d, km in gerar_dados.ROTAS}


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource(show_spinner="Treinando os dois modelos (cerca de 10 s, só na primeira vez)...")
def treinar_modelos():
    """Mesma divisão e semente de regressao.main()."""
    X, y = rg.carregar_dados()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=rg.FRACAO_TESTE, random_state=rg.SEMENTE)
    modelos = {}
    for chave, modelo in rg.criar_modelos().items():
        modelo.fit(X_tr, y_tr)
        modelos[chave] = modelo
    return modelos, X_te, y_te


@st.cache_data
def ler_csv(nome: str, **kw):
    return pd.read_csv(R / nome, encoding="utf-8", **kw)


def exigir_dados():
    if not rg.ARQ_DADOS.exists():
        st.error("Os dados ainda não foram gerados. No terminal, rode:\n\n`python gerar_dados.py`")
        st.stop()


def exigir_resultados():
    exigir_dados()
    if not (R / "metricas.csv").exists():
        st.error("Os resultados ainda não foram gerados. No terminal, rode:\n\n`python regressao.py`")
        st.stop()


def barras_pertinencia(graus: dict):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.barh(list(graus), list(graus.values()), color=[rg.CORES_FAIXAS[f] for f in graus])
    ax.set_xlim(0, 1)
    ax.set_xlabel("grau de pertinência μ")
    for i, v in enumerate(graus.values()):
        ax.text(min(v + 0.02, 0.85), i, f"{v:.2f}", va="center")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Página 1 - Visão geral
# --------------------------------------------------------------------------
def pagina_visao_geral():
    st.title("✈️ Preço de passagens aéreas")
    st.caption("Questão 2 de Lógica Nebulosa: Regressão Linear vs Gradient Boosting")
    exigir_dados()

    st.header("Objetivo")
    st.write("Prever o **preço (R$)** de uma passagem, uma variável **contínua**, a partir das características "
             "do voo, comparando uma Regressão Linear (referência) com um Gradient Boosting.")
    st.warning(AVISO_SINTETICO)

    st.header("Dados")
    df = pd.read_csv(rg.ARQ_DADOS, encoding="utf-8")
    c1, c2, c3 = st.columns(3)
    c1.metric("Passagens", f"{len(df):,}".replace(",", "."))
    c2.metric("Preço mediano", brl(df[rg.ALVO].median()))
    c3.metric("Faixa de preço", f"{brl(df[rg.ALVO].min())} a {brl(df[rg.ALVO].max())}")
    st.dataframe(df.head(10), hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df[rg.ALVO], bins=60, color="#4c72b0")
    for lim in rg.LIMITES_CRISP:
        ax.axvline(lim, color="black", ls=":", lw=1)
    ax.set_xlabel("preço (R$)")
    ax.set_ylabel("passagens")
    ax.set_title("Distribuição do preço (linhas pontilhadas = limites das faixas crisp)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.header("A regra que gera o preço")
    st.code(gerar_dados.__doc__.split("Regra do preço (todos os fatores multiplicam):")[1].strip("\n"),
            language="text")
    st.caption("Como a regra é conhecida, dá para conferir se o modelo aprendeu a coisa certa "
               "(veja a importância das variáveis na página de comparação).")

    st.header("Fluxo")
    st.markdown("`passagens.csv` → `divisão 80/20 (semente 42)` → `StandardScaler + OneHotEncoder` → "
                "`Regressão Linear | Gradient Boosting` → `MAE, MSE, RMSE, R²` → `faixas crisp x fuzzy`")


# --------------------------------------------------------------------------
# Página 2 - Prever o preço
# --------------------------------------------------------------------------
def pagina_prever():
    st.title("Prever o preço de uma passagem")
    exigir_dados()
    modelos, _, _ = treinar_modelos()
    st.info("Monte um voo e veja o preço previsto pelos dois modelos e a pertinência fuzzy nas faixas "
            "barata / média / cara. É uma estimativa sobre dados sintéticos.")

    c1, c2, c3 = st.columns(3)
    rota = c1.selectbox("Rota", list(ROTAS))
    companhia = c1.selectbox("Companhia", list(gerar_dados.COMPANHIAS))
    classe = c1.radio("Classe", list(gerar_dados.CLASSES), horizontal=True)
    antecedencia = c2.slider("Antecedência da compra (dias)", 0, 180, 30)
    escalas = c2.radio("Escalas", [0, 1, 2], horizontal=True)
    alta = c2.checkbox("Alta temporada (dez, jan, jul)")
    dia = c3.selectbox("Dia da semana", gerar_dados.DIAS_SEMANA, index=4)
    periodo = c3.selectbox("Período do dia", list(gerar_dados.PERIODOS), index=1)

    voo = pd.DataFrame([{"distancia_km": ROTAS[rota], "antecedencia_dias": antecedencia, "escalas": escalas,
                         "alta_temporada": int(alta), "companhia": companhia, "classe": classe,
                         "dia_semana": dia, "periodo_dia": periodo}])[rg.NUMERICAS + rg.CATEGORICAS]
    previsto = {c: float(m.predict(voo)[0]) for c, m in modelos.items()}
    regra = float(gerar_dados.preco_sem_ruido(voo)[0])

    st.subheader("Preço previsto")
    cols = st.columns(3)
    cols[0].metric(rg.NOMES_MODELOS["linear"], brl(previsto["linear"]), f"{previsto['linear'] - regra:+.0f} vs regra",
                   delta_color="off")
    cols[1].metric(rg.NOMES_MODELOS["boosting"], brl(previsto["boosting"]),
                   f"{previsto['boosting'] - regra:+.0f} vs regra", delta_color="off")
    cols[2].metric("Regra verdadeira (sem ruído)", brl(regra))
    st.caption("A regra verdadeira só existe porque os dados são sintéticos; o preço real teria ainda ~10% de ruído. "
               "Compare: em geral a Regressão Linear fica mais longe da regra que o Gradient Boosting.")

    st.subheader("Faixa de preço: crisp x fuzzy (Gradient Boosting)")
    p = previsto["boosting"]
    graus = {f: float(g) for f, g in rg.pertinencias(p).items()}
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Crisp:** a passagem é **{rg.faixa_crisp(p)}** (pertinência 1 nessa faixa e 0 nas outras).")
        st.write("**Fuzzy:** " + ", ".join(f"{g:.0%} {f}" for f, g in graus.items() if g > 0) + ".")
        fig = barras_pertinencia(graus)
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        fig = rg.figura_pertinencia([p])
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------------------------------
# Página 3 - Comparação
# --------------------------------------------------------------------------
def pagina_comparacao():
    st.title("Comparação dos modelos")
    exigir_resultados()
    metricas = ler_csv("metricas.csv", index_col="Modelo")

    st.header("Métricas no teste (1.000 passagens)")
    st.dataframe(metricas.style.format({"MAE": "{:.2f}", "MSE": "{:,.0f}", "RMSE": "{:.2f}", "R2": "{:.3f}"})
                 .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#c8e6c9")
                 .highlight_max(subset=["R2"], color="#c8e6c9"))
    melhor = metricas["R2"].idxmax()
    st.caption(f"Verde = melhor valor da coluna. Melhor modelo: **{melhor}**.")
    st.markdown(
        "- **MAE**: erro médio absoluto, em R$.\n"
        "- **MSE**: média dos erros ao quadrado (R²$); pune muito os erros grandes.\n"
        "- **RMSE**: raiz do MSE, de volta em R$. RMSE > MAE indica alguns erros grandes.\n"
        "- **R²**: fração da variação do preço explicada pelo modelo (1 = perfeito).")

    st.header("Previsto x real")
    st.image(str(R / "previsto_vs_real.png"))
    st.caption("Quanto mais perto da diagonal, melhor. A Regressão Linear desenha uma curva porque a regra é "
               "multiplicativa e não linear.")
    st.header("Resíduos")
    st.image(str(R / "residuos.png"))
    st.header("Importância das variáveis")
    st.image(str(R / "importancia_variaveis.png"))
    st.caption("Distância, classe e antecedência no topo: bate com a regra que gerou os dados.")

    st.header("Maiores erros do Gradient Boosting")
    prev = ler_csv("previsoes_teste.csv")
    prev["erro_abs"] = (prev["preco_real"] - prev["previsto_boosting"]).abs()
    st.dataframe(prev.sort_values("erro_abs", ascending=False).head(10)[
        ["distancia_km", "antecedencia_dias", "classe", "companhia", "preco_real", "previsto_boosting", "erro_abs"]
    ].round(2), hide_index=True)


# --------------------------------------------------------------------------
# Página 4 - Contínuo x discreto e fuzzy
# --------------------------------------------------------------------------
def pagina_fuzzy():
    st.title("Contínuo x discreto e lógica nebulosa")
    exigir_resultados()
    prev = ler_csv("previsoes_teste.csv")

    st.header("Discretizar perde informação")
    st.write(f"A regressão devolve **{prev['previsto_boosting'].round(2).nunique()} valores distintos** em "
             f"{len(prev)} passagens. Cortando em faixas crisp (barata < R$ {rg.LIMITES_CRISP[0]:.0f} ≤ média < "
             f"R$ {rg.LIMITES_CRISP[1]:.0f} ≤ cara) sobram 3 valores, e o problema vira uma classificação de 3 classes.")
    st.image(str(R / "continuo_vs_discreto.png"))
    acerto = (prev["faixa_real"] == prev["faixa_prevista"]).mean()
    st.metric("Acerto de faixa ao discretizar a previsão", f"{acerto:.1%}")

    st.header("Experimente: quanto custa?")
    preco = st.slider("Preço (R$)", 0, 2500, 600, step=10)
    graus = {f: float(g) for f, g in rg.pertinencias(preco).items()}
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write(f"**Crisp:** {rg.faixa_crisp(preco)}")
        for f, g in graus.items():
            st.write(f"μ {f}: **{g:.2f}**")
            st.progress(min(max(g, 0.0), 1.0))
        st.caption("Em R$ 590 e R$ 609 o crisp diz classes diferentes; o fuzzy diz 0,5 e 0,5. "
                   "Os graus sempre somam 1.")
    with c2:
        fig = rg.figura_pertinencia([float(preco)])
        st.pyplot(fig)
        plt.close(fig)

    st.header("Exemplos reais do teste")
    st.dataframe(ler_csv("exemplos_pertinencia.csv").round(3), hide_index=True)
    st.info("A regressão é a forma **não defuzzificada** do problema: na Q1 o modelo calcula um grau e o corte "
            "vira rótulo; na Q2 o resultado já é o valor contínuo e a classificação é uma etapa posterior.")


# --------------------------------------------------------------------------
# Página 5 - Entenda o código
# --------------------------------------------------------------------------
def mostrar_funcao(funcao):
    st.code(inspect.getsource(funcao), language="python")


def pagina_codigo():
    st.title("Entenda o código")
    st.markdown("Cada etapa, com o arquivo e a função responsável. Os trechos são lidos do código real.")

    with st.expander("1. Como os dados são gerados - `gerar_dados.py` → `preco_sem_ruido()`", expanded=True):
        st.write("A regra multiplicativa do preço; o ruído lognormal (~10%) é somado depois.")
        mostrar_funcao(gerar_dados.preco_sem_ruido)
    with st.expander("2. Como os dados são lidos - `regressao.py` → `carregar_dados()`"):
        st.write("**X** = características do voo; **y** = preço em R$.")
        mostrar_funcao(rg.carregar_dados)
    with st.expander("3. Pré-processamento - `regressao.py` → `preprocessador()`"):
        st.write("Numéricas padronizadas, categóricas em colunas 0/1. Fica dentro do `Pipeline`, então é "
                 "ajustado só no treino (sem vazamento).")
        mostrar_funcao(rg.preprocessador)
    with st.expander("4. Os modelos - `regressao.py` → `criar_modelos()`"):
        mostrar_funcao(rg.criar_modelos)
    with st.expander("5. Divisão e treino - `app.py` → `treinar_modelos()` (igual a `regressao.main()`)"):
        mostrar_funcao(treinar_modelos.__wrapped__)
    with st.expander("6. Métricas - `regressao.py` → `calcular_metricas()`"):
        mostrar_funcao(rg.calcular_metricas)
    with st.expander("7. Faixas crisp - `regressao.py` → `faixa_crisp()`"):
        mostrar_funcao(rg.faixa_crisp)
    with st.expander("8. Pertinência fuzzy - `regressao.py` → `pertinencias()`"):
        st.write("'média' é 1 menos as outras duas, por isso a soma é sempre 1.")
        mostrar_funcao(rg.pertinencias)
    with st.expander("9. Fluxo completo - `regressao.py` → `main()`"):
        mostrar_funcao(rg.main)


# --------------------------------------------------------------------------
PAGINAS = {
    "Visão geral": pagina_visao_geral,
    "Prever preço": pagina_prever,
    "Comparação dos modelos": pagina_comparacao,
    "Contínuo x fuzzy": pagina_fuzzy,
    "Entenda o código": pagina_codigo,
}

st.set_page_config(page_title="Preço de passagens", page_icon="✈️", layout="wide")
pagina = st.sidebar.radio("Navegação", list(PAGINAS))
st.sidebar.caption("Regressão Linear vs Gradient Boosting")
PAGINAS[pagina]()
