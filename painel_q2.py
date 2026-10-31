"""Q2 no Streamlit: montar um voo, prever o preço e ver a pertinência fuzzy."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

Q2 = Path(__file__).resolve().parent / "questao2_regressao"
if str(Q2) not in sys.path:
    sys.path.insert(0, str(Q2))

import gerar_dados  # noqa: E402
import regressao as rg  # noqa: E402

ROTAS = {f"{o} → {d} ({km} km)": km for o, d, km in gerar_dados.ROTAS}


def brl(v):
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource(show_spinner="Treinando os dois modelos (cerca de 10 s)...")
def _treinar(assinatura):
    X, y = rg.carregar_dados()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=rg.FRACAO_TESTE, random_state=rg.SEMENTE)
    modelos = {}
    for chave, m in rg.criar_modelos().items():
        m.fit(X_tr, y_tr)
        modelos[chave] = m
    return modelos


def secao_prever():
    if not rg.ARQ_DADOS.exists():
        st.warning("Os dados ainda não existem. Rode o passo 1 desta página (gerar dados).")
        return
    modelos = _treinar(rg.ARQ_DADOS.stat().st_mtime)
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
    prev = {c: float(m.predict(voo)[0]) for c, m in modelos.items()}
    regra = float(gerar_dados.preco_sem_ruido(voo)[0])
    m1, m2, m3 = st.columns(3)
    m1.metric(rg.NOMES_MODELOS["linear"], brl(prev["linear"]))
    m2.metric(rg.NOMES_MODELOS["boosting"], brl(prev["boosting"]))
    m3.metric("Regra verdadeira (sem ruído)", brl(regra))
    p = prev["boosting"]
    graus = {f: float(g) for f, g in rg.pertinencias(p).items()}
    a, b = st.columns(2)
    a.write(f"**Crisp:** {rg.faixa_crisp(p)}")
    a.write("**Fuzzy:** " + ", ".join(f"{g:.0%} {f}" for f, g in graus.items() if g > 0))
    a.bar_chart(pd.Series(graus, name="pertinência"))
    fig = rg.figura_pertinencia([p])
    b.pyplot(fig)
    plt.close(fig)
