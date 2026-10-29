import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import gerar_dados
import regressao as rg


def test_geracao_e_reprodutivel():
    assert gerar_dados.gerar(300).equals(gerar_dados.gerar(300))
    assert not gerar_dados.gerar(300).equals(gerar_dados.gerar(300, semente=7))


def test_dados_validos():
    df = gerar_dados.gerar(2000)
    assert (df["preco"] > 0).all() and df.notna().all().all()
    assert df["antecedencia_dias"].between(0, 180).all()
    assert set(df["escalas"]) <= {0, 1, 2}


def test_regra_do_preco_num_caso_conhecido():
    voo = pd.DataFrame([{"distancia_km": 1000, "companhia": "VoaBrasil", "classe": "econômica",
                         "antecedencia_dias": 10_000, "escalas": 0, "dia_semana": "seg",
                         "alta_temporada": 0, "periodo_dia": "tarde"}])
    assert np.isclose(gerar_dados.preco_sem_ruido(voo)[0], 90 + 0.38 * 1000)   # todos os fatores = 1


def test_metricas_calculadas_a_mao():
    y, p = np.array([100.0, 200.0, 300.0]), np.array([110.0, 190.0, 330.0])
    m = rg.calcular_metricas(y, p)
    assert np.isclose(m["MAE"], (10 + 10 + 30) / 3)
    assert np.isclose(m["MSE"], (100 + 100 + 900) / 3)
    assert np.isclose(m["RMSE"], np.sqrt(m["MSE"]))
    assert np.isclose(m["R2"], 1 - 1100 / 20000)


def test_faixas_crisp():
    assert list(rg.faixa_crisp([100, 599.99, 600, 1199, 1200, 5000])) == \
        ["barata", "barata", "média", "média", "cara", "cara"]


def test_pertinencias_particao_e_cruzamento_nos_limites_crisp():
    precos = np.linspace(0, 3000, 301)
    graus = rg.pertinencias(precos)
    assert np.allclose(sum(graus.values()), 1.0)
    assert all(((g >= 0) & (g <= 1)).all() for g in graus.values())
    assert np.isclose(rg.pertinencias(600)["barata"], 0.5) and np.isclose(rg.pertinencias(1200)["cara"], 0.5)


def test_boosting_aprende_bem_e_supera_linear():
    df = gerar_dados.gerar(3000)
    X, y = df[rg.NUMERICAS + rg.CATEGORICAS], df[rg.ALVO].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    r2 = {c: rg.calcular_metricas(yte, m.fit(Xtr, ytr).predict(Xte))["R2"] for c, m in rg.criar_modelos().items()}
    assert r2["boosting"] > 0.9 and r2["boosting"] > r2["linear"]


def test_saidas_geradas_se_existirem():
    arq = rg.R / "metricas.csv"
    if not arq.exists():
        pytest.skip("rode python regressao.py")
    m = pd.read_csv(arq, index_col="Modelo")
    assert np.allclose(m["RMSE"], np.sqrt(m["MSE"]))
