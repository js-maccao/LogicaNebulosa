import numpy as np
import pandas as pd
import pytest

import analise_fuzzy as fz
import modelos


def test_particao_fuzzy_soma_1():
    p = np.linspace(0, 1, 101)
    graus = fz.pertinencias(p)
    assert np.allclose(sum(graus.values()), 1.0)
    assert all(((g >= 0) & (g <= 1)).all() for g in graus.values())


@pytest.mark.parametrize("p, esperado", [
    (0.0, {"não cachorro": 1, "incerto": 0, "cachorro": 0}),
    (0.25, {"não cachorro": 1, "incerto": 0, "cachorro": 0}),
    (0.5, {"não cachorro": 0, "incerto": 1, "cachorro": 0}),
    (0.625, {"não cachorro": 0, "incerto": 0.5, "cachorro": 0.5}),
    (1.0, {"não cachorro": 0, "incerto": 0, "cachorro": 1}),
])
def test_pontos_da_funcao(p, esperado):
    graus = fz.pertinencias(p)
    for termo, valor in esperado.items():
        assert np.isclose(graus[termo], valor)


def test_termo_dominante():
    assert list(fz.termo_dominante([0.1, 0.45, 0.55, 0.9])) == ["não cachorro", "incerto", "incerto", "cachorro"]


def test_tabela_zonas_conta_todas_as_imagens():
    n = 10
    p = np.linspace(0.02, 0.98, n)
    df = pd.DataFrame({"rotulo": [0, 1] * 5})
    for chave in modelos.NOMES_MODELOS:
        df[f"proba_cachorro_{chave}"] = p
        df[f"previsto_{chave}"] = (p >= 0.5).astype(int)
    zonas = fz.tabela_zonas(df)
    assert (zonas.groupby("Modelo")["Imagens"].sum() == n).all()
    assert (zonas["Acertos"] + zonas["Erros"] == zonas["Imagens"]).all()


def test_metricas_por_limiar_extremos():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    df = fz.metricas_por_limiar(y, p, [0.05, 0.5])
    assert df.loc[0.05, "Recall"] == 1.0 and df.loc[0.05, "Precisão"] == 0.5   # tudo vira cachorro
    assert df.loc[0.5, "Acurácia"] == 1.0


def test_auc_perfeita():
    df = pd.DataFrame({"rotulo": [0, 0, 1, 1]})
    for chave in modelos.NOMES_MODELOS:
        df[f"proba_cachorro_{chave}"] = [0.1, 0.2, 0.8, 0.9]
    assert (fz.tabela_auc(df)["ROC-AUC"] == 1.0).all()
