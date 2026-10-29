"""Testes de fumaça da interface e do caminho de classificação (usam os modelos reais)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import configuracao as cfg
import modelos
from pre_processamento import extrair_atributos, preparar_imagem

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(not modelos.caminho_modelo("knn").exists(), reason="modelos ainda não treinados")
PAGINAS = ["Visão geral", "Classificar imagem", "Comparação dos modelos", "Matrizes de confusão", "Entenda o código"]


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_roda_sem_erro(pagina):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(pagina).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("chave", list(modelos.NOMES_MODELOS))
def test_classificar_imagem_real_devolve_probabilidades_validas(chave):
    imagem = next(cfg.PASTA_CACHORROS.rglob("*.jpg"), None)
    if imagem is None:
        pytest.skip("base de imagens não encontrada")
    preparar = preparar_imagem if modelos.ENTRADA_MODELO[chave] == "pixels" else extrair_atributos
    x = preparar(imagem).reshape(1, -1)
    modelo = modelos.carregar_modelo(chave)
    proba = modelos.probabilidades(modelo, x)
    assert abs(sum(proba.values()) - 1) < 1e-9
    assert int(modelo.predict(x)[0]) == max(proba, key=proba.get)
