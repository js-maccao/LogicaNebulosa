import numpy as np
import pytest

import configuracao as cfg
import modelos


def _dados_pequenos():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.2, 0.05, (30, cfg.N_ATRIBUTOS)), rng.normal(0.8, 0.05, (30, cfg.N_ATRIBUTOS))])
    y = np.array([0] * 30 + [1] * 30)
    return X.astype(np.float32), y


@pytest.mark.parametrize("chave", list(modelos.NOMES_MODELOS))
def test_salvar_e_carregar_dao_mesmas_previsoes(tmp_path, monkeypatch, chave):
    monkeypatch.setattr(cfg, "PASTA_MODELOS", tmp_path)
    X, y = _dados_pequenos()
    modelo = modelos.criar_modelos()[chave].fit(X, y)
    modelos.salvar_modelo(modelo, chave)
    carregado = modelos.carregar_modelo(chave)
    assert np.array_equal(modelo.predict(X), carregado.predict(X))
    proba = modelos.probabilidades(carregado, X[:1])
    assert set(proba) == {0, 1} and abs(sum(proba.values()) - 1) < 1e-9
    assert list(carregado.classes_) == [0, 1]


def test_modelo_inexistente_da_erro_claro(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PASTA_MODELOS", tmp_path)
    with pytest.raises(FileNotFoundError, match="treinar.py"):
        modelos.carregar_modelo("knn")


def test_parametros_pedidos():
    assert modelos.criar_knn().named_steps["knn"].n_neighbors == 5
    rf = modelos.criar_random_forest()
    assert rf.n_estimators == 100 and rf.random_state == 42
    svm = modelos.criar_svm_atributos().named_steps["svm"].estimator
    assert svm.kernel == "rbf" and svm.C == cfg.SVM_C


def test_todo_modelo_tem_entrada_definida():
    assert set(modelos.ENTRADA_MODELO) == set(modelos.NOMES_MODELOS)
    assert set(modelos.ENTRADA_MODELO.values()) <= {"pixels", "atributos"}


def test_modelos_reais_salvos_se_existirem():
    if not modelos.caminho_modelo("knn").exists():
        pytest.skip("modelos ainda não treinados")
    for chave in modelos.NOMES_MODELOS:
        m = modelos.carregar_modelo(chave)
        n = cfg.N_ATRIBUTOS if modelos.ENTRADA_MODELO[chave] == "pixels" else cfg.N_ATRIBUTOS_EXTRAIDOS
        assert m.predict(np.zeros((1, n), dtype=np.float32)).shape == (1,)
