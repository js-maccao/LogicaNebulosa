import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import avaliar

Y_REAL = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PREV = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])   # VP=3 FN=1 FP=1 VN=5


def test_metricas_calculadas_a_mao():
    m = avaliar.calcular_metricas(Y_REAL, Y_PREV)
    precisao, recall = 3 / 4, 3 / 4
    assert m["Acurácia"] == 0.8
    assert m["Precisão"] == precisao and m["Recall"] == recall
    assert np.isclose(m["F1-score"], 2 * precisao * recall / (precisao + recall))
    assert np.isclose(m["F2-score"], 5 * precisao * recall / (4 * precisao + recall))


def test_zero_division_nao_quebra():
    m = avaliar.calcular_metricas(np.array([1, 1, 0]), np.array([0, 0, 0]))
    assert m["Precisão"] == 0 and m["Recall"] == 0 and m["F1-score"] == 0


def test_matriz_de_confusao_ordem_e_valores():
    m = avaliar.matriz_confusao(Y_REAL, Y_PREV)
    assert m.tolist() == [[5, 1], [1, 3]]            # linhas = real (0,1); colunas = previsto (0,1)
    assert avaliar.vp_vn_fp_fn(m) == {"VP": 3, "VN": 5, "FP": 1, "FN": 1}


def test_matriz_tem_2x2_mesmo_sem_uma_classe():
    assert avaliar.matriz_confusao([1, 1], [1, 1]).shape == (2, 2)


def test_tabela_metricas_uma_linha_por_modelo():
    df = avaliar.tabela_metricas({"A": (Y_REAL, Y_PREV), "B": (Y_REAL, Y_REAL)})
    assert list(df.index) == ["A", "B"] and list(df.columns) == avaliar.METRICAS
    assert df.loc["B"].eq(1.0).all()


def test_probabilidade_usa_classes_e_nao_posicao():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array([1, 1, 0, 0])                        # classe 1 aparece primeiro nos dados
    modelo = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert list(modelo.classes_) == [0, 1]            # mas classes_ vem ordenado
    p = avaliar.probabilidade_de_cachorro(modelo, np.array([[0.05], [5.05]]))
    assert p.tolist() == [1.0, 0.0]


def test_erros_por_grupo():
    meta = pd.DataFrame({"grupo": ["cachorro", "cachorro", "cat", "cat"], "rotulo": [1, 1, 0, 0]})
    df = avaliar.erros_por_grupo(meta, {"M": np.array([1, 0, 1, 0])}).set_index("Grupo")
    assert df.loc["cachorro", "Erros M"] == 1
    assert df.loc["cat", "Erros M"] == 1
    assert df.loc["cow", "Erros M"] == 0
