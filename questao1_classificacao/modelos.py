"""Criação, gravação e carregamento dos três modelos (KNN, Random Forest e SVM)."""
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import configuracao as cfg

NOMES_MODELOS = {"knn": "KNN", "random_forest": "Random Forest", "svm_atributos": "SVM (cor + forma)"}
# representação da imagem que cada modelo recebe (ver pre_processamento.py)
ENTRADA_MODELO = {"knn": "pixels", "random_forest": "pixels", "svm_atributos": "atributos"}


def criar_knn() -> Pipeline:
    """KNN: as 5 imagens de treino mais próximas votam.

    A probabilidade é a fração de votos, então só pode valer 0; 0,2; 0,4 ... 1.
    O StandardScaler fica dentro do Pipeline para ser ajustado só com o treino.
    """
    return Pipeline([
        ("padronizar", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=cfg.KNN_VIZINHOS)),
    ])


def criar_random_forest() -> RandomForestClassifier:
    """100 árvores de decisão; a probabilidade é a média delas. Não precisa padronizar."""
    return RandomForestClassifier(n_estimators=cfg.RF_ARVORES, random_state=cfg.SEMENTE, n_jobs=-1)


def criar_svm_atributos() -> Pipeline:
    """SVM com kernel RBF sobre os atributos de cor e forma.

    O SVC sozinho dá a distância até a fronteira, não uma probabilidade. O
    CalibratedClassifierCV ajusta uma sigmoide (Platt) que converte essa distância
    em probabilidade, usando validação cruzada interna só no treino.
    """
    return Pipeline([
        ("padronizar", StandardScaler()),
        ("svm", CalibratedClassifierCV(SVC(kernel="rbf", C=cfg.SVM_C), ensemble=False)),
    ])


def criar_modelos() -> dict:
    return {"knn": criar_knn(), "random_forest": criar_random_forest(), "svm_atributos": criar_svm_atributos()}


def caminho_modelo(chave: str):
    return cfg.PASTA_MODELOS / f"{chave}.joblib"


def salvar_modelo(modelo, chave: str):
    cfg.PASTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo, caminho_modelo(chave))


def carregar_modelo(chave: str):
    arq = caminho_modelo(chave)
    if not arq.exists():
        raise FileNotFoundError(f"Modelo '{chave}' não encontrado em {arq}. Rode: python treinar.py")
    return joblib.load(arq)


def coluna_da_classe(modelo, rotulo: int) -> int:
    """Posição da classe `rotulo` nas colunas de predict_proba().

    A ordem das colunas segue `modelo.classes_`, então procuro pelo valor
    em vez de supor que a coluna 1 é cachorro.
    """
    return int(np.where(modelo.classes_ == rotulo)[0][0])


def probabilidades(modelo, X) -> dict:
    """{rótulo: probabilidade} para a primeira linha de X."""
    proba = modelo.predict_proba(X)[0]
    return {int(classe): float(p) for classe, p in zip(modelo.classes_, proba)}
