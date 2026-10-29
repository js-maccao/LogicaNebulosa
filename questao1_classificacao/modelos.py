"""Criação, gravação e carregamento dos três modelos (KNN, Random Forest e SVM)."""
import json
import os

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
# Que representação da imagem cada modelo recebe (ver pre_processamento.py):
#   "pixels"    -> 4096 tons de cinza (preparar_imagem)
#   "atributos" -> histograma de cor HSV + HOG (extrair_atributos)
ENTRADA_MODELO = {"knn": "pixels", "random_forest": "pixels", "svm_atributos": "atributos"}


def criar_knn() -> Pipeline:
    """KNN com padronização dos atributos.

    KNN classifica por DISTÂNCIA: acha as k imagens de treino mais parecidas
    (menor distância entre os vetores de pixels) e deixa-as "votarem". A classe
    com mais votos vence; a proporção de votos vira a probabilidade
    (com k=5 os valores possíveis são 0, 0.2, 0.4, 0.6, 0.8 e 1.0).

    StandardScaler (média 0, desvio 1 em cada pixel) fica DENTRO do Pipeline, então
    é ajustado só com os dados de treino e reaplicado igual em qualquer imagem nova
    (sem vazamento de dados do teste).
    """
    return Pipeline([
        ("padronizar", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=cfg.KNN_VIZINHOS)),
    ])


def criar_random_forest() -> RandomForestClassifier:
    """Random Forest: 100 árvores de decisão treinadas em amostras diferentes.

    Cada árvore faz perguntas do tipo "o pixel 1234 é mais claro que 0.4?" até
    chegar a uma decisão. A floresta combina as árvores (média das probabilidades).
    Não precisa de padronização, pois as árvores só comparam cada pixel com um limiar.
    random_state=42 torna o treino reprodutível. n_jobs=-1 usa todos os núcleos.
    """
    return RandomForestClassifier(n_estimators=cfg.RF_ARVORES, random_state=cfg.SEMENTE, n_jobs=-1)


def criar_svm_atributos() -> Pipeline:
    """SVM (máquina de vetores de suporte) sobre atributos de cor e forma.

    O SVM procura a fronteira que separa as duas classes com a maior "margem"
    possível. O kernel RBF permite que essa fronteira seja curva. C controla o
    quanto a fronteira se dobra para acertar cada exemplo de treino (C maior =
    mais ajustada, maior risco de decorar o treino); C=1 foi o melhor na validação.

    O SVM, por si só, devolve uma DISTÂNCIA até a fronteira, não uma probabilidade.
    CalibratedClassifierCV aprende (com validação cruzada interna, só no treino)
    a converter essa distância em probabilidade entre 0 e 1 (calibração sigmoide/Platt).
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
    """Grava em arquivo temporário e só então renomeia: nunca deixa um .joblib pela metade."""
    cfg.PASTA_MODELOS.mkdir(exist_ok=True)
    destino = caminho_modelo(chave)
    temporario = destino.with_suffix(".tmp")
    joblib.dump(modelo, temporario)
    os.replace(temporario, destino)


def carregar_modelo(chave: str, pasta=None):
    """Carrega um modelo salvo; erro claro se ainda não foi treinado."""
    arq = (pasta or cfg.PASTA_MODELOS) / f"{chave}.joblib"
    if not arq.exists():
        raise FileNotFoundError(f"Modelo '{chave}' não encontrado em {arq}. Rode: python treinar.py")
    return joblib.load(arq)


def carregar_info_modelos():
    arq = cfg.PASTA_MODELOS / "info_modelos.json"
    if not arq.exists():
        raise FileNotFoundError("Informações dos modelos não encontradas. Rode: python treinar.py")
    return json.loads(arq.read_text(encoding="utf-8"))


def probabilidades(modelo, X) -> dict:
    """Devolve {rótulo: probabilidade} para a primeira linha de X.

    A ordem das colunas de predict_proba() é a de `modelo.classes_`; por isso
    procuramos cada classe pelo valor em vez de supor "coluna 1 = cachorro".
    """
    proba = modelo.predict_proba(X)[0]
    return {int(classe): float(p) for classe, p in zip(modelo.classes_, proba)}


def coluna_da_classe(modelo, rotulo: int) -> int:
    """Posição da classe `rotulo` nas colunas de predict_proba()."""
    return int(np.where(modelo.classes_ == rotulo)[0][0])
