"""Treina, valida e avalia KNN, Random Forest e SVM.

Uso:  python treinar.py        (antes, rode: python preparar_dados.py)

Etapas:
  1. Lê a divisão e transforma as imagens em X (pixels para KNN/RF, cor + HOG para o SVM).
  2. Treina só com o treino e mede na validação.
  3. Validação cruzada (5 folds) em treino + validação, como informação extra.
  4. Treina os modelos finais com treino + validação e salva.
  5. Avalia uma única vez no teste.
"""
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

import avaliar
import configuracao as cfg
import modelos
from preparar_dados import carregar_divisao, montar_X_y

R = cfg.PASTA_RESULTADOS


def validacao_cruzada(Xs_dev: dict, y_dev, grupos_dev) -> pd.DataFrame:
    """5 folds estratificados pelo grupo, os mesmos para os três modelos.

    Xs_dev = {"pixels": X, "atributos": X}, porque cada modelo usa a sua entrada.
    """
    kw = dict(pos_label=1, zero_division=0)
    pontuadores = {
        "Acurácia": make_scorer(accuracy_score),
        "Precisão": make_scorer(precision_score, **kw),
        "Recall": make_scorer(recall_score, **kw),
        "F1-score": make_scorer(f1_score, **kw),
        "F2-score": make_scorer(fbeta_score, beta=2, **kw),
    }
    folds = list(StratifiedKFold(cfg.FOLDS_VALIDACAO_CRUZADA, shuffle=True,
                                 random_state=cfg.SEMENTE).split(grupos_dev, grupos_dev))
    linhas = []
    for chave, modelo in modelos.criar_modelos().items():
        res = cross_validate(modelo, Xs_dev[modelos.ENTRADA_MODELO[chave]], y_dev, cv=folds, scoring=pontuadores)
        linha = {"Modelo": modelos.NOMES_MODELOS[chave]}
        for m in avaliar.METRICAS:
            valores = res[f"test_{m}"]
            linha[f"{m} (média)"], linha[f"{m} (desvio)"] = valores.mean(), valores.std()
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("Modelo")


def main():
    R.mkdir(exist_ok=True)
    df = carregar_divisao()
    print("Transformando imagens em números...")
    Xs = {}
    for entrada in sorted(set(modelos.ENTRADA_MODELO.values())):
        Xs[entrada], y = montar_X_y(df, entrada)
        print(f"X {entrada}: {Xs[entrada].shape} ({Xs[entrada].dtype}), "
              f"valores em [{Xs[entrada].min():.2f}, {Xs[entrada].max():.2f}]")
    print(f"y: {y.shape}")

    def X_de(chave):
        return Xs[modelos.ENTRADA_MODELO[chave]]

    conjunto = df["conjunto"].to_numpy()
    treino = conjunto == "treino"
    val = conjunto == "validacao"
    teste = conjunto == "teste"
    dev = ~teste                                                  # treino + validação
    print(f"treino={treino.sum()} validação={val.sum()} teste={teste.sum()} (teste ainda não usado)")

    # --- 2. treino -> validação ---------------------------------------------
    prev_val = {}
    for chave, modelo in modelos.criar_modelos().items():
        X = X_de(chave)
        modelo.fit(X[treino], y[treino])
        prev_val[modelos.NOMES_MODELOS[chave]] = (y[val], modelo.predict(X[val]))
    metricas_val = avaliar.tabela_metricas(prev_val)
    print("\nVALIDAÇÃO (modelos treinados só com o treino):")
    print(metricas_val.round(4))

    # --- 3. validação cruzada ------------------------------------------------
    print("\nValidação cruzada (5 folds) em treino+validação...")
    cv = validacao_cruzada({e: X[dev] for e, X in Xs.items()}, y[dev], df.loc[dev, "grupo"].to_numpy())
    print(cv.round(4))

    # --- 4. modelos finais: treino + validação -------------------------------
    finais = {}
    for chave, modelo in modelos.criar_modelos().items():
        finais[chave] = clone(modelo).fit(X_de(chave)[dev], y[dev])
    for chave, modelo in finais.items():
        modelos.salvar_modelo(modelo, chave)

    # --- 5. teste (uma única avaliação) --------------------------------------
    meta_teste = df[teste].reset_index(drop=True)
    y_teste = y[teste]
    saida = meta_teste[["caminho", "grupo", "raca_especie", "rotulo"]].copy()
    prev_teste, matrizes = {}, {}
    for chave, modelo in finais.items():
        nome = modelos.NOMES_MODELOS[chave]
        X_teste = X_de(chave)[teste]
        previsto = modelo.predict(X_teste)
        prev_teste[nome] = (y_teste, previsto)
        matrizes[nome] = avaliar.matriz_confusao(y_teste, previsto)
        saida[f"previsto_{chave}"] = previsto
        saida[f"proba_cachorro_{chave}"] = avaliar.probabilidade_de_cachorro(modelo, X_teste)
    metricas_teste = avaliar.tabela_metricas(prev_teste)
    erros = avaliar.erros_por_grupo(meta_teste, {n: p for n, (_, p) in prev_teste.items()})
    print("\nTESTE FINAL (modelos treinados com treino+validação):")
    print(metricas_teste.round(4))
    for nome, m in matrizes.items():
        print(f"\nMatriz de confusão - {nome} (linhas=real, colunas=previsto; ordem 0,1):\n{m}",
              avaliar.vp_vn_fp_fn(m))
    print("\nErros por grupo no teste:\n", erros.to_string(index=False))

    # --- gravação dos resultados ---------------------------------------------
    metricas_val.to_csv(R / "metricas_validacao.csv", encoding="utf-8")
    metricas_teste.to_csv(R / "metricas_teste.csv", encoding="utf-8")
    cv.to_csv(R / "validacao_cruzada.csv", encoding="utf-8")
    erros.to_csv(R / "erros_por_grupo_teste.csv", index=False, encoding="utf-8")
    saida.to_csv(R / "previsoes_teste.csv", index=False, encoding="utf-8")
    (R / "matrizes_teste.json").write_text(json.dumps(
        {n: {"matriz": m.tolist(), **avaliar.vp_vn_fp_fn(m)} for n, m in matrizes.items()},
        indent=2, ensure_ascii=False), encoding="utf-8")
    for chave in finais:
        nome = modelos.NOMES_MODELOS[chave]
        fig = avaliar.desenhar_matriz(matrizes[nome], f"{nome} - teste final")
        fig.savefig(R / f"matriz_{chave}.png", dpi=150)
    avaliar.desenhar_comparacao(metricas_teste, "Métricas no teste final").savefig(
        R / "comparacao_metricas_teste.png", dpi=150)

    info = {
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
        "lado_imagem": cfg.LADO_IMAGEM, "escala_de_cinza": True, "normalizacao": "pixels / 255 (0 a 1)",
        "classes": {str(k): v for k, v in cfg.NOMES_CLASSES.items()},
        "semente": cfg.SEMENTE,
        "knn": {"n_neighbors": cfg.KNN_VIZINHOS, "padronizacao": "StandardScaler (ajustado no treino)"},
        "random_forest": {"n_estimators": cfg.RF_ARVORES, "random_state": cfg.SEMENTE},
        "svm_atributos": {"C": cfg.SVM_C, "kernel": "rbf", "probabilidade": "CalibratedClassifierCV (sigmoide)",
                          "atributos": f"histograma HSV {cfg.BINS_HSV} + HOG (células {cfg.CELULA_HOG}px, "
                                       f"{cfg.ORIENTACOES_HOG} orientações) sobre {cfg.LADO_ATRIBUTOS}x{cfg.LADO_ATRIBUTOS}",
                          "n_atributos": cfg.N_ATRIBUTOS_EXTRAIDOS},
        "treino_final_com": int(dev.sum()), "teste": int(teste.sum()),
        "versoes": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__},
    }
    (cfg.PASTA_MODELOS / "info_modelos.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nModelos salvos em", cfg.PASTA_MODELOS, "| resultados em", R)


if __name__ == "__main__":
    main()
