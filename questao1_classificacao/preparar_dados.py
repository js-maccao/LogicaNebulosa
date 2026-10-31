"""Lê as pastas de imagens, monta a tabela, valida os arquivos e divide os dados.

Uso:  python preparar_dados.py

Gera em resultados/:
  tabela_imagens.csv   -> uma linha por imagem válida (caminho, rótulo, grupo, raça/espécie)
  divisao.csv          -> a que conjunto cada imagem pertence (treino/validacao/teste)
  resumo_dados.json    -> contagens, corrompidas, duplicatas, verificações
  duplicatas.csv       -> (só se houver) arquivos idênticos
"""
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import configuracao as cfg
from pre_processamento import extrair_varios, preparar_varias

ARQ_TABELA = cfg.PASTA_RESULTADOS / "tabela_imagens.csv"
ARQ_DIVISAO = cfg.PASTA_RESULTADOS / "divisao.csv"
ARQ_RESUMO = cfg.PASTA_RESULTADOS / "resumo_dados.json"
ARQ_DUPLICATAS = cfg.PASTA_RESULTADOS / "duplicatas.csv"


def _listar_arquivos(pasta):
    """Imagens da pasta e subpastas, em ordem alfabética (para a tabela sair sempre igual)."""
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
    return sorted(p for p in pasta.rglob("*")
                  if p.is_file() and p.suffix.lower() in cfg.EXTENSOES_IMAGEM)


def montar_tabela() -> pd.DataFrame:
    """Uma linha por imagem. O rótulo vem da pasta de origem."""
    linhas = []
    for p in _listar_arquivos(cfg.PASTA_CACHORROS):
        raca = re.sub(r"^n\d+-", "", p.parent.name)          # 'n02085620-Chihuahua' -> 'Chihuahua'
        linhas.append((p, cfg.ROTULO_CACHORRO, cfg.GRUPO_CACHORRO, raca))
    for p in _listar_arquivos(cfg.PASTA_NAO_CACHORROS):
        especie = p.name.split("_")[0]                       # 'cat_0001.jpeg' -> 'cat'
        if especie not in cfg.ESPECIES_NAO_CACHORRO:
            print(f"AVISO: prefixo inesperado '{especie}' em {p.name}", file=sys.stderr)
        linhas.append((p, cfg.ROTULO_NAO_CACHORRO, especie, especie))

    df = pd.DataFrame(linhas, columns=["caminho", "rotulo", "grupo", "raca_especie"])
    # caminho relativo, para o CSV funcionar em outro computador
    df["caminho"] = df["caminho"].map(lambda p: p.relative_to(cfg.RAIZ_BASES).as_posix())
    return df


def caminho_absoluto(caminho_relativo: str):
    return cfg.RAIZ_BASES / caminho_relativo


def validar_arquivos(df: pd.DataFrame):
    """Separa imagens legíveis das corrompidas e calcula o hash de cada arquivo."""
    validas, corrompidas, hashes = [], [], []
    for caminho in df["caminho"]:
        arq = caminho_absoluto(caminho)
        try:
            with Image.open(arq) as img:
                img.verify()                                 # detecta arquivo truncado/inválido
            validas.append(True)
            hashes.append(hashlib.sha256(arq.read_bytes()).hexdigest())
        except Exception as erro:                            # qualquer falha = imagem ilegível
            validas.append(False)
            hashes.append(None)
            corrompidas.append({"caminho": caminho, "erro": repr(erro)})
    df = df.assign(hash_arquivo=hashes)
    return df[np.array(validas)].reset_index(drop=True), corrompidas


def achar_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Arquivos com conteúdo idêntico (mesmo hash). Só lista, não remove nada."""
    dup = df[df.duplicated("hash_arquivo", keep=False)].sort_values("hash_arquivo")
    return dup[["hash_arquivo", "caminho", "rotulo", "grupo"]]


def dividir(df: pd.DataFrame) -> pd.DataFrame:
    """20% do total vai para o teste; dos 80% que sobram, 20% vão para a validação.

    Estratifico pelo grupo (cachorro + 5 espécies) para que cada espécie apareça
    na mesma proporção nos três conjuntos, e não só cada classe.
    """
    indices = np.arange(len(df))
    idx_dev, idx_teste = train_test_split(
        indices, test_size=cfg.FRACAO_TESTE, stratify=df["grupo"], random_state=cfg.SEMENTE)
    idx_treino, idx_val = train_test_split(
        idx_dev, test_size=cfg.FRACAO_VALIDACAO, stratify=df["grupo"].iloc[idx_dev],
        random_state=cfg.SEMENTE)

    conjunto = pd.Series("", index=df.index)
    conjunto.iloc[idx_treino] = "treino"
    conjunto.iloc[idx_val] = "validacao"
    conjunto.iloc[idx_teste] = "teste"
    return df.assign(conjunto=conjunto)


def verificar_divisao(df: pd.DataFrame) -> dict:
    """Confere que nenhuma imagem ficou em dois conjuntos."""
    partes = {c: set(df.loc[df["conjunto"] == c, "caminho"]) for c in ["treino", "validacao", "teste"]}
    assert not partes["treino"] & partes["validacao"], "imagem em treino e validação"
    assert not partes["treino"] & partes["teste"], "imagem em treino e teste"
    assert not partes["validacao"] & partes["teste"], "imagem em validação e teste"
    assert sum(len(p) for p in partes.values()) == len(df), "imagem sem conjunto"
    assert df["caminho"].is_unique, "caminho repetido na tabela"

    # duplicatas exatas em conjuntos diferentes seriam vazamento do teste
    conjuntos_por_hash = df.groupby("hash_arquivo")["conjunto"].nunique()
    vazamentos = int((conjuntos_por_hash > 1).sum())
    return {"sem_sobreposicao_de_caminhos": True, "duplicatas_entre_conjuntos": vazamentos}


def montar_X_y(df: pd.DataFrame, entrada: str = "pixels"):
    """X: uma linha por imagem; y: rótulos 0/1.

    entrada="pixels"    -> 4096 colunas, tons de cinza (KNN e Random Forest)
    entrada="atributos" -> 1892 colunas, cor HSV + HOG (SVM)
    """
    caminhos = [caminho_absoluto(c) for c in df["caminho"]]
    if entrada == "pixels":
        X, n_colunas = preparar_varias(caminhos), cfg.N_ATRIBUTOS
    elif entrada == "atributos":
        X, n_colunas = extrair_varios(caminhos), cfg.N_ATRIBUTOS_EXTRAIDOS
    else:
        raise ValueError(f"entrada desconhecida: {entrada!r}")
    y = df["rotulo"].to_numpy()
    assert X.shape == (len(df), n_colunas) and len(y) == len(X)
    return X, y


def carregar_divisao() -> pd.DataFrame:
    if not ARQ_DIVISAO.exists():
        raise FileNotFoundError("Divisão não encontrada. Rode primeiro: python preparar_dados.py")
    return pd.read_csv(ARQ_DIVISAO, encoding="utf-8")


def contagens(df: pd.DataFrame) -> dict:
    por_conjunto = {}
    for c in ["treino", "validacao", "teste"]:
        sub = df[df["conjunto"] == c]
        por_conjunto[c] = {
            "total": int(len(sub)),
            "por_classe": {str(k): int(v) for k, v in sub["rotulo"].value_counts().sort_index().items()},
            "por_grupo": {k: int(v) for k, v in sub["grupo"].value_counts().items()},
        }
    return {
        "total_validas": int(len(df)),
        "por_classe": {str(k): int(v) for k, v in df["rotulo"].value_counts().sort_index().items()},
        "por_grupo": {k: int(v) for k, v in df["grupo"].value_counts().items()},
        "racas_distintas": int(df.loc[df["grupo"] == cfg.GRUPO_CACHORRO, "raca_especie"].nunique()),
        "por_conjunto": por_conjunto,
    }


def main():
    cfg.PASTA_RESULTADOS.mkdir(exist_ok=True)
    df = montar_tabela()
    print(f"Arquivos de imagem encontrados: {len(df)}  (por classe: {df['rotulo'].value_counts().to_dict()})")

    df, corrompidas = validar_arquivos(df)
    print(f"Imagens válidas: {len(df)} | corrompidas/ilegíveis ignoradas: {len(corrompidas)}")
    for c in corrompidas:
        print("  IGNORADA:", c)

    esperado = {"cachorro": 500, **{e: 100 for e in cfg.ESPECIES_NAO_CACHORRO}}
    real = df["grupo"].value_counts().to_dict()
    if real != esperado:
        print(f"AVISO: contagem por grupo difere do esperado {esperado}: {real}")

    dup = achar_duplicatas(df)
    if len(dup):
        dup.to_csv(ARQ_DUPLICATAS, index=False, encoding="utf-8")
        entre_classes = int((dup.groupby("hash_arquivo")["rotulo"].nunique() > 1).sum())
        print(f"AVISO: {len(dup)} arquivos em grupos de duplicatas exatas "
              f"({entre_classes} grupos misturam as duas classes). Nada foi removido; ver {ARQ_DUPLICATAS.name}")
    else:
        print("Duplicatas exatas (hash SHA-256 do arquivo): nenhuma, dentro ou entre as classes.")

    df = dividir(df)
    verificacao = verificar_divisao(df)
    print("Verificação da divisão:", verificacao)

    df.drop(columns=["conjunto"]).to_csv(ARQ_TABELA, index=False, encoding="utf-8")
    df.to_csv(ARQ_DIVISAO, index=False, encoding="utf-8")

    resumo = {
        **contagens(df),
        "corrompidas_ignoradas": corrompidas,
        "arquivos_em_duplicatas": int(len(dup)),
        "verificacao_divisao": verificacao,
        "semente": cfg.SEMENTE,
    }
    ARQ_RESUMO.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nContagem por conjunto e grupo:")
    print(pd.crosstab(df["grupo"], df["conjunto"])[["treino", "validacao", "teste"]])


if __name__ == "__main__":
    main()
