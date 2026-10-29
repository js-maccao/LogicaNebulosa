import pandas as pd

import configuracao as cfg
import preparar_dados as prep


def _tabela_falsa():
    linhas = [(f"c/{i}.jpg", 1, "cachorro", "raca") for i in range(100)]
    for esp in cfg.ESPECIES_NAO_CACHORRO:
        linhas += [(f"n/{esp}_{i}.jpeg", 0, esp, esp) for i in range(20)]
    df = pd.DataFrame(linhas, columns=["caminho", "rotulo", "grupo", "raca_especie"])
    df["hash_arquivo"] = df["caminho"]               # todos distintos
    return df


def test_tamanhos_e_ausencia_de_sobreposicao():
    df = prep.dividir(_tabela_falsa())
    contagem = df["conjunto"].value_counts()
    assert contagem["teste"] == 40 and contagem["validacao"] == 32 and contagem["treino"] == 128
    assert prep.verificar_divisao(df)["sem_sobreposicao_de_caminhos"]


def test_estratificacao_por_grupo():
    df = prep.dividir(_tabela_falsa())
    teste = df[df["conjunto"] == "teste"]["grupo"].value_counts()
    assert teste["cachorro"] == 20 and all(teste[e] == 4 for e in cfg.ESPECIES_NAO_CACHORRO)


def test_divisao_e_reprodutivel():
    a, b = prep.dividir(_tabela_falsa()), prep.dividir(_tabela_falsa())
    assert a["conjunto"].equals(b["conjunto"])


def test_divisao_salva_nao_tem_sobreposicao():
    if not prep.ARQ_DIVISAO.exists():
        return
    df = prep.carregar_divisao()
    assert prep.verificar_divisao(df.assign(hash_arquivo=df["caminho"]))["sem_sobreposicao_de_caminhos"]
    assert df["conjunto"].value_counts().to_dict() == {"treino": 640, "validacao": 160, "teste": 200}
