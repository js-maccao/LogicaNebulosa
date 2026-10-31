"""Q3 no Streamlit: treino ao vivo, replay da galinha e treino passo a passo.

Só importa as funções de questao3_reforco_galinha (treinar.py, avaliar.py, galinha_env.py);
as regras do jogo e a equação do Q-learning continuam lá.
"""
import copy
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q3 = Path(__file__).resolve().parent / "questao3_reforco_galinha"
if str(Q3) not in sys.path:
    sys.path.insert(0, str(Q3))

import avaliar  # noqa: E402
import galinha_env  # noqa: E402
import treinar  # noqa: E402

PASTA_PAINEL = os.path.join(treinar.PASTA, "checkpoints_painel")


def treinar_por_blocos(episodios, do_zero, pasta_destino, checkpoint_cada=1000, bloco=250):
    """Treina como treinar.treinar(), mas devolve o progresso a cada `bloco` episódios.
    Salva ultimo.pkl e ep_N.pkl em `pasta_destino`. Se a execução for interrompida, salva o que tem."""
    ultimo = os.path.join(pasta_destino, "ultimo.pkl")
    env = galinha_env.TravessiaGalinha()
    if not do_zero and os.path.exists(ultimo):
        q, epsilon, ep0, historico = treinar.carregar_checkpoint(ultimo)
    elif not do_zero and os.path.exists(treinar.ARQUIVO_ULTIMO) and pasta_destino != treinar.PASTA_CHECKPOINTS:
        q, epsilon, ep0, historico = treinar.carregar_checkpoint(treinar.ARQUIVO_ULTIMO)
    else:
        q, epsilon, ep0, historico = treinar.nova_tabela_q(), treinar.EPSILON_INICIAL, 0, []
    ep = ep0
    try:
        for ep in range(ep0 + 1, ep0 + episodios + 1):
            total = treinar.rodar_episodio(env, q, epsilon)
            epsilon = max(treinar.EPSILON_MINIMO, epsilon * treinar.EPSILON_DECAIMENTO)
            historico.append((total, env.resultado))
            if ep % checkpoint_cada == 0:
                treinar.salvar_checkpoint(os.path.join(pasta_destino, f"ep_{ep}.pkl"), q, epsilon, ep, historico)
            if ep % bloco == 0 or ep == ep0 + episodios:
                ult = historico[-treinar.JANELA:]
                yield {"episodio": ep, "pontos": sum(t for t, _ in ult) / len(ult),
                       "chegou": 100 * sum(1 for _, r in ult if r == "chegou") / len(ult),
                       "epsilon": epsilon, "estados": len(q)}
    finally:
        if ep > ep0:
            treinar.salvar_checkpoint(ultimo, q, epsilon, ep, historico)


def fazer_backup_checkpoints():
    """Copia checkpoints/ para checkpoints_backup_<hora>/ antes de sobrescrever."""
    destino = os.path.join(treinar.PASTA, "checkpoints_backup_" + time.strftime("%Y%m%d_%H%M%S"))
    shutil.copytree(treinar.PASTA_CHECKPOINTS, destino)
    return destino


def listar_checkpoints():
    """Todos os .pkl disponíveis (pasta oficial e pasta do painel)."""
    itens = []
    for pasta in (treinar.PASTA_CHECKPOINTS, PASTA_PAINEL):
        if os.path.isdir(pasta):
            for nome in sorted(os.listdir(pasta)):
                if nome.endswith(".pkl"):
                    itens.append(os.path.join(pasta, nome))
    return itens


def rotulo_checkpoint(caminho):
    return os.path.join(os.path.basename(os.path.dirname(caminho)), os.path.basename(caminho))


def gravar_replay(caminho_ck, semente):
    """Joga uma partida só com o que a galinha aprendeu. Devolve (q, passos, resultado, env)."""
    q, *_ = treinar.carregar_checkpoint(caminho_ck)
    env = galinha_env.TravessiaGalinha()
    passos, resultado = avaliar.jogar_partida(env, avaliar.politica_gulosa(q), semente)
    return q, passos, resultado, env


def quadro(env, passos, i, titulo=""):
    """Figura do instante i da partida, com o rastro percorrido até ali."""
    fig, ax = plt.subplots(figsize=(4.2, 5.6))
    rastro = [(p["linha"], p["coluna"]) for p in passos[:i + 1]]
    avaliar.desenhar_quadro(ax, env, passos[i], rastro, titulo)
    fig.tight_layout()
    return fig


def episodio_didatico(q, epsilon, semente):
    """Uma partida registrando cada passo: explorou?, ação, recompensa, Q antes e depois.
    Trabalha numa CÓPIA da tabela Q (nada é gravado)."""
    q = copy.deepcopy(q)
    random.seed(semente)
    env = galinha_env.TravessiaGalinha()
    estado = env.reset()
    passos = [{"linha": env.linha, "coluna": env.coluna, "carros": {l: list(c) for l, c in env.carros.items()},
               "estado": estado}]
    registros, terminou = [], False
    while not terminou:
        acao, explorou = treinar.escolher_acao(q, estado, epsilon)
        novo, recompensa, terminou = env.passo(acao)
        antigo, novo_q = treinar.aprender(q, estado, acao, recompensa, novo, terminou)
        registros.append({"estado": estado, "acao": galinha_env.NOMES_ACOES[acao], "explorou": explorou,
                          "recompensa": recompensa, "Q_antes": antigo, "Q_depois": novo_q})
        passos.append({"linha": env.linha, "coluna": env.coluna,
                       "carros": {l: list(c) for l, c in env.carros.items()}, "estado": novo})
        estado = novo
    return env, passos, registros, env.resultado
