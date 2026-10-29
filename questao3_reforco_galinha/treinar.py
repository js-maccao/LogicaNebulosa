"""Treino da galinha com Q-learning (tabela Q, sem bibliotecas de RL).

USO
    python treinar.py                     continua do ultimo checkpoint (ou comeca do zero)
    python treinar.py --do-zero           recomeca do zero
    python treinar.py --episodios 20000   episodios DESTA sessao (padrao 10000)
    python treinar.py --checkpoint-cada 1000
"""
import argparse
import os
import pickle
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # gera o grafico em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np

from galinha_env import TravessiaGalinha, NOMES_ACOES

N_ACOES = len(NOMES_ACOES)

# --- Hiperparametros ---
ALFA = 0.1                   # taxa de aprendizado
GAMA = 0.95                  # desconto do futuro
EPSILON_INICIAL = 1.0        # comeca 100% aleatoria
EPSILON_MINIMO = 0.05        # nunca para de explorar por completo
EPSILON_DECAIMENTO = 0.9995  # epsilon *= isso a cada episodio

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_CHECKPOINTS = os.path.join(PASTA, "checkpoints")
ARQUIVO_ULTIMO = os.path.join(PASTA_CHECKPOINTS, "ultimo.pkl")
ARQUIVO_GRAFICO = os.path.join(PASTA, "resultados", "curva_aprendizado.png")
JANELA = 500                 # janela da media movel


# ===== AQUI A TABELA Q =====
def nova_tabela_q():
    """estado -> array com 5 valores (um por acao). Estado novo comeca com zeros."""
    return defaultdict(lambda: np.zeros(N_ACOES))


# ===== AQUI ELE ESCOLHE A ACAO =====
def escolher_acao(q, estado, epsilon):
    """Epsilon-greedy. Devolve (acao, explorou).
    Com chance epsilon explora (acao ao acaso), senao usa o que aprendeu."""
    if random.random() < epsilon:
        return random.randrange(N_ACOES), True
    valores = q[estado]
    # desempate aleatorio entre as melhores (no comeco todas valem 0)
    melhores = np.flatnonzero(valores == valores.max())
    return int(random.choice(melhores)), False


# ===== AQUI ELE APRENDE (equacao do Q-learning) =====
def aprender(q, estado, acao, recompensa, novo_estado, terminou):
    """Corrige um pouquinho o valor Q[estado][acao]. Devolve (valor_antigo, valor_novo)."""
    antigo = float(q[estado][acao])
    futuro = 0.0 if terminou else np.max(q[novo_estado])
    alvo = recompensa + GAMA * futuro
    q[estado][acao] += ALFA * (alvo - antigo)
    return antigo, float(q[estado][acao])


def rodar_episodio(env, q, epsilon):
    """Uma partida inteira, aprendendo a cada passo. Devolve os pontos feitos."""
    estado = env.reset()
    total = 0
    terminou = False
    while not terminou:
        acao, _ = escolher_acao(q, estado, epsilon)
        novo_estado, recompensa, terminou = env.passo(acao)
        aprender(q, estado, acao, recompensa, novo_estado, terminou)
        estado = novo_estado
        total += recompensa
    return total


# ===== AQUI O CHECKPOINT =====
def salvar_checkpoint(caminho, q, epsilon, episodio, historico):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "wb") as f:
        # dict(q): defaultdict com lambda nao pode ser salvo em pickle
        pickle.dump({"q": dict(q), "epsilon": epsilon,
                     "episodio": episodio, "historico": historico}, f)


def carregar_checkpoint(caminho):
    with open(caminho, "rb") as f:
        dados = pickle.load(f)
    q = nova_tabela_q()
    q.update(dados["q"])
    return q, dados["epsilon"], dados["episodio"], dados["historico"]


# ===== AQUI AS METRICAS =====
def media_movel(valores, janela=JANELA):
    v = np.array(valores, dtype=float)
    janela = max(1, min(janela, len(v)))
    return np.convolve(v, np.ones(janela) / janela, mode="valid"), janela


def salvar_grafico(historico):
    pontos = [p for p, _ in historico]
    chegou = [100.0 if r == "chegou" else 0.0 for _, r in historico]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, dados, titulo in ((ax1, pontos, "Pontos por episodio"),
                              (ax2, chegou, "% de vezes que chegou")):
        suave, j = media_movel(dados)
        ax.plot(range(j, j + len(suave)), suave)
        ax.set_title(f"{titulo} (media dos ultimos {j})")
        ax.grid(alpha=0.3)
    ax2.set_xlabel("episodio")
    fig.tight_layout()
    os.makedirs(os.path.dirname(ARQUIVO_GRAFICO), exist_ok=True)
    fig.savefig(ARQUIVO_GRAFICO, dpi=120)
    plt.close(fig)
    print("Grafico salvo em", ARQUIVO_GRAFICO)


def treinar(episodios=10000, do_zero=False, checkpoint_cada=5000):
    env = TravessiaGalinha()

    if not do_zero and os.path.exists(ARQUIVO_ULTIMO):
        q, epsilon, ep_inicial, historico = carregar_checkpoint(ARQUIVO_ULTIMO)
        print(f"Retomando do checkpoint: episodio {ep_inicial}, epsilon {epsilon:.3f}")
    else:
        q, epsilon, ep_inicial, historico = nova_tabela_q(), EPSILON_INICIAL, 0, []
        print("Comecando do zero")

    for ep in range(ep_inicial + 1, ep_inicial + episodios + 1):
        total = rodar_episodio(env, q, epsilon)   # joga uma partida inteira e aprende

        epsilon = max(EPSILON_MINIMO, epsilon * EPSILON_DECAIMENTO)
        historico.append((total, env.resultado))

        if ep % 1000 == 0:
            ult = historico[-JANELA:]
            media = sum(t for t, _ in ult) / len(ult)
            chegou = 100 * sum(1 for _, r in ult if r == "chegou") / len(ult)
            print(f"ep {ep:6d} | pontos {media:7.1f} | chegou {chegou:5.1f}% | eps {epsilon:.3f} | estados {len(q)}")

        if ep % checkpoint_cada == 0:
            salvar_checkpoint(os.path.join(PASTA_CHECKPOINTS, f"ep_{ep}.pkl"),
                              q, epsilon, ep, historico)
            salvar_checkpoint(ARQUIVO_ULTIMO, q, epsilon, ep, historico)

    # salva o estado final (mesmo que nao seja multiplo de checkpoint_cada)
    salvar_checkpoint(ARQUIVO_ULTIMO, q, epsilon, ep_inicial + episodios, historico)
    salvar_grafico(historico)
    return q, historico


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--do-zero", action="store_true")
    p.add_argument("--episodios", type=int, default=10000)
    p.add_argument("--checkpoint-cada", type=int, default=5000)
    a = p.parse_args()
    treinar(a.episodios, a.do_zero, a.checkpoint_cada)
