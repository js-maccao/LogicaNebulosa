"""Avaliacao do agente treinado: gera as figuras pedidas na Questao 3 (b).

Saidas (pasta resultados/):
    curva_aprendizado.png      evolucao do treino (pontos e % de sucesso por episodio)
    evolucao_checkpoints.png   cada checkpoint jogando SEM explorar (epsilon = 0) x agente aleatorio
    evolucao_checkpoints.csv   a mesma avaliacao em tabela
    caminho_final.png          caminho final encontrado pelo agente, passo a passo
    caminho_evolucao.png       mesma partida jogada por checkpoints de epocas diferentes
    caminho_final.txt          tabela do caminho final (estado, acao, recompensa)

USO
    python avaliar.py                                       usa o checkpoint mais treinado
    python avaliar.py --checkpoint checkpoints/ep_20000.pkl
    python avaliar.py --semente 7                           sorteia outra partida para o caminho
"""
import argparse
import glob
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from galinha_env import TravessiaGalinha, NOMES_ACOES
from treinar import carregar_checkpoint, media_movel, PASTA, PASTA_CHECKPOINTS, N_ACOES

PASTA_RESULTADOS = os.path.join(PASTA, "resultados")
N_AVALIACAO = 1000           # partidas por checkpoint na avaliacao


def listar_checkpoints():
    """Checkpoints ep_N.pkl ordenados pelo numero do episodio."""
    caminhos = glob.glob(os.path.join(PASTA_CHECKPOINTS, "ep_*.pkl"))
    return sorted(caminhos, key=lambda c: int(re.search(r"ep_(\d+)", c).group(1)))


# ===== AQUI ELA JOGA SO COM O QUE APRENDEU (epsilon = 0) =====
def politica_gulosa(q):
    def escolher(estado):
        return int(np.argmax(q.get(estado, np.zeros(N_ACOES))))
    return escolher


def politica_aleatoria(_estado):
    return random.randrange(N_ACOES)


def jogar_partida(env, escolher, semente):
    """Joga uma partida com carros sorteados pela semente. Devolve o caminho percorrido."""
    random.seed(semente)
    estado = env.reset()
    caminho = [{"linha": env.linha, "coluna": env.coluna,
                "carros": {l: list(c) for l, c in env.carros.items()},
                "estado": estado, "acao": None, "recompensa": 0}]
    terminou = False
    while not terminou:
        acao = escolher(estado)
        estado, recompensa, terminou = env.passo(acao)
        caminho.append({"linha": env.linha, "coluna": env.coluna,
                        "carros": {l: list(c) for l, c in env.carros.items()},
                        "estado": estado, "acao": acao, "recompensa": recompensa})
    return caminho, env.resultado


def avaliar(escolher, n=N_AVALIACAO, semente_base=10_000):
    """Taxa de sucesso, pontos medios e passos medios em n partidas (sementes fixas)."""
    env = TravessiaGalinha()
    chegou, pontos, passos = 0, [], []
    for i in range(n):
        caminho, resultado = jogar_partida(env, escolher, semente_base + i)
        chegou += resultado == "chegou"
        pontos.append(sum(p["recompensa"] for p in caminho))
        passos.append(len(caminho) - 1)
    return 100 * chegou / n, float(np.mean(pontos)), float(np.mean(passos))


# ===== FIGURAS =====
def grafico_treino(historico):
    pontos = [p for p, _ in historico]
    chegou = [100.0 if r == "chegou" else 0.0 for _, r in historico]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, dados, titulo in ((ax1, pontos, "Pontos por episodio"),
                              (ax2, chegou, "% de vezes que chegou")):
        suave, j = media_movel(dados)
        ax.plot(range(j, j + len(suave)), suave)
        ax.set_title(f"{titulo} durante o treino (media movel de {j})")
        ax.grid(alpha=0.3)
    ax2.set_xlabel("episodio")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_RESULTADOS, "curva_aprendizado.png"), dpi=120)
    plt.close(fig)


def grafico_checkpoints(linhas, base):
    eps = [e for e, *_ in linhas]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(eps, [s for _, s, _, _ in linhas], marker="o", ms=3, label="agente treinado (epsilon = 0)")
    ax1.axhline(base[0], color="gray", ls="--", label="agente aleatorio")
    ax1.set_ylabel("% de travessias")
    ax1.set_ylim(-3, 103)
    ax1.set_title(f"Desempenho de cada checkpoint ({N_AVALIACAO} partidas cada)")
    ax2.plot(eps, [p for _, _, p, _ in linhas], marker="o", ms=3)
    ax2.axhline(base[1], color="gray", ls="--")
    ax2.set_ylabel("pontos medios")
    ax2.set_xlabel("episodios de treino")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax1.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_RESULTADOS, "evolucao_checkpoints.png"), dpi=120)
    plt.close(fig)


def desenhar_quadro(ax, env, passo, rastro, titulo):
    """Um 'print' do jogo: pistas, carros, rastro da galinha e a galinha."""
    L, C = env.linhas, env.colunas
    for l in range(L):
        cor = "#aaa9a2" if l == 0 else "#5c965a" if l == L - 1 else "#3a3c44"
        ax.add_patch(Rectangle((0, l), C, 1, color=cor))
    for l, colunas in passo["carros"].items():
        cor = "#d64541" if env.direcao(l) == 1 else "#3478d2"
        for c in colunas:
            ax.add_patch(Rectangle((c + 0.1, l + 0.2), 0.8, 0.6, color=cor))
    if len(rastro) > 1:
        xs = [c + 0.5 for _, c in rastro]
        ys = [l + 0.5 for l, _ in rastro]
        ax.plot(xs, ys, color="#f2c230", lw=2, alpha=0.9)
    ax.plot(passo["coluna"] + 0.5, passo["linha"] + 0.5, "o", ms=9,
            mfc="white", mec="#f29628", mew=2)
    ax.set_xlim(0, C)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titulo, fontsize=9)


def figura_caminho(env, caminho, resultado, rotulo, arquivo):
    """Filme quadro a quadro + resumo do trajeto completo."""
    n = len(caminho)
    cols = min(8, n + 1)
    lins = int(np.ceil((n + 1) / cols))
    fig, axs = plt.subplots(lins, cols, figsize=(1.9 * cols, 2.7 * lins))
    axs = np.atleast_1d(axs).ravel()
    rastro = []
    for i, passo in enumerate(caminho):
        rastro.append((passo["linha"], passo["coluna"]))
        titulo = "inicio" if passo["acao"] is None else f"passo {i}: {NOMES_ACOES[passo['acao']]}"
        desenhar_quadro(axs[i], env, passo, rastro, titulo)
    # ultimo quadro: so o trajeto completo, sem carros
    resumo = dict(caminho[-1], carros={})
    desenhar_quadro(axs[n], env, resumo, rastro, f"trajeto completo\n({resultado}, {n - 1} passos)")
    for ax in axs[n + 1:]:
        ax.axis("off")
    fig.suptitle(f"Caminho encontrado pelo agente - {rotulo}\n"
                 "vermelho: carro indo p/ direita | azul: carro indo p/ esquerda | amarelo: rastro da galinha",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=120)
    plt.close(fig)


def figura_evolucao_caminho(env, semente, checkpoints):
    """A mesma partida (mesma semente) jogada por checkpoints de epocas diferentes."""
    fig, axs = plt.subplots(1, len(checkpoints), figsize=(2.6 * len(checkpoints), 4))
    for ax, ck in zip(np.atleast_1d(axs), checkpoints):
        q, _, ep, _ = carregar_checkpoint(ck)
        caminho, resultado = jogar_partida(env, politica_gulosa(q), semente)
        rastro = [(p["linha"], p["coluna"]) for p in caminho]
        desenhar_quadro(ax, env, caminho[-1], rastro,
                        f"episodio {ep}\n{resultado}, {len(caminho) - 1} passos")
    fig.suptitle(f"Evolucao do caminho na mesma partida (semente {semente})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(PASTA_RESULTADOS, "caminho_evolucao.png"), dpi=120)
    plt.close(fig)


def salvar_tabela_caminho(caminho, resultado, arquivo):
    with open(arquivo, "w", encoding="utf-8") as f:
        f.write(f"Resultado: {resultado} em {len(caminho) - 1} passos | "
                f"pontos: {sum(p['recompensa'] for p in caminho)}\n\n")
        f.write(f"{'passo':>5}  {'acao':<9} {'posicao (l,c)':<14} {'recompensa':>10}  "
                "estado (linha, coluna, frente_ocupada, frente_vindo, aqui_vindo)\n")
        for i, p in enumerate(caminho):
            acao = "-" if p["acao"] is None else NOMES_ACOES[p["acao"]]
            f.write(f"{i:>5}  {acao:<9} {str((p['linha'], p['coluna'])):<14} "
                    f"{p['recompensa']:>10}  {p['estado']}\n")


def main(caminho_ck, semente):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    checkpoints = listar_checkpoints()
    if caminho_ck is None:
        if not checkpoints:
            raise SystemExit("Nenhum checkpoint encontrado. Rode antes: python treinar.py")
        caminho_ck = checkpoints[-1]
    q, epsilon, episodio, historico = carregar_checkpoint(caminho_ck)
    print(f"Checkpoint: {caminho_ck} (episodio {episodio}, {len(q)} estados na tabela Q)")

    # 1) evolucao durante o treino
    grafico_treino(historico)

    # 2) cada checkpoint avaliado sem exploracao, contra a linha de base aleatoria
    base = avaliar(politica_aleatoria)
    print(f"\n{'agente':<22}{'chegou':>8}{'pontos':>9}{'passos':>8}")
    print(f"{'aleatorio':<22}{base[0]:>7.1f}%{base[1]:>9.1f}{base[2]:>8.1f}")
    linhas = []
    for ck in checkpoints:
        qc, _, ep, _ = carregar_checkpoint(ck)
        s, p, n = avaliar(politica_gulosa(qc))
        linhas.append((ep, s, p, n))
        print(f"{'episodio ' + str(ep):<22}{s:>7.1f}%{p:>9.1f}{n:>8.1f}")
    if linhas:
        grafico_checkpoints(linhas, base)
    with open(os.path.join(PASTA_RESULTADOS, "evolucao_checkpoints.csv"), "w", encoding="utf-8") as f:
        f.write("agente,episodio,chegou_pct,pontos_medios,passos_medios\n")
        f.write(f"aleatorio,0,{base[0]:.1f},{base[1]:.1f},{base[2]:.1f}\n")
        for ep, s, p, n in linhas:
            f.write(f"treinado,{ep},{s:.1f},{p:.1f},{n:.1f}\n")

    # 3) caminho final encontrado pelo agente
    env = TravessiaGalinha()
    caminho, resultado = jogar_partida(env, politica_gulosa(q), semente)
    figura_caminho(env, caminho, resultado, f"episodio {episodio}, semente {semente}",
                   os.path.join(PASTA_RESULTADOS, "caminho_final.png"))
    salvar_tabela_caminho(caminho, resultado, os.path.join(PASTA_RESULTADOS, "caminho_final.txt"))
    print(f"\nCaminho final (semente {semente}): {resultado} em {len(caminho) - 1} passos")
    print("Acoes:", " -> ".join(NOMES_ACOES[p["acao"]] for p in caminho[1:]))

    # 4) a mesma partida em epocas diferentes do treino
    if len(checkpoints) >= 2:
        idx = np.unique(np.linspace(0, len(checkpoints) - 1, 5).round().astype(int))
        figura_evolucao_caminho(env, semente, [checkpoints[i] for i in idx])

    print(f"\nFiguras salvas em {PASTA_RESULTADOS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="padrao: o checkpoint mais treinado")
    p.add_argument("--semente", type=int, default=1, help="sorteio dos carros da partida mostrada")
    a = p.parse_args()
    main(a.checkpoint, a.semente)
