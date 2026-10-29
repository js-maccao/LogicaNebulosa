"""Visualizador da galinha em pygame (mostra a galinha treinada jogando).

USO
    python visualizador.py                                         usa checkpoints/ultimo.pkl
    python visualizador.py --checkpoint checkpoints/ep_5000.pkl    escolhe o checkpoint
    python visualizador.py --velocidade 6                          passos por segundo

TECLAS  ESPACO pausa | CIMA/BAIXO velocidade | R reinicia a partida | ESC sai
"""
import argparse
import os
import sys
import time

import numpy as np
import pygame

from galinha_env import TravessiaGalinha, NOMES_ACOES
from treinar import (carregar_checkpoint, salvar_checkpoint, salvar_grafico, nova_tabela_q,
                     escolher_acao, aprender, rodar_episodio, ARQUIVO_ULTIMO, PASTA_CHECKPOINTS,
                     EPSILON_INICIAL, EPSILON_MINIMO, EPSILON_DECAIMENTO, JANELA)
from visual import Cena



def jogar(caminho, velocidade=3.0):
    if not os.path.exists(caminho):
        sys.exit(f"Checkpoint nao encontrado: {caminho}\nRode antes: python treinar.py")
    q, _, episodio, _ = carregar_checkpoint(caminho)

    env = TravessiaGalinha()
    cena = Cena(env, "Galinha - jogando com o que aprendeu")
    relogio = pygame.time.Clock()

    estado = env.reset()
    cena.sincronizar()
    tempo = 0.0            # relogio proprio: para a pausa congelar a animacao
    inicio_passo = 0.0     # quando o passo atual comecou
    fim_em = None          # quando a partida acabou
    pausado = False
    estado_visto, q_visto, acao = estado, None, None   # o que ela viu e escolheu no ultimo passo
    partidas = vitorias = 0

    rodando = True
    while rodando:
        dur = 1000 / velocidade                         # duracao de cada passo em ms
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                rodando = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    rodando = False
                elif e.key == pygame.K_SPACE:
                    pausado = not pausado
                elif e.key == pygame.K_UP:
                    velocidade = min(30.0, velocidade * 1.5)
                elif e.key == pygame.K_DOWN:
                    velocidade = max(0.5, velocidade / 1.5)
                elif e.key == pygame.K_r:
                    estado = env.reset()
                    cena.sincronizar()
                    inicio_passo, fim_em = tempo, None
                    estado_visto, q_visto, acao = estado, None, None

        t = min(1.0, (tempo - inicio_passo) / dur)      # progresso da animacao (0 a 1)
        if not pausado:
            if fim_em is not None:
                if tempo - fim_em > dur + 1000:         # mostra o resultado e recomeca
                    estado = env.reset()
                    cena.sincronizar()
                    inicio_passo, fim_em = tempo, None
                    estado_visto, q_visto, acao = estado, None, None
            elif t >= 1.0:                              # animacao acabou: proximo passo
                cena.sincronizar()                      # foto do ANTES
                estado_visto = estado
                q_visto = q[estado].copy()
                # ===== AQUI ELA USA O QUE APRENDEU (sem explorar) =====
                acao = int(np.argmax(q_visto))
                estado, _, terminou = env.passo(acao)
                inicio_passo = tempo
                if terminou:
                    partidas += 1
                    vitorias += int(env.resultado == "chegou")
                    fim_em = tempo
            t = min(1.0, (tempo - inicio_passo) / dur)

        pct = 100 * vitorias / partidas if partidas else 0
        cena.desenhar_mapa(t, env.resultado)
        cena.desenhar_painel(
            "Galinha Q-learning",
            f"Checkpoint: episódio {episodio}" + ("   [PAUSADO]" if pausado else ""),
            estado_visto, q_visto, acao,
            [("Partidas", partidas),
             ("Chegou", f"{vitorias} ({pct:.0f}%)"),
             ("Velocidade", f"{velocidade:.1f} passos/s")],
            ["ESPAÇO pausa  |  R reinicia", "CIMA/BAIXO velocidade  |  ESC sai"],
        )
        pygame.display.flip()
        dt = relogio.tick(60)
        if not pausado:
            tempo += dt

    pygame.quit()


def treinar_ao_vivo(do_zero=False, checkpoint_cada=1000, velocidade=3.0):
    """Treina a galinha mostrando tudo na tela. T liga/desliga o modo TURBO."""
    env = TravessiaGalinha()
    if not do_zero and os.path.exists(ARQUIVO_ULTIMO):
        q, epsilon, episodio, historico = carregar_checkpoint(ARQUIVO_ULTIMO)
    else:
        q, epsilon, episodio, historico = nova_tabela_q(), EPSILON_INICIAL, 0, []

    def salvar(so_ultimo=False):
        if not so_ultimo:
            salvar_checkpoint(os.path.join(PASTA_CHECKPOINTS, f"ep_{episodio}.pkl"),
                              q, epsilon, episodio, historico)
        salvar_checkpoint(ARQUIVO_ULTIMO, q, epsilon, episodio, historico)

    def registrar(total):
        """Fim de um episodio: epsilon cai, historico cresce, checkpoint automatico."""
        nonlocal epsilon, episodio
        epsilon = max(EPSILON_MINIMO, epsilon * EPSILON_DECAIMENTO)
        historico.append((total, env.resultado))
        episodio += 1
        if episodio % checkpoint_cada == 0:
            salvar()

    cena = Cena(env, "Galinha - treinando ao vivo")
    relogio = pygame.time.Clock()

    estado = env.reset()
    cena.sincronizar()
    tempo = 0.0
    inicio_passo = 0.0
    fim_em = None
    pausado = False
    turbo = False
    aviso_ate = 0                  # mostra "checkpoint salvo" ate esse instante (tempo real)
    total = 0                      # pontos do episodio atual
    estado_visto, q_visto, acao = estado, None, None
    ultimo = None                  # (explorou, acao, q_antigo, q_novo, recompensa) do ultimo passo

    rodando = True
    while rodando:
        dur = 1000 / velocidade
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                rodando = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    rodando = False
                elif e.key == pygame.K_SPACE:
                    pausado = not pausado
                elif e.key == pygame.K_UP:
                    velocidade = min(30.0, velocidade * 1.5)
                elif e.key == pygame.K_DOWN:
                    velocidade = max(0.5, velocidade / 1.5)
                elif e.key == pygame.K_s:
                    salvar()
                    aviso_ate = pygame.time.get_ticks() + 1500
                elif e.key == pygame.K_t:
                    turbo = not turbo
                    if not turbo:                       # volta ao modo animado com partida nova
                        estado = env.reset()
                        cena.sincronizar()
                        inicio_passo, fim_em, total = tempo, None, 0
                    estado_visto, q_visto, acao, ultimo = (None, None, None, None) if turbo else (estado, None, None, None)

        if not pausado:
            if turbo:
                # TURBO: roda varios episodios por quadro, sem animar
                inicio = time.perf_counter()
                while time.perf_counter() - inicio < 0.015:
                    registrar(rodar_episodio(env, q, epsilon))
                t = 1.0
            else:
                t = min(1.0, (tempo - inicio_passo) / dur)
                if fim_em is not None:
                    if tempo - fim_em > dur + 500:      # mostra o resultado e recomeca
                        estado = env.reset()
                        cena.sincronizar()
                        inicio_passo, fim_em, total = tempo, None, 0
                        estado_visto, q_visto, acao = estado, None, None
                elif t >= 1.0:                          # animacao acabou: proximo passo
                    cena.sincronizar()                  # foto do ANTES
                    estado_visto = estado
                    q_visto = q[estado].copy()
                    acao, explorou = escolher_acao(q, estado, epsilon)
                    novo_estado, recompensa, terminou = env.passo(acao)
                    # ===== AQUI ELE APRENDE (mesma funcao do treino normal) =====
                    antigo, novo = aprender(q, estado, acao, recompensa, novo_estado, terminou)
                    ultimo = (explorou, acao, antigo, novo, recompensa)
                    estado = novo_estado
                    total += recompensa
                    inicio_passo = tempo
                    if terminou:
                        registrar(total)
                        fim_em = tempo
                t = min(1.0, (tempo - inicio_passo) / dur)
        else:
            t = 1.0 if turbo else min(1.0, (tempo - inicio_passo) / dur)

        # ---- desenho ----
        ult = historico[-JANELA:]
        pct = 100 * sum(1 for _, r in ult if r == "chegou") / len(ult) if ult else 0
        media = sum(p for p, _ in ult) / len(ult) if ult else 0
        modo = "TURBO" if turbo else f"{velocidade:.1f} passos/s"
        if pygame.time.get_ticks() < aviso_ate:
            sub = "Checkpoint salvo!"
        else:
            sub = f"Treino ao vivo  |  {modo}" + ("  [PAUSADO]" if pausado else "")
        extra = []
        if ultimo is not None:
            explorou, a, antigo, novo, recompensa = ultimo
            extra.append(("Ação: EXPLORANDO (aleatória)" if explorou else "Ação: USANDO O QUE SABE",
                          (240, 190, 70) if explorou else (88, 190, 120)))
            extra.append((f"Q[{NOMES_ACOES[a]}]: {antigo:.1f} → {novo:.1f}   (recompensa {recompensa:+d})",
                          (222, 226, 236)))
        cena.desenhar_mapa(t, "turbo" if turbo else env.resultado)
        cena.desenhar_painel(
            "Galinha aprendendo", sub, estado_visto, q_visto, acao,
            [("Episódio", episodio),
             ("Epsilon", f"{epsilon:.3f}"),
             (f"Chegou (últ. {JANELA})", f"{pct:.0f}%"),
             (f"Pontos (últ. {JANELA})", f"{media:.1f}"),
             ("Estados vistos", len(q))],
            ["ESPAÇO pausa | T turbo | S salva", "CIMA/BAIXO velocidade | ESC sai"],
            extra,
        )
        pygame.display.flip()
        dt = relogio.tick(60)
        if not pausado:
            tempo += dt

    salvar(so_ultimo=True)
    if historico:
        salvar_grafico(historico)
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=ARQUIVO_ULTIMO)
    p.add_argument("--velocidade", type=float, default=3.0, help="passos por segundo")
    p.add_argument("--treinar", action="store_true", help="treina ao vivo na tela")
    p.add_argument("--do-zero", action="store_true", help="(com --treinar) ignora o checkpoint")
    p.add_argument("--checkpoint-cada", type=int, default=1000, help="(com --treinar)")
    a = p.parse_args()
    if a.treinar:
        treinar_ao_vivo(a.do_zero, a.checkpoint_cada, a.velocidade)
    else:
        jogar(a.checkpoint, a.velocidade)
