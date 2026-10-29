import os

import numpy as np
import pytest

import galinha_env as ge
import treinar
from galinha_env import TravessiaGalinha


def _env_sem_carros():
    env = TravessiaGalinha()
    env.carros = {l: [] for l in env.carros}
    return env


# ---------------- ambiente ----------------
def test_estado_tem_5_componentes_e_comeca_na_calcada():
    env = TravessiaGalinha()
    estado = env.reset()
    assert len(estado) == 5 and estado[:2] == (0, env.colunas // 2)
    assert all(v in (0, 1) for v in estado[2:])


def test_atropelada_ao_andar_para_cima_de_um_carro():
    env = _env_sem_carros()
    env.carros[1] = [env.coluna]                       # carro logo acima
    _, recompensa, terminou = env.passo(ge.CIMA)
    assert terminou and env.resultado == "atropelada"
    assert recompensa == ge.R_PASSO + ge.R_AVANCO + ge.R_ATROPELADA


def test_atropelada_quando_o_carro_chega():
    env = _env_sem_carros()
    env.linha = env.linha_max = 1
    env.carros[1] = [(env.coluna + 1) % env.colunas]   # pista 1 anda para a esquerda: chega na galinha
    _, recompensa, terminou = env.passo(ge.PARADA)
    assert terminou and env.resultado == "atropelada" and recompensa == ge.R_PASSO + ge.R_ATROPELADA


def test_chegada_da_bonus_e_termina():
    env = _env_sem_carros()
    env.linha = env.linha_max = env.linhas - 2
    _, recompensa, terminou = env.passo(ge.CIMA)
    assert terminou and env.resultado == "chegou"
    assert recompensa == ge.R_PASSO + ge.R_AVANCO + ge.R_OBJETIVO


def test_paredes_e_limite_de_passos():
    env = _env_sem_carros()
    env.passo(ge.BAIXO)                                # passo 1
    assert env.linha == 0                              # não sai do mapa
    env.max_passos = 3
    assert not env.passo(ge.PARADA)[2]                 # passo 2
    assert env.passo(ge.PARADA)[2] and env.resultado == "tempo"


def test_sensores_do_estado():
    env = _env_sem_carros()
    c = env.coluna
    env.carros[1] = [c]
    assert env.estado()[2] == 1                        # frente_ocupada
    env.carros[1] = [(c + 1) % env.colunas]            # pista 1 vai para a esquerda: chega em c
    assert env.estado()[3] == 1                        # frente_vindo


# ---------------- agente ----------------
def test_equacao_do_q_learning():
    q = treinar.nova_tabela_q()
    s, s2 = (0, 4, 0, 0, 0), (1, 4, 0, 0, 0)
    q[s2][:] = [0, 10, 3, 0, 0]
    antigo, novo = treinar.aprender(q, s, ge.CIMA, recompensa=-1, novo_estado=s2, terminou=False)
    esperado = 0 + treinar.ALFA * (-1 + treinar.GAMA * 10 - 0)
    assert antigo == 0 and np.isclose(novo, esperado)


def test_estado_terminal_nao_olha_o_futuro():
    q = treinar.nova_tabela_q()
    s, s2 = (9, 4, 0, 0, 0), (10, 4, 0, 0, 0)
    q[s2][:] = 999
    _, novo = treinar.aprender(q, s, ge.CIMA, recompensa=104, novo_estado=s2, terminou=True)
    assert np.isclose(novo, treinar.ALFA * 104)


def test_epsilon_guloso():
    q = treinar.nova_tabela_q()
    s = (3, 2, 0, 1, 0)
    q[s][ge.ESQUERDA] = 5
    assert all(treinar.escolher_acao(q, s, 0.0) == (ge.ESQUERDA, False) for _ in range(50))
    assert all(treinar.escolher_acao(q, s, 1.0)[1] for _ in range(50))


def test_checkpoint_salva_e_carrega(tmp_path):
    q = treinar.nova_tabela_q()
    q[(1, 2, 0, 0, 1)][3] = 7.5
    arq = tmp_path / "ck.pkl"
    treinar.salvar_checkpoint(str(arq), q, 0.3, 42, [(10, "chegou")])
    q2, eps, ep, hist = treinar.carregar_checkpoint(str(arq))
    assert eps == 0.3 and ep == 42 and hist == [(10, "chegou")]
    assert np.array_equal(q2[(1, 2, 0, 0, 1)], q[(1, 2, 0, 0, 1)])


def test_agente_treinado_atravessa_quase_sempre():
    if not os.path.exists(treinar.ARQUIVO_ULTIMO):
        pytest.skip("sem checkpoint treinado")
    import avaliar
    q, *_ = treinar.carregar_checkpoint(treinar.ARQUIVO_ULTIMO)
    sucesso, _, _ = avaliar.avaliar(avaliar.politica_gulosa(q), n=300)
    aleatorio, _, _ = avaliar.avaliar(avaliar.politica_aleatoria, n=300)
    assert sucesso > 90 and aleatorio < 5


def test_visual_desenha_sem_erro(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from visual import Cena
    env = TravessiaGalinha()
    cena = Cena(env, "teste")
    estado = env.reset()
    cena.sincronizar()
    cena.desenhar_mapa(0.5, None)
    cena.desenhar_painel("t", "s", estado, np.zeros(5), 0, [("a", 1)], ["ajuda"])
    pygame.quit()
