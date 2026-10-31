"""Testes do painel (streamlit.testing). Rodar da raiz: python -m pytest tests"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
PAGINAS = ["Início", "Q1 · Classificação", "Q2 · Regressão", "Q3 · Galinha", "Terminal", "Voltar ao zero"]


def abrir(pagina):
    at = AppTest.from_file(str(RAIZ / "painel.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value(pagina).run()
    return at


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_abre_sem_erro(pagina):
    at = abrir(pagina)
    assert not at.exception, [e.value for e in at.exception]


def test_executar_processo_captura_saida():
    import painel_exec as ex
    codigo, texto, _ = ex.executar_processo(RAIZ, ["-c", "print('ola')"])
    assert codigo == 0 and "ola" in texto


def test_treino_nao_toca_checkpoints_oficiais(tmp_path):
    import painel_q3 as q3
    antes = Path(q3.treinar.ARQUIVO_ULTIMO).read_bytes()
    infos = list(q3.treinar_por_blocos(100, True, str(tmp_path), checkpoint_cada=50, bloco=50))
    assert infos[-1]["episodio"] == 100 and (tmp_path / "ultimo.pkl").exists()
    assert Path(q3.treinar.ARQUIVO_ULTIMO).read_bytes() == antes


def test_replay_e_passo_a_passo():
    import painel_q3 as q3
    q, passos, resultado, env = q3.gravar_replay(q3.treinar.ARQUIVO_ULTIMO, 1)
    assert len(passos) > 1 and resultado in ("chegou", "atropelada", "tempo")
    _, p, regs, _ = q3.episodio_didatico(q3.treinar.nova_tabela_q(), 1.0, 3)
    assert len(p) == len(regs) + 1


def test_reset_apaga_so_gerados_e_restaura(tmp_path, monkeypatch):
    import painel_reset as rs
    q = tmp_path / "q"
    (q / "res").mkdir(parents=True)
    (q / "res" / "README.md").write_text("fica")
    (q / "res" / "a.csv").write_text("1")
    monkeypatch.setattr(rs, "RAIZ", tmp_path)
    monkeypatch.setattr(rs, "BACKUPS", tmp_path / "bk")
    monkeypatch.setitem(rs.ALVOS, "q1", [("teste", q / "res")])
    zip_, n = rs.zerar(["q1"])
    assert n == 1 and not (q / "res" / "a.csv").exists() and (q / "res" / "README.md").exists()
    rs.restaurar(zip_)
    assert (q / "res" / "a.csv").read_text() == "1"
