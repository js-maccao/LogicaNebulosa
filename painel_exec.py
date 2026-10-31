"""Execução de comandos do projeto com log ao vivo no Streamlit.

Só roda comandos de uma LISTA FIXA (COMANDOS): nunca texto digitado pelo usuário.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent
Q1 = RAIZ / "questao1_classificacao"
Q2 = RAIZ / "questao2_regressao"
Q3 = RAIZ / "questao3_reforco_galinha"

# chave -> (pasta, argumentos depois de "python", texto equivalente no PowerShell)
COMANDOS = {
    "q1_preparar": (Q1, ["preparar_dados.py"], "cd questao1_classificacao; python preparar_dados.py"),
    "q1_treinar": (Q1, ["treinar.py"], "cd questao1_classificacao; python treinar.py"),
    "q1_fuzzy": (Q1, ["analise_fuzzy.py"], "cd questao1_classificacao; python analise_fuzzy.py"),
    "q1_testes": (Q1, ["-m", "pytest", "-q"], "cd questao1_classificacao; python -m pytest"),
    "q2_gerar": (Q2, ["gerar_dados.py"], "cd questao2_regressao; python gerar_dados.py"),
    "q2_regressao": (Q2, ["regressao.py"], "cd questao2_regressao; python regressao.py"),
    "q2_testes": (Q2, ["-m", "pytest", "-q"], "cd questao2_regressao; python -m pytest"),
    "q3_avaliar": (Q3, ["avaliar.py"], "cd questao3_reforco_galinha; python avaliar.py"),
    "q3_testes": (Q3, ["-m", "pytest", "-q"], "cd questao3_reforco_galinha; python -m pytest"),
}
# abrem janela própria e não bloqueiam o painel
COMANDOS_JANELA = {
    "q3_pygame": (Q3, ["visualizador.py"], "cd questao3_reforco_galinha; python visualizador.py"),
}
ROTULOS = {
    "q1_preparar": "Q1 · preparar_dados.py", "q1_treinar": "Q1 · treinar.py", "q1_fuzzy": "Q1 · analise_fuzzy.py",
    "q1_testes": "Q1 · pytest", "q2_gerar": "Q2 · gerar_dados.py", "q2_regressao": "Q2 · regressao.py",
    "q2_testes": "Q2 · pytest", "q3_avaliar": "Q3 · avaliar.py", "q3_testes": "Q3 · pytest",
}


def _ambiente():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def executar_processo(pasta, args, destino=None, max_linhas=400):
    """Roda `python -u <args>` em `pasta` e escreve o log em `destino` (st.empty) linha a linha.
    Devolve (codigo_de_saida, texto_completo, segundos)."""
    inicio = time.time()
    linhas = []
    proc = subprocess.Popen([sys.executable, "-u", *args], cwd=str(pasta), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            env=_ambiente())
    try:
        for linha in proc.stdout:
            linhas.append(linha.rstrip("\n"))
            if destino is not None:
                destino.code("\n".join(linhas[-max_linhas:]), language="text")
        proc.wait()
    finally:
        if proc.poll() is None:      # o usuário interrompeu a execução da página
            proc.terminate()
    return proc.returncode, "\n".join(linhas), time.time() - inicio


def botao_comando(chave, rotulo=None, args_extra=(), mostrar_cmd=True):
    """Botão que executa um comando da lista fixa, com log ao vivo. Devolve True se rodou com sucesso agora."""
    pasta, args, ps = COMANDOS[chave]
    if mostrar_cmd:
        st.caption(f"Equivalente no terminal: `{ps}`")
    guardado = st.session_state.get(f"log_{chave}")
    rodar = st.button(f"▶ Executar {rotulo or ROTULOS[chave]}", key=f"btn_{chave}")
    area = st.empty()
    if rodar:
        with st.spinner("Executando..."):
            codigo, texto, seg = executar_processo(pasta, [*args, *args_extra], area)
        st.session_state[f"log_{chave}"] = (codigo, texto, seg)
        (st.success if codigo == 0 else st.error)(
            f"{'Concluído' if codigo == 0 else 'Falhou (código ' + str(codigo) + ')'} em {seg:.1f} s")
        return codigo == 0
    if guardado:
        codigo, texto, seg = guardado
        area.code(texto, language="text")
        (st.success if codigo == 0 else st.error)(f"Última execução: {seg:.1f} s (código {codigo})")
    return False


def abrir_janela(chave, args_extra=()):
    """Abre um programa com janela própria (pygame) sem travar o painel."""
    pasta, args, _ = COMANDOS_JANELA[chave]
    subprocess.Popen([sys.executable, *args, *args_extra], cwd=str(pasta), env=_ambiente())
