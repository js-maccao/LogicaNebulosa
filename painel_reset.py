"""Volta o projeto ao estágio zero: apaga dados gerados, modelos, resultados e checkpoints.

Nunca apaga código, imagens-fonte, README nem testes. Antes de apagar, guarda um .zip em
backups_reset/ (fora do git) para poder desfazer.
"""
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
BACKUPS = RAIZ / "backups_reset"
Q1, Q2, Q3 = (RAIZ / "questao1_classificacao", RAIZ / "questao2_regressao", RAIZ / "questao3_reforco_galinha")

# (rótulo, pasta). O README.md de cada pasta é preservado.
ALVOS = {
    "q1": [("Q1 · modelos treinados", Q1 / "modelos"), ("Q1 · resultados", Q1 / "resultados")],
    "q2": [("Q2 · dados sintéticos", Q2 / "dados"), ("Q2 · resultados", Q2 / "resultados")],
    "q3": [("Q3 · checkpoints oficiais", Q3 / "checkpoints"),
           ("Q3 · checkpoints do painel", Q3 / "checkpoints_painel"),
           ("Q3 · resultados", Q3 / "resultados")],
}
NOMES = {"q1": "Q1 · Classificação", "q2": "Q2 · Regressão", "q3": "Q3 · Galinha"}
PRESERVAR = {"README.md", ".gitkeep"}


def arquivos(questoes):
    """Lista (rótulo, caminho) do que seria apagado."""
    achados = []
    for q in questoes:
        for rotulo, pasta in ALVOS[q]:
            if pasta.is_dir():
                for f in sorted(pasta.iterdir()):
                    if f.name not in PRESERVAR:
                        achados.append((rotulo, f))
    return achados


def _tamanho(f):
    return f.stat().st_size if f.is_file() else sum(x.stat().st_size for x in f.rglob("*") if x.is_file())


def resumo(questoes):
    """{rótulo: (nº de itens, bytes)}"""
    r = {}
    for rotulo, f in arquivos(questoes):
        n, b = r.get(rotulo, (0, 0))
        r[rotulo] = (n + 1, b + _tamanho(f))
    return r


def zerar(questoes, backup=True):
    """Apaga e devolve (caminho do backup ou None, quantidade apagada)."""
    itens = arquivos(questoes)
    destino = None
    if backup and itens:
        BACKUPS.mkdir(exist_ok=True)
        destino = BACKUPS / f"reset_{datetime.now():%Y%m%d_%H%M%S}.zip"
        with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
            for _, f in itens:
                for x in ([f] if f.is_file() else [y for y in f.rglob("*") if y.is_file()]):
                    z.write(x, x.relative_to(RAIZ))
    for _, f in itens:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    return destino, len(itens)


def restaurar(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAIZ)
        return len(z.namelist())


def listar_backups():
    return sorted(BACKUPS.glob("reset_*.zip"), reverse=True) if BACKUPS.exists() else []
