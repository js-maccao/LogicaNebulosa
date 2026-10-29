"""Coloca a pasta do projeto no caminho de importação dos testes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
