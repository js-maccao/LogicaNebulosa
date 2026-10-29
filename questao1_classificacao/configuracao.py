"""Configuração central do projeto: caminhos das bases, tamanho da imagem, semente.

Se as pastas das imagens mudarem de lugar, basta alterar este arquivo.
"""
from pathlib import Path

# --- Caminhos ---------------------------------------------------------------
PASTA_PROJETO = Path(__file__).resolve().parent
RAIZ_BASES = PASTA_PROJETO                              # as bases ficam dentro do projeto
PASTA_CACHORROS = RAIZ_BASES / "BaseCachorro"           # rótulo 1 (subpastas por raça)
PASTA_NAO_CACHORROS = RAIZ_BASES / "BaseNãoCachorros"   # rótulo 0 (espécie = prefixo do nome)
PASTA_MODELOS = PASTA_PROJETO / "modelos"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png"}

# --- Rótulos ----------------------------------------------------------------
ROTULO_CACHORRO = 1
ROTULO_NAO_CACHORRO = 0
NOMES_CLASSES = {ROTULO_NAO_CACHORRO: "Não cachorro", ROTULO_CACHORRO: "Cachorro"}
GRUPO_CACHORRO = "cachorro"
ESPECIES_NAO_CACHORRO = ["cat", "chicken", "cow", "horse", "sheep"]
GRUPOS = [GRUPO_CACHORRO] + ESPECIES_NAO_CACHORRO
NOMES_GRUPOS_PT = {
    "cachorro": "cachorro", "cat": "gato", "chicken": "galinha",
    "cow": "vaca", "horse": "cavalo", "sheep": "ovelha",
}

# --- Pré-processamento --------------------------------------------------------
LADO_IMAGEM = 64                                        # imagens viram 64 x 64 pixels
N_ATRIBUTOS = LADO_IMAGEM * LADO_IMAGEM                 # 4096 números por imagem

# --- Atributos de cor e forma (3º modelo) ---------------------------------------
LADO_ATRIBUTOS = 128                                    # imagem colorida 128 x 128 antes de extrair
BINS_HSV = (8, 4, 4)                                    # matiz x saturação x brilho = 128 caixas
CELULA_HOG = 16                                         # células de 16 x 16 pixels -> grade 8 x 8
ORIENTACOES_HOG = 9                                     # direções de borda entre 0 e 180 graus
_CELULAS = LADO_ATRIBUTOS // CELULA_HOG
N_ATRIBUTOS_COR = BINS_HSV[0] * BINS_HSV[1] * BINS_HSV[2]                    # 128
N_ATRIBUTOS_HOG = (_CELULAS - 1) ** 2 * 4 * ORIENTACOES_HOG                  # 7*7 blocos * 36 = 1764
N_ATRIBUTOS_EXTRAIDOS = N_ATRIBUTOS_COR + N_ATRIBUTOS_HOG

# --- Divisão dos dados --------------------------------------------------------
# A semente fixa torna o sorteio da divisão (e das florestas) reprodutível:
# rodar de novo gera exatamente os mesmos conjuntos e os mesmos modelos.
SEMENTE = 42
FRACAO_TESTE = 0.20          # 20% do total  -> teste final
FRACAO_VALIDACAO = 0.20      # 20% do que sobrou -> validação

# --- Hiperparâmetros (valores iniciais pedidos; não ajustados com o teste) -----
KNN_VIZINHOS = 5
RF_ARVORES = 100
SVM_C = 1.0                  # escolhido na VALIDAÇÃO entre 1, 3 e 10 (nunca no teste)
FOLDS_VALIDACAO_CRUZADA = 5

# --- Upload na interface ------------------------------------------------------
TAMANHO_MAX_UPLOAD_MB = 10
EXTENSOES_UPLOAD = ["jpg", "jpeg", "png", "bmp", "webp"]
