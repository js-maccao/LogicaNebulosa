"""Transforma uma imagem em um vetor de números.

POR QUE NÚMEROS? KNN e Random Forest só entendem tabelas numéricas: cada linha é
um exemplo e cada coluna é um atributo. Uma imagem "bruta" (arquivo JPEG) é uma
sequência compactada de bytes, sem significado direto para esses algoritmos.
Por isso lemos os pixels e os transformamos em 4096 números (64 x 64 pixels em
tons de cinza), cada um entre 0 (preto) e 1 (branco).

IMPORTANTE: esta é a ÚNICA função de preparação usada no projeto inteiro
(treino, validação, teste e imagem enviada pelo usuário). Só os pixels entram:
nome, extensão, pasta, tamanho do arquivo e proporção original nunca são
usados como atributo.
"""
import numpy as np
from PIL import Image, ImageOps

from configuracao import (BINS_HSV, CELULA_HOG, LADO_ATRIBUTOS, LADO_IMAGEM, N_ATRIBUTOS_COR,
                          ORIENTACOES_HOG)


def preparar_imagem(fonte) -> np.ndarray:
    """Abre uma imagem e devolve um vetor float32 de shape (LADO_IMAGEM**2,).

    `fonte` pode ser um caminho (str/Path) ou um arquivo aberto (ex.: upload).

    Passos:
      1. Abrir com Pillow.
      2. Corrigir a orientação gravada no EXIF (fotos de celular "deitadas").
      3. Padronizar o modo de cor: PNG/WebP com transparência é colocado sobre
         fundo branco; CMYK, paleta (P) etc. são convertidos para RGB.
      4. Converter para escala de cinza ("L").
      5. Redimensionar para 64 x 64 (a imagem é "esticada" para esse quadrado;
         a proporção original não é usada como atributo).
      6. Virar vetor NumPy e normalizar de 0..255 para 0..1.
    """
    with Image.open(fonte) as img:
        img = ImageOps.exif_transpose(img)           # passo 2
        img = _remover_transparencia(img)            # passo 3
        img = img.convert("L")                       # passo 4
        img = img.resize((LADO_IMAGEM, LADO_IMAGEM), Image.Resampling.LANCZOS)  # passo 5
        pixels = np.asarray(img, dtype=np.float32)   # shape (64, 64), valores 0..255
    return (pixels / 255.0).reshape(-1)              # passo 6: vetor com 4096 posições


def _remover_transparencia(img: Image.Image) -> Image.Image:
    """Devolve a imagem em RGB; áreas transparentes viram branco."""
    tem_transparencia = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if tem_transparencia:
        rgba = img.convert("RGBA")
        fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(fundo, rgba).convert("RGB")
    return img.convert("RGB")


def preparar_varias(caminhos) -> np.ndarray:
    """Aplica `preparar_imagem` a uma lista de caminhos -> matriz (n_imagens, 4096)."""
    return np.vstack([preparar_imagem(c) for c in caminhos])


# ============================================================================
# Atributos de COR e FORMA (usados pelo 3º modelo, o SVM)
# ============================================================================
# Pixels crus dizem "o pixel 1234 é claro", o que muda toda vez que o cachorro
# se mexe na foto. Os atributos abaixo resumem a imagem de um jeito mais estável:
#   - histograma de cor HSV: QUAIS cores aparecem e em que proporção (não onde);
#   - HOG (histograma de gradientes orientados): em cada região, para que lado
#     apontam as bordas. Capta contornos (orelhas, focinho, patas) e textura (pelo).
def extrair_atributos(fonte) -> np.ndarray:
    """Abre uma imagem e devolve o vetor float32 [cor HSV (128) | HOG (1764)]."""
    with Image.open(fonte) as img:
        img = ImageOps.exif_transpose(img)
        img = _remover_transparencia(img)
        img = img.resize((LADO_ATRIBUTOS, LADO_ATRIBUTOS), Image.Resampling.LANCZOS)
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        cinza = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return np.concatenate([histograma_hsv(hsv), hog(cinza)]).astype(np.float32)


def histograma_hsv(hsv: np.ndarray) -> np.ndarray:
    """Conta os pixels em caixas de (matiz, saturação, brilho); soma = 1."""
    caixas = [np.minimum((hsv[..., i] * n).astype(int), n - 1) for i, n in enumerate(BINS_HSV)]
    indice = (caixas[0] * BINS_HSV[1] + caixas[1]) * BINS_HSV[2] + caixas[2]
    hist = np.bincount(indice.ravel(), minlength=N_ATRIBUTOS_COR).astype(np.float32)
    return hist / hist.sum()


def hog(cinza: np.ndarray) -> np.ndarray:
    """HOG simplificado, feito só com NumPy.

    1. Gradiente em x e y (diferença entre vizinhos): força e direção da borda.
    2. A imagem é dividida em células 16x16; cada célula vira um histograma de
       9 direções (0 a 180 graus), em que cada pixel vota com a força da borda.
    3. Blocos de 2x2 células são normalizados (L2), então o resultado não muda
       com o brilho/contraste da foto.
    """
    gx = np.zeros_like(cinza)
    gy = np.zeros_like(cinza)
    gx[:, 1:-1] = cinza[:, 2:] - cinza[:, :-2]
    gy[1:-1, :] = cinza[2:, :] - cinza[:-2, :]
    forca = np.hypot(gx, gy)
    direcao = np.rad2deg(np.arctan2(gy, gx)) % 180
    caixa = np.minimum((direcao / (180 / ORIENTACOES_HOG)).astype(int), ORIENTACOES_HOG - 1)

    n = cinza.shape[0] // CELULA_HOG
    linha_celula = np.arange(cinza.shape[0]) // CELULA_HOG
    coluna_celula = np.arange(cinza.shape[1]) // CELULA_HOG
    celula = linha_celula[:, None] * n + coluna_celula[None, :]
    hist = np.bincount((celula * ORIENTACOES_HOG + caixa).ravel(), weights=forca.ravel(),
                       minlength=n * n * ORIENTACOES_HOG).reshape(n, n, ORIENTACOES_HOG)

    blocos = np.concatenate([hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]], axis=2)
    norma = np.sqrt((blocos ** 2).sum(axis=2, keepdims=True) + 1e-6)
    return (blocos / norma).ravel()


def extrair_varios(caminhos) -> np.ndarray:
    """Aplica `extrair_atributos` a uma lista de caminhos -> matriz (n_imagens, 1892)."""
    return np.vstack([extrair_atributos(c) for c in caminhos])
