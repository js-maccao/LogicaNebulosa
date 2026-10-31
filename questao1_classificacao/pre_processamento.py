"""Transforma uma imagem em um vetor de números.

São duas representações:
  - preparar_imagem:   64 x 64 em tons de cinza -> 4096 valores (KNN e Random Forest)
  - extrair_atributos: histograma de cor HSV + HOG -> 1892 valores (SVM)

As mesmas funções são usadas no treino, no teste e na imagem enviada pela
interface. Só os pixels entram: nome, pasta e tamanho do arquivo não são usados.
"""
import numpy as np
from PIL import Image, ImageOps

from configuracao import (BINS_HSV, CELULA_HOG, LADO_ATRIBUTOS, LADO_IMAGEM, N_ATRIBUTOS_COR,
                          ORIENTACOES_HOG)


def abrir_rgb(fonte) -> Image.Image:
    """Abre a imagem (caminho ou arquivo aberto), corrige a rotação do EXIF e devolve em RGB.

    Fotos de celular costumam vir "deitadas" com a rotação só anotada no EXIF.
    Transparência (PNG/WebP) vira fundo branco.
    """
    with Image.open(fonte) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(fundo, rgba).convert("RGB")
        return img.convert("RGB")


def preparar_imagem(fonte) -> np.ndarray:
    """Imagem -> vetor float32 com 4096 valores entre 0 (preto) e 1 (branco).

    A imagem é esticada para 64 x 64 (a proporção original se perde).
    """
    img = abrir_rgb(fonte).convert("L")
    img = img.resize((LADO_IMAGEM, LADO_IMAGEM), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    return pixels.reshape(-1)


def preparar_varias(caminhos) -> np.ndarray:
    return np.vstack([preparar_imagem(c) for c in caminhos])


# Atributos de cor e forma (SVM)
# Pixels crus mudam toda vez que o animal se mexe na foto. Aqui a imagem é resumida
# de um jeito mais estável:
#   - histograma HSV: quais cores aparecem e em que proporção (não importa onde);
#   - HOG: em cada região, para que lado apontam as bordas (contorno e textura do pelo).
def extrair_atributos(fonte) -> np.ndarray:
    """Imagem -> vetor float32 [cor HSV (128) | HOG (1764)]."""
    img = abrir_rgb(fonte).resize((LADO_ATRIBUTOS, LADO_ATRIBUTOS), Image.Resampling.LANCZOS)
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
    """HOG simplificado, só com NumPy.

    1. Gradiente em x e y (diferença entre vizinhos): força e direção da borda.
    2. Células de 16x16 pixels; cada uma vira um histograma de 9 direções (0 a 180 graus),
       em que cada pixel vota com a força da borda.
    3. Blocos de 2x2 células são normalizados, para não depender do brilho/contraste da foto.
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
    return np.vstack([extrair_atributos(c) for c in caminhos])
