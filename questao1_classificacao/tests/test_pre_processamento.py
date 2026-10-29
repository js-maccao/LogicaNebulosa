import numpy as np
from PIL import Image

import configuracao as cfg
from pre_processamento import preparar_imagem


def _salvar(tmp_path, nome, img):
    caminho = tmp_path / nome
    img.save(caminho)
    return caminho


def test_formato_e_faixa_de_valores(tmp_path):
    caminho = _salvar(tmp_path, "a.jpg", Image.new("RGB", (300, 120), (200, 30, 90)))
    x = preparar_imagem(caminho)
    assert x.shape == (cfg.N_ATRIBUTOS,)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_branco_e_preto(tmp_path):
    branco = preparar_imagem(_salvar(tmp_path, "b.png", Image.new("RGB", (50, 50), "white")))
    preto = preparar_imagem(_salvar(tmp_path, "p.png", Image.new("RGB", (50, 50), "black")))
    assert np.allclose(branco, 1.0) and np.allclose(preto, 0.0)


def test_modos_de_cor_diferentes_funcionam(tmp_path):
    base = Image.new("RGB", (40, 40), (10, 200, 10))
    for modo in ["RGBA", "L", "CMYK", "P"]:
        nome = "c.tif" if modo == "CMYK" else f"{modo}.png"
        x = preparar_imagem(_salvar(tmp_path, nome, base.convert(modo)))
        assert x.shape == (cfg.N_ATRIBUTOS,)


def test_transparencia_vira_fundo_branco(tmp_path):
    transparente = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
    assert np.allclose(preparar_imagem(_salvar(tmp_path, "t.png", transparente)), 1.0)


def test_mesmo_resultado_para_caminho_e_arquivo_aberto(tmp_path):
    caminho = _salvar(tmp_path, "c.png", Image.effect_noise((80, 60), 50).convert("RGB"))
    with open(caminho, "rb") as f:
        assert np.array_equal(preparar_imagem(caminho), preparar_imagem(f))


def test_orientacao_exif_e_corrigida(tmp_path):
    img = Image.new("L", (40, 20), 0)
    img.paste(255, (0, 0, 20, 20))                    # metade esquerda branca
    exif = Image.Exif()
    exif[0x0112] = 6                                  # "gire 90 graus para exibir"
    caminho = tmp_path / "e.jpg"
    img.convert("RGB").save(caminho, exif=exif)
    x = preparar_imagem(caminho).reshape(cfg.LADO_IMAGEM, cfg.LADO_IMAGEM)
    assert x[:32].mean() > 0.8 and x[32:].mean() < 0.2   # agora a parte branca está em cima
