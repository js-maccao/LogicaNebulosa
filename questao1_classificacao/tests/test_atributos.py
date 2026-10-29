import numpy as np
from PIL import Image

import configuracao as cfg
from pre_processamento import extrair_atributos, histograma_hsv, hog


def _salvar(tmp_path, nome, img):
    caminho = tmp_path / nome
    img.save(caminho)
    return caminho


def test_formato_e_tipo(tmp_path):
    x = extrair_atributos(_salvar(tmp_path, "a.png", Image.effect_noise((200, 90), 60).convert("RGB")))
    assert x.shape == (cfg.N_ATRIBUTOS_EXTRAIDOS,) and x.dtype == np.float32
    assert np.isfinite(x).all()


def test_histograma_de_cor_soma_1(tmp_path):
    x = extrair_atributos(_salvar(tmp_path, "c.png", Image.new("RGB", (60, 60), (200, 30, 90))))
    cor = x[:cfg.N_ATRIBUTOS_COR]
    assert np.isclose(cor.sum(), 1.0)
    assert (cor == 1.0).sum() == 1                    # imagem de uma só cor -> uma única caixa cheia


def test_imagem_lisa_nao_tem_bordas():
    assert np.allclose(hog(np.full((cfg.LADO_ATRIBUTOS, cfg.LADO_ATRIBUTOS), 0.5, dtype=np.float32)), 0.0)


def test_hog_nao_muda_com_contraste():
    rng = np.random.default_rng(0)
    img = rng.random((cfg.LADO_ATRIBUTOS, cfg.LADO_ATRIBUTOS)).astype(np.float32)
    assert np.allclose(hog(img), hog(img * 0.5), atol=1e-3)   # normalização por bloco


def test_hog_detecta_direcao_da_borda():
    lado = cfg.LADO_ATRIBUTOS
    vertical = np.zeros((lado, lado), dtype=np.float32)
    vertical[:, lado // 2:] = 1.0                     # borda vertical -> gradiente horizontal (0 graus)
    blocos = hog(vertical).reshape(-1, 4, cfg.ORIENTACOES_HOG).sum(axis=(0, 1))
    assert blocos.argmax() == 0


def test_deterministico(tmp_path):
    caminho = _salvar(tmp_path, "d.jpg", Image.effect_noise((80, 80), 40).convert("RGB"))
    with open(caminho, "rb") as f:
        assert np.array_equal(extrair_atributos(caminho), extrair_atributos(f))


def test_histograma_hsv_direto():
    hsv = np.zeros((10, 10, 3), dtype=np.float32)
    h = histograma_hsv(hsv)
    assert h.shape == (cfg.N_ATRIBUTOS_COR,) and h[0] == 1.0
