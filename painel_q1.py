"""Q1 no Streamlit: enviar uma imagem e ver o que cada modelo responde (reaproveita os módulos da Q1)."""
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

Q1 = Path(__file__).resolve().parent / "questao1_classificacao"
if str(Q1) not in sys.path:
    sys.path.insert(0, str(Q1))

import configuracao as cfg  # noqa: E402
import modelos  # noqa: E402
import pre_processamento  # noqa: E402


@st.cache_resource
def _modelo(chave, assinatura):
    """`assinatura` (data do arquivo) invalida o cache quando o modelo é retreinado ou apagado."""
    return modelos.carregar_modelo(chave)


def _assinatura(chave):
    arq = modelos.caminho_modelo(chave)
    return arq.stat().st_mtime if arq.exists() else None


def modelos_prontos():
    return [c for c in modelos.NOMES_MODELOS if modelos.caminho_modelo(c).exists()]


def _entrada(chave, dados):
    if modelos.ENTRADA_MODELO[chave] == "pixels":
        return pre_processamento.preparar_imagem(BytesIO(dados)).reshape(1, -1)
    return pre_processamento.extrair_atributos(BytesIO(dados)).reshape(1, -1)


def secao_classificar():
    st.info("As probabilidades são **estimativas, não garantias**. O modelo só conhece 5 espécies de não "
            "cachorros (gato, galinha, vaca, cavalo, ovelha): lobo, pessoa ou objeto podem confundi-lo.")
    prontos = modelos_prontos()
    if not prontos:
        st.warning("Nenhum modelo treinado. Rode os passos 1 e 2 desta página (preparar dados e treinar).")
        return
    escolha = st.radio("Algoritmo", prontos, format_func=modelos.NOMES_MODELOS.get, horizontal=True)
    arquivo = st.file_uploader("Envie uma imagem", type=cfg.EXTENSOES_UPLOAD)
    if arquivo is None:
        return
    dados = arquivo.getvalue()
    if len(dados) > cfg.TAMANHO_MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"O arquivo tem mais de {cfg.TAMANHO_MAX_UPLOAD_MB} MB.")
        return
    try:
        with Image.open(BytesIO(dados)) as img:
            img.verify()
        x = _entrada(escolha, dados)
        modelo = _modelo(escolha, _assinatura(escolha))
        proba = modelos.probabilidades(modelo, x)
        classe = int(modelo.predict(x)[0])
    except Exception as e:
        st.error(f"Não foi possível classificar este arquivo: {e}")
        return
    col_img, col_res = st.columns(2)
    col_img.image(dados, caption="Imagem enviada")
    with col_res:
        st.subheader(f"{modelos.NOMES_MODELOS[escolha]}: {cfg.NOMES_CLASSES[classe]}")
        for rotulo in (cfg.ROTULO_CACHORRO, cfg.ROTULO_NAO_CACHORRO):
            st.write(f"**{cfg.NOMES_CLASSES[rotulo]}**: {proba[rotulo]:.1%}")
            st.progress(float(proba[rotulo]))
        if escolha == "knn":
            st.caption(f"No KNN a probabilidade é a fração dos {cfg.KNN_VIZINHOS} vizinhos mais parecidos.")
    st.subheader("Os modelos lado a lado")
    linhas = []
    for c in prontos:
        p = modelos.probabilidades(_modelo(c, _assinatura(c)), _entrada(c, dados))
        linhas.append({"Modelo": modelos.NOMES_MODELOS[c], "P(cachorro)": p[cfg.ROTULO_CACHORRO],
                       "Decisão": cfg.NOMES_CLASSES[int(p[cfg.ROTULO_CACHORRO] >= 0.5)]})
    st.dataframe(pd.DataFrame(linhas).style.format({"P(cachorro)": "{:.1%}"}), hide_index=True)
    with st.expander("O que o modelo enxerga"):
        if modelos.ENTRADA_MODELO[escolha] == "pixels":
            st.image(x.reshape(cfg.LADO_IMAGEM, cfg.LADO_IMAGEM), clamp=True, width=256,
                     caption=f"{cfg.LADO_IMAGEM}x{cfg.LADO_IMAGEM} pixels em cinza")
        else:
            st.bar_chart(pd.Series(x[0, :cfg.N_ATRIBUTOS_COR], name="proporção de pixels (histograma HSV)"))
