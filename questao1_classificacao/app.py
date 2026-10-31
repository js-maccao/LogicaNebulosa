"""Interface Streamlit:  streamlit run app.py

Não treina nada: só lê os modelos de modelos/ e os resultados de resultados/.
"""
import inspect
import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import avaliar
import configuracao as cfg
import modelos
import pre_processamento
import preparar_dados
import treinar

R = cfg.PASTA_RESULTADOS
AVISO_LIMITACOES = (
    "**Limitações importantes**\n"
    "- \"Não cachorro\" é formado por apenas **5 espécies**: gato, galinha, vaca, cavalo e ovelha. "
    "O modelo não viu lobos, raposas, pessoas, objetos nem paisagens e pode errar com eles.\n"
    "- As duas classes vêm de **bases diferentes** (Stanford Dogs e Animals-10), com resoluções, formatos "
    "e estilos de foto diferentes. O modelo pode ter aprendido características da base (brilho, qualidade) "
    "em vez do animal.\n"
    "- As técnicas (pixels em cinza 64 x 64 para KNN/Random Forest; cor + HOG para o SVM) são didáticas e simples; "
    "não têm qualidade profissional."
)


# cache: cada arquivo é lido do disco uma vez só
@st.cache_resource
def carregar_modelo_em_cache(chave: str):
    return modelos.carregar_modelo(chave)


@st.cache_data
def ler_json(nome: str):
    return json.loads((R / nome).read_text(encoding="utf-8"))


@st.cache_data
def ler_csv(nome: str, **kw):
    return pd.read_csv(R / nome, encoding="utf-8", **kw)


def exigir_resultados():
    """Interrompe a página se o treinamento ainda não foi feito."""
    faltando = [n for n in ["resumo_dados.json", "metricas_teste.csv", "matrizes_teste.json"]
                if not (R / n).exists()]
    if faltando or not modelos.caminho_modelo("knn").exists():
        st.error("Os modelos ainda não foram treinados. No terminal, rode:\n\n"
                 "`python preparar_dados.py` e depois `python treinar.py`.")
        st.stop()


# --------------------------------------------------------------------------
# Página 1 - Visão geral
# --------------------------------------------------------------------------
def pagina_visao_geral():
    st.title("🐶 Cachorro ou não cachorro?")
    st.caption("Projeto acadêmico de Machine Learning: comparação entre KNN, Random Forest e SVM")
    exigir_resultados()
    resumo = ler_json("resumo_dados.json")

    st.header("Objetivo")
    st.write("Classificar uma imagem em duas classes: **Cachorro** (classe positiva, rótulo 1) e "
             "**Não cachorro** (rótulo 0), comparando três algoritmos com cinco métricas e matrizes de confusão.")

    st.header("Dados")
    c1, c2 = st.columns(2)
    c1.markdown("**Cachorro** - *Stanford Dogs Dataset* (Khosla et al., 2011), derivado do ImageNet, "
                "para pesquisa e educação. 500 imagens de 120 raças.")
    c2.markdown("**Não cachorro** - *Animals-10* (Rapidata, Hugging Face). Apenas **gato, galinha, vaca, "
                "cavalo e ovelha**, 100 imagens de cada. Licença não verificada; consultar a página do dataset.")
    st.write("O rótulo de cada imagem vem da **pasta de origem**. Raça/espécie é guardada só para análise de erros "
             "e nunca entra como atributo do modelo.")

    st.subheader("Imagens por grupo e por conjunto")
    tabela = pd.DataFrame(resumo["por_conjunto"]).T["por_grupo"].apply(pd.Series).fillna(0).astype(int)
    tabela = tabela[cfg.GRUPOS]
    tabela["Total"] = tabela.sum(axis=1)
    tabela.loc["Todos"] = tabela.sum()
    tabela.index = ["Treino", "Validação", "Teste", "Todos"]
    tabela.columns = [cfg.NOMES_GRUPOS_PT.get(c, c).capitalize() for c in tabela.columns]
    st.dataframe(tabela)
    st.caption(f"Corrompidas ignoradas: {len(resumo['corrompidas_ignoradas'])} | "
               f"arquivos duplicados: {resumo['arquivos_em_duplicatas']} | semente = {resumo['semente']}. "
               "A divisão é estratificada por grupo.")

    st.header("Algoritmos")
    st.write("**KNN** (k=5, com padronização dos pixels) e **Random Forest** (100 árvores) sobre os pixels; "
             "**SVM** (kernel RBF, C=1) sobre atributos de **cor (histograma HSV)** e **forma (HOG)**.")

    st.header("Fluxo")
    st.markdown("`Imagens` → `Preparação (64x64 cinza | cor + HOG)` → `Treinamento` → `Avaliação` → `Previsão`")
    st.warning(AVISO_LIMITACOES)


# --------------------------------------------------------------------------
# Página 2 - Classificar imagem
# --------------------------------------------------------------------------
def validar_upload(arquivo):
    """Devolve (bytes, mensagem_de_erro)."""
    dados = arquivo.getvalue()
    if len(dados) > cfg.TAMANHO_MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"O arquivo tem mais de {cfg.TAMANHO_MAX_UPLOAD_MB} MB."
    try:
        with Image.open(BytesIO(dados)) as img:
            img.verify()
    except Exception:
        return None, "Não foi possível abrir este arquivo como imagem (formato não suportado ou arquivo corrompido)."
    return dados, None


def entrada_do_modelo(chave: str, dados: bytes) -> np.ndarray:
    """Mesma preparação usada no treino desse modelo."""
    if modelos.ENTRADA_MODELO[chave] == "pixels":
        return pre_processamento.preparar_imagem(BytesIO(dados)).reshape(1, -1)
    return pre_processamento.extrair_atributos(BytesIO(dados)).reshape(1, -1)


def pagina_classificar():
    st.title("Classificar uma imagem")
    exigir_resultados()
    st.info("As probabilidades são **estimativas do modelo, não garantias de acerto**. "
            "O modelo só conhece 5 espécies de não cachorros (gato, galinha, vaca, cavalo, ovelha).")

    escolha = st.radio("Algoritmo", list(modelos.NOMES_MODELOS), format_func=modelos.NOMES_MODELOS.get,
                       horizontal=True)
    arquivo = st.file_uploader("Envie uma imagem", type=cfg.EXTENSOES_UPLOAD)
    if arquivo is None:
        return
    dados, erro = validar_upload(arquivo)
    if erro:
        st.error(erro)
        return

    try:
        x = entrada_do_modelo(escolha, dados)
        modelo = carregar_modelo_em_cache(escolha)
        proba = modelos.probabilidades(modelo, x)                             # {classe: probabilidade}
        classe = int(modelo.predict(x)[0])
    except Exception as e:
        st.error(f"Erro ao classificar a imagem: {e}")
        return

    col_img, col_res = st.columns([1, 1])
    col_img.image(dados, caption="Imagem enviada")
    with col_res:
        st.subheader(f"Previsão ({modelos.NOMES_MODELOS[escolha]}): {cfg.NOMES_CLASSES[classe]}")
        for rotulo in (cfg.ROTULO_CACHORRO, cfg.ROTULO_NAO_CACHORRO):
            p = proba[rotulo]
            st.write(f"**{cfg.NOMES_CLASSES[rotulo]}**: {p:.1%}")
            st.progress(p)
        if escolha == "knn":
            st.caption(f"No KNN a probabilidade é a fração dos {cfg.KNN_VIZINHOS} vizinhos mais parecidos "
                       "que pertencem à classe, por isso só assume valores como 0%, 20%, 40%...")
    with st.expander("O que o modelo \"enxerga\""):
        if modelos.ENTRADA_MODELO[escolha] == "pixels":
            st.image(x.reshape(cfg.LADO_IMAGEM, cfg.LADO_IMAGEM), clamp=True, width=256,
                     caption=f"{cfg.LADO_IMAGEM}x{cfg.LADO_IMAGEM} pixels em tons de cinza (4096 números)")
        else:
            st.write(f"{cfg.N_ATRIBUTOS_EXTRAIDOS} números: {cfg.N_ATRIBUTOS_COR} do histograma de cor HSV "
                     f"e {cfg.N_ATRIBUTOS_HOG} do HOG (direção das bordas em cada região da imagem).")
            st.bar_chart(pd.Series(x[0, :cfg.N_ATRIBUTOS_COR], name="proporção de pixels"))


# --------------------------------------------------------------------------
# Página 3 - Comparação
# --------------------------------------------------------------------------
def pagina_comparacao():
    st.title("Comparação dos modelos")
    exigir_resultados()
    teste = ler_csv("metricas_teste.csv", index_col="Modelo")
    validacao = ler_csv("metricas_validacao.csv", index_col="Modelo")

    st.header("Teste final (200 imagens, modelos treinados com treino + validação)")
    st.dataframe(teste.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"))
    st.caption("Verde = maior valor da coluna. Cada métrica tem seu melhor modelo; nenhum é necessariamente "
               "melhor em tudo.")
    st.pyplot(avaliar.desenhar_comparacao(teste, "Métricas no teste final"))

    st.header("Validação (160 imagens, modelos treinados só com o treino)")
    st.dataframe(validacao.style.format("{:.4f}"))
    st.caption("Usada durante o desenvolvimento. Não confundir com o teste final.")

    if (R / "validacao_cruzada.csv").exists():
        with st.expander("Validação cruzada (5 folds) - informação complementar"):
            st.dataframe(ler_csv("validacao_cruzada.csv", index_col="Modelo").style.format("{:.4f}"))

    st.info("Com cerca de 200 imagens de teste, uma diferença de poucos pontos entre os modelos pode ser "
            "ruído estatístico. A escolha do modelo depende do comportamento desejado: se perder um cachorro "
            "é pior que um alarme falso, vale olhar o **recall** e o **F2**; se alarmes falsos incomodam mais, "
            "a **precisão**.")


# --------------------------------------------------------------------------
# Página 4 - Matrizes de confusão
# --------------------------------------------------------------------------
def pagina_matrizes():
    st.title("Matrizes de confusão (teste final)")
    exigir_resultados()
    matrizes = ler_json("matrizes_teste.json")
    colunas = st.columns(len(matrizes))
    for col, (nome, dados) in zip(colunas, matrizes.items()):
        col.pyplot(avaliar.desenhar_matriz(np.array(dados["matriz"]), f"{nome} - teste final"))
        col.write(f"VP={dados['VP']} · VN={dados['VN']} · FP={dados['FP']} · FN={dados['FN']}")

    st.markdown(
        "- **VP (verdadeiro positivo)**: era cachorro e o modelo previu cachorro.\n"
        "- **VN (verdadeiro negativo)**: não era cachorro e o modelo previu não cachorro.\n"
        "- **FP (falso positivo)**: não era cachorro, mas o modelo previu cachorro.\n"
        "- **FN (falso negativo)**: era cachorro, mas o modelo previu não cachorro.\n\n"
        "Linhas = classe real; colunas = classe prevista.")

    st.header("Erros por grupo no teste")
    st.dataframe(ler_csv("erros_por_grupo_teste.csv"), hide_index=True)
    st.caption("Para os grupos de não cachorro, cada erro é um animal tomado por cachorro (falso positivo); "
               "para cachorro, é um cachorro não reconhecido (falso negativo).")

    st.header("Exemplos reais de erros")
    previsoes = ler_csv("previsoes_teste.csv")
    chave = st.radio("Modelo", list(modelos.NOMES_MODELOS), format_func=modelos.NOMES_MODELOS.get, horizontal=True)
    erros = previsoes[previsoes[f"previsto_{chave}"] != previsoes["rotulo"]].copy()
    erros["confianca"] = erros.apply(
        lambda l: l[f"proba_cachorro_{chave}"] if l[f"previsto_{chave}"] == 1 else 1 - l[f"proba_cachorro_{chave}"],
        axis=1)
    erros = erros.sort_values("confianca", ascending=False).head(8)
    st.caption("Os 8 erros em que o modelo estava mais \"confiante\".")
    cols = st.columns(4)
    for i, (_, linha) in enumerate(erros.iterrows()):
        with cols[i % 4]:
            arq = preparar_dados.caminho_absoluto(linha["caminho"])
            if arq.exists():
                st.image(str(arq))
            st.caption(f"Real: {cfg.NOMES_CLASSES[linha['rotulo']]} ({linha['raca_especie']})  \n"
                       f"Previsto: {cfg.NOMES_CLASSES[linha[f'previsto_{chave}']]}  \n"
                       f"P(cachorro) = {linha[f'proba_cachorro_{chave}']:.1%}")


# --------------------------------------------------------------------------
# Página 5 - Entenda o código
# --------------------------------------------------------------------------
def mostrar_funcao(funcao):
    st.code(inspect.getsource(funcao), language="python")


def pagina_codigo():
    st.title("Entenda o código")
    st.markdown("Cada etapa, com o arquivo e a função responsável. Os trechos abaixo são lidos do código real.")

    with st.expander("1. Como os dados são carregados - `preparar_dados.py` → `montar_tabela()`", expanded=True):
        st.write("Percorre `BaseCachorro` (subpastas de raça) e `BaseNãoCachorros`. O **rótulo vem da pasta**: "
                 "1 para cachorro, 0 para não cachorro. `validar_arquivos()` detecta imagens corrompidas e "
                 "`achar_duplicatas()` compara hashes SHA-256.")
        mostrar_funcao(preparar_dados.montar_tabela)
    with st.expander("2. Como as imagens viram números - `pre_processamento.py` → `preparar_imagem()`"):
        st.write("Os algoritmos só entendem números. A imagem vira um vetor de 4096 valores entre 0 e 1. "
                 "A **mesma função** é usada no treino, no teste e no upload.")
        mostrar_funcao(pre_processamento.abrir_rgb)
        mostrar_funcao(pre_processamento.preparar_imagem)
    with st.expander("3. Atributos de cor e forma do SVM - `pre_processamento.py` → `extrair_atributos()`"):
        st.write("Histograma de cor HSV (quais cores aparecem) + HOG (para onde apontam as bordas em cada região).")
        mostrar_funcao(pre_processamento.extrair_atributos)
        mostrar_funcao(pre_processamento.hog)
    with st.expander("4. O que são X e y - `preparar_dados.py` → `montar_X_y()`"):
        st.write("**X** é a matriz de atributos (uma linha por imagem, 4096 colunas de pixels). "
                 "**y** é o vetor de respostas corretas (0 ou 1).")
        mostrar_funcao(preparar_dados.montar_X_y)
    with st.expander("5. Como os dados são divididos - `preparar_dados.py` → `dividir()`"):
        st.write("20% teste; dos 80% restantes, 20% validação; o resto é treino. Estratificado por grupo, "
                 "`random_state=42`. `verificar_divisao()` confirma que nenhuma imagem repete.")
        mostrar_funcao(preparar_dados.dividir)
    with st.expander("6. Onde os modelos são criados e treinados - `modelos.py` e `treinar.py`"):
        st.write("`criar_knn()`, `criar_random_forest()` e `criar_svm_atributos()` definem os modelos; "
                 "`treinar.py → main()` chama `modelo.fit(X, y)`. A versão final é treinada com treino + validação.")
        mostrar_funcao(modelos.criar_knn)
        mostrar_funcao(modelos.criar_random_forest)
        mostrar_funcao(modelos.criar_svm_atributos)
    with st.expander("7. Onde as previsões e probabilidades são calculadas"):
        st.write("`modelo.predict(X)` devolve a classe; `modelo.predict_proba(X)` devolve as probabilidades, "
                 "na ordem de `modelo.classes_`. Em `treinar.py` (teste) e `app.py → pagina_classificar()` (upload).")
        mostrar_funcao(modelos.probabilidades)
    with st.expander("8. Onde as métricas são calculadas - `avaliar.py` → `calcular_metricas()`"):
        mostrar_funcao(avaliar.calcular_metricas)
    with st.expander("9. Onde as matrizes de confusão são geradas - `avaliar.py` → `matriz_confusao()`"):
        mostrar_funcao(avaliar.matriz_confusao)
        st.write("Chamada em `treinar.py → main()` com os rótulos reais do teste e as previsões do modelo final.")
    with st.expander("10. Como os modelos são salvos e carregados - `modelos.py`"):
        st.write("`salvar_modelo()` usa `joblib.dump` e `carregar_modelo()` usa "
                 "`joblib.load`. No Streamlit, `st.cache_resource` evita reler o arquivo a cada clique.")
        mostrar_funcao(modelos.salvar_modelo)
        mostrar_funcao(modelos.carregar_modelo)
    with st.expander("11. Fluxo completo do treinamento - `treinar.py` → `main()`"):
        mostrar_funcao(treinar.main)


# --------------------------------------------------------------------------
PAGINAS = {
    "Visão geral": pagina_visao_geral,
    "Classificar imagem": pagina_classificar,
    "Comparação dos modelos": pagina_comparacao,
    "Matrizes de confusão": pagina_matrizes,
    "Entenda o código": pagina_codigo,
}

st.set_page_config(page_title="Cachorro ou não cachorro", page_icon="🐶", layout="wide")
pagina = st.sidebar.radio("Navegação", list(PAGINAS))
st.sidebar.caption("KNN vs Random Forest vs SVM")
PAGINAS[pagina]()
