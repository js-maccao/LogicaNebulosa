"""Gera apresentacao.pdf (slides 16:9) com os modelos, resultados e discussões das 3 questões.

Uso:  python gerar_apresentacao.py

Os números e figuras são LIDOS dos resultados gerados por cada questão; rode antes:
  questao1_classificacao:   python preparar_dados.py && python treinar.py && python analise_fuzzy.py
  questao2_regressao:       python gerar_dados.py && python regressao.py
  questao3_reforco_galinha: python avaliar.py
"""
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False      # "R$" não vira fórmula
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

ALUNO = ""                       # <- preencha com o seu nome
TURMA = ""                       # <- opcional
REPOSITORIO = ""                 # <- link do GitHub, depois de publicar

RAIZ = Path(__file__).resolve().parent
Q1 = RAIZ / "questao1_classificacao"
Q2 = RAIZ / "questao2_regressao"
Q3 = RAIZ / "questao3_reforco_galinha"
R1, R2, R3 = Q1 / "resultados", Q2 / "resultados", Q3 / "resultados"
SAIDA = RAIZ / "apresentacao.pdf"

LARG, ALT = 13.333, 7.5
AZUL = "#1f3a5f"
VERMELHO = "#c0392b"
CINZA = "#555555"
_pagina = [0]


# --------------------------------------------------------------------------
# Utilitários de desenho
# --------------------------------------------------------------------------
def br(v, casas=3):
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def novo_slide(titulo, secao=""):
    _pagina[0] += 1
    fig = plt.figure(figsize=(LARG, ALT))
    fig.add_artist(Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=AZUL))
    fig.text(0.035, 0.942, titulo, color="white", fontsize=23, weight="bold", va="center")
    if secao:
        fig.text(0.965, 0.942, secao, color="#c9d6e6", fontsize=13, ha="right", va="center")
    fig.text(0.965, 0.025, str(_pagina[0]), color=CINZA, fontsize=10, ha="right")
    fig.text(0.035, 0.025, "Trabalho Final - Lógica Nebulosa", color="#999999", fontsize=9)
    return fig


def texto(fig, itens, x=0.04, y=0.84, largura=0.92, tamanho=15, espaco=1.45):
    """Lista de tópicos. Item começando com '-' vira sub-tópico; '!' vira destaque; '' pula linha."""
    chars = int(largura * LARG * 72 / (0.56 * tamanho))
    altura_linha = tamanho * espaco / 72 / ALT
    for item in itens:
        if item == "":
            y -= altura_linha * 0.5
            continue
        nivel, cor, peso, marcador = 0, "black", "normal", "•"
        if item.startswith("-"):
            nivel, item, marcador, cor = 1, item[1:].strip(), "–", "#333333"
        elif item.startswith("!"):
            item, cor, peso, marcador = item[1:].strip(), VERMELHO, "bold", "»"
        recuo = 0.025 * nivel
        tam = tamanho - 1.5 * nivel
        limite = int(chars * (0.86 if peso == "bold" else 1.0) * tamanho / tam)   # negrito é mais largo
        linhas = textwrap.wrap(item, max(20, limite - 4 * nivel))
        for i, linha in enumerate(linhas):
            if i == 0:
                fig.text(x + recuo, y, marcador, fontsize=tam, color=cor, va="top", weight=peso)
            fig.text(x + recuo + 0.018, y, linha, fontsize=tam, color=cor, va="top", weight=peso)
            y -= altura_linha
        y -= altura_linha * 0.25
    return y


def imagem(fig, caminho, rect, legenda=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if Path(caminho).exists():
        ax.imshow(plt.imread(str(caminho)))
    else:
        ax.text(0.5, 0.5, f"(figura não encontrada:\n{Path(caminho).name})", ha="center", va="center")
    if legenda:
        ax.set_title(legenda, fontsize=10, color=CINZA)
    return ax


def tabela(fig, df, rect, tamanho=13, destacar_coluna_max=(), destacar_coluna_min=(), larguras=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    celulas = [[str(v) for v in linha] for linha in df.astype(object).values]
    t = ax.table(cellText=celulas, colLabels=list(df.columns), loc="center", cellLoc="center",
                 colWidths=larguras)
    t.auto_set_font_size(False)
    t.set_fontsize(tamanho)
    t.scale(1, 1.9)
    for (i, j), cel in t.get_celld().items():
        cel.set_edgecolor("#bbbbbb")
        if i == 0:
            cel.set_facecolor(AZUL)
            cel.set_text_props(color="white", weight="bold")
        elif i % 2 == 0:
            cel.set_facecolor("#f2f5f9")
    return t


def destacar_celulas(t, df, colunas, maior=True):
    for nome in colunas:
        j = list(df.columns).index(nome)
        valores = pd.to_numeric(df[nome].astype(str).str.replace(".", "").str.replace(",", "."), errors="coerce")
        i = int((valores.idxmax() if maior else valores.idxmin()))
        linha = list(df.index).index(i) + 1
        t[linha, j].set_text_props(weight="bold", color=VERMELHO)


def quadro(fig, rect, linhas, tamanho=14, cor_fundo="#fff6e5", cor_borda="#f2a93b"):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=cor_fundo, ec=cor_borda, lw=2))
    ax.text(0.5, 0.5, "\n".join(linhas), ha="center", va="center", fontsize=tamanho, transform=ax.transAxes,
            linespacing=1.6)


def salvar(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------
def capa(pdf):
    _pagina[0] += 1
    fig = plt.figure(figsize=(LARG, ALT))
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=AZUL))
    fig.text(0.5, 0.74, "Universidade Carioca", color="#c9d6e6", fontsize=22, ha="center")
    fig.text(0.5, 0.60, "Trabalho Final - Lógica Nebulosa", color="white", fontsize=38, ha="center", weight="bold")
    fig.text(0.5, 0.52, "Aprendizado de Máquina e Lógica Nebulosa", color="white", fontsize=22, ha="center")
    fig.text(0.5, 0.40, "Q1 Classificação  •  Q2 Regressão  •  Q3 Aprendizado por Reforço",
             color="#f2c230", fontsize=17, ha="center")
    fig.text(0.5, 0.24, f"Aluno(a): {ALUNO or '______________________'}", color="white", fontsize=17, ha="center")
    fig.text(0.5, 0.18, "Professor: Prof. Mardônio França" + (f"   |   Turma: {TURMA}" if TURMA else ""),
             color="white", fontsize=15, ha="center")
    fig.text(0.5, 0.12, date.today().strftime("%d/%m/%Y"), color="#c9d6e6", fontsize=13, ha="center")
    salvar(pdf, fig)


def visao_geral(pdf):
    fig = novo_slide("Visão geral do trabalho")
    texto(fig, [
        "Questão 1 - Classificação: cachorro x não cachorro em imagens, com 3 modelos (KNN, Random Forest, SVM).",
        "-Saída DISCRETA (0 ou 1); a probabilidade de cada previsão é lida como grau de pertinência fuzzy.",
        "Questão 2 - Regressão: preço de passagens aéreas (Regressão Linear x Gradient Boosting).",
        "-Saída CONTÍNUA (R$); o valor já carrega a gradação que a lógica fuzzy formaliza.",
        "Questão 3 - Aprendizado por reforço: galinha atravessando uma rua com carros (Q-learning tabular).",
        "-Variação do labirinto: obstáculos móveis e sorteados; o agente aprende uma política.",
        "",
        "!Fio condutor: a realidade raramente é 0 ou 1. Os modelos produzem GRAUS (probabilidade, valor contínuo, "
        "valor Q) e a decisão crisp é uma etapa final que descarta parte dessa informação.",
        "",
        "Tudo em Python (scikit-learn, NumPy, pandas, matplotlib); código, dados, testes e README em cada pasta.",
    ], tamanho=16)
    salvar(pdf, fig)


# ---------------- Questão 1 ----------------
def q1_dados(pdf):
    fig = novo_slide("Q1 - Problema e dados", "Questão 1 · Classificação")
    texto(fig, [
        "Classificação binária: Cachorro (1, positivo) x Não cachorro (0).",
        "1.000 imagens:",
        "-500 cachorros: Stanford Dogs (120 raças).",
        "-500 não cachorros: Animals-10 (gato, galinha, vaca, cavalo, ovelha; 100 de cada).",
        "Rótulo vem da pasta; 0 corrompidas, 0 duplicatas (SHA-256).",
        "Divisão estratificada por grupo (semente 42):",
        "-treino 640 | validação 160 | teste 200.",
        "-O teste é usado UMA vez, no final.",
        "!Limitação: só 5 espécies e duas bases diferentes (viés de fonte).",
    ], largura=0.47, tamanho=14)
    divisao = pd.read_csv(R1 / "divisao.csv", encoding="utf-8")
    exemplos = divisao.groupby("grupo").head(1).set_index("grupo")
    nomes = {"cachorro": "cachorro", "cat": "gato", "chicken": "galinha", "cow": "vaca", "horse": "cavalo",
             "sheep": "ovelha"}
    for k, grupo in enumerate(["cachorro", "cat", "chicken", "cow", "horse", "sheep"]):
        ax = fig.add_axes([0.53 + (k % 3) * 0.152, 0.47 - (k // 3) * 0.38, 0.14, 0.34])
        ax.axis("off")
        with Image.open(Q1 / exemplos.loc[grupo, "caminho"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(nomes[grupo], fontsize=11)
    salvar(pdf, fig)


def q1_modelos(pdf):
    fig = novo_slide("Q1 - Representação das imagens e modelos", "Questão 1 · Classificação")
    df = pd.DataFrame([
        ["KNN (k=5)", "pixels cinza 64×64 (4096)", "5 vizinhos mais próximos votam", "fração de votos (0; 0,2; ...)"],
        ["Random Forest (100)", "pixels cinza 64×64 (4096)", "árvores de decisão votam", "média das árvores"],
        ["SVM RBF (C=1)", "cor HSV (128) + HOG (1764)", "fronteira de margem máxima", "distância calibrada (Platt)"],
    ], columns=["Modelo", "Entrada", "Como decide", "Probabilidade"])
    tabela(fig, df, [0.04, 0.55, 0.92, 0.28], tamanho=13, larguras=[0.18, 0.27, 0.28, 0.27])
    texto(fig, [
        "Pixels crus mudam quando o animal se desloca na foto; o 3º modelo usa atributos mais estáveis:",
        "-Histograma HSV: QUAIS cores aparecem (pasto, céu, pelagem), não onde.",
        "-HOG (implementado em NumPy): direção das bordas em cada célula 16×16 → contornos e textura.",
        "StandardScaler dentro do Pipeline (ajustado só no treino); C do SVM escolhido na VALIDAÇÃO.",
        "Modelo final treinado com treino + validação (800 imagens) e avaliado uma vez no teste.",
    ], y=0.47, tamanho=14)
    salvar(pdf, fig)


def q1_metricas(pdf):
    fig = novo_slide("Q1 - Métricas no teste (200 imagens)", "Questão 1 · Classificação")
    m = pd.read_csv(R1 / "metricas_teste.csv", index_col="Modelo", encoding="utf-8")
    auc = pd.read_csv(R1 / "auc_teste.csv", index_col="Modelo", encoding="utf-8")
    cv = pd.read_csv(R1 / "validacao_cruzada.csv", index_col="Modelo", encoding="utf-8")
    df = m.join(auc)
    df["Acur. CV"] = cv["Acurácia (média)"]
    mostrar = df.reset_index().rename(columns={"F1-score": "F1", "F2-score": "F2"})
    for c in mostrar.columns[1:]:
        mostrar[c] = mostrar[c].map(lambda v: br(v, 3))
    t = tabela(fig, mostrar, [0.03, 0.6, 0.94, 0.23], tamanho=13,
               larguras=[0.2] + [0.1] * (len(mostrar.columns) - 1))
    destacar_celulas(t, mostrar, list(mostrar.columns[1:]))
    melhor_auc = auc["ROC-AUC"].idxmax()
    texto(fig, [
        "Em vermelho, o melhor de cada métrica. F2 = (5·P·R)/(4·P + R): o recall pesa 4×. Acur. CV = validação cruzada.",
        f"O {melhor_auc} lidera em acurácia, precisão, ROC-AUC e validação cruzada.",
        "-A ROC-AUC usa a probabilidade inteira (todos os limiares): mede quão bem o modelo ORDENA as imagens.",
        "O Random Forest tem o maior F1/F2 no limiar 0,5; o KNN tem o maior recall, mas muitos alarmes falsos.",
        "!Com 200 imagens, a margem de erro da acurácia é ≈ ±7 pontos: diferenças pequenas não são conclusivas.",
        "Desempenho modesto: representações simples, poucos dados; os três ficam acima do acaso (50%).",
    ], y=0.52, tamanho=14)
    salvar(pdf, fig)


def q1_matrizes(pdf):
    fig = novo_slide("Q1 - Matrizes de confusão (teste)", "Questão 1 · Classificação")
    for k, chave in enumerate(["knn", "random_forest", "svm_atributos"]):
        imagem(fig, R1 / f"matriz_{chave}.png", [0.02 + k * 0.325, 0.25, 0.31, 0.6])
    texto(fig, [
        "Linhas = classe real; colunas = prevista. FP = animal tomado por cachorro; FN = cachorro perdido.",
        "KNN: muitos FP (chama quase tudo de cachorro). SVM: menos FP (mais preciso), porém mais FN.",
    ], y=0.2, tamanho=14)
    salvar(pdf, fig)


def q1_fuzzy(pdf):
    fig = novo_slide("Q1 c) Probabilidade da predição = grau de pertinência", "Questão 1 · Lógica fuzzy")
    imagem(fig, R1 / "pertinencia_fuzzy.png", [0.0, 0.07, 0.56, 0.8])
    texto(fig, [
        "predict_proba(X) devolve P(cachorro) para cada imagem.",
        "Lógica clássica: a imagem É ou NÃO É cachorro (0 ou 1).",
        "Lógica fuzzy: μ_cachorro = P(cachorro) e μ_não = 1 − P.",
        "-P = 0,74 → 'bastante cachorro'; P = 0,52 → quase totalmente ambíguo.",
        "Três termos linguísticos sobre P(cachorro): 'não cachorro', 'incerto', 'cachorro' (somam 1).",
        "!predict (corte em 0,5) é uma DEFUZZIFICAÇÃO: transforma o grau em sim/não e joga fora a incerteza.",
    ], x=0.57, largura=0.41, y=0.84, tamanho=14)
    salvar(pdf, fig)


def q1_zona(pdf):
    fig = novo_slide("Q1 c) A incerteza diz quando confiar na resposta", "Questão 1 · Lógica fuzzy")
    z = pd.read_csv(R1 / "zona_incerta.csv", encoding="utf-8")
    z["valor"] = z.apply(lambda l: f"{br(100 * l['Acurácia'], 1)}% ({l['Imagens']})", axis=1)
    tab = z.pivot(index="Modelo", columns="Termo", values="valor")[["não cachorro", "incerto", "cachorro"]]
    tab = tab.reset_index()
    tab.columns = ["Modelo", "'não cachorro'", "'incerto'", "'cachorro'"]
    tabela(fig, tab, [0.03, 0.6, 0.5, 0.22], tamanho=12)
    fig.text(0.03, 0.56, "acerto da decisão crisp (nº de imagens) por termo dominante", fontsize=10, color=CINZA)
    svm = z[z["Modelo"].str.startswith("SVM")].set_index("Termo")
    texto(fig, [
        f"SVM: pertinência clara → {br(100 * svm.loc['não cachorro', 'Acurácia'], 0)}% e "
        f"{br(100 * svm.loc['cachorro', 'Acurácia'], 0)}% de acerto;",
        f"-zona incerta → {br(100 * svm.loc['incerto', 'Acurácia'], 0)}% (quase cara ou coroa).",
        "O grau de pertinência mede a confiança; a saída crisp não.",
        "Um sistema fuzzy poderia responder 'não sei → revisar' na zona incerta.",
        "KNN: só 6 valores de P (pertinência em degraus); SVM: 200 valores distintos.",
    ], x=0.03, y=0.5, largura=0.5, tamanho=13)
    imagem(fig, R1 / "exemplos_pertinencia.png", [0.54, 0.05, 0.45, 0.82])
    salvar(pdf, fig)


def q1_limiar(pdf):
    fig = novo_slide("Q1 c) O limiar 0,5 é uma escolha (fronteira crisp)", "Questão 1 · Lógica fuzzy")
    imagem(fig, R1 / "metricas_vs_limiar.png", [0.02, 0.33, 0.96, 0.53])
    texto(fig, [
        "Mover o corte troca precisão por recall: baixar o limiar encontra mais cachorros (F2 ↑), com mais alarmes falsos.",
        "A fronteira entre 'cachorro' e 'não cachorro' no espaço de atributos não é nítida: o próprio conceito é difuso",
        "-(alguns gatos e cães pequenos se parecem mais entre si do que um chihuahua com um são-bernardo).",
    ], y=0.28, tamanho=14)
    salvar(pdf, fig)


# ---------------- Questão 2 ----------------
def q2_dados(pdf):
    fig = novo_slide("Q2 - Problema, dados e modelos", "Questão 2 · Regressão")
    texto(fig, [
        "Prever o PREÇO (R$) de uma passagem aérea: variável contínua.",
        "5.000 passagens SINTÉTICAS (gerar_dados.py): regra conhecida + ruído de ≈10%.",
        "-Não há base pública offline; a regra conhecida permite checar o que o modelo aprendeu.",
        "-Variáveis: distância, antecedência, escalas, alta temporada, companhia, classe, dia da semana, período.",
        "preço = (90 + 0,38·km) × companhia × classe × (1 + 1,2·e^(−dias/12)) × escalas × dia × temporada × período × ruído",
        "a) Divisão treino 80% (4.000) / teste 20% (1.000), semente 42.",
        "Pré-processamento no Pipeline: StandardScaler (numéricas) + OneHotEncoder (categóricas).",
        "Modelos:",
        "-Regressão Linear: soma de pesos × atributos (referência).",
        "-Gradient Boosting: 400 árvores pequenas, cada uma corrige o erro das anteriores (não linear, interações).",
    ], tamanho=15)
    salvar(pdf, fig)


def q2_metricas(pdf):
    fig = novo_slide("Q2 b) Métricas no teste e previsto x real", "Questão 2 · Regressão")
    m = pd.read_csv(R2 / "metricas.csv", index_col="Modelo", encoding="utf-8").reset_index()
    mostrar = pd.DataFrame({
        "Modelo": m["Modelo"], "MAE (R$)": m["MAE"].map(lambda v: br(v, 2)),
        "MSE (R$²)": m["MSE"].map(lambda v: br(v, 0)), "RMSE (R$)": m["RMSE"].map(lambda v: br(v, 2)),
        "R²": m["R2"].map(lambda v: br(v, 3))})
    tabela(fig, mostrar, [0.03, 0.66, 0.45, 0.17], tamanho=12, larguras=[0.3, 0.17, 0.19, 0.17, 0.12])
    b = m.set_index("Modelo").loc["Gradient Boosting"]
    texto(fig, [
        f"MAE: erro médio absoluto → o Boosting erra ≈ R$ {br(b['MAE'], 0)} por passagem.",
        "MSE: média dos erros²; pune erros grandes (unidade R$²).",
        f"RMSE = √MSE, de volta em R$ (R$ {br(b['RMSE'], 0)}).",
        f"R²: {br(100 * b['R2'], 1)}% da variação do preço explicada; o resto ≈ ruído dos dados.",
        "A Linear perde porque a regra é multiplicativa e não linear (curvatura no gráfico).",
    ], x=0.03, y=0.6, largura=0.45, tamanho=13)
    imagem(fig, R2 / "previsto_vs_real.png", [0.48, 0.06, 0.51, 0.8])
    salvar(pdf, fig)


def q2_diagnostico(pdf):
    fig = novo_slide("Q2 b) Resíduos e importância das variáveis", "Questão 2 · Regressão")
    imagem(fig, R2 / "residuos.png", [0.01, 0.2, 0.55, 0.66])
    imagem(fig, R2 / "importancia_variaveis.png", [0.56, 0.2, 0.43, 0.66])
    texto(fig, [
        "Resíduos do Boosting em torno de zero, sem padrão. A Linear mostra curvatura: falta não linearidade.",
        "Distância, classe e antecedência são as mais importantes, exatamente como na regra que gerou os dados.",
    ], y=0.17, tamanho=14)
    salvar(pdf, fig)


def q2_continuo(pdf):
    fig = novo_slide("Q2 c) Resultado contínuo x classificação discreta", "Questão 2 · Regressão")
    imagem(fig, R2 / "continuo_vs_discreto.png", [0.01, 0.33, 0.98, 0.53])
    prev = pd.read_csv(R2 / "previsoes_teste.csv", encoding="utf-8")
    distintos = prev["previsto_boosting"].nunique()
    texto(fig, [
        f"Q1: saída discreta, 1 entre 2 rótulos. Q2: saída contínua, um número real ({distintos} valores distintos em "
        f"{len(prev)} passagens).",
        "Discretizando em faixas (barata < R$ 600 ≤ média < R$ 1.200 ≤ cara), a regressão vira classificação de 3 classes.",
        "-Perde-se informação: R$ 610 e R$ 1.150 viram a mesma 'média'; R$ 590 e R$ 609 caem em classes diferentes.",
        "-Os erros de faixa aparecem só entre faixas vizinhas, justamente nas fronteiras arbitrárias.",
    ], y=0.29, tamanho=13)
    salvar(pdf, fig)


def q2_fuzzy(pdf):
    fig = novo_slide("Q2 d) A lógica nebulosa já está no resultado contínuo", "Questão 2 · Lógica fuzzy")
    imagem(fig, R2 / "pertinencia_preco.png", [0.0, 0.36, 0.58, 0.5])
    ex = pd.read_csv(R2 / "exemplos_pertinencia.csv", encoding="utf-8")
    mostrar = pd.DataFrame({
        "Previsto": ex["previsto_boosting"].map(lambda v: "R$ " + br(v, 0)), "Faixa crisp": ex["faixa_prevista"],
        "μ barata": ex["mu_barata"].map(lambda v: br(v, 2)), "μ média": ex["mu_média"].map(lambda v: br(v, 2)),
        "μ cara": ex["mu_cara"].map(lambda v: br(v, 2))})
    tabela(fig, mostrar, [0.59, 0.4, 0.39, 0.45], tamanho=11)
    texto(fig, [
        "O valor contínuo carrega GRAU: R$ 1.150 é 'um pouco média (0,58) e um pouco cara (0,42)', não uma coisa ou outra.",
        "Crisp: R$ 590 é barata e R$ 609 é média. Fuzzy: ambas ≈ 0,5/0,5, praticamente iguais (a leitura humana correta).",
        "!A regressão é o problema 'não defuzzificado': na Q1 o corte 0,5 transforma o grau em rótulo; aqui o grau é a saída.",
        "Há ainda a incerteza do próprio número: RMSE ≈ R$ 193 → 'R$ 1.150' é, na prática, algo entre ~R$ 960 e ~R$ 1.340.",
    ], y=0.33, tamanho=13)
    salvar(pdf, fig)


# ---------------- Questão 3 ----------------
def q3_ambiente(pdf):
    fig = novo_slide("Q3 a) Ambiente: galinha atravessando a rua", "Questão 3 · Reforço")
    texto(fig, [
        "Grade 11 × 8: linha 0 = calçada (início), 10 = grama (chegada), 1 a 9 = pistas.",
        "-2 carros por pista; pistas alternam o sentido; carros sorteados a cada episódio.",
        "Estado: (linha, coluna, frente_ocupada, frente_vindo, aqui_vindo).",
        "-11 × 8 × 2 × 2 × 2 = 704 estados (em vez de ~10¹³ com o mapa inteiro).",
        "Ações: CIMA, BAIXO, ESQUERDA, DIREITA, PARADA.",
        "Fim do episódio: atropelada, chegou ou 100 passos.",
        "É um 'labirinto' com paredes móveis: não existe caminho fixo, e o agente aprende uma POLÍTICA.",
    ], largura=0.55, tamanho=14)
    df = pd.DataFrame([["cada passo", "−1"], ["linha mais alta que todas as anteriores", "+5"],
                       ["atropelada (fim)", "−50"], ["chegou do outro lado (fim)", "+100"]],
                      columns=["Evento", "Recompensa"])
    tabela(fig, df, [0.58, 0.5, 0.39, 0.3], tamanho=12, larguras=[0.7, 0.3])
    fig.text(0.775, 0.45,"galinha_env.py (só as regras, sem aprendizado)", fontsize=10, color=CINZA, ha="center")
    salvar(pdf, fig)


def q3_agente(pdf):
    fig = novo_slide("Q3 a) Agente: Q-learning tabular", "Questão 3 · Reforço")
    quadro(fig, [0.12, 0.62, 0.76, 0.18],
           ["Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]"], tamanho=22)
    texto(fig, [
        "Tabela Q: estado → 5 valores (um por ação) = quanto vale cada ação naquele estado.",
        "α = 0,1 (taxa de aprendizado): corrige 10% do erro a cada passo.",
        "γ = 0,95 (desconto): recompensas futuras valem um pouco menos que as imediatas.",
        "Epsilon-greedy: com chance ε faz ação aleatória (explora); senão, a de maior Q (usa o que aprendeu).",
        "-ε começa em 1,0 e decai ×0,9995 por episódio até 0,05 (≈ episódio 6.000).",
        "Implementado do zero, sem bibliotecas de RL (treinar.py: escolher_acao, aprender, rodar_episodio).",
    ], y=0.55, tamanho=15)
    salvar(pdf, fig)


def q3_evolucao(pdf):
    fig = novo_slide("Q3 b) Evolução do aprendizado ao longo dos episódios", "Questão 3 · Reforço")
    imagem(fig, R3 / "curva_aprendizado.png", [0.0, 0.2, 0.5, 0.62], "durante o treino (com exploração ε)")
    imagem(fig, R3 / "evolucao_checkpoints.png", [0.5, 0.2, 0.5, 0.62], "cada checkpoint SEM exploração, 1.000 partidas")
    ev = pd.read_csv(R3 / "evolucao_checkpoints.csv", encoding="utf-8")
    al = ev[ev["agente"] == "aleatorio"].iloc[0]
    tr = ev[ev["agente"] == "treinado"].set_index("episodio")
    final = tr.index.max()
    marcos = [e for e in (1000, 3000, 5000) if e in tr.index]
    texto(fig, [
        f"Travessias sem exploração - aleatório: {br(al['chegou_pct'], 1)}% | " +
        " | ".join(f"ep. {e:,}".replace(",", ".") + f": {br(tr.loc[e, 'chegou_pct'], 1)}%" for e in marcos) +
        f" | ep. {final:,}".replace(",", ".") + f": {br(tr.loc[final, 'chegou_pct'], 1)}%",
        "No treino a curva fica em ≈76% porque 5% das ações continuam aleatórias (ε mínimo).",
    ], y=0.16, tamanho=13)
    salvar(pdf, fig)


def q3_caminho(pdf):
    fig = novo_slide("Q3 b) Caminho final encontrado pelo agente", "Questão 3 · Reforço")
    resumo = (R3 / "caminho_final.txt").read_text(encoding="utf-8").splitlines()[0]
    imagem(fig, R3 / "caminho_final.png", [0.01, 0.13, 0.98, 0.75])
    texto(fig, [f"{resumo}. Recua (BAIXO), espera (PARADA) e desvia: ninguém programou isso."],
          y=0.115, tamanho=13)
    salvar(pdf, fig)


def q3_caminho_evolucao(pdf):
    fig = novo_slide("Q3 b) A mesma partida jogada por épocas diferentes", "Questão 3 · Reforço")
    imagem(fig, R3 / "caminho_evolucao.png", [0.01, 0.3, 0.98, 0.56])
    texto(fig, [
        "Mesmos carros (mesma semente); só muda a tabela Q carregada de cada checkpoint.",
        "Início: sobe direto e é atropelada. Meio do treino: vai mais longe, mas cai na mesma armadilha. Final: atravessa.",
        "Relação com o fuzzy: os valores Q são graus contínuos de 'quão boa' é cada ação; o argmax é a decisão crisp.",
    ], y=0.27, tamanho=14)
    salvar(pdf, fig)


def conclusao(pdf):
    fig = novo_slide("Conclusões")
    texto(fig, [
        "Q1: três classificadores comparados por matriz de confusão, acurácia, precisão, recall, F1, F2 e ROC-AUC.",
        "-O SVM com cor + HOG foi o melhor no geral, mas com resultados modestos; a incerteza das previsões é alta.",
        "-A probabilidade é um grau de pertinência: os erros se concentram onde ela é ambígua.",
        "Q2: Gradient Boosting com R² ≈ 0,96 no teste, contra 0,76 da Regressão Linear.",
        "-O preço contínuo já é graduado; discretizar em faixas cria fronteiras artificiais.",
        "-Funções de pertinência expressam 'um pouco média e um pouco cara'.",
        "Q3: Q-learning aprendeu a atravessar em 100% das partidas de avaliação (contra 0% do aleatório).",
        "",
        "!Lógica nebulosa: classificação, regressão e reforço produzem GRAUS; a decisão crisp é só o último passo.",
        "!Manter o grau permite medir a confiança, ajustar limiares e responder 'não sei' quando for o caso.",
    ], tamanho=16)
    salvar(pdf, fig)


def como_executar(pdf):
    fig = novo_slide("Repositório e como executar")
    texto(fig, [
        f"Repositório: {REPOSITORIO or '(link do GitHub)'}",
        "pip install -r requirements.txt   (Python 3.10+)",
        "Q1: cd questao1_classificacao → python preparar_dados.py → python treinar.py → python analise_fuzzy.py",
        "-Interface: streamlit run app.py",
        "Q2: cd questao2_regressao → python gerar_dados.py → python regressao.py",
        "Q3: cd questao3_reforco_galinha → python avaliar.py | python visualizador.py (pygame)",
        "-Treinar do zero: python treinar.py --do-zero --episodios 20000 --checkpoint-cada 1000",
        "Testes: python -m pytest em cada pasta (69 testes no total).",
        "Esta apresentação: python gerar_apresentacao.py",
    ], tamanho=15)
    salvar(pdf, fig)


def main():
    slides = [capa, visao_geral, q1_dados, q1_modelos, q1_metricas, q1_matrizes, q1_fuzzy, q1_zona, q1_limiar,
              q2_dados, q2_metricas, q2_diagnostico, q2_continuo, q2_fuzzy,
              q3_ambiente, q3_agente, q3_evolucao, q3_caminho, q3_caminho_evolucao, conclusao, como_executar]
    with PdfPages(SAIDA) as pdf:
        for slide in slides:
            slide(pdf)
        info = pdf.infodict()
        info["Title"] = "Trabalho Final - Lógica Nebulosa"
        info["Author"] = ALUNO or "Aluno(a)"
    print(f"{len(slides)} slides gravados em {SAIDA}")


if __name__ == "__main__":
    main()
