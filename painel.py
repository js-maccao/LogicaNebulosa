"""Painel de controle:  streamlit run painel.py

Executa, por botões, os mesmos comandos que se rodariam no terminal e mostra o que cada
um gera (log ao vivo, tabelas e figuras). Só roda comandos de uma lista fixa (painel_exec.py).
"""
import json

import pandas as pd
import streamlit as st

import painel_exec as ex

st.set_page_config(page_title="Painel - Lógica Nebulosa", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------
# utilitários de exibição
# --------------------------------------------------------------------------
def tabela_csv(caminho, titulo=None, **kw):
    if caminho.exists():
        if titulo:
            st.subheader(titulo)
        st.dataframe(pd.read_csv(caminho, encoding="utf-8", **kw), hide_index=True)
    else:
        st.info(f"`{caminho.name}` ainda não existe. Execute o passo acima.")


def figuras(pasta, nomes, colunas=2):
    cols = st.columns(colunas)
    for i, nome in enumerate(nomes):
        arq = pasta / nome
        if arq.exists():
            cols[i % colunas].image(str(arq), caption=nome)
        else:
            cols[i % colunas].info(f"`{nome}` ainda não existe.")


def passo(titulo, explicacao, expandido=False):
    exp = st.expander(titulo, expanded=expandido)
    exp.write(explicacao)
    return exp


# --------------------------------------------------------------------------
def pagina_inicio():
    st.title("🧪 Painel de controle")
    st.write("Aqui você roda, **sem abrir o terminal**, tudo o que o trabalho faz: preparar dados, treinar, "
             "analisar e ver a galinha aprender. Cada passo mostra o log do comando e o que ele produziu.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### Q1 · Classificação\n`preparar_dados` → `treinar` → `analise_fuzzy`")
    c2.markdown("### Q2 · Regressão\n`gerar_dados` → `regressao`")
    c3.markdown("### Q3 · Reforço\ntreino ao vivo, galinha jogando e treino passo a passo")
    st.info("Ao lado de cada botão aparece o comando equivalente no terminal. As interfaces de resultados "
            "continuam disponíveis: `streamlit run app.py` dentro de `questao1_classificacao` ou "
            "`questao2_regressao` (portas diferentes: `--server.port 8502`).")
    st.warning("Um passo longo (como treinar a Q1, cerca de 1 min) mantém a página ocupada. "
               "Clicar em outra aba do menu interrompe a execução.")


# --------------------------------------------------------------------------
def pagina_q1():
    st.title("Q1 · Classificação (cachorro ou não)")
    R = ex.Q1 / "resultados"

    with passo("1. Preparar os dados", "Lê as pastas de imagens, valida (arquivo corrompido, duplicata por SHA-256) "
                                       "e divide em treino, validação e teste, estratificado por grupo.", True):
        if ex.botao_comando("q1_preparar"):
            st.rerun()
        arq = R / "resumo_dados.json"
        if arq.exists():
            resumo = json.loads(arq.read_text(encoding="utf-8"))
            tab = pd.DataFrame(resumo["por_conjunto"]).T["por_grupo"].apply(pd.Series).fillna(0).astype(int)
            tab["Total"] = tab.sum(axis=1)
            st.subheader("Imagens por conjunto e grupo")
            st.dataframe(tab)
            st.caption(f"Corrompidas ignoradas: {len(resumo['corrompidas_ignoradas'])} · "
                       f"duplicatas: {resumo['arquivos_em_duplicatas']} · semente {resumo['semente']}")

    with passo("2. Treinar KNN, Random Forest e SVM", "Transforma as imagens em números, treina os 3 modelos, "
                                                       "mede validação, validação cruzada e teste (cerca de 1 min)."):
        if ex.botao_comando("q1_treinar"):
            st.rerun()
        tabela_csv(R / "metricas_teste.csv", "Métricas no teste")
        figuras(R, ["comparacao_metricas_teste.png", "curvas_roc.png"])

    with passo("3. Análise fuzzy (probabilidade = pertinência)", "Lê as probabilidades do teste e calcula "
                                                                  "o acerto por zona (não cachorro / incerto / cachorro) e as métricas por limiar."):
        if ex.botao_comando("q1_fuzzy"):
            st.rerun()
        figuras(R, ["pertinencia_fuzzy.png", "metricas_vs_limiar.png", "exemplos_pertinencia.png"])
        tabela_csv(R / "zona_incerta.csv", "Acerto por zona")

    with passo("4. Classificar uma imagem (upload)", "Envie uma foto e veja a probabilidade de cada modelo treinado.", True):
        import painel_q1
        painel_q1.secao_classificar()

    with passo("5. Testes automáticos", "Roda os testes da Q1."):
        ex.botao_comando("q1_testes")


# --------------------------------------------------------------------------
def pagina_q2():
    st.title("Q2 · Regressão (preço de passagens)")
    R = ex.Q2 / "resultados"
    dados = ex.Q2 / "dados" / "passagens.csv"

    with passo("1. Gerar os dados sintéticos", "Cria as passagens a partir da regra do preço + ruído de 10%.", True):
        n = st.number_input("Quantidade de passagens (--n)", 500, 50000, 5000, step=500)
        if ex.botao_comando("q2_gerar", args_extra=("--n", str(int(n)))):
            st.rerun()
        if dados.exists():
            df = pd.read_csv(dados, encoding="utf-8")
            st.write(f"{len(df)} passagens · preço mediano R$ {df['preco'].median():,.0f}")
            st.dataframe(df.head(10), hide_index=True)
            st.bar_chart(df["preco"].round(-2).value_counts().sort_index())

    with passo("2. Treinar e avaliar", "Divide 80/20, treina Regressão Linear e Gradient Boosting, calcula "
                                       "MAE, MSE, RMSE e R², faixas crisp e pertinência fuzzy (cerca de 10 s)."):
        if not dados.exists():
            st.warning("Gere os dados primeiro (passo 1).")
        if ex.botao_comando("q2_regressao"):
            st.rerun()
        tabela_csv(R / "metricas.csv", "Métricas no teste")
        figuras(R, ["previsto_vs_real.png", "residuos.png", "importancia_variaveis.png",
                    "continuo_vs_discreto.png", "pertinencia_preco.png"])

    with passo("3. Prever o preço de um voo", "Monte um voo e veja o preço previsto pelos dois modelos e a "
                                              "pertinência fuzzy.", True):
        import painel_q2
        painel_q2.secao_prever()

    with passo("4. Testes automáticos", "Roda os testes da Q2."):
        ex.botao_comando("q2_testes")


# --------------------------------------------------------------------------
def pagina_q3():
    import painel_q3 as q3   # só aqui: coloca a pasta da Q3 no sys.path

    st.title("Q3 · A galinha atravessando a rua")
    aba1, aba2, aba3, aba4 = st.tabs(["🎓 Treinar ao vivo", "🐔 Ver a galinha jogar",
                                      "🔬 Passo a passo (Q-learning)", "📊 Avaliar checkpoints"])

    with aba1:
        st.write("Treina com o Q-learning do projeto e atualiza o gráfico enquanto aprende.")
        c1, c2, c3 = st.columns(3)
        episodios = c1.slider("Episódios", 500, 30000, 3000, step=500)
        cada = c2.select_slider("Checkpoint a cada", [500, 1000, 2500, 5000], value=1000)
        do_zero = c3.checkbox("Começar do zero", value=True)
        sobrescrever = st.checkbox("Gravar na pasta oficial `checkpoints/` (sobrescreve; faz backup antes)")
        destino = q3.treinar.PASTA_CHECKPOINTS if sobrescrever else q3.PASTA_PAINEL
        st.caption(f"Os checkpoints serão gravados em `{destino}`. "
                   + ("" if sobrescrever else "A pasta oficial não é alterada."))
        if st.button("▶ Treinar"):
            if sobrescrever:
                st.info(f"Backup criado em `{q3.fazer_backup_checkpoints()}`")
            metricas = st.empty()
            barra = st.progress(0.0)
            grafico = st.empty()
            linhas = []
            for info in q3.treinar_por_blocos(episodios, do_zero, destino, cada):
                linhas.append(info)
                barra.progress(min(1.0, len(linhas) * 250 / episodios))
                with metricas.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Episódio", info["episodio"])
                    m2.metric("% chegou (últimos 500)", f"{info['chegou']:.1f}%")
                    m3.metric("Epsilon", f"{info['epsilon']:.3f}")
                    m4.metric("Estados na tabela Q", info["estados"])
                grafico.line_chart(pd.DataFrame(linhas).set_index("episodio")[["chegou"]])
            barra.progress(1.0)
            st.success("Treino concluído. Veja o resultado na aba “Ver a galinha jogar”.")

    with aba2:
        cks = q3.listar_checkpoints()
        if not cks:
            st.info("Nenhum checkpoint encontrado. Treine primeiro.")
        else:
            nomes = {q3.rotulo_checkpoint(c): c for c in cks}
            padrao = next((i for i, n in enumerate(nomes) if n.endswith("ultimo.pkl")), 0)
            c1, c2, c3 = st.columns(3)
            escolha = c1.selectbox("Checkpoint", list(nomes), index=padrao)
            semente = c2.number_input("Semente (sorteia os carros)", 0, 99999, 1)
            vel = c3.slider("Velocidade (passos/s)", 1, 10, 3)
            q, passos, resultado, env = q3.gravar_replay(nomes[escolha], int(semente))
            i = st.slider("Passo", 0, len(passos) - 1, 0, key=f"passo_{escolha}_{semente}")
            tocar = st.button("▶ Tocar a partida")
            esq, dir_ = st.columns([1, 1])
            area_fig, area_q = esq.empty(), dir_.empty()

            def mostrar(k):
                atual = passos[k]
                area_fig.pyplot(q3.quadro(env, passos, k, f"passo {k}/{len(passos) - 1}"))
                import matplotlib.pyplot as plt
                plt.close("all")
                vals = q.get(atual["estado"])
                with area_q.container():
                    st.markdown(f"**Estado:** `{atual['estado']}`  \n(linha, coluna, carro à frente, carro chegando à frente, carro chegando em mim)")
                    if vals is not None:
                        st.bar_chart(pd.Series(vals, index=q3.galinha_env.NOMES_ACOES, name="valor Q"))
                    else:
                        st.warning("Estado nunca visto no treino: todos os valores Q são 0.")
                    if k + 1 < len(passos):
                        st.write(f"Ação escolhida: **{q3.galinha_env.NOMES_ACOES[passos[k + 1]['acao']]}**")
                    else:
                        st.write(f"Resultado final: **{resultado}**")

            if tocar:
                import time
                for k in range(len(passos)):
                    mostrar(k)
                    time.sleep(1 / vel)
            else:
                mostrar(i)
            st.caption(f"Resultado da partida: **{resultado}** em {len(passos) - 1} passos.")

    with aba3:
        st.write("Uma partida de treino em câmera lenta: a cada passo, a galinha **explora** (ação aleatória) ou "
                 "**usa o que sabe**, recebe a recompensa e corrige o valor Q. Nada é gravado.")
        cks = q3.listar_checkpoints()
        c1, c2, c3 = st.columns(3)
        origem = c1.selectbox("Tabela Q inicial", ["Zerada (galinha novata)"] + [q3.rotulo_checkpoint(c) for c in cks])
        epsilon = c2.slider("Epsilon (chance de explorar)", 0.0, 1.0, 0.5, step=0.05)
        semente = c3.number_input("Semente", 0, 99999, 7, key="sem_didatica")
        if origem.startswith("Zerada"):
            q0 = q3.treinar.nova_tabela_q()
        else:
            q0 = q3.treinar.carregar_checkpoint({q3.rotulo_checkpoint(c): c for c in cks}[origem])[0]
        env, passos, registros, resultado = q3.episodio_didatico(q0, epsilon, int(semente))
        k = st.slider("Passo", 1, len(registros), 1)
        esq, dir_ = st.columns(2)
        import matplotlib.pyplot as plt
        esq.pyplot(q3.quadro(env, passos, k, f"depois do passo {k}"))
        plt.close("all")
        r = registros[k - 1]
        with dir_:
            st.markdown("**Modo:** " + ("🎲 EXPLORANDO (ação aleatória)" if r["explorou"] else "🧠 USANDO O QUE SABE"))
            st.write(f"Estado: `{r['estado']}` · ação: **{r['acao']}** · recompensa: **{r['recompensa']}**")
            st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max Q(s',a') - Q(s,a)\,]")
            st.write(f"Q antes: `{r['Q_antes']:.3f}` → Q depois: `{r['Q_depois']:.3f}` "
                     f"(α = {q3.treinar.ALFA}, γ = {q3.treinar.GAMA})")
        st.caption(f"Resultado da partida: {resultado}. Recompensa total: {sum(x['recompensa'] for x in registros)}")
        st.dataframe(pd.DataFrame(registros), hide_index=True)

    with aba4:
        st.write("Joga as mesmas 1.000 partidas com cada checkpoint (sem exploração) e compara com o agente aleatório.")
        st.warning("Isto roda `avaliar.py` e regrava as figuras de `questao3_reforco_galinha/resultados/` "
                   "usando os checkpoints oficiais.")
        ex.botao_comando("q3_avaliar")
        R = ex.Q3 / "resultados"
        tabela_csv(R / "evolucao_checkpoints.csv", "Evolução dos checkpoints")
        figuras(R, ["curva_aprendizado.png", "evolucao_checkpoints.png", "caminho_final.png", "caminho_evolucao.png"])
        st.divider()
        st.write("Janela do pygame (a versão original do jogo, fora do navegador):")
        if st.button("Abrir visualizador pygame"):
            ex.abrir_janela("q3_pygame")
            st.success("Janela aberta. ESC fecha.")
        ex.botao_comando("q3_testes")


# --------------------------------------------------------------------------
def pagina_terminal():
    st.title("Terminal")
    st.write("Escolha um comando do projeto e veja a saída. É uma **lista fixa**, não aceita texto livre.")
    chave = st.selectbox("Comando", list(ex.COMANDOS), format_func=lambda k: ex.ROTULOS[k])
    pasta, args, ps = ex.COMANDOS[chave]
    st.code(ps, language="powershell")
    if st.button("▶ Executar", key="term_run"):
        area = st.empty()
        codigo, _, seg = ex.executar_processo(pasta, args, area)
        (st.success if codigo == 0 else st.error)(f"Código de saída {codigo} em {seg:.1f} s")


def pagina_reiniciar():
    import painel_reset as rs

    st.title("♻️ Voltar ao zero")
    st.write("Apaga tudo o que foi **gerado** (dados sintéticos, modelos, resultados e checkpoints) para você "
             "refazer cada estágio na prática. **Código, imagens-fonte, README e testes não são tocados.**")
    escolhidas = st.multiselect("O que zerar", list(rs.NOMES), default=list(rs.NOMES), format_func=rs.NOMES.get)
    res = rs.resumo(escolhidas)
    if res:
        st.dataframe(pd.DataFrame([{"Grupo": k, "Itens": n, "Tamanho (MB)": round(b / 1e6, 2)}
                                   for k, (n, b) in res.items()]), hide_index=True)
    else:
        st.success("Nada para apagar: já está no zero.")
    backup = st.checkbox("Guardar um backup (.zip) antes de apagar", value=True)
    st.caption("O backup fica em `backups_reset/` e pode ser restaurado abaixo.")
    confirma = st.text_input("Digite ZERAR para confirmar")
    if st.button("🗑 Zerar agora", disabled=not (res and confirma.strip() == "ZERAR")):
        destino, n = rs.zerar(escolhidas, backup)
        st.cache_resource.clear()
        st.success(f"{n} itens apagados." + (f" Backup: `{destino}`" if destino else ""))
        st.info("Ordem para refazer: Q1 (preparar → treinar → fuzzy), Q2 (gerar → regressão), Q3 (treinar ao vivo).")

    backups = rs.listar_backups()
    if backups:
        st.divider()
        st.subheader("Restaurar um backup")
        b = st.selectbox("Backup", backups, format_func=lambda x: x.name)
        if st.button("↩ Restaurar"):
            st.cache_resource.clear()
            st.success(f"{rs.restaurar(b)} arquivos restaurados.")


PAGINAS = {"Início": pagina_inicio, "Q1 · Classificação": pagina_q1, "Q2 · Regressão": pagina_q2,
           "Q3 · Galinha": pagina_q3, "Terminal": pagina_terminal, "Voltar ao zero": pagina_reiniciar}
pagina = st.sidebar.radio("Navegação", list(PAGINAS))
st.sidebar.caption("Executa o projeto sem o terminal")
PAGINAS[pagina]()
