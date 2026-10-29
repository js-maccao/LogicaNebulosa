# Trabalho Final: Lógica Nebulosa

**Universidade Carioca · Aprendizado de Máquina e Lógica Nebulosa · Prof. Mardônio França**

Enunciado: [`fromalizacaoTrabalho.pdf`](fromalizacaoTrabalho.pdf) · Apresentação: [`apresentacao.pdf`](apresentacao.pdf)

| Questão | Pasta | Problema | Modelos | Principal resultado |
|---|---|---|---|---|
| 1. Classificação | [`questao1_classificacao/`](questao1_classificacao/README.md) | Imagem: cachorro x não cachorro | KNN, Random Forest, SVM (cor + HOG) | SVM: acurácia 0,64 e ROC-AUC 0,73 no teste; probabilidade analisada como pertinência fuzzy |
| 2. Regressão | [`questao2_regressao/`](questao2_regressao/README.md) | Preço de passagens aéreas (dados sintéticos) | Regressão Linear, Gradient Boosting | Gradient Boosting: R² 0,957, MAE R$ 108 |
| 3. Aprendizado por reforço | [`questao3_reforco_galinha/`](questao3_reforco_galinha/README.md) | Galinha atravessando uma rua com carros (labirinto com obstáculos móveis) | Q-learning tabular (do zero) | 100% de travessias sem exploração, contra 0% do agente aleatório |

Cada pasta tem um `README.md` com o modelo, os resultados e a discussão pedida no enunciado, além de testes automáticos (`tests/`).

## Instalação

Requer **Python 3.10+** (testado com Python 3.14 no Windows).

```powershell
git clone <URL-deste-repositorio>
cd LogicaNebulosa
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Como executar cada questão

Os resultados (figuras e CSVs) já estão versionados. Os comandos abaixo recriam tudo do zero.

```powershell
# Questão 1: Classificação (imagens incluídas no repositório)
cd questao1_classificacao
python preparar_dados.py        # valida e divide as 1.000 imagens
python treinar.py               # treina KNN, Random Forest e SVM; métricas e matrizes   (~1 min)
python analise_fuzzy.py         # item c): probabilidade como pertinência fuzzy, ROC, limiares
streamlit run app.py            # interface para classificar uma imagem (opcional)
cd ..

# Questão 2: Regressão
cd questao2_regressao
python gerar_dados.py           # gera dados/passagens.csv (sintético, semente 42)
python regressao.py             # treina, avalia e gera os gráficos
cd ..

# Questão 3: Aprendizado por reforço
cd questao3_reforco_galinha
python avaliar.py               # avalia os checkpoints já treinados e gera as figuras
python visualizador.py          # galinha treinada jogando (janela pygame)
# treinar do zero (sobrescreve os checkpoints): python treinar.py --do-zero --episodios 20000 --checkpoint-cada 1000
cd ..

# Apresentação em PDF (lê os resultados das 3 questões)
python gerar_apresentacao.py
```

## Testes

```powershell
cd questao1_classificacao;   python -m pytest; cd ..    # 49 testes
cd questao2_regressao;       python -m pytest; cd ..    #  8 testes
cd questao3_reforco_galinha; python -m pytest; cd ..    # 12 testes
```

## Lógica nebulosa nas três questões

- **Q1:** a probabilidade `predict_proba` é um **grau de pertinência** ao conjunto "cachorro". O corte em 0,5 é uma defuzzificação. Os erros se concentram onde a pertinência é ambígua (zona "incerta").
- **Q2:** o preço previsto é **contínuo** e já carrega gradação. Funções de pertinência mostram que R$ 1.150 é "um pouco média e um pouco cara", enquanto faixas crisp criam fronteiras artificiais.
- **Q3:** os valores Q são graus contínuos de "quão boa" é cada ação; o `argmax` é a decisão crisp.

## Dados

- **Q1:** Stanford Dogs (cachorros) e Animals-10 (não cachorros), ~24 MB, incluídos em `questao1_classificacao/BaseCachorro` e `BaseNãoCachorros`. Fontes e licenças estão no README da questão.
- **Q2:** sintéticos, gerados por `gerar_dados.py` com uma regra documentada.
- **Q3:** não usa dados: o ambiente é simulado em `galinha_env.py`.

## Publicar no GitHub (entrega)

O repositório local já está criado e com o primeiro commit. Para publicar:

1. Crie um repositório **público** vazio em https://github.com/new (sem README).
2. No terminal, na pasta `C:\dev\LogicaNebulosa`:

   ```powershell
   git remote add origin https://github.com/<seu-usuario>/<nome-do-repo>.git
   git push -u origin main
   ```

3. Preencha `ALUNO` (e, se quiser, `TURMA` e `REPOSITORIO`) no topo de `gerar_apresentacao.py`, rode `python gerar_apresentacao.py`, faça commit e push.
4. Envie o link do repositório e o `apresentacao.pdf` para mardfranca@gmail.com.
