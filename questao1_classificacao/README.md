# Questão 1: Cachorro ou não cachorro? KNN vs Random Forest vs SVM

Trabalho final de Lógica Nebulosa (Universidade Carioca, Prof. Mardônio França). **Questão 1: Classificação.**

O projeto classifica imagens em **Cachorro** (classe positiva, rótulo 1) ou **Não cachorro** (rótulo 0). São comparados **três modelos** com matriz de confusão, acurácia, precisão, recall, F1, F2 e ROC-AUC, e a probabilidade de cada previsão é analisada como **grau de pertinência fuzzy**. Há também uma interface em Streamlit para classificar uma imagem nova.

## 1. Como executar (Windows, PowerShell)

```powershell
cd questao1_classificacao
pip install -r ..\requirements.txt      # ou: pip install -r requirements.txt

python preparar_dados.py     # lê as pastas, valida, divide -> resultados\divisao.csv
python treinar.py            # treina os 3 modelos, avalia e salva modelos\ e resultados\   (~1 min)
python analise_fuzzy.py      # item c): probabilidades como pertinência fuzzy, ROC, limiares
streamlit run app.py         # interface (classificar imagem, comparar modelos, matrizes...)
python -m pytest             # 49 testes automáticos
```

Os resultados já estão versionados em `resultados/`. Os modelos (`modelos/*.joblib`) não são versionados; para usar a interface, rode `treinar.py` uma vez.

## 2. Dados

As imagens estão **dentro do projeto** (os caminhos ficam em `configuracao.py`):

| Pasta | Rótulo | Conteúdo | Fonte |
|---|---|---|---|
| `BaseCachorro\` | 1 | 500 imagens `.jpg` em 120 subpastas de raça | *Stanford Dogs Dataset* (Khosla et al., 2011), derivado do ImageNet; uso para pesquisa e educação |
| `BaseNãoCachorros\` | 0 | 500 imagens `.jpeg`: 100 de cada espécie (gato, galinha, vaca, cavalo, ovelha) | *Animals-10* da Rapidata (Hugging Face: `Rapidata/Animals-10`), exportadas por `BaseNãoCachorros\exportar_nao_cachorros.py`. **Licença não verificada; consulte a página do dataset.** |

- O rótulo vem da **pasta de origem**. A raça/espécie é guardada só como metadado (para análise de erros e estratificação), **nunca como atributo**.
- Verificações de `preparar_dados.py`: 1000 imagens válidas, 0 corrompidas, 0 duplicatas exatas (SHA-256), seja dentro de uma classe ou entre classes.
- Divisão (semente 42, estratificada por **grupo**): treino 640, validação 160, teste 200, sem imagens repetidas entre os conjuntos.

| Grupo | Treino | Validação | Teste |
|---|---|---|---|
| cachorro | 320 | 80 | 100 |
| gato, galinha, vaca, cavalo, ovelha (cada) | 64 | 16 | 20 |

## 3. Como a imagem vira números (`pre_processamento.py`)

Os algoritmos só entendem tabelas de números. Há **duas representações**:

1. **Pixels** (`preparar_imagem`, usada por KNN e Random Forest): corrige a orientação EXIF → trata transparência/CMYK → converte para escala de cinza → redimensiona para 64×64 → gera um vetor de **4096** valores entre 0 e 1.
2. **Atributos de cor e forma** (`extrair_atributos`, usada pelo SVM): a imagem colorida em 128×128 vira **1892** valores:
   - **histograma de cor HSV** (8 matizes × 4 saturações × 4 brilhos = 128 caixas): *quais* cores aparecem, não *onde*;
   - **HOG** (*histogram of oriented gradients*), implementado em NumPy: em cada célula de 16×16 pixels, para que direção apontam as bordas (9 direções). Blocos de 2×2 células são normalizados, então o resultado não muda com o brilho da foto. Ao todo, 7×7 blocos × 36 = 1764 valores.

Os pixels crus mudam quando o animal se desloca na foto; cor e bordas são mais estáveis. Por isso o 3º modelo foi acrescentado.

## 4. Modelos (`modelos.py`)

| Modelo | Entrada | Como decide | De onde vem a probabilidade |
|---|---|---|---|
| **KNN** (k=5) + StandardScaler | pixels | acha as 5 imagens de treino mais próximas e vota | fração de votos: só 0, 0,2, …, 1 |
| **Random Forest** (100 árvores) | pixels | cada árvore faz perguntas do tipo "pixel 1234 > 0,4?" | média das árvores |
| **SVM** RBF (C=1) + StandardScaler | cor + HOG | fronteira com a maior margem entre as classes | distância à fronteira **calibrada** (sigmoide de Platt, `CalibratedClassifierCV`) |

k=5 e 100 árvores são os valores iniciais; C=1 foi escolhido na **validação** (entre 1, 3 e 10). Nenhuma decisão usou o teste.

## 5. Métricas (classe positiva = Cachorro)

- **Acurácia**: fração de acertos.
- **Precisão**: dos que o modelo chamou de cachorro, quantos eram cachorro.
- **Recall**: dos cachorros reais, quantos o modelo encontrou.
- **F1**: média harmônica de precisão e recall.
- **F2**: como o F1, mas o recall pesa 4× mais (β=2); serve quando perder um cachorro (FN) é pior que um alarme falso (FP).
- **ROC-AUC**: probabilidade de uma imagem de cachorro receber P(cachorro) maior que uma de não cachorro. Usa a probabilidade inteira, não depende do limiar 0,5.

Matriz de confusão: linhas = classe real, colunas = classe prevista.

| | Previsto: Não cachorro | Previsto: Cachorro |
|---|---|---|
| **Real: Não cachorro** | VN | **FP** (alarme falso) |
| **Real: Cachorro** | **FN** (cachorro perdido) | VP |

## 6. Resultados

**Validação** (160 imagens; modelos treinados só com o treino)

| Modelo | Acurácia | Precisão | Recall | F1 | F2 |
|---|---|---|---|---|---|
| KNN | 0,4750 | 0,4815 | 0,6500 | 0,5532 | 0,6075 |
| Random Forest | 0,6000 | 0,5976 | 0,6125 | 0,6049 | 0,6095 |
| SVM (cor + forma) | **0,6938** | **0,6962** | **0,6875** | **0,6918** | **0,6892** |

**Validação cruzada** (5 folds em treino + validação), acurácia média: KNN 0,554 (±0,020), Random Forest 0,625 (±0,021), SVM 0,681 (±0,018).

**Teste final** (200 imagens; modelos treinados com treino + validação; avaliados uma única vez)

| Modelo | Acurácia | Precisão | Recall | F1 | F2 | ROC-AUC |
|---|---|---|---|---|---|---|
| KNN | 0,5400 | 0,5328 | **0,6500** | 0,5856 | 0,6226 | 0,5700 |
| Random Forest | 0,6200 | 0,6176 | 0,6300 | **0,6238** | **0,6275** | 0,6534 |
| SVM (cor + forma) | **0,6400** | **0,6591** | 0,5800 | 0,6170 | 0,5943 | **0,7297** |

| Modelo | VP | VN | FP | FN |
|---|---|---|---|---|
| KNN | 65 | 43 | 57 | 35 |
| Random Forest | 63 | 61 | 39 | 37 |
| SVM (cor + forma) | 58 | 70 | 30 | 42 |

**Erros por grupo no teste** (cada grupo de não cachorro tem 20 imagens; cachorro tem 100)

| Grupo | KNN | Random Forest | SVM |
|---|---|---|---|
| cachorro | 35 | 37 | 42 |
| gato | 12 | 8 | 10 |
| galinha | 12 | 9 | 7 |
| vaca | 12 | 3 | 2 |
| cavalo | 11 | 13 | 9 |
| ovelha | 10 | 6 | 2 |

### Leitura dos resultados

- Os três modelos ficam acima do acaso (50%), mas longe de um classificador profissional: as representações são simples e a base é pequena.
- O **SVM com cor + HOG** foi o melhor na validação, na validação cruzada, na acurácia de teste e, principalmente, na **ROC-AUC** (0,73 contra 0,65 e 0,57). Isso indica que as probabilidades dele *ordenam* melhor as imagens.
- **No teste, porém, a vantagem em acurácia é pequena** (0,64 contra 0,62 do RF). Com 200 imagens, a margem de erro de uma acurácia é de cerca de ±7 pontos (95%), então essa diferença isolada não é conclusiva. A AUC, a validação e a validação cruzada apontam na mesma direção.
- Cada modelo erra de um jeito diferente. O KNN chama muitos animais de cachorro (57 FP; maior recall). O SVM é o mais "conservador" (menos FP, maior precisão, mas perde mais cachorros: 42 FN). Se perder um cachorro for o erro mais grave (F2), o RF no limiar 0,5 é a melhor escolha. A figura `metricas_vs_limiar.png` mostra que baixar o limiar do SVM aumenta o recall dele.
- Vaca e ovelha foram fáceis para o SVM (cores de pasto/céu, formas grandes); gato e cavalo foram os mais confundidos com cachorro.

## 7. Item c) Probabilidade da predição e lógica fuzzy (`analise_fuzzy.py`)

**Onde está a probabilidade:** `modelo.predict_proba(X)` devolve P(não cachorro) e P(cachorro) para cada imagem (em `treinar.py`, gravadas em `resultados/previsoes_teste.csv`). `modelo.predict(X)` é apenas o corte: "cachorro se P(cachorro) ≥ 0,5".

**O aspecto fuzzy:** na lógica clássica (*crisp*), uma imagem **é** ou **não é** cachorro (pertinência 0 ou 1). A probabilidade do modelo, no entanto, é um **grau de pertinência** ao conjunto "cachorro":

> μ_cachorro(imagem) = P(cachorro), μ_não_cachorro(imagem) = 1 − P(cachorro)

Uma imagem com P = 0,74 é "bastante cachorro"; com P = 0,52 é quase totalmente ambígua, ainda que a saída crisp diga "cachorro" nos dois casos. O `predict` com corte em 0,5 é uma **defuzzificação**: ele transforma um grau contínuo em uma resposta sim/não e descarta a incerteza.

Para tornar isso explícito, `analise_fuzzy.py` define três **termos linguísticos** sobre P(cachorro). Eles formam uma partição fuzzy: em qualquer ponto, as três pertinências somam 1.

| Termo | Função de pertinência |
|---|---|
| "não cachorro" | 1 até 0,25, cai linearmente até 0 em 0,50 |
| "incerto" | triângulo 0,25 → 1 em 0,50 → 0 em 0,75 |
| "cachorro" | 0 até 0,50, sobe linearmente até 1 em 0,75 |

**Resultado** (`zona_incerta.csv`): acerto da decisão crisp conforme o termo dominante de cada imagem de teste.

| Modelo | "não cachorro" | "incerto" | "cachorro" |
|---|---|---|---|
| KNN | 60,7% (28 imgs) | 50,0% (108) | 57,8% (64) |
| Random Forest | 65,7% (35) | 59,4% (133) | 68,8% (32) |
| SVM (cor + forma) | **80,4%** (51) | 53,4% (116) | **75,8%** (33) |

- No SVM, quando a pertinência é clara o modelo acerta 76% a 80% das vezes; na zona **incerta**, acerta 53%, o que é praticamente um cara ou coroa. **O grau de pertinência informa o quanto confiar em cada resposta**, e a saída crisp joga essa informação fora.
- A maioria das imagens (58% no SVM) cai na zona incerta, o que é coerente com um modelo de 64% de acurácia. As probabilidades calibradas do SVM ficam entre 0,18 e 0,74: o modelo "sabe que não sabe".
- O KNN só produz 6 valores de probabilidade (0, 0,2, …, 1), uma pertinência "em degraus". RF e SVM produzem graus mais contínuos (54 e 200 valores distintos no teste).
- `metricas_vs_limiar.png` mostra que o limiar 0,5 é uma escolha arbitrária: mover o corte troca precisão por recall. Em vez de um sistema crisp, um sistema fuzzy poderia responder "cachorro", "não cachorro" ou "**não sei, revisar manualmente**".
- A própria categoria é difusa: alguns gatos e cachorros pequenos se parecem mais entre si do que um chihuahua se parece com um são-bernardo. As fronteiras do conceito "cachorro" no espaço de atributos não são nítidas.

Figuras: `pertinencia_fuzzy.png`, `distribuicao_probabilidades.png`, `exemplos_pertinencia.png` (imagens reais com μ), `metricas_vs_limiar.png`, `curvas_roc.png`.

## 8. Limitações

- "Não cachorro" tem só 5 espécies. Lobos, raposas, pessoas, objetos e paisagens nunca foram vistos.
- As classes vêm de **bases diferentes** (Stanford Dogs e Animals-10), com resolução e estilo de foto diferentes. O modelo pode aprender a base em vez do animal (viés de fonte).
- Pixels cinza e HOG/HSV são representações simples. Redes convolucionais pré-treinadas passariam de 95%, mas exigem bibliotecas e pesos externos.
- Poucos dados (1000 imagens); os resultados variam com a divisão.

## 9. Estrutura

```
questao1_classificacao/
├── configuracao.py        caminhos, tamanhos, semente, hiperparâmetros
├── pre_processamento.py   imagem -> pixels (4096) | cor HSV + HOG (1892)
├── preparar_dados.py      lê pastas, valida, detecta duplicatas, divide, monta X e y
├── modelos.py             cria, salva e carrega KNN, Random Forest e SVM
├── treinar.py             treino, validação, CV, modelo final, teste
├── avaliar.py             métricas, matrizes de confusão, erros por grupo
├── analise_fuzzy.py       item c): pertinência fuzzy, zona incerta, limiares, ROC
├── app.py                 interface Streamlit (5 páginas)
├── tests/                 pytest (49 testes)
├── docs/GUIA_APRESENTACAO.md
├── BaseCachorro/  BaseNãoCachorros/   imagens
├── modelos/ (gerado)  resultados/ (gerado, versionado)
```

## 10. Referências

- Khosla, A.; Jayadevaprakash, N.; Yao, B.; Fei-Fei, L. *Novel dataset for Fine-Grained Image Categorization: Stanford Dogs.* CVPR Workshop FGVC, 2011. http://vision.stanford.edu/aditya86/ImageNetDogs/
- Rapidata. *Animals-10* (Hugging Face): https://huggingface.co/datasets/Rapidata/Animals-10
- Dalal, N.; Triggs, B. *Histograms of Oriented Gradients for Human Detection.* CVPR, 2005.
- Platt, J. *Probabilistic Outputs for Support Vector Machines.* 1999.
- Zadeh, L. A. *Fuzzy sets.* Information and Control, 8(3), 1965.
- Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
