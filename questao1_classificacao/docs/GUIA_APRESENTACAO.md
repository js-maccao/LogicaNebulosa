# Guia de apresentação: Questão 1 (6 a 9 minutos)

Os números citados vêm de `resultados/` (gerados por `treinar.py` e `analise_fuzzy.py`). Se retreinar, confira os valores novos.

## 1. Roteiro sugerido

1. **Problema (30 s)**: classificar uma imagem em cachorro (1) ou não cachorro (0). Positivo = cachorro.
2. **Dados (1 min)**: 500 cachorros (Stanford Dogs, 120 raças) e 500 não cachorros (Animals-10: gato, galinha, vaca, cavalo, ovelha). O rótulo vem da pasta. Sem duplicatas e sem arquivos corrompidos. Admita logo a limitação: só 5 espécies e duas bases diferentes.
3. **Imagem → números (1 min)**: duas representações.
   - `preparar_imagem()`: cinza, 64×64, 4096 números (KNN e RF).
   - `extrair_atributos()`: histograma de cor HSV + HOG (direção das bordas), 1892 números (SVM).
   - Mostre a seção "O que o modelo enxerga" na página Classificar.
4. **Divisão (45 s)**: 640 treino / 160 validação / 200 teste, estratificado por grupo, semente 42. O teste só é usado no final.
5. **Três modelos (1 min)**: KNN (os vizinhos mais parecidos votam), Random Forest (100 árvores votam) e SVM (fronteira de margem máxima sobre cor + forma, com probabilidade calibrada).
6. **Resultados (1,5 min)**: tabela do teste, matrizes e erros por espécie. Seja franco:
   - desempenho modesto (SVM 0,64, RF 0,62, KNN 0,54 de acurácia);
   - o SVM vence com folga na ROC-AUC (0,73) e na validação cruzada (0,68), mas no teste a diferença em acurácia para o RF não é conclusiva com 200 imagens.
7. **Item c) Fuzzy (2 min)**: `pertinencia_fuzzy.png`, depois a tabela da zona incerta, depois `exemplos_pertinencia.png`, depois `metricas_vs_limiar.png`. Frase-chave: *"P(cachorro) é um grau de pertinência; o corte em 0,5 é uma defuzzificação que joga fora a incerteza."*
8. **Demonstração (1 min)**: enviar uma imagem na página 2 e comentar as probabilidades nos três modelos.
9. **Limitações (30 s)**: viés entre bases, poucas espécies, representações simples.

## 2. Perguntas prováveis e respostas curtas

### Dados e preparação

**Por que converter a imagem em números?** Os algoritmos trabalham com tabelas numéricas. Cada pixel (ou cada atributo) vira uma coluna.

**Por que 64×64 e cinza?** Reduz o custo e deixa todas as imagens com o mesmo número de colunas. Em troca, perde cor e detalhe. Essa limitação motivou o 3º modelo.

**O que é o HOG?** *Histogram of Oriented Gradients*. Para cada pixel, calcula-se a diferença de brilho com os vizinhos (o gradiente): a força e a direção da borda. A imagem é dividida em células de 16×16 e cada célula vira um histograma de 9 direções. Blocos de 2×2 células são normalizados, o que torna o resultado independente do brilho. Ele capta contornos (orelhas, focinho, patas) e textura (pelo). Está implementado em `pre_processamento.hog()`, só com NumPy.

**E o histograma HSV?** Conta quantos pixels há de cada combinação matiz/saturação/brilho (8×4×4 = 128 caixas). Diz *quais* cores existem, não *onde* estão; por isso não muda se o animal se mexer na foto. Pasto verde e céu azul, por exemplo, são típicos de vaca e ovelha.

**O que são X e y?** X: matriz de atributos (1 linha por imagem). y: respostas corretas (0 ou 1).

**Treino, validação e teste?** Treino: o modelo aprende. Validação: serve para comparar e decidir durante o desenvolvimento (por exemplo, C=1 do SVM). Teste: é a nota final, usada uma vez, com dados que nunca influenciaram nenhuma decisão.

**Por que treinar o modelo final com treino + validação?** Mais dados (800 em vez de 640) sem encostar no teste. As decisões já tinham sido tomadas.

**Por que estratificar por grupo?** Para que as 6 categorias (cachorro e 5 espécies) apareçam na mesma proporção nos três conjuntos.

**Para que serve `random_state=42`?** Torna o sorteio reprodutível. 42 é só uma convenção.

### Modelos

**Como o KNN decide?** Acha as 5 imagens de treino mais próximas (distância euclidiana) e faz uma votação. A probabilidade é a fração de votos, então só pode valer 0%, 20%, 40%...

**Por que `StandardScaler`?** KNN e SVM dependem de distância; padronizar dá peso comparável a cada coluna. Ele fica dentro do `Pipeline` para ser ajustado só com o treino (evita vazamento).

**Como o Random Forest decide?** Cada árvore faz perguntas sobre pixels e é treinada em uma amostra diferente; a probabilidade é a média das 100 árvores. Não precisa de padronização.

**Como o SVM decide?** Procura a fronteira que separa as classes com a maior margem. O kernel RBF permite fronteiras curvas, e C controla o quanto a fronteira se dobra para acertar o treino.

**O SVM dá probabilidade?** Não diretamente: ele dá a distância até a fronteira. O `CalibratedClassifierCV` aprende, com validação cruzada interna sobre o treino, uma sigmoide que converte essa distância em probabilidade (calibração de Platt).

**Diferença entre classe prevista e probabilidade?** A classe é a decisão final (`predict`). A probabilidade (`predict_proba`) é a estimativa do modelo para cada classe. A classe prevista é a de maior probabilidade, o que não garante acerto.

### Métricas

**Precisão, recall, F1, F2?** Ver a seção 5 do README. O F2 pesa mais o recall: é útil quando perder um cachorro é pior do que um alarme falso.

**E a ROC-AUC?** É a chance de um cachorro qualquer receber P(cachorro) maior que um não cachorro qualquer: 0,5 = acaso, 1 = perfeito. Ela avalia a probabilidade em todos os limiares de uma vez. Por isso diferencia mais os modelos do que a acurácia no limiar 0,5.

**Qual modelo é o melhor?** Depende do critério:
- O **SVM** tem a maior acurácia, precisão e ROC-AUC, e também foi o melhor na validação e na validação cruzada.
- O **RF** tem o maior F1/F2 no limiar 0,5.
- O **KNN** tem o maior recall, mas muitos falsos positivos.

Com 200 imagens, diferenças de poucos pontos em acurácia podem ser ruído (±7 pontos).

**Por que os resultados são modestos?** Representações simples, só 640 imagens de treino, fundo e iluminação muito variados, possível viés entre as duas bases e só 5 espécies de não cachorro.

### Item c): lógica fuzzy

**Onde está a probabilidade de cada predição?** Em `treinar.py`, `avaliar.probabilidade_de_cachorro()` chama `predict_proba` e grava a coluna `proba_cachorro_<modelo>` em `resultados/previsoes_teste.csv`. Na interface, `modelos.probabilidades()` faz o mesmo para a imagem enviada.

**Qual é o aspecto fuzzy?**
- Na lógica clássica, a pertinência a "cachorro" é 0 ou 1. A probabilidade do modelo é um **grau de pertinência** entre 0 e 1: μ_cachorro = P(cachorro).
- Uma imagem pode ser "0,7 cachorro e 0,3 não cachorro".
- O `predict` (corte em 0,5) é uma **defuzzificação**: converte o grau numa resposta crisp.

**O que são os termos "não cachorro / incerto / cachorro"?** São três conjuntos fuzzy definidos sobre o eixo P(cachorro), com funções trapezoidais e triangulares (`analise_fuzzy.pertinencias()`). Eles formam uma partição: as pertinências somam 1. "Incerto" tem pico em 0,5.

**O que a tabela da zona incerta mostra?** No SVM, as imagens com pertinência clara têm 76% a 80% de acerto. As da zona incerta têm 53%, quase cara ou coroa. **O grau de pertinência mede o quanto confiar na resposta.** Um sistema fuzzy poderia responder "não sei, revise manualmente" nesses casos.

**Por que o limiar 0,5 é arbitrário?** `metricas_vs_limiar.png` mostra que mover o corte troca precisão por recall. Se perder um cachorro for grave, baixe o limiar. A fronteira crisp é uma escolha, não uma propriedade dos dados.

**Probabilidade e pertinência fuzzy são a mesma coisa?** Não exatamente. Probabilidade mede a chance de um evento incerto (a imagem *é* ou *não é* cachorro, e não sabemos qual); pertinência mede o grau com que algo pertence a um conceito vago. Aqui usamos a probabilidade como grau de pertinência porque ambas expressam uma transição gradual entre as classes, e é essa leitura que o enunciado pede.

**Por que o KNN tem uma pertinência "em degraus"?** Com k=5, só há 6 valores possíveis (0, 0,2, …, 1). RF e SVM produzem graus mais contínuos.

### Outras

**Isso funcionaria com um lobo?** Não há garantia: o modelo nunca viu lobos.

**Como evitou vazamento de dados?** A divisão foi feita uma vez e verificada sem sobreposição; o escalonador é ajustado só no treino; o teste foi usado uma vez; não há duplicatas exatas entre conjuntos; o C do SVM foi escolhido na validação.

**Por que `classes_`?** `predict_proba` devolve as colunas na ordem de `modelo.classes_`; o código procura a classe pelo valor em vez de supor a posição.

## 3. Mapa do código

| Arquivo | Função | Responsabilidade |
|---|---|---|
| `configuracao.py` | (constantes) | caminhos, 64×64, semente 42, k=5, 100 árvores, C=1, parâmetros do HOG |
| `pre_processamento.py` | `preparar_imagem` | imagem → vetor 4096 (EXIF, cinza, 64×64, 0..1) |
| `pre_processamento.py` | `extrair_atributos`, `histograma_hsv`, `hog` | imagem → 128 de cor + 1764 de HOG |
| `preparar_dados.py` | `montar_tabela` | percorre as pastas, define rótulo/grupo/raça |
| `preparar_dados.py` | `validar_arquivos`, `achar_duplicatas` | corrompidas e duplicatas (SHA-256) |
| `preparar_dados.py` | `dividir`, `verificar_divisao` | 640/160/200 estratificado; confere sobreposição |
| `preparar_dados.py` | `montar_X_y(df, entrada)` | cria X (pixels ou atributos) e y |
| `modelos.py` | `criar_knn`, `criar_random_forest`, `criar_svm_atributos` | define os modelos; `ENTRADA_MODELO` diz a entrada de cada um |
| `modelos.py` | `salvar_modelo`, `carregar_modelo` | `joblib.dump` / `joblib.load` |
| `modelos.py` | `probabilidades` | `predict_proba` mapeado por `classes_` |
| `treinar.py` | `main` | fluxo completo (treino → validação → CV → final → teste) |
| `treinar.py` | `validacao_cruzada` | 5 folds estratificados, os mesmos para os 3 modelos |
| `avaliar.py` | `calcular_metricas` | as 5 métricas do `sklearn.metrics` |
| `avaliar.py` | `matriz_confusao`, `desenhar_matriz` | `confusion_matrix` e gráfico |
| `avaliar.py` | `erros_por_grupo` | erros por espécie |
| `analise_fuzzy.py` | `pertinencias`, `termo_dominante` | funções de pertinência fuzzy sobre P(cachorro) |
| `analise_fuzzy.py` | `tabela_zonas`, `metricas_por_limiar`, `tabela_auc` | zona incerta, limiares, ROC-AUC |
| `app.py` | `pagina_*`, `entrada_do_modelo` | as 5 páginas; `pagina_classificar` faz a previsão do upload |

## 4. Funções do scikit-learn

- `fit(X, y)`: o modelo aprende com os exemplos (o KNN guarda os exemplos, a floresta constrói as árvores, o SVM encontra os vetores de suporte).
- `predict(X)`: classe prevista de cada linha (corte em 0,5).
- `predict_proba(X)`: probabilidades, uma coluna por classe, na ordem de `classes_`.
- `CalibratedClassifierCV(SVC(), ensemble=False)`: transforma a distância do SVM em probabilidade.
- `confusion_matrix(y_real, y_prev, labels=[0, 1])`: conta VN, FP, FN, VP.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score(beta=2)`: métricas. `pos_label=1` define cachorro como positivo.
- `roc_auc_score`, `roc_curve`: curva ROC e área sob ela.
- `train_test_split(..., stratify=..., random_state=42)`: divisão estratificada e reprodutível.
