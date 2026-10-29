# resultados/

Gerada por `python preparar_dados.py`, `python treinar.py` e `python analise_fuzzy.py`. A pasta é versionada no Git para que os resultados possam ser vistos sem treinar.

| Arquivo | Conteúdo |
|---|---|
| `tabela_imagens.csv` | Uma linha por imagem válida: caminho (relativo às bases), rótulo, grupo, raça/espécie, hash |
| `divisao.csv` | A mesma tabela com a coluna `conjunto` (treino / validacao / teste) |
| `resumo_dados.json` | Contagens por classe, grupo e conjunto; corrompidas; duplicatas; verificações |
| `metricas_validacao.csv` | 5 métricas na **validação** (modelos treinados só com o treino) |
| `metricas_teste.csv` | 5 métricas no **teste final** (modelos treinados com treino + validação) |
| `validacao_cruzada.csv` | 5 folds em treino + validação (informação complementar) |
| `matrizes_teste.json`, `matriz_*.png` | Matrizes de confusão do teste final |
| `erros_por_grupo_teste.csv` | Erros de cada modelo por grupo (cachorro, cat, chicken, cow, horse, sheep) |
| `previsoes_teste.csv` | Previsão e P(cachorro) de cada modelo para cada imagem de teste |
| `comparacao_metricas_teste.png` | Gráfico de barras das métricas no teste |
| **Análise fuzzy (`analise_fuzzy.py`)** | |
| `pertinencia_fuzzy.png` | Funções de pertinência "não cachorro / incerto / cachorro" e onde caem as imagens de teste |
| `distribuicao_probabilidades.png` | Histograma de P(cachorro) por classe real, por modelo |
| `metricas_vs_limiar.png` | Métricas variando o limiar de decisão |
| `exemplos_pertinencia.png` | Imagens reais com os graus de pertinência |
| `curvas_roc.png`, `auc_teste.csv` | Curva ROC e ROC-AUC de cada modelo |
| `zona_incerta.csv` | Acertos e erros por termo linguístico dominante |
| `pertinencia_teste.csv` | Graus μ de cada imagem de teste, por modelo |
