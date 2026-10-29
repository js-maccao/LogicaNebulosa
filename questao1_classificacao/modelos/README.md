# modelos/

Gerada por `python treinar.py` (não versionada no Git; para recriar, rode `python preparar_dados.py` e depois `python treinar.py`):

- `knn.joblib`: Pipeline (StandardScaler + KNeighborsClassifier, k=5), treinado com treino + validação, sobre os pixels.
- `random_forest.joblib`: RandomForestClassifier (100 árvores, `random_state=42`), mesmo treino, sobre os pixels.
- `svm_atributos.joblib`: Pipeline (StandardScaler + SVC RBF com C=1, calibrado por `CalibratedClassifierCV`), mesmo treino, sobre os atributos de cor HSV + HOG.
- `info_modelos.json`: parâmetros, tamanho da imagem, semente e versões das bibliotecas usadas.

Os arquivos `.joblib` dependem da versão do scikit-learn (ver `info_modelos.json`). Em outra versão, rode `python treinar.py` de novo.
