# Guia de estudo: Galinha atravessando a rua com Q-learning

> **Como usar este arquivo no NotebookLM:** cole o conteúdo inteiro como uma fonte. Depois peça coisas como:
> - "Faça um quiz de 10 perguntas sobre este material, uma por vez, e corrija minhas respostas."
> - "Aja como um professor exigente e me faça perguntas sobre o projeto, uma de cada vez."
> - "Me explique a equação do Q-learning como se eu tivesse 12 anos."
> - "Crie um roteiro de 5 minutos de apresentação a partir deste material."
> - "Quais são as 5 perguntas mais difíceis que o professor poderia fazer e como respondê-las?"
>
> Se quiser, adicione também os arquivos `galinha_env.py`, `treinar.py`, `visual.py` e `visualizador.py` como fontes extras.

---

## 1. O projeto em 30 segundos

Uma galinha precisa atravessar uma rua de **9 pistas** com carros em movimento. Ninguém ensina a ela as regras nem como desviar. Ela joga milhares de partidas, **erra, recebe notas (recompensas) e vai corrigindo uma tabela de números** que guarda o que aprendeu. No fim, atravessa a rua de forma consistente.

A técnica se chama **Q-learning**, um tipo de **aprendizado por reforço**. Não usa redes neurais nem bibliotecas de IA: só Python, `numpy` (para os vetores), `matplotlib` (gráfico) e `pygame` (visual).

**A frase central para a apresentação:**
> "A galinha não recebe instruções. Ela recebe notas. Por tentativa e erro, ela descobre sozinha quais ações rendem mais pontos em cada situação, e guarda isso numa tabela."

---

## 2. Glossário (os termos que o professor pode usar)

| Termo | Significado simples | Onde está no código |
|---|---|---|
| **Aprendizado por reforço** | Aprender por tentativa e erro, guiado por notas, sem exemplos prontos | O projeto todo |
| **Agente** | Quem aprende e decide (a galinha) | `treinar.py` |
| **Ambiente** | O mundo onde o agente age (rua, carros, regras) | `galinha_env.py` |
| **Estado** | A "foto" da situação que o agente usa para decidir | `estado()` em `galinha_env.py` |
| **Ação** | O que o agente pode fazer (5 opções) | `NOMES_ACOES` |
| **Recompensa** | A nota que o ambiente dá após cada ação | constantes `R_...` |
| **Episódio** | Uma partida completa, do início até acabar | `rodar_episodio()` |
| **Tabela Q** | A memória do agente: valor de cada ação em cada estado | `nova_tabela_q()` |
| **Valor Q** | Quantos pontos o agente espera somar, a partir daqui, se fizer aquela ação naquele estado | números na tabela |
| **Epsilon (ε)** | Chance de agir ao acaso para explorar | `escolher_acao()` |
| **Alfa (α)** | Taxa de aprendizado: o quanto cada experiência altera o que já sabia | `ALFA = 0.1` |
| **Gama (γ)** | Desconto: o quanto o futuro importa | `GAMA = 0.95` |
| **Exploração vs. aproveitamento** | Tentar algo novo vs. usar o que já sabe que funciona | epsilon-greedy |
| **Checkpoint** | "Save game" do treino | `salvar_checkpoint()` |
| **Política** | A regra de decisão do agente. Aqui: "em cada estado, escolha a ação de maior valor Q" | `np.argmax` |

---

## 3. O jogo (o ambiente)

Arquivo: **`galinha_env.py`**, classe `TravessiaGalinha`. **Nenhum aprendizado acontece aqui.** É só o mundo e suas regras.

### O mapa
- Grade de **11 linhas × 8 colunas**.
- Linha 0 = calçada de início. Linha 10 = chegada. Linhas 1 a 9 = **9 pistas** com carros.
- A galinha começa na linha 0, coluna 4 (no meio).

```
linha 10  chegada (grama)
linha 9   ┐
  ...     │ 9 pistas de rua
linha 1   ┘
linha 0   início (calçada)
```

### Os carros
- Cada pista tem **2 carros**. Cada carro é só um número: **a coluna** onde ele está.
- Pistas **pares** vão para a **direita** (+1); **ímpares** para a **esquerda** (−1). No visual: carros quentes (vermelho/laranja/amarelo) vão para a direita; frios (azul/verde-água) para a esquerda.
- Quando um carro sai por um lado, **reaparece do outro** (*wrap-around*), com o operador `%`.
- Em cada **reset** (partida nova), as posições dos carros são **sorteadas de novo**. Isso é importante: a galinha **não decora um mapa**, ela aprende uma regra geral.

### As 5 ações
Cada ação é só um número: `CIMA=0, BAIXO=1, ESQUERDA=2, DIREITA=3, PARADA=4`. A IA não "entende" movimento: ela escolhe um número e o ambiente traduz.

### Um turno (`passo()`) acontece nesta ordem
1. A galinha se move (travada dentro do mapa).
2. Se ela caiu em cima de um carro → **atropelada**.
3. Se chegou na última linha → **chegou**.
4. Senão, **os carros andam**. Se algum caiu em cima dela → **atropelada**.

Checar duas vezes (passos 2 e 4) evita um bug: sem isso, um carro poderia "atravessar" a galinha sem pegar nela.

Também existe um **limite de 100 passos** por partida (resultado `"tempo"`) para evitar partidas infinitas.

### O jogo é por turnos
A cada `passo()`, galinha e carros andam **uma casa**. Isso simplifica muito o aprendizado. A animação suave que se vê na tela é só um efeito visual (ver seção 12).

---

## 4. Recompensas: como o jogo "fala" com a IA

O ambiente devolve uma nota a cada passo. É a **única** forma de comunicar o que é bom ou ruim.

| Situação | Nota | Por quê |
|---|---|---|
| Cada passo dado | **−1** | Cria pressa. Sem isso, enrolar não custaria nada. |
| Chegar numa linha mais alta que todas as anteriores | **+5** | Incentiva avançar. Só vale em linha **nova**, senão ela ganharia pontos subindo e descendo. |
| Ser atropelada | **−50** | Evitar carros. Fim da partida. |
| Chegar do outro lado | **+100** | O objetivo. Fim da partida. |

```python
R_PASSO = -1
R_AVANCO = 5
R_ATROPELADA = -50
R_OBJETIVO = 100
```

**Ponto-chave:** ninguém diz "desvie dos carros". A IA só quer **somar o máximo de pontos**. Desviar de carros e atravessar a rua é uma **consequência** de como as notas foram desenhadas.

**A proporção entre as notas importa.** Se ser atropelada custasse só −1, ela se jogaria na frente de qualquer carro. Se custasse −10000, teria tanto medo que nunca tentaria atravessar. Com −50 contra +100, vale arriscar quando a chance de passar é boa.

**Linha de base:** uma galinha que age totalmente ao acaso faz em média **−51 pontos** por partida e **chega em 0% das vezes**. Isso mostra que atravessar não acontece por sorte.

---

## 5. O estado: o que a galinha "enxerga"

Esta é a parte mais importante de entender.

### O problema
Se o estado fosse o mapa inteiro (posição de todos os carros), existiriam trilhões de situações. Cada pista tem 28 arranjos possíveis para 2 carros em 8 colunas; com 9 pistas isso dá 28⁹ ≈ **10,6 trilhões** de arranjos de carros, vezes 88 posições da galinha. A tabela nunca caberia na memória e a galinha **nunca veria a mesma situação duas vezes**, então não teria como aprender com a experiência passada.

### A solução
Damos um estado **pequeno**, com só **5 números**, uma tupla:

```
(linha, coluna, frente_ocupada, frente_vindo, aqui_vindo)
```

| Número | Pergunta que responde |
|---|---|
| `linha` | Em que linha estou? (0 a 10) |
| `coluna` | Em que coluna estou? (0 a 7) |
| `frente_ocupada` | **Agora** tem um carro na casa logo acima de mim? (0 ou 1) |
| `frente_vindo` | No **próximo turno** vai chegar um carro na casa logo acima de mim? |
| `aqui_vindo` | No **próximo turno** vai chegar um carro **em cima de mim**, na minha pista? |

Total de estados possíveis: 11 × 8 × 2 × 2 × 2 = **704**. Em treino, ela chega a visitar cerca de 510 a 530 deles. Por serem poucos, eles **se repetem muito**, e é essa repetição que permite aprender.

```python
def estado(self):
    l, c = self.linha, self.coluna
    frente_ocupada = int(self.tem_carro(l + 1, c))
    frente_vindo = int(self.tem_carro(l + 1, (c - self.direcao(l + 1)) % self.colunas))
    aqui_vindo = int(self.tem_carro(l, (c - self.direcao(l)) % self.colunas))
    return (l, c, frente_ocupada, frente_vindo, aqui_vindo)
```

### Por que `(c - direcao) % colunas`?
Um carro que está em `c - direcao` e anda na direção da pista vai parar exatamente em `c` no próximo turno. Então essa conta responde "alguém está vindo para cá?".

### Exemplo para saber ler um estado
A galinha está na linha 3, coluna 2. Na linha 4 (pista que anda para a direita) há um carro na coluna 1.

```
linha 4   .  C→ .  .  .  .  .  .     ← o carro vai para a coluna 2 no próximo turno
linha 3   .  .  G  .  .  .  .  .
```

- Agora não há carro acima dela → `frente_ocupada = 0`
- O carro vai ocupar a casa de cima no próximo turno → `frente_vindo = 1`
- Nada vem na pista dela → `aqui_vindo = 0`
- Estado: **`(3, 2, 0, 1, 0)`**

Nesse estado, subir é ruim: quando ela chegar na linha 4, o carro também chega.

Mais exemplos de leitura:
- `(3, 2, 1, 0, 0)` → tem carro **agora** na casa de cima. Evitar **CIMA**.
- `(5, 3, 0, 0, 1)` → um carro vem **na minha pista** direto para mim. O perigo é ficar **parada** (andar para o lado ou para cima a salva).
- O `1` é um **sim/não**, não "quantos carros".

### Limitação (boa de mencionar)
A galinha é **míope** de propósito: não vê carros nos lados nem a duas casas de distância. Mesmo assim aprende bem. Isso mostra o poder de escolher bem o estado. Também explica por que ela às vezes ainda é atropelada.

---

## 6. A tabela Q: a memória da galinha

Arquivo: **`treinar.py`**, função `nova_tabela_q()`.

```python
def nova_tabela_q():
    return defaultdict(lambda: np.zeros(N_ACOES))
```

É um **dicionário**: a **chave** é o estado (a tupla) e o **valor** são **5 números**, um por ação.

```
estado (3,2,0,1,0) → [ CIMA: -40.0,  BAIXO: 2.0,  ESQ: 5.0,  DIR: 3.0,  PARADA: 1.0 ]
```

Cada número responde: **"se eu estou nesse estado e faço essa ação, quantos pontos espero somar daqui até o fim da partida?"** Isso é o **valor Q**.

- No começo, tudo é **0**: ela não sabe nada.
- O `defaultdict` cria uma linha de zeros sozinho na primeira vez que um estado aparece.
- Tamanho máximo: 704 estados × 5 ações = **3.520 números**. É isso que ela "aprende".

### Como ela decide (a política)
Em um estado, olha a linha da tabela e escolhe o **maior valor**:

```python
acao = int(np.argmax(q[estado]))
```

`argmax` devolve a **posição** do maior número. **Toda a "inteligência" é isso**: olhar uma linha de 5 números e escolher o maior.

---

## 7. Epsilon-greedy: explorar ou aproveitar

Arquivo: **`treinar.py`**, função `escolher_acao()`.

Dilema: se ela só usar o que já sabe, pode nunca descobrir um caminho melhor. Se só explorar, nunca usa o que aprendeu.

```python
def escolher_acao(q, estado, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACOES), True       # explorou: ação ao acaso
    valores = q[estado]
    melhores = np.flatnonzero(valores == valores.max())
    return int(random.choice(melhores)), False        # usou o que sabe
```

- Com probabilidade **epsilon**, age ao **acaso** (explora).
- Senão, escolhe a ação de **maior valor Q** (aproveita).
- O **desempate aleatório** existe porque no começo todos os valores são 0. Sem ele ela sempre escolheria a primeira ação (CIMA) e nunca testaria as outras.

### Como o epsilon muda
```python
EPSILON_INICIAL = 1.0         # começa 100% aleatória
EPSILON_MINIMO = 0.05         # nunca para de explorar por completo
EPSILON_DECAIMENTO = 0.9995   # a cada episódio: epsilon = epsilon × 0.9995
```

O epsilon chega em 0,05 por volta do episódio **6.000**. Início = curiosa, fim = confiante. Por isso, no começo do treino ela morre em quase todas as partidas: está jogando ao acaso.

---

## 8. A equação do Q-learning: onde ela aprende

Arquivo: **`treinar.py`**, função `aprender()`. É o coração do projeto.

```python
def aprender(q, estado, acao, recompensa, novo_estado, terminou):
    antigo = float(q[estado][acao])
    futuro = 0.0 if terminou else np.max(q[novo_estado])
    alvo = recompensa + GAMA * futuro
    q[estado][acao] += ALFA * (alvo - antigo)
    return antigo, float(q[estado][acao])
```

Em fórmula:

```
Q[estado][ação]  ←  Q[estado][ação]  +  ALFA × ( alvo − Q[estado][ação] )

alvo = recompensa + GAMA × (melhor valor Q do novo estado)
```

### Em português
> "O que eu achava que valia" é ajustado um pouquinho na direção de "o que realmente aconteceu agora + o que espero ganhar daqui para frente".

### Cada parte
- **`alvo`**: a nova estimativa, formada pelo que ganhou agora (`recompensa`) mais o melhor que espera do estado seguinte (`GAMA × max(Q[novo_estado])`).
- **`alvo − antigo`**: o **erro** da estimativa anterior.
- **`ALFA = 0.1`**: só corrige **10% do erro** por vez. Aprendizado suave, não dá chilique com uma experiência isolada.
- **`GAMA = 0.95`**: o futuro vale quase tanto quanto o presente. Um prêmio de 100 que está a 10 passos de distância vale hoje 100 × 0,95¹⁰ ≈ **60**. É isso que a faz aceitar perder 1 ponto por passo para ganhar +100 lá na frente.
- **Se o jogo acabou** (`terminou`): não existe futuro, então `alvo = recompensa`.

### Exemplo numérico 1: aprendendo que algo é ruim
Ela faz CIMA com um carro na frente e é atropelada.
- recompensa = −1 (passo) −50 (atropelada) = **−51**, e o jogo acabou → `alvo = −51`
- `Q = 0 + 0.1 × (−51 − 0) =` **−5,1**

Na próxima vez em que isso acontecer: `−5,1 + 0,1 × (−51 − (−5,1))` = **−9,7**. Depois −13,8... até se aproximar de −51. Ela vai "desconfiando" cada vez mais dessa ação nesse estado.

### Exemplo numérico 2: o futuro entrando na conta
Num estado, ela faz CIMA: recompensa = −1 (passo) + 5 (linha nova) = +4. O melhor valor do novo estado é 80. Valor antigo Q = 70.
- `alvo = 4 + 0,95 × 80 = 80`
- `Q = 70 + 0,1 × (80 − 70) =` **71**

### Como o +100 "se espalha para trás"
O estado logo antes da chegada aprende que há um prêmio ali. Depois o estado anterior aprende que leva a um estado bom (por causa do termo `GAMA × max(Q[novo_estado])`), e assim sucessivamente. **É assim que a informação do fim do jogo chega ao começo.** Isso só acontece depois de muitas partidas.

---

## 9. O loop de treino

Arquivo: **`treinar.py`**, funções `rodar_episodio()` e `treinar()`.

```python
def rodar_episodio(env, q, epsilon):
    estado = env.reset()
    total = 0
    terminou = False
    while not terminou:
        acao, _ = escolher_acao(q, estado, epsilon)               # 1. escolhe
        novo_estado, recompensa, terminou = env.passo(acao)       # 2. age e vê o resultado
        aprender(q, estado, acao, recompensa, novo_estado, terminou)  # 3. aprende
        estado = novo_estado
        total += recompensa
    return total
```

### O ciclo de cada passo (decorar!)
1. **Olha** o estado.
2. **Escolhe** uma ação (explorar ou aproveitar).
3. **Age**; o ambiente devolve o novo estado e a recompensa.
4. **Aprende**: corrige o valor Q usando a equação.
5. Repete até a partida acabar. Aí vem a partida seguinte, com o epsilon um pouco menor.

O treino padrão roda **10.000 episódios**. Ao fim de cada um: `epsilon = max(0.05, epsilon × 0.9995)`.

---

## 10. Checkpoints: salvar e retomar

Arquivo: **`treinar.py`**, `salvar_checkpoint()` e `carregar_checkpoint()`.

Um checkpoint é um **save game**: guarda a tabela Q, o epsilon, o número do episódio e o histórico, em arquivo `.pkl` (formato `pickle`, que salva objetos Python).

- `checkpoints/ultimo.pkl`: sempre o mais recente.
- `checkpoints/ep_5000.pkl`, `ep_10000.pkl`...: fotografias em momentos diferentes.
- **Por que serve:** treina uma vez e reutiliza; retoma de onde parou; e permite **comparar a galinha em épocas diferentes** (ex.: `ep_1000` desastrada contra `ep_10000` esperta).
- Detalhe técnico: `defaultdict` com `lambda` não pode ser salvo em pickle, por isso salvamos `dict(q)` e reconstruímos ao carregar.
- Por padrão, o treino **retoma** do `ultimo.pkl`. Para recomeçar: `--do-zero`.

---

## 11. Métricas e resultados

Arquivo: **`treinar.py`**, `media_movel()` e `salvar_grafico()` → gera `resultados/curva_aprendizado.png`.

Como cada partida varia muito, usamos **média móvel** (a média das últimas 500), que revela a tendência.

### Evolução típica (do meu teste, 10.000 episódios)

| Episódio | Epsilon | Pontos (média) | Chegou |
|---|---|---|---|
| 1.000 | 0,606 | −49 | 0% |
| 3.000 | 0,223 | −25 | 9% |
| 4.000 | 0,135 | +14 a +27 | 31 a 38% |
| 6.000 | 0,050 | +83 | 71% |
| 10.000 | 0,050 | +87 | 73% |

### Duas leituras importantes
1. **Aleatória vs. treinada:** a aleatória faz −51 pontos e 0% de sucesso. A treinada passa de +85 e ~75%.
2. **Por que não chega perto de 100% durante o treino?** Porque o treino **ainda tem 5% de jogadas aleatórias** (epsilon mínimo). Uma única jogada aleatória na frente de um carro já é fatal. Medindo a mesma galinha treinada **sem exploração** (epsilon = 0), a taxa de sucesso foi de **cerca de 96%** (74% com epsilon 0,05, no mesmo teste). **É por isso que o visualizador mostra a galinha bem melhor que o gráfico.**
3. Os ~3 a 4% de falha restantes vêm da limitação do estado (míope) e do limite de 100 passos.

(Os números exatos variam um pouco a cada treino, porque o sorteio dos carros é aleatório.)

---

## 12. A parte visual (pygame)

Dois arquivos, com responsabilidades separadas:

- **`visual.py`** (classe `Cena`): **só desenha**. Mapa, carros, galinha, painel. Não tem regra de jogo nem aprendizado.
- **`visualizador.py`**: controla o fluxo (loop, teclado, tempo). Tem duas funções: `jogar()` (galinha treinada jogando) e `treinar_ao_vivo()` (treino na tela).

### A animação é só um efeito visual
O jogo é por turnos, e `passo()` é **instantâneo**. Para parecer natural, a `Cena` guarda uma **"foto do antes"** (`sincronizar()`) e desenha o **meio do caminho** entre antes e depois (progresso `t` de 0 a 1). Carros deslizam e a galinha dá um pulinho. **Isso não afeta o aprendizado.**

### O painel lateral (ótimo para explicar a IA)
- **"O que ela vê (estado)":** as 3 perguntas do estado, com bolinha vermelha e "SIM" quando há perigo.
- **"Valores Q deste estado":** a **linha da tabela Q** daquele estado, em barras. Verde = ação boa, vermelha = ação ruim. **A ação destacada é a de maior valor (o `argmax`).**
- **Placar:** episódio, epsilon, % de sucesso, pontos médios, estados vistos.
- No treino ao vivo: mostra se a ação foi **"EXPLORANDO (aleatória)"** ou **"USANDO O QUE SABE"**, e a linha `Q[CIMA]: 0.0 → -0.1 (recompensa -1)`, que é **a equação acontecendo em tempo real**.

### O modo TURBO (tecla T)
Treinar 10.000 episódios com animação levaria horas. O turbo roda vários episódios por quadro **sem desenhar nem esperar**. **É o mesmo algoritmo, só sem animação.** O treino ao vivo e o do terminal usam a **mesma** função `aprender()`.

---

## 13. Mapa do código: onde está cada coisa

| Pergunta do professor | Onde mostrar |
|---|---|
| Onde está o jogo / as regras? | `galinha_env.py`, classe `TravessiaGalinha`, método `passo()` |
| Onde são definidas as recompensas? | `galinha_env.py`, constantes `R_PASSO`, `R_AVANCO`, `R_ATROPELADA`, `R_OBJETIVO` |
| O que a IA "enxerga"? | `galinha_env.py`, método `estado()` |
| Onde está a memória da IA? | `treinar.py`, `nova_tabela_q()` (**AQUI A TABELA Q**) |
| Onde ela escolhe a ação? | `treinar.py`, `escolher_acao()` (**AQUI ELE ESCOLHE A AÇÃO**) |
| Onde acontece o aprendizado? | `treinar.py`, `aprender()` (**AQUI ELE APRENDE**) |
| Onde fica o loop de treino? | `treinar.py`, `rodar_episodio()` e `treinar()` |
| Onde ficam os hiperparâmetros? | `treinar.py`, topo do arquivo: `ALFA`, `GAMA`, `EPSILON_*` |
| Onde salva e retoma o treino? | `treinar.py`, `salvar_checkpoint()` / `carregar_checkpoint()` (**AQUI O CHECKPOINT**) |
| Onde estão as métricas e o gráfico? | `treinar.py`, `media_movel()` e `salvar_grafico()` (**AQUI AS MÉTRICAS**) |
| Como mostrar que ela só usa o que aprendeu? | `visualizador.py`, `jogar()`, linha `np.argmax(q_visto)` (**AQUI ELA USA O QUE APRENDEU**) |
| Onde vejo o treino ao vivo? | `visualizador.py`, `treinar_ao_vivo()` |
| Onde fica a parte gráfica? | `visual.py`, classe `Cena` |

**Dica:** nos arquivos, procure por `AQUI` para achar os pontos principais.

### Estrutura de arquivos
```
galinha_env.py     o mundo: regras, estado, recompensas (sem aprendizado)
treinar.py         o cérebro: tabela Q, epsilon, equação, checkpoints, gráfico
visual.py          só desenha (mapa, carros, galinha, painel)
visualizador.py    joga com o que aprendeu / treina ao vivo
avaliar.py         avalia os checkpoints e gera as figuras de resultado
checkpoints/       saves do treino (.pkl)
resultados/        figuras (curva_aprendizado.png, evolucao_checkpoints.png, caminho_final.png...)
tests/             testes automáticos (python -m pytest)
```

### Quem chama quem
```
visualizador.py ──usa──> visual.py (desenho)
       │
       ├──usa──> treinar.py (tabela Q, aprender, checkpoints)
       │
       └──usa──> galinha_env.py (o jogo)

treinar.py ──usa──> galinha_env.py
```

---

## 14. Comandos

```powershell
python treinar.py --do-zero                    # treina do zero (10000 episódios) e gera o gráfico
python treinar.py --episodios 20000            # continua treinando do último checkpoint
python treinar.py --do-zero --checkpoint-cada 1000   # salva um checkpoint a cada 1000 episódios

python visualizador.py                         # galinha treinada jogando (usa ultimo.pkl)
python visualizador.py --checkpoint checkpoints/ep_2000.pkl   # compara um checkpoint antigo
python visualizador.py --treinar --do-zero     # treino ao vivo, do zero
python visualizador.py --treinar               # treino ao vivo, retomando do último checkpoint
```

```powershell
python avaliar.py                              # gera as figuras da Questão 3 (b) em resultados/
python avaliar.py --semente 7                  # mostra o caminho de outra partida
```

`avaliar.py` faz cada checkpoint jogar 1.000 partidas **sem exploração** (ε = 0) e compara com um agente aleatório. Ele gera: `curva_aprendizado.png` (treino), `evolucao_checkpoints.png` (sucesso por checkpoint), `caminho_final.png` (o caminho do agente final, quadro a quadro), `caminho_evolucao.png` (a mesma partida jogada por épocas diferentes) e `caminho_final.txt` (estado, ação e recompensa de cada passo). Resultado do checkpoint final (56.000 episódios): **100% de travessias** nas 1.000 partidas de avaliação, contra 0% do agente aleatório.

**Teclas:** `ESPAÇO` pausa · `↑/↓` velocidade · `R` reinicia a partida (modo jogar) · `T` turbo e `S` salva (treino ao vivo) · `ESC` sai (e salva, no treino ao vivo).

---

## 15. Roteiro de apresentação (≈ 6 a 8 minutos)

**1. Problema (30 s).** "Uma galinha precisa atravessar uma rua com carros. Ninguém programou como desviar. Ela vai aprender sozinha."

**2. O jogo (1 min)** — `galinha_env.py`. "Grade 11×8, 9 pistas, carros que vão e voltam. Cada ação é um número. O ambiente não aprende nada, só tem as regras."

**3. Recompensas (1 min).** "A única comunicação com a IA são notas: −1 por passo, +5 por avançar, −50 se atropelada, +100 se chegar. Ninguém diz 'desvie'. Ela só quer pontos."

**4. Estado (1 min).** "Ela não vê o mapa todo. Vê 5 números: onde está e 3 perguntas de sim/não sobre carros chegando. Isso reduz trilhões de situações para 704." (Mostre o painel do visualizador.)

**5. Tabela Q e epsilon (1 min).** "A memória é uma tabela: para cada estado, 5 números, um por ação. No começo tudo é zero. Ela explora ao acaso (epsilon alto) e vai usando mais o que sabe."

**6. A equação (1,5 min).** Mostre a função `aprender()`. "Ela corrige 10% do erro entre o que achava e o que aconteceu, somando o futuro esperado." Use o exemplo de −5,1 do item 8.

**7. Demonstração ao vivo (2 min).**
- `python visualizador.py --treinar --do-zero`
- Deixe 2 ou 3 episódios no modo animado: ela morre, aparece "EXPLORANDO" e `Q[...]: 0.0 → -0.1`.
- Aperte **T** por uns 5 segundos. O episódio sobe, o epsilon cai.
- Aperte **T** de novo: agora "USANDO O QUE SABE" e ela atravessa.

**8. Resultados (30 s).** Mostre `curva_aprendizado.png`. "De 0% a ~75% durante o treino. Sem as jogadas aleatórias, passa de 95%."

**9. Limitações (30 s).** "Estado míope; uma tabela só funciona porque o problema é pequeno. Para problemas maiores usaríamos redes neurais (Deep Q-Learning)."

**Plano B se a demo falhar:** rode `python visualizador.py` (sem `--treinar`), que só joga com o último checkpoint, e mostre o gráfico.

---

## 16. Perguntas que o professor pode fazer (e respostas)

**1. Isso é inteligência artificial de verdade? De onde vem o modelo?**
É aprendizado por reforço clássico. Não há modelo externo nem API: a "IA" é uma tabela de 704 × 5 números que o próprio programa preenche jogando. Foi escrita do zero com Python e numpy.

**2. A galinha decorou o mapa?**
Não. A cada partida os carros são sorteados de novo. Ela aprende uma **regra geral** sobre o estado local ("se há carro chegando em cima, não fico parada").

**3. Por que usar uma tabela e não uma rede neural?**
Porque o estado é pequeno (704 possibilidades). Para estados enormes (imagens, por exemplo) a tabela seria impraticável e usaríamos redes neurais (Deep Q-Learning).

**4. Por que o epsilon começa em 1 e cai?**
No começo ela não sabe nada, então precisa explorar. Conforme aprende, vale mais usar o que sabe. Ele para em 0,05 para nunca parar de testar alternativas.

**5. Por que o desconto gama?**
Para ela valorizar o futuro, mas dar mais peso ao que está perto. É o que permite aceitar um custo pequeno agora (−1 por passo) por um prêmio grande depois (+100). Com gama = 0,95, um prêmio a 10 passos vale ~60.

**6. Por que o treino não chega a 100% de sucesso?**
Duas razões: (a) o treino mantém 5% de jogadas aleatórias; sem elas, o sucesso passa de 95%; (b) o estado é míope e não vê carros pelos lados.

**7. O que muda se eu mudar as recompensas?**
Muda o comportamento. Se atropelar custasse pouco, ela se arriscaria mais. Se custasse muito, ficaria com medo. Ela otimiza exatamente o que as notas pedem, não o que queremos.

**8. O que é o "alfa"?**
A taxa de aprendizado: o quanto cada experiência nova altera o valor antigo. 0,1 = ajusta 10% do erro por vez. Muito alto → instável; muito baixo → lento.

**9. Como você sabe que ela realmente aprendeu e não foi sorte?**
O gráfico mostra a evolução. E a linha de base: uma galinha 100% aleatória faz −51 pontos e chega em 0% das vezes. A treinada faz +85 e chega em mais de 95% dos casos sem exploração.

**10. O que é um episódio? E um checkpoint?**
Episódio = uma partida completa. Checkpoint = um "save game" do treino (tabela Q + epsilon + progresso), que permite retomar e comparar fases diferentes.

**11. (Difícil) O que significa dizer que o Q-learning é "off-policy"?**
Que a atualização usa o **melhor** valor do próximo estado (`max`), e não a ação que ela de fato vai escolher (que pode ser aleatória por causa do epsilon). Assim ela aprende a política ótima mesmo explorando.

**12. (Difícil) O que garante que a tabela converge?**
Com estados e ações discretos, visitando todos os pares muitas vezes e com taxa de aprendizado adequada, o Q-learning converge para os valores ótimos. Na prática, aqui, é isso que mostra o gráfico estabilizando.

---

## 17. Autoteste: responda sem olhar

1. Quais são as 5 ações e por que cada uma é só um número?
2. Cite as 4 recompensas e seus valores.
3. Por que usar só 5 números como estado em vez do mapa todo?
4. Leia o estado `(3, 2, 0, 1, 0)` e diga qual ação seria perigosa.
5. O que guarda cada linha da tabela Q?
6. Como a galinha escolhe uma ação quando o epsilon vale 0?
7. O que acontece no começo do treino, quando tudo vale 0?
8. Escreva a equação do Q-learning e explique `alfa`, `gama` e `alvo`.
9. Se Q antigo = 0, recompensa = −51 e o jogo acabou, qual o novo Q? (Resposta: −5,1)
10. Por que a taxa de sucesso no gráfico é menor que no visualizador?
11. Para que serve um checkpoint?
12. O que é o modo turbo e ele muda o algoritmo?

**Respostas curtas:**
1. CIMA, BAIXO, ESQUERDA, DIREITA, PARADA (0 a 4); a IA só escolhe números e o ambiente traduz. 2. −1 passo, +5 linha nova, −50 atropelada, +100 chegada. 3. Para ter só 704 situações que se repetem, em vez de trilhões. 4. CIMA (um carro chega na casa de cima). 5. O valor esperado (pontos futuros) de cada uma das 5 ações naquele estado. 6. Escolhe a ação de maior valor Q (`argmax`). 7. Epsilon alto, age ao acaso e a tabela vai se preenchendo. 8. Q ← Q + α(alvo − Q), com alvo = recompensa + γ·max Q(novo); α = quanto corrige por vez, γ = peso do futuro, alvo = a nova estimativa. 9. −5,1. 10. O treino tem 5% de ações aleatórias; o visualizador usa só o que aprendeu. 11. Salvar e retomar o treino e comparar épocas. 12. Treina sem animar; é o mesmo algoritmo.

---

## 18. Limitações e ideias de melhoria

- **Estado míope:** acrescentar carros dos lados ou a duas casas de distância aumentaria o sucesso (mas também o tamanho da tabela).
- **Tabela não escala:** problemas com muitos estados exigem redes neurais (Deep Q-Learning).
- **Epsilon fixo no fim:** poderia chegar a 0 no final para aproveitar totalmente o aprendizado.
- **Recompensa por avanço:** o bônus de +5 ajuda a guiar, mas poderia causar comportamentos estranhos se mal calibrado.
- **Mundo simples:** carros com velocidade única e por turnos. Um mundo contínuo seria bem mais difícil.
- **Generalização:** mudar o tamanho do mapa exige retreinar (o estado inclui linha e coluna).

---

## 19. Resumo para revisar 5 minutos antes

- **Aprendizado por reforço:** o agente age, o ambiente devolve **estado novo + recompensa**, e o agente corrige o que sabe.
- **Estado:** 5 números `(linha, coluna, frente_ocupada, frente_vindo, aqui_vindo)` → 704 estados.
- **Tabela Q:** dicionário `estado → 5 valores`. Decisão = `argmax`.
- **Epsilon-greedy:** com chance ε explora; senão usa a tabela. ε: 1,0 → 0,05 (em ~6.000 episódios).
- **Equação:** `Q ← Q + 0,1 × (recompensa + 0,95 × max Q(novo) − Q)`.
- **Resultado:** aleatória −51 pts / 0%; treinada ~+87 pts / ~75% no treino e **~96% sem exploração**.
- **Código:** `galinha_env.py` (mundo) · `treinar.py` (cérebro) · `visual.py` (desenho) · `visualizador.py` (demo).
