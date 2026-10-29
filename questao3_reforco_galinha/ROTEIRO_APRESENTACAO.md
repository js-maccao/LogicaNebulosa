# Roteiro da apresentação (fala + código)

Duração estimada: 12 a 15 minutos, com a demonstração.

**Como usar:** o texto entre aspas é o que você fala. Os blocos "Mostrar no código" dizem qual arquivo abrir, em qual linha, e o que dizer sobre cada linha. Os números de linha são da versão atual dos arquivos. Se você editar algum arquivo, confira de novo.

**Antes de começar:**
- Abra no VS Code (ou outro editor) os arquivos `galinha_env.py`, `treinar.py`, `avaliar.py` e `visualizador.py`, cada um numa aba.
- Deixe um terminal aberto na pasta `questao3_reforco_galinha`.
- No VS Code, `Ctrl+G` vai direto para uma linha. Use isso quando o professor perguntar "onde está isso?".
- Atalho: procure por `AQUI` (`Ctrl+F`). Os pontos principais do código têm comentários `# ===== AQUI ... =====`.

---

## Slide 1: Capa

> "Bom dia/boa noite. Meu trabalho é sobre a Questão 3, aprendizado por reforço. O professor pediu ênfase em uma das questões, e eu escolhi essa. Em vez do labirinto sugerido, fiz uma galinha que precisa atravessar uma rua cheia de carros. A ideia é a mesma de um labirinto, só que os obstáculos se movem."

---

## Slide 2: O problema

> "Aprendizado por reforço é aprender por tentativa e erro. Não existe um conjunto de exemplos com a resposta certa, como na classificação. O agente, que aqui é a galinha, age no ambiente, recebe uma nota e ajusta o que sabe."

> "Ninguém programou a galinha para desviar de carros. Ela só recebe notas. Depois de milhares de partidas, ela descobre sozinha o que dá mais pontos."

> "Não usei nenhuma biblioteca de aprendizado por reforço. O algoritmo está escrito do zero em Python, com numpy para os vetores, matplotlib para os gráficos e pygame para a parte visual."

---

## Slide 3: O ambiente

> "O ambiente fica todo em um arquivo, o `galinha_env.py`. Ele só tem as regras do jogo. Não existe aprendizado nele."

**Mostrar no código: `galinha_env.py`**

| Linha | O que é | O que falar |
|---|---|---|
| 1–3 | docstring | "Este arquivo só tem as regras. O aprendizado fica no `treinar.py`." |
| 16 | `class TravessiaGalinha` | "A classe que representa o jogo." |
| 17 | `__init__(linhas=11, colunas=8, carros_por_pista=2, max_passos=100)` | "O mapa tem 11 linhas por 8 colunas, 2 carros por pista e limite de 100 passos por partida." |
| 25–27 | `direcao()` | "Pista par anda para a direita (+1), ímpar para a esquerda (−1)." |
| 29 | `reset()` | "Começa uma partida nova." |
| 31–32 | `self.linha = 0`, `self.coluna = self.colunas // 2` | "A galinha começa na linha 0, no meio (coluna 4)." |
| 33 | `self.linha_max = 0` | "Guarda a linha mais alta que ela já alcançou. Uso isso para o bônus de avanço." |
| 37–40 | `self.carros = {... random.sample(...)}` | "Os carros são sorteados de novo a cada partida. Por isso ela não decora um mapa; ela aprende uma regra geral." |
| 43–44 | `tem_carro()` | "Responde se há carro numa casa." |
| 46–49 | `mover_carros()` | "Cada carro anda uma casa. O `% self.colunas` faz o carro que sai de um lado voltar pelo outro." |
| 63 | `passo(acao)` | "Um turno do jogo. Recebe a ação e devolve o novo estado, a recompensa e se a partida terminou." |
| 68–79 | movimento | "Primeiro a galinha se move. As linhas 78 e 79 travam a posição dentro do mapa." |
| 86–90 | colisão e chegada | "Depois verifica se ela caiu em cima de um carro, ou se chegou na última linha." |
| 92–95 | carros andam | "Se não acabou, os carros andam e eu verifico a colisão de novo. Verificar duas vezes evita que um carro passe através da galinha sem pegá-la." |
| 102–103 | `"tempo"` | "Se passou de 100 passos, acaba por tempo, para não ter partida infinita." |
| 106 | `return self.estado(), recompensa, terminou` | "É isso que o agente recebe a cada passo." |

---

## Slide 4: Estados

> "Este é o ponto mais importante do projeto. Se o estado fosse o mapa inteiro, com a posição de todos os carros, seriam cerca de 10 trilhões de combinações. A galinha nunca veria a mesma situação duas vezes e não teria como aprender."

> "Então o estado tem só 5 números: onde ela está, e três perguntas de sim ou não sobre os carros perto dela. Isso dá 704 estados possíveis, que se repetem muito. É essa repetição que permite aprender."

**Mostrar no código: `galinha_env.py`, linhas 51–60**

| Linha | Código | O que falar |
|---|---|---|
| 53 | `l, c = self.linha, self.coluna` | "Posição atual da galinha." |
| 55 | `frente_ocupada = int(self.tem_carro(l + 1, c))` | "Tem carro na casa logo acima, agora?" |
| 57 | `frente_vindo = int(self.tem_carro(l + 1, (c - self.direcao(l + 1)) % self.colunas))` | "Vai chegar carro na casa de cima no próximo turno? Um carro que está em `c − direção` vai parar em `c` quando andar. Então eu olho essa casa." |
| 59 | `aqui_vindo = int(self.tem_carro(l, (c - self.direcao(l)) % self.colunas))` | "A mesma conta, mas na minha pista: vai chegar carro em cima de mim?" |
| 60 | `return (l, c, frente_ocupada, frente_vindo, aqui_vindo)` | "O estado é uma tupla. Ela vira a chave da tabela Q." |

> "Exemplo: o estado (3, 2, 0, 1, 0) quer dizer linha 3, coluna 2, não há carro acima agora, mas vai chegar um no próximo turno. Subir agora seria perigoso."

---

## Slide 5: Ações e recompensas

**Mostrar no código: `galinha_env.py`**

| Linha | Código | O que falar |
|---|---|---|
| 6 | `CIMA, BAIXO, ESQUERDA, DIREITA, PARADA = 0, 1, 2, 3, 4` | "As 5 ações. Para o agente, cada ação é só um número de 0 a 4." |
| 10 | `R_PASSO = -1` | "Cada passo custa 1 ponto. Isso cria pressa; senão ficar parada não custaria nada." |
| 11 | `R_AVANCO = 5` | "+5 quando ela chega numa linha mais alta do que já esteve. Só em linha nova, senão ela ganharia pontos subindo e descendo." |
| 12 | `R_ATROPELADA = -50` | "Ser atropelada tira 50 e acaba a partida." |
| 13 | `R_OBJETIVO = 100` | "Chegar dá +100 e acaba a partida." |
| 66 | `recompensa = R_PASSO` | "Todo passo começa com −1." |
| 82–84 | `if self.linha > self.linha_max: ... recompensa += R_AVANCO` | "Aqui entra o bônus de avanço." |
| 98–101 | `recompensa += R_ATROPELADA` / `R_OBJETIVO` | "E aqui a nota final da partida." |

> "Repare que eu nunca escrevi 'desvie do carro'. A galinha só quer somar pontos. Desviar é consequência das notas que eu escolhi."

> "A proporção importa: se atropelar custasse só −1, ela se jogaria na frente dos carros. Se custasse −10.000, ela teria medo de sair da calçada."

---

## Slide 6: Tabela Q e epsilon-greedy

> "O agente fica no `treinar.py`. A memória dele é a tabela Q."

**Mostrar no código: `treinar.py`**

| Linha | Código | O que falar |
|---|---|---|
| 25 | `ALFA = 0.1` | "Taxa de aprendizado." |
| 26 | `GAMA = 0.95` | "Desconto do futuro." |
| 27–29 | `EPSILON_INICIAL = 1.0`, `EPSILON_MINIMO = 0.05`, `EPSILON_DECAIMENTO = 0.9995` | "Começa 100% aleatória, cai um pouco a cada episódio e nunca fica abaixo de 5%." |
| 38–41 | `nova_tabela_q()` → `defaultdict(lambda: np.zeros(N_ACOES))` | "A tabela Q é um dicionário. A chave é o estado, o valor é um vetor com 5 números, um por ação. Quando aparece um estado novo, o `defaultdict` cria uma linha de zeros sozinho." |
| 45 | `escolher_acao(q, estado, epsilon)` | "Aqui ela decide o que fazer." |
| 48–49 | `if random.random() < epsilon: return random.randrange(N_ACOES), True` | "Com chance epsilon, ela explora: faz uma ação aleatória." |
| 50 | `valores = q[estado]` | "Senão, pega a linha da tabela desse estado." |
| 52–53 | `melhores = np.flatnonzero(valores == valores.max())` e `random.choice(melhores)` | "E escolhe a ação de maior valor. Se houver empate, sorteia. No começo está tudo em zero; sem o sorteio, ela sempre escolheria CIMA e nunca testaria as outras." |

> "Cada número da tabela responde: se eu estou neste estado e faço esta ação, quantos pontos espero somar daqui até o fim? A tabela tem no máximo 704 × 5 = 3.520 números. É isso que ela aprende."

---

## Slide 7: A equação do Q-learning

> "Este é o coração do trabalho: a função `aprender`."

**Mostrar no código: `treinar.py`, linhas 56–63**

| Linha | Código | O que falar |
|---|---|---|
| 59 | `antigo = float(q[estado][acao])` | "O que ela achava que essa ação valia." |
| 60 | `futuro = 0.0 if terminou else np.max(q[novo_estado])` | "O melhor que ela espera do estado seguinte. Se a partida acabou, não tem futuro: é zero." |
| 61 | `alvo = recompensa + GAMA * futuro` | "A nova estimativa: o que ganhou agora mais o futuro descontado." |
| 62 | `q[estado][acao] += ALFA * (alvo - antigo)` | "A correção. `alvo − antigo` é o erro; ela corrige 10% desse erro." |

> "Exemplo: ela sobe com um carro na frente e é atropelada. Recompensa: −1 do passo e −50 da batida, total −51. O jogo acabou, então o alvo é −51. Q era 0. Novo Q = 0 + 0,1 × (−51 − 0) = −5,1. Na próxima vez vai para −9,7, depois −13,8, e assim ela vai desconfiando cada vez mais dessa ação naquele estado."

> "O gama de 0,95 faz o futuro contar: um prêmio de 100 a 10 passos de distância vale hoje uns 60. É por isso que ela aceita perder 1 ponto por passo para ganhar 100 lá na frente. E é pelo termo `max Q(novo_estado)` que o +100 da chegada vai se espalhando para trás, de estado em estado."

---

## Slide 8: O loop de treino

**Mostrar no código: `treinar.py`**

| Linha | Código | O que falar |
|---|---|---|
| 66 | `rodar_episodio(env, q, epsilon)` | "Uma partida inteira." |
| 68 | `estado = env.reset()` | "Começa a partida." |
| 72 | `acao, _ = escolher_acao(q, estado, epsilon)` | "1. Escolhe a ação." |
| 73 | `novo_estado, recompensa, terminou = env.passo(acao)` | "2. Age no ambiente e recebe o resultado." |
| 74 | `aprender(q, estado, acao, recompensa, novo_estado, terminou)` | "3. Aprende, com a equação." |
| 75 | `estado = novo_estado` | "4. Segue para o próximo passo." |
| 122 | `treinar(...)` | "O treino completo." |
| 132–133 | `for ep in range(...)`: `rodar_episodio(...)` | "Um episódio atrás do outro." |
| 135 | `epsilon = max(EPSILON_MINIMO, epsilon * EPSILON_DECAIMENTO)` | "Depois de cada episódio, o epsilon diminui." |
| 136 | `historico.append((total, env.resultado))` | "Guardo os pontos e o resultado de cada episódio. É daqui que sai o gráfico." |
| 144–147 | `salvar_checkpoint(...)` | "A cada 1.000 episódios, salvo um checkpoint." |
| 81–86 | `salvar_checkpoint()` | "O checkpoint é um save game com a tabela Q, o epsilon, o episódio e o histórico, em `.pkl`. Na linha 85 eu salvo `dict(q)` porque um `defaultdict` com `lambda` não pode ir para o pickle." |

> "Treinei 56.000 episódios, salvando um checkpoint a cada 1.000. É com esses checkpoints que eu mostro a evolução."

---

## Slide 9: Evolução durante o treino

> "Esta é a curva do item (b): pontos e porcentagem de travessias por episódio, com média móvel de 500, porque cada partida varia muito."

> "Até uns 2.000 episódios, ela joga quase ao acaso e morre sempre. Entre 2.000 e 6.000, o epsilon cai e ela começa a usar a tabela: a curva sobe rápido. Depois estabiliza em cerca de 76%."

> "Por que não chega a 100% aqui? Porque o treino sempre mantém 5% de jogadas aleatórias, e uma única jogada aleatória na frente de um carro já mata."

**Mostrar no código**
- `treinar.py`, linhas 98–101, `media_movel()`: "`np.convolve` com uma janela de 500 faz a média móvel."
- `avaliar.py`, linha 81, `grafico_treino()`: "Gera esta figura a partir do histórico salvo no checkpoint."

---

## Slide 10: Desempenho de cada checkpoint

> "Para medir o que ela realmente aprendeu, tirei a exploração. Cada checkpoint joga as mesmas 1.000 partidas com epsilon zero, só usando a tabela. Comparo com um agente totalmente aleatório."

> "O aleatório chega 0% das vezes. Com 1.000 episódios de treino, 2,5%. Com 5.000, 94%. O checkpoint final, com 56.000, chega em 100% das 1.000 partidas."

**Mostrar no código: `avaliar.py`**

| Linha | Código | O que falar |
|---|---|---|
| 41–44 | `politica_gulosa(q)` → `np.argmax(...)` | "Só escolhe a ação de maior valor Q. Sem exploração." |
| 47–48 | `politica_aleatoria` | "A linha de base: ação totalmente aleatória." |
| 51–53 | `jogar_partida(env, escolher, semente)`, `random.seed(semente)` | "Cada partida usa uma semente fixa. Os carros são os mesmos para todos os checkpoints, então a comparação é justa." |
| 68 | `avaliar(escolher, n=1000, semente_base=10_000)` | "Joga 1.000 partidas e calcula a taxa de sucesso, os pontos médios e os passos médios." |
| 211–215 | laço `for ck in checkpoints:` dentro de `main()` | "Avalia cada um dos 56 checkpoints." |

---

## Slide 11: Caminho final

> "Esta é a outra parte do item (b): o caminho final que o agente encontrou. Cada quadro é um passo da partida. Vermelho são carros indo para a direita, azul para a esquerda, amarelo é o rastro da galinha."

> "Ela atravessa em 21 passos. Repare que nos passos 1 e 3 ela desce para fugir de um carro, nos passos 6, 12 e 19 fica parada esperando, e várias vezes desvia para o lado. Ninguém ensinou isso."

> "Diferente de um labirinto, aqui não existe um único caminho ótimo fixo, porque os carros mudam a cada partida. O que o agente aprendeu é uma política: como reagir ao trânsito em cada situação."

**Mostrar no código / arquivos**
- `resultados/caminho_final.txt`: a tabela com estado, ação e recompensa de cada passo. Bom para mostrar que a recompensa da chegada é 104 (−1 do passo + 5 da linha nova + 100 do objetivo).
- `avaliar.py`, linha 51, `jogar_partida()`: "Joga a partida e guarda a posição, os carros, o estado, a ação e a recompensa a cada passo."
- `avaliar.py`, linha 117, `desenhar_quadro()`: "Desenha um quadro: pistas, carros, rastro e galinha."
- `avaliar.py`, linha 141, `figura_caminho()`: "Junta os quadros na figura."
- Para mostrar outra partida ao vivo: `python avaliar.py --semente 7`.

---

## Slide 12: Evolução do caminho

> "Aqui é a mesma partida, com os mesmos carros, jogada por checkpoints de épocas diferentes. Só muda a tabela Q carregada. No episódio 1.000, ela sobe direto e é atropelada. Nos episódios 15.000 a 42.000 ela já vai bem mais longe, mas cai na mesma armadilha da pista 7. Só o agente final escapa e atravessa."

**Mostrar no código:** `avaliar.py`, linha 166, `figura_evolucao_caminho()`: "Carrega 5 checkpoints espaçados e joga a mesma semente com cada um."

---

## Slide 13: Conclusões

> "Uma tabela de 3.520 números bastou para atravessar a rua em 99% das partidas. O que fez funcionar foi escolher bem o estado, pequeno o suficiente para se repetir, e calibrar bem as recompensas."

> "As limitações: a galinha é míope, não vê carros dos lados, e por isso ainda falha em cerca de 1% das vezes. E uma tabela só funciona porque o problema é pequeno. Para problemas grandes, como jogar a partir da imagem da tela, seria preciso uma rede neural no lugar da tabela, o Deep Q-Learning."

---

## Slide 14: Execução e demonstração

> "O repositório tem um README com tudo. Agora vou mostrar ao vivo."

**Demonstração (2 minutos):**

1. No terminal: `python visualizador.py --treinar --do-zero`
2. Deixe passar 2 ou 3 episódios no modo animado. Aponte no painel:
   - "EXPLORANDO (aleatória)": ela está jogando ao acaso.
   - A linha `Q[CIMA]: 0.0 → -0.1 (recompensa -1)`: "É a equação acontecendo em tempo real."
3. Aperte **T** (turbo) por uns 5 a 10 segundos. "O turbo roda vários episódios por quadro sem desenhar. É o mesmo algoritmo, só sem animação." O episódio sobe e o epsilon cai.
4. Aperte **T** de novo. Agora aparece "USANDO O QUE SABE" e ela atravessa.
5. `ESC` sai e salva.

**Plano B**, se algo falhar: `python visualizador.py`, que só joga com o checkpoint final (`ultimo.pkl`, ep. 56.717).

> Atenção: o modo `--treinar` salva checkpoints na pasta `checkpoints/` e sobrescreve o `ultimo.pkl`. Se você rodar a demonstração e depois quiser o plano B com o agente final, use `python visualizador.py --checkpoint checkpoints/ep_56000.pkl`.

---

## "Onde está no código o que aparece na tela?"

Esta tabela é para as perguntas sobre o visualizador.

| O que aparece na tela | Onde está | O que responder |
|---|---|---|
| A rua, os carros e a galinha | `visual.py`, `desenhar_mapa()` (l. 171) | "Só desenha. Lê as posições de `env.linha`, `env.coluna` e `env.carros`." |
| Carros deslizando, galinha pulando | `visual.py`, `sincronizar()` (l. 76) e `desenhar_mapa()` | "O jogo é por turnos. Guardo uma foto do antes e desenho o meio do caminho. Não afeta o aprendizado." |
| "ATROPELADA!", "CHEGOU!" | `visual.py`, `BANNERS` (l. 33), usado na l. 201 | "Mostra `env.resultado`, que é definido em `galinha_env.py`, linhas 88, 90 e 95." |
| Painel "O QUE ELA VÊ (ESTADO)" | `visual.py`, l. 233 | "São os 3 números de sim/não do estado, que vêm de `galinha_env.py`, `estado()`, linhas 55–59." |
| Painel "VALORES Q DESTE ESTADO" | `visual.py`, l. 252; o valor vem de `visualizador.py`, l. 76 / 188 (`q_visto = q[estado].copy()`) | "É a linha da tabela Q do estado atual. Verde é positivo, vermelho é negativo." |
| A ação destacada no painel | `visualizador.py`, l. 78 (`np.argmax`) no modo jogar; l. 189 (`escolher_acao`) no treino | "É a ação de maior valor Q. No treino, pode ser aleatória por causa do epsilon." |
| "EXPLORANDO" / "USANDO O QUE SABE" | `visualizador.py`, l. 216; vem do `True`/`False` de `escolher_acao` (`treinar.py`, l. 49 e 53) | "Mostra se a ação foi sorteada (epsilon) ou tirada da tabela." |
| `Q[CIMA]: 0.0 → -0.1 (recompensa -1)` | `visualizador.py`, l. 192 chama `aprender()`; texto na l. 218 | "É o valor antes e depois da equação, a linha 62 do `treinar.py`." |
| Episódio e Epsilon no placar | `visualizador.py`, `registrar()` (l. 120–124) | "Mesmo decaimento do treino normal: `epsilon = max(0.05, epsilon × 0.9995)`." |
| "Chegou (últ. 500)" e "Pontos (últ. 500)" | `visualizador.py`, l. 225 | "Média dos últimos 500 episódios do histórico." |
| "Estados vistos" | `visualizador.py`, l. 227, `len(q)` | "Quantas linhas a tabela Q já tem. No máximo 704." |
| Modo TURBO | `visualizador.py`, l. 163 (tecla T) e l. 174–175 | "Roda `rodar_episodio()` do `treinar.py` em loop por 15 ms a cada quadro, sem desenhar." |
| "Chegou X (Y%)" no modo jogar | `visualizador.py`, l. 94 | "Conta as partidas que terminaram com `env.resultado == "chegou"`." |

## "Onde está..." (perguntas gerais)

| Pergunta | Resposta |
|---|---|
| Onde estão os estados? | `galinha_env.py`, `estado()`, l. 51–60 |
| Onde estão as ações? | `galinha_env.py`, l. 6–7; aplicadas em `passo()`, l. 69–77 |
| Onde estão as recompensas? | `galinha_env.py`, l. 10–13; somadas em `passo()`, l. 66, 84, 99, 101 |
| Onde está a tabela Q? | `treinar.py`, `nova_tabela_q()`, l. 39–41 |
| Onde está o epsilon-greedy? | `treinar.py`, `escolher_acao()`, l. 45–53 |
| Onde está a equação do Q-learning? | `treinar.py`, `aprender()`, l. 57–63 (a linha central é a 62) |
| Onde estão alfa, gama e epsilon? | `treinar.py`, l. 25–29 |
| Onde é o loop de treino? | `treinar.py`, `rodar_episodio()` l. 66–77 e `treinar()` l. 122–152 |
| Onde o epsilon diminui? | `treinar.py`, l. 135 (e `visualizador.py`, l. 123) |
| Onde salva o checkpoint? | `treinar.py`, `salvar_checkpoint()` l. 81–86, chamado nas l. 144–150 |
| Onde gera a curva de aprendizado? | `treinar.py`, `salvar_grafico()` l. 104; `avaliar.py`, `grafico_treino()` l. 81 |
| Onde mede sem exploração? | `avaliar.py`, `politica_gulosa()` l. 41 e `avaliar()` l. 68 |
| Onde gera o caminho final? | `avaliar.py`, `jogar_partida()` l. 51 e `figura_caminho()` l. 141 |
| Onde a galinha treinada joga na tela? | `visualizador.py`, `jogar()` l. 26, ação na l. 78 |

## Perguntas difíceis que podem aparecer

**"Isso é off-policy? Por quê?"**
> "Sim. Na linha 60 do `treinar.py` eu uso o `max` do próximo estado, e não a ação que ela vai realmente escolher, que pode ser aleatória. Então ela aprende a melhor política mesmo enquanto explora."

**"Por que tabela e não rede neural?"**
> "Porque o estado tem só 704 possibilidades e cabe numa tabela. Com estados enormes, como imagens, precisaria de uma rede neural."

**"Ela decorou o mapa?"**
> "Não. Os carros são sorteados a cada partida, na linha 38 do `galinha_env.py`. Ela aprende regras sobre a situação local, como 'se vem carro em cima de mim, não fico parada'."

**"Como você sabe que não é sorte?"**
> "Pelo agente aleatório: nas mesmas 1.000 partidas, ele chega 0% das vezes, e o treinado chega 99%."

**"Por que o resultado do treino (77%) é diferente da avaliação (99%)?"**
> "No treino, 5% das ações são aleatórias, e uma ação aleatória na frente de um carro é fatal. Na avaliação, o epsilon é zero."

**"O que acontece se mudar as recompensas?"**
> "O comportamento muda. Ela otimiza exatamente o que as notas pedem. Se atropelar custasse pouco, ela se arriscaria. Se custasse demais, ficaria com medo de atravessar."

**"O que é o alfa?"**
> "A taxa de aprendizado. Com 0,1, cada experiência corrige 10% do erro. Muito alto deixa instável; muito baixo deixa lento."

**"Onde entra a lógica nebulosa?"** (a Questão 3 não pede, mas pode aparecer)
> "O valor Q não é um 'sim ou não' sobre a ação ser boa: é um grau contínuo. A política escolhe o maior, mas os valores mostram que várias ações podem ser 'mais ou menos boas' ao mesmo tempo. É uma ideia parecida com a de grau de pertinência."
