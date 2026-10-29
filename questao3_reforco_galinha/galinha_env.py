"""Ambiente da galinha atravessando a rua.
Este arquivo so contem as REGRAS DO JOGO. Nao existe aprendizado aqui.
"""
import random

CIMA, BAIXO, ESQUERDA, DIREITA, PARADA = 0, 1, 2, 3, 4
NOMES_ACOES = ["CIMA", "BAIXO", "ESQUERDA", "DIREITA", "PARADA"]

# --- Recompensas ---
R_PASSO = -1          # custo de cada passo (incentiva ser rapida)
R_AVANCO = 5          # bonus ao chegar numa linha mais alta do que ja esteve
R_ATROPELADA = -50    # atropelada (fim da partida)
R_OBJETIVO = 100      # chegou do outro lado (fim da partida)


class TravessiaGalinha:
    def __init__(self, linhas=11, colunas=8, carros_por_pista=2, max_passos=100):
        self.linhas = linhas
        self.colunas = colunas
        self.carros_por_pista = carros_por_pista
        self.max_passos = max_passos
        self.reset()


    def direcao(self, linha):
        """Pistas pares vao para a direita (+1), impares para a esquerda (-1)."""
        return 1 if linha % 2 == 0 else -1

    def reset(self):
        """Comeca uma partida nova: galinha na linha 0, no meio, carros sorteados."""
        self.linha = 0
        self.coluna = self.colunas // 2
        self.linha_max = 0   # linha mais alta que ela ja alcancou
        self.passos = 0
        self.resultado = None  # None = jogando | "atropelada" | "chegou"
        # carros[linha] = lista com a coluna de cada carro daquela pista
        self.carros = {
            l: random.sample(range(self.colunas), self.carros_por_pista)
            for l in range(1, self.linhas - 1)
        }
        return self.estado()

    def tem_carro(self, linha, coluna):
        return coluna in self.carros.get(linha, [])

    def mover_carros(self):
        for l in self.carros:
            d = self.direcao(l)
            self.carros[l] = [(c + d) % self.colunas for c in self.carros[l]]

    def estado(self):
        """Estado pequeno: (linha, coluna, frente_ocupada, frente_vindo, aqui_vindo)."""
        l, c = self.linha, self.coluna
        # carro na casa logo acima, agora
        frente_ocupada = int(self.tem_carro(l + 1, c))
        # carro na pista de cima que chega nessa casa no proximo turno
        frente_vindo = int(self.tem_carro(l + 1, (c - self.direcao(l + 1)) % self.colunas))
        # carro na minha pista que chega na minha casa no proximo turno
        aqui_vindo = int(self.tem_carro(l, (c - self.direcao(l)) % self.colunas))
        return (l, c, frente_ocupada, frente_vindo, aqui_vindo)


    def passo(self, acao):
        """Um turno. Devolve (novo_estado, recompensa, terminou)."""
        self.passos += 1
        recompensa = R_PASSO

        # 1) galinha se move
        if acao == CIMA:
            self.linha += 1
        elif acao == BAIXO:
            self.linha -= 1
        elif acao == ESQUERDA:
            self.coluna -= 1
        elif acao == DIREITA:
            self.coluna += 1
        # PARADA: nao faz nada
        self.linha = max(0, min(self.linhas - 1, self.linha))
        self.coluna = max(0, min(self.colunas - 1, self.coluna))

        # bonus por chegar numa linha nova (mais alta que todas as anteriores)
        if self.linha > self.linha_max:
            self.linha_max = self.linha
            recompensa += R_AVANCO

        # 2) bateu num carro? chegou do outro lado?
        if self.tem_carro(self.linha, self.coluna):
            self.resultado = "atropelada"
        elif self.linha == self.linhas - 1:
            self.resultado = "chegou"
        else:
            # 3) os carros andam; algum caiu em cima dela?
            self.mover_carros()
            if self.tem_carro(self.linha, self.coluna):
                self.resultado = "atropelada"

        # fim de partida: soma a nota final
        if self.resultado == "atropelada":
            recompensa += R_ATROPELADA
        elif self.resultado == "chegou":
            recompensa += R_OBJETIVO
        elif self.passos >= self.max_passos:
            self.resultado = "tempo"

        terminou = self.resultado is not None
        return self.estado(), recompensa, terminou


    def desenhar(self):
        """Desenha o mapa no terminal (linha de cima = chegada)."""
        for l in range(self.linhas - 1, -1, -1):
            texto = ""
            for c in range(self.colunas):
                if l == self.linha and c == self.coluna:
                    texto += " G "
                elif self.tem_carro(l, c):
                    texto += " C "
                else:
                    texto += " . "
            print(f"{l:2d} {texto}")
        print()
