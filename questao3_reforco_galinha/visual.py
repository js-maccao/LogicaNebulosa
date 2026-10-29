"""Parte VISUAL do jogo: so desenha. Nao tem regra do jogo nem aprendizado."""
import math

import pygame

from galinha_env import NOMES_ACOES

TAM = 60                 # tamanho de cada casa em pixels
LARGURA_PAINEL = 320

COR_ASFALTO = [(56, 58, 66), (52, 54, 62)]
COR_CALCADA = (172, 170, 162)
COR_CALCADA_LINHA = (150, 148, 140)
COR_GRAMA = (92, 150, 90)
COR_DIVISORIA = (120, 122, 130)
COR_PAINEL = (24, 26, 33)
COR_TITULO = (255, 255, 255)
COR_TEXTO = (222, 226, 236)
COR_SUAVE = (130, 140, 162)
COR_SIM = (232, 84, 84)
COR_NAO = (66, 72, 88)
COR_POSITIVO = (88, 190, 120)
COR_NEGATIVO = (224, 92, 92)
COR_SELECAO = (44, 52, 74)
COR_VIDRO = (38, 52, 72)

# Carros que vao para a DIREITA = tons quentes. Para a ESQUERDA = tons frios.
PALETA_DIREITA = [(214, 69, 65), (232, 119, 34), (226, 172, 40)]
PALETA_ESQUERDA = [(52, 120, 210), (40, 170, 190), (70, 96, 196)]
SETA_DIREITA = (88, 66, 68)
SETA_ESQUERDA = (60, 72, 98)

BANNERS = {
    "atropelada": ("ATROPELADA!", (232, 84, 84)),
    "chegou": ("CHEGOU!", (88, 190, 120)),
    "tempo": ("TEMPO ESGOTADO", (240, 190, 70)),
    "turbo": ("TURBO", (240, 190, 70)),
}


def fonte(tamanho, negrito=False):
    return pygame.font.SysFont("segoeui,arial,helvetica", tamanho, bold=negrito)


def escurecer(cor, f=0.65):
    return tuple(int(v * f) for v in cor)


def clarear(cor, f=0.35):
    return tuple(int(v + (255 - v) * f) for v in cor)


class Cena:
    def __init__(self, env, titulo_janela):
        self.env = env
        self.larg_mapa = env.colunas * TAM
        self.altura = env.linhas * TAM
        pygame.init()
        self.tela = pygame.display.set_mode((self.larg_mapa + LARGURA_PAINEL, self.altura))
        pygame.display.set_caption(titulo_janela)
        self.f_titulo = fonte(28, True)
        self.f_secao = fonte(15, True)
        self.f_texto = fonte(19)
        self.f_negrito = fonte(19, True)
        self.f_pequena = fonte(15)
        self.f_banner = fonte(54, True)
        self.sprites = {}
        self.sombra = pygame.Surface((38, 14), pygame.SRCALPHA)
        pygame.draw.ellipse(self.sombra, (0, 0, 0, 80), (0, 0, 38, 14))
        self.sincronizar()

    # ------------------------------------------------------------------
    # ANIMACAO: o jogo anda por turnos, entao guardamos uma "foto do antes"
    # (posicoes antes do passo) e desenhamos o meio do caminho (t de 0 a 1).
    # ------------------------------------------------------------------
    def sincronizar(self):
        """Tira a foto do ANTES. Chame logo antes de env.passo() (e apos reset)."""
        self.pos_ant = (self.env.linha, self.env.coluna)
        self.carros_ant = {l: list(cs) for l, cs in self.env.carros.items()}

    # ------------------------------------------------------------------
    # MAPA
    # ------------------------------------------------------------------
    def _y(self, linha):
        return (self.env.linhas - 1 - linha) * TAM   # linha 0 fica embaixo

    def _fundo(self):
        env, tela, W = self.env, self.tela, self.larg_mapa
        for l in range(env.linhas):
            y = self._y(l)
            if l == 0:                                  # calcada de inicio
                pygame.draw.rect(tela, COR_CALCADA, (0, y, W, TAM))
                for x in range(0, W, TAM):
                    pygame.draw.line(tela, COR_CALCADA_LINHA, (x, y), (x, y + TAM))
                pygame.draw.line(tela, COR_CALCADA_LINHA, (0, y + TAM // 2), (W, y + TAM // 2))
                tela.blit(self.f_pequena.render("INÍCIO", True, (110, 108, 100)), (6, y + TAM - 20))
            elif l == env.linhas - 1:                   # grama de chegada
                pygame.draw.rect(tela, COR_GRAMA, (0, y, W, TAM))
                for i in range(W // 10):                # faixa quadriculada
                    for j in range(2):
                        if (i + j) % 2 == 0:
                            pygame.draw.rect(tela, (245, 245, 245), (i * 10, y + TAM // 2 - 10 + j * 10, 10, 10))
                tela.blit(self.f_pequena.render("CHEGADA", True, (235, 245, 235)), (6, y + 4))
            else:                                       # pista
                pygame.draw.rect(tela, COR_ASFALTO[l % 2], (0, y, W, TAM))
                d = env.direcao(l)
                cor_seta = SETA_DIREITA if d == 1 else SETA_ESQUERDA
                for k in range(1, env.colunas, 3):      # setinhas da direcao da pista
                    cx, cy = k * TAM + TAM // 2, y + TAM // 2
                    pts = [(cx - 6 * d, cy - 8), (cx + 4 * d, cy), (cx - 6 * d, cy + 8)]
                    pygame.draw.lines(tela, cor_seta, False, pts, 3)
        # divisorias entre pistas
        for l in range(1, env.linhas - 1):
            y = self._y(l)
            if l == env.linhas - 2:                     # borda com a grama: linha continua
                pygame.draw.line(tela, COR_DIVISORIA, (0, y), (W, y), 2)
            else:                                       # entre pistas: tracejada
                for x in range(0, W, 32):
                    pygame.draw.rect(tela, COR_DIVISORIA, (x + 6, y - 1, 18, 2))
        pygame.draw.line(tela, COR_DIVISORIA, (0, self._y(1) + TAM), (W, self._y(1) + TAM), 2)

    def _sprite_carro(self, cor, direcao):
        chave = (cor, direcao)
        if chave not in self.sprites:
            corpo = pygame.Surface((52, 40), pygame.SRCALPHA)
            for rx in (9, 35):                          # rodas
                pygame.draw.rect(corpo, (18, 18, 22), (rx, 1, 10, 6), border_radius=2)
                pygame.draw.rect(corpo, (18, 18, 22), (rx, 33, 10, 6), border_radius=2)
            pygame.draw.rect(corpo, cor, (0, 4, 52, 32), border_radius=9)
            pygame.draw.rect(corpo, escurecer(cor), (0, 4, 52, 32), width=2, border_radius=9)
            pygame.draw.rect(corpo, clarear(cor), (17, 9, 18, 22), border_radius=5)   # teto
            pygame.draw.rect(corpo, COR_VIDRO, (37, 10, 7, 20), border_radius=3)      # para-brisa
            pygame.draw.rect(corpo, COR_VIDRO, (11, 10, 5, 20), border_radius=3)      # vidro de tras
            for fy in (8, 26):
                pygame.draw.rect(corpo, (255, 240, 150), (49, fy, 3, 6))              # farois
                pygame.draw.rect(corpo, (150, 25, 25), (0, fy, 3, 6))                 # lanternas
            if direcao == -1:                           # espelha: frente do carro para a esquerda
                corpo = pygame.transform.flip(corpo, True, False)
            sprite = pygame.Surface((60, 46), pygame.SRCALPHA)
            pygame.draw.rect(sprite, (0, 0, 0, 70), (5, 9, 52, 32), border_radius=9)  # sombra
            sprite.blit(corpo, (2, 2))
            self.sprites[chave] = sprite
        return self.sprites[chave]

    def _galinha(self, cx, cy, pulo, morta):
        tela = self.tela
        if morta:                                       # achatada, olhos X, penas
            pygame.draw.ellipse(tela, (232, 232, 226), (cx - 21, cy - 9, 42, 20))
            pygame.draw.ellipse(tela, (190, 190, 184), (cx - 21, cy - 9, 42, 20), 2)
            pygame.draw.ellipse(tela, (222, 60, 60), (cx - 6, cy - 12, 12, 7))
            for ox in (-7, 7):
                pygame.draw.line(tela, (30, 30, 30), (cx + ox - 3, cy - 4), (cx + ox + 3, cy + 2), 2)
                pygame.draw.line(tela, (30, 30, 30), (cx + ox - 3, cy + 2), (cx + ox + 3, cy - 4), 2)
            for ox, oy in [(-27, -10), (25, -12), (-23, 12), (27, 10), (2, -22)]:
                pygame.draw.ellipse(tela, (250, 250, 250), (cx + ox - 5, cy + oy - 2, 10, 5))
            return
        y = cy - pulo
        for ox, oy, r in [(-5, -17, 4), (0, -20, 5), (5, -17, 4)]:                    # crista
            pygame.draw.circle(tela, (222, 60, 60), (cx + ox, y + oy), r)
        pygame.draw.circle(tela, (248, 248, 244), (cx, y), 16)                         # corpo
        pygame.draw.circle(tela, (196, 196, 190), (cx, y), 16, 2)
        pygame.draw.ellipse(tela, (228, 228, 222), (cx - 21, y - 2, 9, 15))            # asas
        pygame.draw.ellipse(tela, (228, 228, 222), (cx + 12, y - 2, 9, 15))
        for ox in (-6, 6):                                                              # olhos
            pygame.draw.circle(tela, (25, 25, 25), (cx + ox, y - 4), 3)
            pygame.draw.circle(tela, (255, 255, 255), (cx + ox + 1, y - 5), 1)
        pygame.draw.polygon(tela, (242, 150, 40), [(cx - 4, y + 1), (cx + 4, y + 1), (cx, y + 8)])  # bico
        for ox in (-6, 6):                                                              # pes
            pygame.draw.line(tela, (242, 150, 40), (cx + ox, y + 16), (cx + ox, y + 21), 2)

    def desenhar_mapa(self, t, resultado=None):
        """t = progresso do passo atual (0 = antes, 1 = depois)."""
        env, tela = self.env, self.tela
        tela.set_clip(pygame.Rect(0, 0, self.larg_mapa, self.altura))
        self._fundo()

        # galinha: posicao interpolada + pulinho
        l0, c0 = self.pos_ant
        cx = (c0 + (env.coluna - c0) * t) * TAM + TAM / 2
        cy = (env.linhas - 1 - (l0 + (env.linha - l0) * t)) * TAM + TAM / 2
        pulo = math.sin(math.pi * t) * 10 if (l0, c0) != (env.linha, env.coluna) else 0
        morta = resultado == "atropelada" and t >= 1
        tela.blit(self.sombra, (cx - 19, cy + 10))

        # carros: deslizam da posicao antiga para a nova (com wrap-around)
        for l, colunas in env.carros.items():
            y = self._y(l)
            d = env.direcao(l)
            paleta = PALETA_DIREITA if d == 1 else PALETA_ESQUERDA
            for i, c1 in enumerate(colunas):
                c0 = self.carros_ant[l][i]
                x_cel = c0 + d * t if c1 != c0 else c0
                px = (x_cel % env.colunas) * TAM
                sprite = self._sprite_carro(paleta[(l + i) % len(paleta)], d)
                for dx in (0, -self.larg_mapa):         # copia do outro lado da borda
                    tela.blit(sprite, (px + dx, y + (TAM - 46) // 2))

        self._galinha(cx, cy, pulo, morta)              # por cima dos carros

        if resultado and t >= 1:
            texto, cor = BANNERS[resultado]
            veu = pygame.Surface((self.larg_mapa, self.altura), pygame.SRCALPHA)
            veu.fill((0, 0, 0, 110))
            tela.blit(veu, (0, 0))
            img = self.f_banner.render(texto, True, cor)
            tela.blit(img, img.get_rect(center=(self.larg_mapa // 2, self.altura // 2)))
        tela.set_clip(None)

    # ------------------------------------------------------------------
    # PAINEL LATERAL
    # ------------------------------------------------------------------
    def _secao(self, titulo, x, y):
        self.tela.blit(self.f_secao.render(titulo, True, COR_SUAVE), (x, y))
        fim = self.larg_mapa + LARGURA_PAINEL - 18
        pygame.draw.line(self.tela, (48, 54, 70), (x, y + 20), (fim, y + 20))
        return y + 28

    def _direita(self, texto, fonte_, cor, y):
        img = fonte_.render(texto, True, cor)
        self.tela.blit(img, (self.larg_mapa + LARGURA_PAINEL - 18 - img.get_width(), y))

    def desenhar_painel(self, titulo, subtitulo, estado, q_vals, acao, placar, ajuda, extra=()):
        tela, x0 = self.tela, self.larg_mapa
        x = x0 + 18
        pygame.draw.rect(tela, COR_PAINEL, (x0, 0, LARGURA_PAINEL, self.altura))
        pygame.draw.line(tela, (60, 66, 82), (x0, 0), (x0, self.altura), 2)

        tela.blit(self.f_titulo.render(titulo, True, COR_TITULO), (x, 14))
        tela.blit(self.f_pequena.render(subtitulo, True, COR_SUAVE), (x, 48))
        y = 76

        # --- o que ela ve (o ESTADO) ---
        y = self._secao("O QUE ELA VÊ (ESTADO)", x, y)
        if estado is None:
            tela.blit(self.f_texto.render("-", True, COR_TEXTO), (x, y))
            y += 22 * 4
        else:
            linha, coluna, fo, fv, av = estado
            tela.blit(self.f_texto.render(f"Posição: linha {linha}, coluna {coluna}", True, COR_TEXTO), (x, y))
            y += 25
            for rotulo, v in [("Carro logo acima agora", fo),
                              ("Carro chegando acima", fv),
                              ("Carro chegando em mim", av)]:
                pygame.draw.circle(tela, COR_SIM if v else COR_NAO, (x + 6, y + 11), 6)
                tela.blit(self.f_texto.render(rotulo, True, COR_TEXTO), (x + 20, y))
                self._direita("SIM" if v else "não", self.f_negrito if v else self.f_texto,
                              COR_SIM if v else COR_SUAVE, y)
                y += 23
        y += 10

        # --- valores Q (a linha da TABELA Q daquele estado) ---
        y = self._secao("VALORES Q DESTE ESTADO", x, y)
        escala = max(1.0, max(abs(v) for v in q_vals)) if q_vals is not None else 1.0
        xz = x + 142                                    # linha do zero
        for i, nome in enumerate(NOMES_ACOES):
            if acao == i:
                pygame.draw.rect(tela, COR_SELECAO, (x - 8, y - 2, LARGURA_PAINEL - 20, 26), border_radius=6)
            cor_nome = COR_TITULO if acao == i else COR_SUAVE
            tela.blit((self.f_negrito if acao == i else self.f_texto).render(nome, True, cor_nome), (x, y))
            if q_vals is not None:
                v = float(q_vals[i])
                larg = int(62 * abs(v) / escala)
                if larg:
                    barra = (xz, y + 4, larg, 14) if v >= 0 else (xz - larg, y + 4, larg, 14)
                    pygame.draw.rect(tela, COR_POSITIVO if v >= 0 else COR_NEGATIVO, barra, border_radius=3)
                self._direita(f"{v:.1f}", self.f_pequena, COR_TEXTO, y + 2)
            pygame.draw.line(tela, (90, 98, 120), (xz, y + 2), (xz, y + 20))
            y += 27
        y += 4
        for texto, cor in extra:                        # linhas extras (usadas no treino ao vivo)
            tela.blit(self.f_pequena.render(texto, True, cor), (x, y))
            y += 19
        y += 6

        # --- placar ---
        y = self._secao("PLACAR", x, y)
        for rotulo, valor in placar:
            tela.blit(self.f_texto.render(rotulo, True, COR_SUAVE), (x, y))
            self._direita(str(valor), self.f_negrito, COR_TEXTO, y)
            y += 24
        y += 8

        # --- legenda das cores dos carros ---
        pygame.draw.rect(tela, PALETA_DIREITA[0], (x, y + 3, 14, 10), border_radius=3)
        tela.blit(self.f_pequena.render("vai p/ direita", True, COR_TEXTO), (x + 20, y))
        pygame.draw.rect(tela, PALETA_ESQUERDA[0], (x + 140, y + 3, 14, 10), border_radius=3)
        tela.blit(self.f_pequena.render("vai p/ esquerda", True, COR_TEXTO), (x + 160, y))

        # --- ajuda (teclas) no rodape ---
        y = self.altura - 14 - 18 * len(ajuda)
        for linha_txt in ajuda:
            tela.blit(self.f_pequena.render(linha_txt, True, COR_SUAVE), (x, y))
            y += 18
