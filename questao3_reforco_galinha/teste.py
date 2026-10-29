from treinar import *

q = nova_tabela_q()
estado = (3, 2, 0, 1, 0)
print("tabela vazia:", q[estado])

# fingindo que ela ja aprendeu algo neste estado
q[estado][0] = -40   # CIMA e pessimo
q[estado][2] = 5     # ESQUERDA e boa
print("depois de 'aprender':", q[estado])

# epsilon = 0: sempre usa o que sabe
escolhas = [escolher_acao(q, estado, 0.0) for _ in range(1000)]
print("epsilon 0   ->", {NOMES_ACOES[a]: escolhas.count(a) for a in range(N_ACOES)})

# epsilon = 1: sempre aleatoria
escolhas = [escolher_acao(q, estado, 1.0) for _ in range(1000)]
print("epsilon 1   ->", {NOMES_ACOES[a]: escolhas.count(a) for a in range(N_ACOES)})
