# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:44:13 2025

@author: bebel
"""
"verifica se é possivel ou nao realizar o movimento"
def condicao(estado):
    md, cd, bd , me, ce, be, = estado
    return (me == 0 or me >= ce) and (md == 0 or md >= cd)

def mov(estado):
    "faz os proximos movimentos possiveis"
    movimentos = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]  #n( missionarios e canibais)
    me, ce, b, md, cd, _= estado 
    proximos_estados = []
    
    for m, c in movimentos:
        if b == 1:  # b indo da esquerda p direita
            novo_me = me - m #tira o m da esq e leva p dir
            novo_ce = ce - c
            novo_md = md + m #add m e c na direita
            novo_cd = cd + c
            novo_b = 0 #qnd ta na dir barco = 0 qnd ta na esq = 1
        else:  # b  direita p esquerda
            novo_me = me + m
            novo_ce = ce + c
            novo_md = md - m
            novo_cd = cd - c
            novo_b = 1
        
        novo_estado = (novo_me, novo_ce, novo_b, novo_md, novo_cd, 1 - novo_b)
        #1-novo_b indica que o barco esta do lado oposto
        if novo_me >= 0 and novo_ce >= 0 and novo_md >= 0 and novo_cd >= 0 and condicao(novo_estado):
            proximos_estados.append(novo_estado)
    
    return proximos_estados

def bfs():
    #Busca em largura 
    estado_inicial = (3, 3, 1, 0, 0, 0)  # (margem esquerda: mionários, cibais, b, margem direita: mionários, cibais, b)
    estado_objetivo = (0, 0, 0, 3, 3, 1)
    #FIFO
    fila = [(estado_inicial, [])]  # (estado atual, caminho percorrido)
    visitados = set()
    
    while fila:
        estado, caminho = fila[0]  #retira o primeiro elemento da fila
        del fila[0] #para nao entrar em looing
        if estado in visitados:
            continue
        visitados.add(estado)
        
        if estado == estado_objetivo:
            return caminho + [estado]
        
        for proximo_estado in mov(estado):
            fila.append((proximo_estado, caminho + [estado]))
    
    return None

def dfs():
    #Busca em largura 
    estado_inicial = (3, 3, 1, 0, 0, 0)  # (margem esquerda: mionários, cibais, b, margem direita: mionários, cibais, b)
    estado_objetivo = (0, 0, 0, 3, 3, 1)
    #LIFO
    fila = [(estado_inicial, [])]  # (estado atual, caminho percorrido)
    visitados = set()
    
    while fila:
        estado, caminho = fila[-1]  #PEGA O ULTIMO ELEMENTO DA FILA
        del fila[-1] #para nao entrar em looing
        if estado in visitados:
            continue
        visitados.add(estado)
        
        if estado == estado_objetivo:
            return caminho + [estado]
        
        for proximo_estado in mov(estado):
            fila.append((proximo_estado, caminho + [estado]))
    
    return None
def resposta(solucao, metodo):
    #tipo de busca usado
    print(f"\nSolução encontrada com {metodo}:")

    if not solucao:
        print("Sem solução")
        return

    for passo, estado in enumerate(solucao):
        me, ce, b_esq, md, cd, b_dir = estado
        print(f"Estado {passo}: ({me}, {ce}, {'Barco' if b_esq else '_'},{md}, {cd}, {'B' if b_dir else '_'})" )

# BSF
solucao = bfs()
resposta(solucao, "BFS")
#DSF 
solucao = dfs()
resposta(solucao, "DFS")

