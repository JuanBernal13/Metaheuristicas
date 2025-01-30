##################################################################
# EMPAQUE DE PIEZAS 3D EN UNA CAJA 5x5x5 CON VISUALIZACIÓN
##################################################################

# Asegurarse de tener PuLP instalado:
# !pip install pulp matplotlib numpy

import pulp
import itertools
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necesario para proyecciones 3D
import numpy as np

# --------------------------------------------------------------
# 1. DATOS DEL PROBLEMA
# --------------------------------------------------------------
L, W, H = 5, 5, 5  # Dimensiones de la caja (x, y, z)

# Tenemos 12 piezas en total:
#  - 6 de tipo A (dimensiones 1x2x4)
#  - 6 de tipo B (dimensiones 2x2x3)
piezas = []
for i in range(6):
    piezas.append(('A', i))  # 6 de tipo A
for i in range(6):
    piezas.append(('B', i))  # 6 de tipo B

# Función para obtener orientaciones "distintas" de un paralelepípedo (a,b,c)
def orientaciones(a, b, c):
    """
    Devuelve una lista (sin repeticiones) de tuplas (dx, dy, dz) 
    con las posibles orientaciones del bloque (a,b,c).
    """
    from itertools import permutations
    return list(set(permutations((a,b,c))))

# Orientaciones posibles según el tipo
orientaciones_A = orientaciones(1, 2, 4)  # se esperan 6 orientaciones
orientaciones_B = orientaciones(2, 2, 3)  # se esperan 3 orientaciones (dos lados iguales)

# --------------------------------------------------------------
# 2. GENERAR LOS "PLACEMENTS" FACTIBLES (ORIENTACIÓN + POSICIÓN)
# --------------------------------------------------------------
placements = {}  # placements[p_idx] = lista de (o_id, x, y, z) factibles
                # donde p_idx es el índice (0..11) en la lista "piezas"

for p_idx, (tipo, _) in enumerate(piezas):
    if tipo == 'A':
        lista_orient = orientaciones_A
    else:
        lista_orient = orientaciones_B
    
    placements_p = []
    for o_id, (dx, dy, dz) in enumerate(lista_orient):
        # Explorar todas las posiciones (x,y,z) donde quepa
        max_x = L - dx
        max_y = W - dy
        max_z = H - dz
        
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                for z in range(max_z + 1):
                    placements_p.append((o_id, x, y, z))
    
    placements[p_idx] = placements_p

# --------------------------------------------------------------
# 3. FORMULACIÓN DEL MODELO EN PuLP
# --------------------------------------------------------------
modelo = pulp.LpProblem("3DBoxPacking", pulp.LpMinimize)

# Variables de decisión X_{p, (o_id,x,y,z)} = 1 si la pieza p 
# se ubica en esa orientación y posición
X = {}
for p_idx in placements:
    for pl in placements[p_idx]:
        X[(p_idx, pl)] = pulp.LpVariable(
            f"X_{p_idx}_{pl}",
            cat=pulp.LpBinary
        )

# Función objetivo: sólo factibilidad, minimizamos 0
modelo.setObjective(pulp.lpSum([]))

# 3.1. Restricción: cada pieza se coloca exactamente en un placement
for p_idx in placements:
    modelo.addConstraint(
        pulp.lpSum(X[(p_idx, pl)] for pl in placements[p_idx]) == 1,
        name=f"colocar_una_vez_pieza_{p_idx}"
    )

# 3.2. Restricción de no solapamiento
# Para cada celda (cx,cy,cz), la suma de las piezas que la ocupan <= 1
for cx in range(L):
    for cy in range(W):
        for cz in range(H):
            # Determinamos qué placements ocupan esta celda
            ocupar_celda = []
            for p_idx, (tipo, _) in enumerate(piezas):
                if tipo == 'A':
                    dims_o = orientaciones_A
                else:
                    dims_o = orientaciones_B
                
                for pl in placements[p_idx]:
                    o_id, px, py, pz = pl
                    dx, dy, dz = dims_o[o_id]
                    
                    # Si (cx,cy,cz) está dentro del bloque [px,px+dx) x [py,py+dy) x [pz,pz+dz)
                    if (px <= cx < px+dx) and \
                       (py <= cy < py+dy) and \
                       (pz <= cz < pz+dz):
                        ocupar_celda.append(X[(p_idx, pl)])
            
            if ocupar_celda:
                modelo.addConstraint(
                    pulp.lpSum(ocupar_celda) <= 1,
                    name=f"no_solape_{cx}{cy}{cz}"
                )

# --------------------------------------------------------------
# 4. RESOLVER EL MODELO
# --------------------------------------------------------------
resultado = modelo.solve(pulp.PULP_CBC_CMD(msg=0))

print("Estatus de la solución:", pulp.LpStatus[resultado])
if pulp.LpStatus[resultado] not in ['Optimal', 'Feasible']:
    print("No se encontró solución factible.")
else:
    print("¡Solución hallada!")
    
    # ----------------------------------------------------------
    # 5. RECUPERAR SOLUCIÓN Y MOSTRARLA
    # ----------------------------------------------------------
    # Para cada pieza, localizamos la tupla (o_id, x, y, z) con X=1
    # y guardamos esa info para poder graficar
    solucion_placements = {}  # p_idx -> (dx,dy,dz, px,py,pz)
    
    for p_idx, (tipo, _) in enumerate(piezas):
        if tipo == 'A':
            dims_o = orientaciones_A
        else:
            dims_o = orientaciones_B
        
        for pl in placements[p_idx]:
            if pulp.value(X[(p_idx, pl)]) > 0.5:
                o_id, px, py, pz = pl
                dx, dy, dz = dims_o[o_id]
                solucion_placements[p_idx] = (dx, dy, dz, px, py, pz)
                break
    
    # Imprimimos en texto la ubicación de cada pieza
    for p_idx, (dx, dy, dz, px, py, pz) in solucion_placements.items():
        print(f"Pieza {p_idx} -> (dim {dx}x{dy}x{dz}), posicion ({px},{py},{pz})")
    
    # ----------------------------------------------------------
    # 6. DIBUJAR EN 3D LA CAJA Y LAS PIEZAS
    # ----------------------------------------------------------
    
    # Configuramos un "canvas" 3D con matplotlib
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ejes: la caja es de 0..5 en x, 0..5 en y, 0..5 en z
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Empaque en caja 5x5x5")
    
    # Generamos colores aleatorios para cada pieza
    random.seed(42)  # para reproducibilidad
    colores = []
    for i in range(len(piezas)):
        color = (random.random(), random.random(), random.random())
        colores.append(color)
    
    # Dibujamos cada pieza como un cuboide (bar3d)
    for p_idx, (dx, dy, dz, px, py, pz) in solucion_placements.items():
        c = colores[p_idx]
        # bar3d(x, y, z, dx, dy, dz) dibuja un cubo en la pos (x,y,z)
        # con dimensiones (dx, dy, dz).
        ax.bar3d(px,    # base x
                  py,    # base y
                  pz,    # base z
                  dx, dy, dz,
                  color=c, alpha=0.6, edgecolor='k')
        
        # Podemos añadir un texto con el índice de la pieza
        ax.text(px + dx/2, py + dy/2, pz + dz/2, 
                f"P{p_idx}", 
                color='black', 
                ha='center', va='center')
    
    plt.show()