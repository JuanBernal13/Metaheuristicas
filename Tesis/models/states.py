##################################################################
# EMPAQUE DE PIEZAS 3D EN UNA CAJA (CON EVITAR "FLOTACIÓN")
# Versión usando PuLP y matplotlib
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
# Puedes cambiar estas dimensiones y probar, por ejemplo 6x6x5, etc.
L, W, H = 5, 5, 5  # Dimensiones de la caja (x, y, z)

# Tenemos 12 piezas en total:
#  - 6 de tipo A (dimensiones 1x2x4)
#  - 6 de tipo B (dimensiones 2x2x3)
piezas = []
for i in range(6):
    piezas.append(('A', i))  # 6 de tipo A
for i in range(6):
    piezas.append(('B', i))  # 6 de tipo B

def orientaciones(a, b, c):
    """
    Devuelve una lista (sin repeticiones) de tuplas (dx, dy, dz) 
    con las posibles orientaciones del bloque (a,b,c).
    """
    from itertools import permutations
    return list(set(permutations((a, b, c))))

# Orientaciones posibles según el tipo
orientaciones_A = orientaciones(1, 2, 4)  # se esperan 6 orientaciones
orientaciones_B = orientaciones(2, 2, 3)  # se esperan 3 orientaciones (2 lados iguales)

# --------------------------------------------------------------
# 2. GENERAR LOS "PLACEMENTS" FACTIBLES (ORIENTACIÓN + POSICIÓN)
# --------------------------------------------------------------
placements = {}  # placements[p_idx] = lista de (o_id, x, y, z) factibles

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
# 2.b) DICCIONARIO PARA SABER QUÉ PLACEMENTS OCUPAN CADA CELDA
#     Esto nos ayuda tanto para la restricción de "no solapamiento"
#     como para la nueva restricción de "no flotación".
# --------------------------------------------------------------
cover_cell = {}  
for cx in range(L):
    for cy in range(W):
        for cz in range(H):
            cover_cell[(cx, cy, cz)] = []
            
# Rellenamos cover_cell[(cx,cy,cz)] con los placements que ocupan esa celda
for p_idx, (tipo, _) in enumerate(piezas):
    if tipo == 'A':
        dims_o = orientaciones_A
    else:
        dims_o = orientaciones_B
    
    for pl in placements[p_idx]:
        o_id, px, py, pz = pl
        dx, dy, dz = dims_o[o_id]
        
        # Recorremos las celdas que esta pieza ocuparía
        for cx in range(px, px + dx):
            for cy in range(py, py + dy):
                for cz in range(pz, pz + dz):
                    cover_cell[(cx, cy, cz)].append( (p_idx, pl) )

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

# 3.2. Restricción de no solapamiento:
#      Para cada celda (cx,cy,cz), la suma de las piezas que la ocupan <= 1
for cx in range(L):
    for cy in range(W):
        for cz in range(H):
            if cover_cell[(cx, cy, cz)]:
                modelo.addConstraint(
                    pulp.lpSum(X[(p_idx, pl)] 
                               for (p_idx, pl) in cover_cell[(cx, cy, cz)]) <= 1,
                    name=f"no_solape_{cx}_{cy}_{cz}"
                )

# 3.3. Restricción de "no flotación":
#     - Si una pieza ocupa celdas con z >= 1, cada celda en su “base”
#       (z mismo) debe tener algo justo debajo (z-1).
#     - Para evitar nombres duplicados, incluimos "o_id" en el nombre.
for p_idx, (tipo, _) in enumerate(piezas):
    # dims_o son las orientaciones posibles (dx,dy,dz)
    if tipo == 'A':
        dims_o = orientaciones_A
    else:
        dims_o = orientaciones_B
    
    for pl in placements[p_idx]:
        o_id, px, py, pz = pl
        dx, dy, dz = dims_o[o_id]
        
        # Sólo nos interesa si pz > 0 (piezas no en el piso)
        if pz > 0:
            # Recorremos la "base" de la pieza
            for bx in range(px, px + dx):
                for by in range(py, py + dy):
                    # La celda justo debajo es (bx, by, pz-1)
                    # Forzamos la restricción
                    nombre_restric = f"soporte_{p_idx}_{o_id}_{px}_{py}_{pz}_{bx}_{by}"
                    modelo.addConstraint(
                        pulp.lpSum(X[(q_idx, pl2)] 
                                   for (q_idx, pl2) in cover_cell[(bx, by, pz-1)]) 
                        >= X[(p_idx, pl)],
                        name=nombre_restric
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
        print(f"Pieza {p_idx} -> dim({dx}x{dy}x{dz}), pos=({px},{py},{pz})")
    
    # ----------------------------------------------------------
    # 6. DIBUJAR EN 3D LA CAJA Y LAS PIEZAS
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Empaque en caja {L}x{W}x{H} (sin flotación)")
    
    # Generamos colores aleatorios para cada pieza
    random.seed(42)  # reproducibilidad
    colores = []
    for i in range(len(piezas)):
        color = (random.random(), random.random(), random.random())
        colores.append(color)
    
    # Dibujamos cada pieza como un cuboide (bar3d)
    for p_idx, (dx, dy, dz, px, py, pz) in solucion_placements.items():
        c = colores[p_idx]
        ax.bar3d(px, py, pz, dx, dy, dz, 
                 color=c, alpha=0.6, edgecolor='k')
        
        # Texto con índice de la pieza
        ax.text(px + dx/2, py + dy/2, pz + dz/2, 
                f"P{p_idx}", 
                color='black', 
                ha='center', va='center')
    
    plt.show()
