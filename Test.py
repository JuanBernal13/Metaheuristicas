import pulp
from itertools import product, permutations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# 1. DEFINICIÓN DEL PROBLEMA Y DATOS BÁSICOS
# =============================================================================

# Creamos el problema de tipo MIP (Mixed Integer Problem).
problem = pulp.LpProblem("3D_Packing", pulp.LpMinimize)

# Dimensiones de la caja
L, W, H = 5, 5, 5

# Piezas: 6 tipo A (1x2x4), 6 tipo B (2x2x3)
# Definimos un identificador para cada una de las 12 piezas
pieces = list(range(12))  # 0..5 -> tipo A, 6..11 -> tipo B

# Para referencia rápida, definimos sus dimensiones "base"
# (largo, ancho, alto) antes de permutar.
dims_A = (1, 2, 4)
dims_B = (2, 2, 3)

# Cantidad de piezas de cada tipo
nA = 6
nB = 6

# =============================================================================
# 2. GENERACIÓN DE TODAS LAS COLOCACIONES POSIBLES PL(p)
# =============================================================================

def unique_permutations(dimensions):
    """
    Devuelve todas las permutaciones únicas de la tupla 'dimensions'.
    (Por ejemplo, (1,2,4) tiene 6 permutaciones, algunas pueden repetirse 
     si hay dimensiones iguales, aquí filtramos duplicados usando set).
    """
    return set(permutations(dimensions))

def generate_all_placements(dimensions, box_dim):
    """
    Genera todas las colocaciones factibles (orientación + posición) 
    dentro de la caja para una pieza con 'dimensions'.
    
    Retorna una lista de tuplas del tipo: 
      ( (lx, ly, lz), (x0, y0, z0) )
    donde (lx, ly, lz) es la orientación,
          (x0, y0, z0) es la posición de la esquina "inferior" de la pieza.
    """
    Lb, Wb, Hb = box_dim
    placements = []
    
    for (lx, ly, lz) in unique_permutations(dimensions):
        # Rango de posiciones factibles
        # Si la pieza es de tamaño (lx, ly, lz),
        # x0 puede ir de 0 a Lb-lx, etc.
        for x0 in range(Lb - lx + 1):
            for y0 in range(Wb - ly + 1):
                for z0 in range(Hb - lz + 1):
                    placements.append(((lx, ly, lz), (x0, y0, z0)))
    
    return placements

# Construimos un diccionario PL(p) que asocia cada pieza p
# con la lista de sus colocaciones factibles.
PL = dict()
for p in pieces:
    if p < nA:
        # Pieza tipo A
        PL[p] = generate_all_placements(dims_A, (L, W, H))
    else:
        # Pieza tipo B
        PL[p] = generate_all_placements(dims_B, (L, W, H))

# =============================================================================
# 3. VARIABLES DE DECISIÓN
# =============================================================================
# X_{p,pl} = 1 si la pieza p se coloca con la colocación pl, 0 si no.
# Para indexar facilmente, haremos un diccionario de variables:
X = {}
for p in pieces:
    for pl_idx, pl in enumerate(PL[p]):
        var_name = f"X_{p}_{pl_idx}"
        # Variable binaria
        X[(p, pl_idx)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)

# No hay costo que minimizar, podemos poner la función objetivo = 0.
problem.setObjective(pulp.lpSum([]))

# =============================================================================
# 4. RESTRICCIÓN: CADA PIEZA SE COLOCA EXACTAMENTE UNA VEZ
# =============================================================================
for p in pieces:
    problem.addConstraint(
        pulp.lpSum([X[(p, pl_idx)] for pl_idx in range(len(PL[p]))]) == 1,
        name=f"place_piece_{p}"
    )

# =============================================================================
# 5. RESTRICCIÓN: NO SOLAPAMIENTO
# =============================================================================
# OPCIÓN A) Basada en celdas unitaria (más directa de entender, 
#          pero puede ser más pesada computacionalmente si la caja es grande).
#
# Para cada celda c (x,y,z) de la caja, 
# la suma de las variables de las piezas que ocupan esa celda ≤ 1.

use_cell_based_approach = True  # Cambia a False para usar la opción B

if use_cell_based_approach:
    # Precomputamos para cada pieza p y cada colocación pl_idx,
    # las celdas que ocuparía (para luego imponer la restricción).
    occupies = dict()
    for p in pieces:
        occupies[p] = dict()
        for pl_idx, ((lx, ly, lz), (x0, y0, z0)) in enumerate(PL[p]):
            cell_list = []
            for dx in range(lx):
                for dy in range(ly):
                    for dz in range(lz):
                        cell_list.append((x0 + dx, y0 + dy, z0 + dz))
            occupies[p][pl_idx] = cell_list
    
    # Ahora para cada celda c, sum(X_{p,pl} que la ocupan) ≤ 1
    for x in range(L):
        for y in range(W):
            for z in range(H):
                # Conjunto de (p, pl_idx) que ocupan la celda (x,y,z)
                cell_occupants = []
                for p in pieces:
                    for pl_idx in range(len(PL[p])):
                        if (x, y, z) in occupies[p][pl_idx]:
                            cell_occupants.append(X[(p, pl_idx)])
                
                if cell_occupants:
                    problem.addConstraint(
                        pulp.lpSum(cell_occupants) <= 1,
                        name=f"no_overlap_cell_{x}_{y}_{z}"
                    )

else:
    # OPCIÓN B) Basada en pares de colocaciones que se solapan.
    # Para cada par p != q, para cada par de colocaciones pl y pl'
    # si se solapan => X_{p,pl} + X_{q,pl'} ≤ 1
    #
    # Primero definimos una función que dado 2 colocaciones, 
    # cheque si se superponen.
    
    def overlap(placement1, placement2):
        # placement = ((lx, ly, lz), (x0, y0, z0))
        (lx1, ly1, lz1), (x01, y01, z01) = placement1
        (lx2, ly2, lz2), (x02, y02, z02) = placement2
        # Chequeo del overlap 3D (intervalos [x0, x0+lx) etc.)
        
        # No hay solapamiento si se separan en algún eje.
        no_overlap = (
            (x01 + lx1 <= x02) or (x02 + lx2 <= x01)  # Separados en X
            or (y01 + ly1 <= y02) or (y02 + ly2 <= y01)  # Separados en Y
            or (z01 + lz1 <= z02) or (z02 + lz2 <= z01)  # Separados en Z
        )
        return not no_overlap
    
    # Construimos la restricción
    for p in pieces:
        for q in pieces:
            if q <= p:
                continue  # evitar duplicados y p=q
            for pl_idx, pl in enumerate(PL[p]):
                for plp_idx, plp in enumerate(PL[q]):
                    if overlap(pl, plp):
                        problem.addConstraint(
                            X[(p, pl_idx)] + X[(q, plp_idx)] <= 1,
                            name=f"no_overlap_{p}_{pl_idx}_{q}_{plp_idx}"
                        )

# =============================================================================
# 6. RESOLVER EL PROBLEMA
# =============================================================================
solver = pulp.PULP_CBC_CMD(msg=True)
problem.solve(solver)

# =============================================================================
# 7. MOSTRAR RESULTADOS
# =============================================================================
print(f"Status: {pulp.LpStatus[problem.status]}")
if pulp.LpStatus[problem.status] == "Optimal":
    placements_selected = {}
    for p in pieces:
        for pl_idx in range(len(PL[p])):
            if pulp.value(X[(p, pl_idx)]) > 0.5:
                (lx, ly, lz), (x0, y0, z0) = PL[p][pl_idx]
                placements_selected[p] = {
                    'type': 'A' if p < nA else 'B',
                    'dimensions': (lx, ly, lz),
                    'position': (x0, y0, z0)
                }
                print(f"Pieza {p} -> Tipo {placements_selected[p]['type']}, "
                      f"Orientación ({lx},{ly},{lz}), Posición ({x0},{y0},{z0})")
else:
    print("No se encontró una solución óptima.")

# =============================================================================
# 8. VISUALIZACIÓN 3D
# =============================================================================
def plot_packing(L, W, H, placements):
    """
    Genera una gráfica 3D de la caja y las piezas colocadas.
    
    Parámetros:
    - L, W, H: dimensiones de la caja.
    - placements: diccionario con información de cada pieza.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibuja la caja
    # Definimos los vértices de la caja
    box_vertices = [
        (0, 0, 0),
        (L, 0, 0),
        (L, W, 0),
        (0, W, 0),
        (0, 0, H),
        (L, 0, H),
        (L, W, H),
        (0, W, H)
    ]
    
    # Define las caras de la caja
    box_faces = [
        [box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]],
        [box_vertices[4], box_vertices[5], box_vertices[6], box_vertices[7]],
        [box_vertices[0], box_vertices[1], box_vertices[5], box_vertices[4]],
        [box_vertices[2], box_vertices[3], box_vertices[7], box_vertices[6]],
        [box_vertices[1], box_vertices[2], box_vertices[6], box_vertices[5]],
        [box_vertices[4], box_vertices[7], box_vertices[3], box_vertices[0]]
    ]
    
    # Añade las caras a la gráfica
    box = Poly3DCollection(box_faces, linewidths=1, edgecolors='k', alpha=0.1)
    box.set_facecolor((0,0,1,0.1))  # Azul transparente
    ax.add_collection3d(box)
    
    # Colores para las piezas
    color_map = {}
    import random
    random.seed(42)  # Para reproducibilidad
    for p in placements:
        color_map[p] = (random.random(), random.random(), random.random(), 0.6)
    
    # Dibuja cada pieza
    for p, info in placements.items():
        lx, ly, lz = info['dimensions']
        x0, y0, z0 = info['position']
        
        # Vértices de la pieza
        piece_vertices = [
            (x0, y0, z0),
            (x0 + lx, y0, z0),
            (x0 + lx, y0 + ly, z0),
            (x0, y0 + ly, z0),
            (x0, y0, z0 + lz),
            (x0 + lx, y0, z0 + lz),
            (x0 + lx, y0 + ly, z0 + lz),
            (x0, y0 + ly, z0 + lz)
        ]
        
        # Define las caras de la pieza
        piece_faces = [
            [piece_vertices[0], piece_vertices[1], piece_vertices[2], piece_vertices[3]],
            [piece_vertices[4], piece_vertices[5], piece_vertices[6], piece_vertices[7]],
            [piece_vertices[0], piece_vertices[1], piece_vertices[5], piece_vertices[4]],
            [piece_vertices[2], piece_vertices[3], piece_vertices[7], piece_vertices[6]],
            [piece_vertices[1], piece_vertices[2], piece_vertices[6], piece_vertices[5]],
            [piece_vertices[4], piece_vertices[7], piece_vertices[3], piece_vertices[0]]
        ]
        
        piece = Poly3DCollection(piece_faces, linewidths=1, edgecolors='k', alpha=0.6)
        piece.set_facecolor(color_map[p])
        ax.add_collection3d(piece)
        
        # Opcional: Añadir etiquetas a las piezas
        # Centro de la pieza para la etiqueta
        cx = x0 + lx / 2
        cy = y0 + ly / 2
        cz = z0 + lz / 2
        ax.text(cx, cy, cz, f"P{p}", color='k', ha='center', va='center')
    
    # Ajusta los límites de la gráfica
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_title('Empaquetamiento 3D de las Piezas en la Caja')
    plt.show()

# Si se encontró una solución óptima, realiza la visualización
if pulp.LpStatus[problem.status] == "Optimal":
    plot_packing(L, W, H, placements_selected)
