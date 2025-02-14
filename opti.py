import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ==========================
# 1. PARÁMETROS DEL PROBLEMA
# ==========================
num_nodes = 10
capacity_truck = 5  # Capacidad del camión en toneladas
depot = 0           # Nodo que actúa como depósito (todas las rutas inician aquí)

np.random.seed(42)  # Para reproducibilidad

# ===============================================
# 2. GENERACIÓN DEL GRAFO Y ARISTAS CON COST/DEMANDA
# ===============================================
graph = {i: {} for i in range(num_nodes)}

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        cost = np.random.randint(5, 30)     # Costo de recorrer la arista
        demand = np.random.randint(1, 4)    # Demanda en toneladas (1 a 3)
        graph[i][j] = {'cost': cost, 'demand': demand}
        graph[j][i] = {'cost': cost, 'demand': demand}

# =======================================
# 3. CREAR GRAFO DE NETWORKX PARA CÁLCULOS
# =======================================
G = nx.Graph()
for node in range(num_nodes):
    G.add_node(node)

# Agregamos las aristas con atributos
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        cost_ij = graph[i][j]['cost']
        demand_ij = graph[i][j]['demand']
        G.add_edge(i, j, cost=cost_ij, demand=demand_ij)

# =====================
# 4. COORDENADAS NODOS
# =====================
coordinates = {
    i: [np.random.randint(0, 50), np.random.randint(0, 50)]
    for i in range(num_nodes)
}

pos = {n: (coordinates[n][0], coordinates[n][1]) for n in G.nodes()}

# =======================================================================
# 5. FUNCIÓN DE CÁLCULO DE RUTA: Todas las rutas parten del mismo DEPÓSITO
# =======================================================================
def shortest_path_cost(u, v, graph_nx, weight='cost'):
    """
    Retorna la distancia (según 'weight') del camino más corto entre u y v.
    Si no hay camino, retorna None.
    """
    try:
        return nx.shortest_path_length(graph_nx, source=u, target=v, weight=weight)
    except nx.NetworkXNoPath:
        return None

def next_feasible_edge(current_node, unserved_edges, current_capacity, G_nx):
    """
    Dada la capacidad actual y un diccionario 'unserved_edges' con las aristas 
    que faltan por servir, encuentra la siguiente arista factible que minimice:
       (distancia en vacío hasta esa arista) + (costo de la arista).
    Retorna:
      (edge_uv, dist_empty, orientation)
         - 'edge_uv' es una tupla (u, v) con u < v
         - 'dist_empty' es el costo de viaje en vacío desde current_node hasta (u o v)
         - 'orientation' indica si conviene llegar a u o a v para servir la arista
    Si no hay arista factible, retorna None.
    """
    best_edge = None
    best_total_cost = float('inf')
    best_dist = 0
    best_orientation = None

    for (u, v), data in unserved_edges.items():
        demand = data['demand']
        cost_arista = data['cost']
        if demand <= current_capacity:
            # Podemos servir esta arista si llegamos a u o a v
            dist_u = shortest_path_cost(current_node, u, G_nx, weight='cost')
            dist_v = shortest_path_cost(current_node, v, G_nx, weight='cost')

            candidates = []
            if dist_u is not None:
                # Costo total = dist_u + cost_arista
                candidates.append(('u', dist_u, dist_u + cost_arista))
            if dist_v is not None:
                candidates.append(('v', dist_v, dist_v + cost_arista))

            for (orient, dist_val, total_val) in candidates:
                if total_val < best_total_cost:
                    best_total_cost = total_val
                    best_edge = (u, v)
                    best_dist = dist_val
                    best_orientation = orient

    if best_edge is None:
        return None
    else:
        return (best_edge, best_dist, best_orientation)

def construct_routes_car_with_depot(G_nx, depot_node, capacity=5):
    """
    Construye rutas para cubrir TODAS las aristas del grafo G_nx (con 'cost' y 'demand'),
    comenzando SIEMPRE desde un mismo depósito (depot_node).
    
    Retorna una lista de rutas, donde cada ruta es un diccionario:
        {
          'path': [ (nA, nB, 'deadhead'/'collect'), ... ],
          'total_cost': float,
          'served_edges': [ (u1,v1), (u2,v2), ... ]  # aristas recolectadas
        }
    """
    # 1. Crear diccionario de aristas sin servir
    #    unserved_edges = { (u,v): {'cost': c, 'demand': d}, ... }
    #    con (u < v) para evitar duplicados
    unserved_edges = {}
    for (u, v) in G_nx.edges():
        uu, vv = min(u, v), max(u, v)
        unserved_edges[(uu, vv)] = {
            'cost': G_nx[u][v]['cost'],
            'demand': G_nx[u][v]['demand']
        }

    routes = []

    # 2. Mientras queden aristas sin servir, construimos una ruta
    while unserved_edges:
        current_route = {
            'path': [],
            'total_cost': 0,
            'served_edges': []
        }
        current_capacity = capacity
        
        # a) Iniciar siempre en el depósito
        current_node = depot_node

        # b) Buscar la siguiente arista factible
        while True:
            nxt = next_feasible_edge(current_node, unserved_edges, current_capacity, G_nx)
            if nxt is None:
                # No hay más aristas factibles (por demanda/capacidad o no hay camino)
                # Terminamos esta ruta
                break
            else:
                (edge_uv, dist_empty, orientation) = nxt
                (u_sel, v_sel) = edge_uv
                cost_sel = unserved_edges[edge_uv]['cost']
                demand_sel = unserved_edges[edge_uv]['demand']

                # 1) Añadimos el costo de viaje en vacío
                if dist_empty > 0:
                    # Movemos el camión a (u_sel) o (v_sel)
                    if orientation == 'u':
                        current_route['path'].append((current_node, u_sel, 'deadhead'))
                    else:
                        current_route['path'].append((current_node, v_sel, 'deadhead'))
                    current_route['total_cost'] += dist_empty

                # 2) Recolectamos la arista
                if orientation == 'u':
                    serving_start = u_sel
                    serving_end = v_sel
                else:
                    serving_start = v_sel
                    serving_end = u_sel

                current_route['path'].append((serving_start, serving_end, 'collect'))
                current_route['total_cost'] += cost_sel
                current_route['served_edges'].append((min(serving_start, serving_end),
                                                      max(serving_start, serving_end)))

                # Actualizamos capacidad
                current_capacity -= demand_sel

                # Eliminamos la arista del conjunto de no servidas
                del unserved_edges[(min(u_sel, v_sel), max(u_sel, v_sel))]

                # El camión ahora está en serving_end
                current_node = serving_end

                # Si la capacidad está agotada, terminamos la ruta
                if current_capacity <= 0:
                    break
        
        # Agregamos la ruta (aunque sea vacía) al listado
        # Nota: si no se sirvió ninguna arista y no hay factibles, se generaría
        #       una ruta vacía; en la práctica, se podría filtrar.
        if len(current_route['served_edges']) > 0:
            routes.append(current_route)
        else:
            # Si no sirvió nada, evitamos rutas en blanco
            break

    return routes


# ========================
# 6. EJECUCIÓN DEL ALGORITMO
# ========================
all_routes = construct_routes_car_with_depot(G, depot, capacity=capacity_truck)

# =================
# 7. VISUALIZACIÓN
# =================
plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Dibujar todas las aristas en gris con etiquetas de cost/demand
nx.draw_networkx_edges(G, pos, edge_color='gray')
edge_labels = {
    (u, v): f"C:{G[u][v]['cost']},D:{G[u][v]['demand']}"
    for (u, v) in G.edges()
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Colores para distinguir rutas
colors = ['red', 'green', 'blue', 'orange', 'magenta', 'brown', 
          'purple', 'olive', 'cyan', 'black']

for i, route_info in enumerate(all_routes):
    c = colors[i % len(colors)]
    # Pintar las aristas servidas
    for (u, v) in route_info['served_edges']:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=c, width=3)

# Resaltar el depósito
nx.draw_networkx_nodes(G, pos, nodelist=[depot], node_color='yellow', node_size=800)

plt.title(f'Capacitated Arc Routing desde un Depósito (Nodo {depot})')
plt.axis('off')
plt.show()

# =================================
# 8. IMPRESIÓN DE RUTAS GENERADAS
# =================================
print(f"\nRutas generadas (todas iniciando en el depósito {depot}):\n")
for idx, rinfo in enumerate(all_routes, start=1):
    print(f"=== Ruta {idx} ===")
    print(f"  Costo total: {rinfo['total_cost']}")
    print(f"  Aristas recolectadas: {rinfo['served_edges']}")
    print("  Secuencia de movimiento:")
    for step in rinfo['path']:
        (a, b, action) = step
        if action == 'collect':
            print(f"    - Recolectando arista ({a}, {b})")
        else:
            print(f"    - Viajando en vacío de {a} a {b}")
    print()
