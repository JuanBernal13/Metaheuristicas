##############################################################################
# EXAMEN FINAL - Optimización Avanzada
# Profesor: Andrés Medaglia
# Asistente: Laura Juliana Sánchez
# ---------------------------------------------------------------------------
# SOLUCIÓN (literal b) - Relajación Lagrangiana de restricciones (3) y
# método del subgradiente en Python + Gurobi, con una HEURÍSTICA inicial
# para obtener la primera cota primal factible. Luego se itera el 
# subgradiente 1,000 veces.
#
# NOTA: 
#  - La heurística inicial consiste en resolver el problema completo de manera
#    rapida (limitando el tiempo de Búsqueda) para obtener una primera 
#    solución factible y, por tanto, una cota primal. Si el solver encuentra
#    una solución en ese tiempo limitado, la usaremos como "best_primal".
#  - Luego, partiendo de mu=0, ejecutamos 1,000 iteraciones del método 
#    del subgradiente para mejorar la cota dual.
##############################################################################

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. DATOS DEL PROBLEMA
# ----------------------------------------------------------------------------
M = 8   # Cantidad de operarios
N = 24  # Cantidad de órdenes

# Tiempos máximos disponibles b_i (Tabla 1)
b = [36, 35, 38, 34, 32, 34, 31, 34]

# Matriz de tiempos requeridos w_{i,j} (Tabla 2), i=0..7, j=0..23
w = np.array([
 [ 8,18,22, 5,11,11,22,11,17,22,11,20,13,13, 7,22,15,22,24, 8, 8,24,18, 8],  # Operario 1
 [24,14,11,15,24, 8,10,15,19,25, 6,13,10,25,19,24,13,12, 5,18,10,24, 8, 5],  # Operario 2
 [22,22,21,22,13,16,21, 5,25,13,12, 9,24, 6,22,24,11,21,11,14,12,10,20, 6],  # Operario 3
 [13, 8,19,12,19,18,10,21, 5, 9,11, 9,22, 8,12,13, 9,25,19,24,22, 6,19,14],  # Operario 4
 [25,16,13, 5,11, 8, 7, 8,25,20,24,20,11, 6,10,10, 6,22,10,10,13,21, 5,19],  # Operario 5
 [19,19, 5,11,22,24,18,11, 6,13,24,24,22, 6,22, 5,14, 6,16,11, 6, 8,18,10],  # Operario 6
 [24,10, 9,10, 6,15, 7,13,20, 8, 7, 9,24, 9,21, 9,11,19,10, 5,23,20, 5,21],  # Operario 7
 [ 6, 9, 9, 5,12,10,16,15,19,18,20,18,16,21,11,12,22,16,21,25, 7,14,16,10]   # Operario 8
])

# Matriz de puntajes p_{i,j} (Tabla 3), i=0..7, j=0..23
p = np.array([
 [25,23,20,16,19,22,20,16,15,22,15,21,20,23,20,22,19,25,25,24,21,17,23,17],  # Operario 1
 [16,19,22,22,19,23,17,24,15,24,18,19,20,24,25,25,19,24,18,21,16,25,15,20],  # Operario 2
 [20,18,23,23,23,17,19,16,24,24,17,23,19,22,23,25,23,18,19,24,20,17,23,23],  # Operario 3
 [16,16,15,23,15,15,25,22,17,20,19,16,17,17,20,17,17,18,16,18,15,25,22,17],  # Operario 4
 [17,23,21,20,24,22,25,17,22,20,16,22,21,23,24,15,22,25,18,19,19,17,22,23],  # Operario 5
 [24,21,23,17,21,19,19,17,18,24,15,15,17,18,15,24,19,21,23,24,17,20,16,21],  # Operario 6
 [18,21,22,23,22,15,18,15,21,22,15,23,21,25,25,23,20,16,25,17,15,15,18,16],  # Operario 7
 [19,24,18,17,21,18,24,25,18,23,21,15,24,23,18,18,23,23,16,20,20,19,25,21]   # Operario 8
])

# ----------------------------------------------------------------------------
# 2. PASO HEURÍSTICO: Resolver el MIP original con un límite de tiempo
#    para intentar obtener una primera cota primal factible.
# ----------------------------------------------------------------------------

heur_model = gp.Model("Initial_Heuristic")
heur_model.Params.LogToConsole = 0
# Limitar el tiempo a, por ejemplo, 5 segundos (o lo que se prefiera)
heur_model.Params.TimeLimit = 5.0

# Variables x[i,j] binaria
x_heur = heur_model.addVars(M, N, vtype=GRB.BINARY, name="x")

# Restricción (2): cada orden j se asigna a exactamente un operario
for j in range(N):
    heur_model.addConstr(gp.quicksum(x_heur[i,j] for i in range(M)) == 1)

# Restricción (3): sum_j w[i,j]*x[i,j] <= b_i
for i in range(M):
    heur_model.addConstr(gp.quicksum(w[i,j]*x_heur[i,j] for j in range(N)) <= b[i])

# Función objetivo (1): maximizar sum_{i,j} p[i,j]*x[i,j]
heur_model.setObjective(gp.quicksum(p[i,j]*x_heur[i,j] for i in range(M) for j in range(N)), 
                        GRB.MAXIMIZE)

heur_model.optimize()

# Inicializamos best_primal con -∞
best_primal = -GRB.INFINITY

if heur_model.SolCount > 0:  # Si encontró alguna solución factible
    # Extraemos su valor de la función objetivo
    best_primal = heur_model.ObjVal

print(">>> Heurística inicial (MIP rápido) finalizada")
if best_primal > -GRB.INFINITY:
    print(f"   * Se obtuvo una solución factible con valor: {best_primal:.2f}")
else:
    print(f"   * No se halló solución en el tiempo dado. best_primal sigue = -∞")

heur_model.dispose()

# ----------------------------------------------------------------------------
# 3. AHORA SÍ, SUBGRADIENTE SOBRE LA RELAJACIÓN LAGRANGIANA
#    Se relajan las restricciones (3): 
#       sum_j w[i,j]*x[i,j] <= b_i
#    con multiplicadores mu_i >= 0.
#
#    L(mu) = max sum_{i,j} (p[i,j] - mu_i*w[i,j]) x[i,j] + sum_i mu_i*b_i
#    s.t. sum_i x[i,j] = 1,   x[i,j] in {0,1}
#
#    Parte "variable" = sum_{i,j} (p[i,j] - mu_i*w[i,j]) x[i,j].
#    A esto se le suma sum_i mu_i*b_i para obtener la cota dual en cada iteración.
# ----------------------------------------------------------------------------

# 3.1 Multiplicadores iniciales mu=0
mu = [0.0]*M

# Control de iteraciones
MAX_ITERS = 1000  # se piden 1,000 iteraciones ahora

dual_history   = []
primal_history = []
best_dual = -GRB.INFINITY  # cota dual

# ----------------------------------------------------------------------------
# 4. Bucle principal de subgradientes
# ----------------------------------------------------------------------------
for k in range(1, MAX_ITERS+1):
    # ----------------------
    # 4.1 Construir Modelo Lagrangiano en Gurobi
    # ----------------------
    model = gp.Model("Lagrangian_relaxation")
    model.Params.LogToConsole = 0

    # Variables x[i,j] binaria
    x_vars = {}
    for i in range(M):
        for j in range(N):
            coef = p[i,j] - mu[i]*w[i,j]  # coef en la función objetivo
            x_vars[(i,j)] = model.addVar(vtype=GRB.BINARY, obj=coef, 
                                         name=f"x_{i}_{j}")

    # Restricción: para cada j, sum_i x[i,j] = 1
    for j in range(N):
        model.addConstr(gp.quicksum(x_vars[(i,j)] for i in range(M)) == 1)

    model.ModelSense = GRB.MAXIMIZE
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.INT_OPTIMAL:
        var_obj_val = model.ObjVal
    else:
        var_obj_val = -1e10

    # Cota dual actual: var_obj_val + sum_i mu[i]*b[i]
    lag_value = var_obj_val + sum(mu[i]*b[i] for i in range(M))
    if lag_value > best_dual:
        best_dual = lag_value

    # ----------------------
    # 4.2 Evaluar factibilidad y actualizar cota primal
    # ----------------------
    # Extraer solución
    x_sol = {}
    for i in range(M):
        for j in range(N):
            x_sol[(i,j)] = x_vars[(i,j)].X

    # Chequear si cumple sum_j w[i,j]*x[i,j] <= b[i]
    feasible = True
    for i in range(M):
        total_time_i = sum(w[i,j]*x_sol[(i,j)] for j in range(N))
        if total_time_i > b[i] + 1e-9:
            feasible = False
            break

    # Si es factible, calculamos su valor primal
    if feasible:
        val_primal = sum(p[i,j]*x_sol[(i,j)] for i in range(M) for j in range(N))
        if val_primal > best_primal:
            best_primal = val_primal

    # Guardar en histórico
    dual_history.append(best_dual)
    # Si no se ha encontrado ninguna solución factible todavía, ponemos 0 o -∞
    if best_primal == -GRB.INFINITY:
        primal_history.append(0.0)  # Para graficar, usamos 0
    else:
        primal_history.append(best_primal)

    # ----------------------
    # 4.3 Subgradiente y actualización de mu
    # ----------------------
    alpha_k = 1.0 / k
    for i in range(M):
        G_i = b[i] - sum(w[i,j]*x_sol[(i,j)] for j in range(N))
        mu[i] = mu[i] + alpha_k*G_i
        # Proyección a no-negativo
        if mu[i] < 0.0:
            mu[i] = 0.0

    model.dispose()

# ----------------------------------------------------------------------------
# 5. RESULTADOS FINALES
# ----------------------------------------------------------------------------
print("\n*** MÉTODO DEL SUBGRADIENTE (con heurística inicial) FINALIZADO ***")
print(f"Iteraciones realizadas: {MAX_ITERS}")
print(f"Mejor cota dual (LR): {best_dual:,.2f}")
if best_primal > -GRB.INFINITY:
    print(f"Mejor cota primal (sol. factible): {best_primal:,.2f}")
    gap = 100.0*(best_primal - best_dual)/best_primal
    print(f"Gap = {gap:,.2f}%")
else:
    print("No se halló ninguna solución primal factible. Gap no definido.")

# ----------------------------------------------------------------------------
# 6. GRÁFICAS DE EVOLUCIÓN
# ----------------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(range(1,MAX_ITERS+1), dual_history,  label="Cota Dual (LR)", color='blue')
plt.plot(range(1,MAX_ITERS+1), primal_history,label="Cota Primal factible", color='red')
plt.xlabel("Iteración (k)")
plt.ylabel("Valor")
plt.title("Evolución de Cotas (Subgradiente + Heurística Inicial)")
plt.legend()
plt.grid(True)
plt.show()
