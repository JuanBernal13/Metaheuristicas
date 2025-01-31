import pulp

def solve_subproblem(lambda_val):
    """
    Resuelve el subproblema Lagrangiano:
       max (1 - 2λ)*x + (2 - 3λ)*y + (λ*10) (constante)
    sujeto a x + y <= 4, x >= 0, y >= 0.
    Devuelve: (x*, y*, valor_objetivo_LR).
    """
    # Creamos un problema de maximización
    subproblem = pulp.LpProblem("Subproblema_Lagrangiano", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable('x', lowBound=0)
    y = pulp.LpVariable('y', lowBound=0)

    # Función objetivo Lagrangiana
    # Constante λ*10 no afecta la optimización de x,y, pero la incluimos para ver la cota
    subproblem += (1 - 2*lambda_val)*x + (2 - 3*lambda_val)*y + (lambda_val*10), "Lagrangian_Objective"

    # Restricciones que MANTENEMOS en el subproblema
    subproblem += x + y <= 4, "Restriccion_1"

    # (La restricción 2: 2x + 3y <= 10 está relajada, se incorpora en el objetivo)
    # No la agregamos aquí, pues la hemos "movido" al objetivo.

    # Resolver
    subproblem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraer resultados
    x_val = pulp.value(x)
    y_val = pulp.value(y)
    obj_val = pulp.value(subproblem.objective)

    return x_val, y_val, obj_val


def main():
    # Parámetros
    max_iterations = 10
    alpha = 0.5         # Step-size para actualizar lambda
    lambda_val = 0.0    # Lambda inicial (>= 0)
    
    best_lagrangian_bound = float('-inf')
    
    print(f"{'Iter':>4} | {'lambda':>7} | {'x*':>5} | {'y*':>5} | {'L(x*,y*,λ)':>12} | {'violacion':>9}")
    print("-" * 60)

    for it in range(1, max_iterations+1):
        # 1) Resolver el subproblema con el lambda actual
        x_star, y_star, lag_obj_value = solve_subproblem(lambda_val)
        
        # 2) Calcular la violación de la restricción relajada: 2x + 3y <= 10
        left_side = 2*x_star + 3*y_star
        violation = max(0, left_side - 10)
        
        # 3) Actualizar lambda con un método subgradiente sencillo
        lambda_val = max(0, lambda_val + alpha * violation)  # forzamos lambda >= 0

        # Mantener registro de la mejor cota Lagrangiana (como es un problema de max, esta es cota superior)
        if lag_obj_value > best_lagrangian_bound:
            best_lagrangian_bound = lag_obj_value
        
        print(f"{it:4d} | {lambda_val:7.3f} | {x_star:5.2f} | {y_star:5.2f} | {lag_obj_value:12.3f} | {violation:9.3f}")

    print("\nMejor cota Lagrangiana obtenida:", best_lagrangian_bound)

if __name__ == "__main__":
    main()
