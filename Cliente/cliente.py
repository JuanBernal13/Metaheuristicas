#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import csv

Z_95 = 1.96  # Para Intervalo de Confianza al 95%

def main():
    url = "http://localhost:8080/api/simulacion"
    
    # Parámetros de entrada
    numero_replicas = 30
    pasos_maximos = 10000
    pacientes_iniciales = 25
    trabajadores_iniciales = 5
    tasa_llegada = 0.5  # Tasa media de llegada (Poisson)

    payload = {
        "numeroReplicas": numero_replicas,
        "pasosMaximos": pasos_maximos,
        "pacientesIniciales": pacientes_iniciales,
        "trabajadoresIniciales": trabajadores_iniciales,
        "tasaLlegada": 5,      # Ajusta si tu DTO se llama "tasaLlegadaPacientes"...
        "tasaSalida": 4.96               # Ejemplo: tasa de salidas voluntarias
    }

    print(f"Enviando POST a {url} con payload: {payload}")

    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        print("Error en la solicitud:", resp.status_code, resp.text)
        return

    resultados = resp.json()  # Se espera un array JSON con 30 objetos (réplicas)
    print(f"Se recibieron {len(resultados)} réplicas.")

    # Validar que todas tengan el mismo número de pasos
    for r in resultados:
        if len(r["tVals"]) != pasos_maximos:
            print("Inconsistencia en tamaño de tVals!")
            return

    # Estados que analizaremos
    estados = ["S","V","E","A","I","U","R","D"]
    nRep = len(resultados)
    nSteps = pasos_maximos

    # Arrays (nRep, nSteps) para cada estado, UCI, vac, mask
    data_estados = {}
    for e in estados:
        data_estados[e] = np.zeros((nRep, nSteps), dtype=float)

    data_uci_occ = np.zeros((nRep, nSteps), dtype=float)
    data_uci_cap = np.zeros((nRep, nSteps), dtype=float)
    data_vac = np.zeros((nRep, nSteps), dtype=float)
    data_mask= np.zeros((nRep, nSteps), dtype=float)

    # Muertes y recuperados totales al final (1 valor por réplica)
    total_muertes = np.zeros(nRep, dtype=float)
    total_recuperados = np.zeros(nRep, dtype=float)

    # Rellenar
    for i, r in enumerate(resultados):
        for e in estados:
            serie = r["conteosEstados"][e]  # list int
            data_estados[e][i,:] = serie

        data_uci_occ[i,:] = r["uciOcupada"]
        data_uci_cap[i,:] = r["uciCapacidadTotal"]
        data_vac[i,:]     = r["tasaVacunacion"]
        data_mask[i,:]    = r["tasaMascarilla"]

        total_muertes[i]    = r["totalMuertes"]
        total_recuperados[i] = r["totalRecuperados"]

    # Función auxiliar para calcular media e IC en cada paso
    def calc_mean_ci(matrix):
        """
        matrix: shape (nRep, nSteps)
        retorna (mean, lower, upper) cada uno shape (nSteps,)
        """
        m = np.mean(matrix, axis=0)
        s = np.std(matrix, axis=0, ddof=1)
        sem = s / np.sqrt(nRep)
        delta = Z_95 * sem
        low = m - delta
        up  = m + delta
        return m, low, up

    # Calculamos estadísticos
    stats_estados = {}
    for e in estados:
        stats_estados[e] = calc_mean_ci(data_estados[e])
    stats_uci_occ = calc_mean_ci(data_uci_occ)
    stats_uci_cap = calc_mean_ci(data_uci_cap)
    stats_vac     = calc_mean_ci(data_vac)
    stats_mask    = calc_mean_ci(data_mask)

    # >>> Graficamos todo:

    # 1) Graficar cada estado en subplots (2 col x 4 filas)
    tvals = np.arange(1, nSteps+1)
    fig1, axes1 = plt.subplots(nrows=4, ncols=2, figsize=(14,16), sharex=True)
    axes1 = axes1.flatten()  # para iterar más fácilmente

    for idx, e in enumerate(estados):
        meanE, lowE, upE = stats_estados[e]
        ax = axes1[idx]
        ax.plot(tvals, meanE, label=f"{e} (media)")
        ax.fill_between(tvals, lowE, upE, alpha=0.2, label="IC 95%")
        ax.set_title(f"Estado {e}")
        ax.set_ylabel("Cantidad")
        ax.set_xlabel("Tiempo (pasos)")
        ax.grid(True)
        ax.legend()

    fig1.tight_layout()
    plt.show()

    # 2) Ocupación vs capacidad UCI
    fig2, ax2 = plt.subplots(figsize=(10,6))
    mOcc, lOcc, uOcc = stats_uci_occ
    mCap, lCap, uCap = stats_uci_cap
    ax2.plot(tvals, mOcc, 'r-', label="UCI ocupada (media)")
    ax2.fill_between(tvals, lOcc, uOcc, color='r', alpha=0.2)
    ax2.plot(tvals, mCap, 'k--', label="UCI capacidad (media)")
    ax2.fill_between(tvals, lCap, uCap, color='k', alpha=0.1)
    ax2.set_title("Ocupación vs Capacidad UCI")
    ax2.set_xlabel("Tiempo (pasos)")
    ax2.set_ylabel("Camas")
    ax2.legend()
    ax2.grid(True)
    plt.show()

    # 3) Tasa Vacunación y Mascarilla
    fig3, ax3 = plt.subplots(figsize=(10,6))
    mVac, lVac, uVac = stats_vac
    mMask, lMask, uMask = stats_mask
    ax3.plot(tvals, mVac*100, 'g-', label="Vacunación (%) (media)")
    ax3.fill_between(tvals, lVac*100, uVac*100, color='g', alpha=0.2)
    ax3.plot(tvals, mMask*100, 'b-', label="Mascarilla (%) (media)")
    ax3.fill_between(tvals, lMask*100, uMask*100, color='b', alpha=0.2)
    ax3.set_title("Tasas de Vacunación y Mascarilla")
    ax3.set_xlabel("Tiempo (pasos)")
    ax3.set_ylabel("Porcentaje (%)")
    ax3.legend()
    ax3.grid(True)
    plt.show()

    # 4) (Opcional) Gráfico de barras del estado final (paso nSteps) [Promedio]
    #    Calculamos la media final de cada estado en step = nSteps
    final_means = {}
    for e in estados:
        mean_, _, _ = stats_estados[e]
        final_means[e] = mean_[-1]  # valor en el último paso
    # Graficamos
    fig4, ax4 = plt.subplots(figsize=(8,5))
    x_estados = np.arange(len(estados))
    barras = [final_means[e] for e in estados]
    ax4.bar(x_estados, barras, color='cyan', alpha=0.7)
    ax4.set_xticks(x_estados)
    ax4.set_xticklabels(estados)
    ax4.set_title("Distribución de Estados al paso final (Promedio)")
    ax4.set_ylabel("Cantidad promedio")
    for i, v in enumerate(barras):
        ax4.text(i, v+0.5, f"{v:.1f}", ha='center', color='black')
    plt.show()

    # 5) Estadísticas globales de muertes y recuperados (final)
    mu_mean = np.mean(total_muertes)
    mu_std  = np.std(total_muertes, ddof=1)
    mu_sem  = mu_std/np.sqrt(nRep)
    mu_low  = mu_mean - Z_95*mu_sem
    mu_up   = mu_mean + Z_95*mu_sem

    rec_mean = np.mean(total_recuperados)
    rec_std  = np.std(total_recuperados, ddof=1)
    rec_sem  = rec_std/np.sqrt(nRep)
    rec_low  = rec_mean - Z_95*rec_sem
    rec_up   = rec_mean + Z_95*rec_sem

    print("\n=== ESTADÍSTICAS GLOBALES (30 réplicas) ===")
    print(f"Muertes totales (media ± IC95%): {mu_mean:.2f} [{mu_low:.2f}, {mu_up:.2f}]")
    print(f"Recuperados totales (media ± IC95%): {rec_mean:.2f} [{rec_low:.2f}, {rec_up:.2f}]")

    # 6) Exportar CSV
    exportar_csv(tvals, estados, stats_estados, stats_uci_occ, stats_uci_cap, stats_vac, stats_mask)

def exportar_csv(tvals, estados, stats_estados, stats_uci_occ, stats_uci_cap, stats_vac, stats_mask, filename="agregado_simulacion.csv"):
    fieldnames = ["Step"]
    for e in estados:
        fieldnames += [f"{e}_mean", f"{e}_low", f"{e}_up"]
    fieldnames += ["UCIOcc_mean","UCIOcc_low","UCIOcc_up",
                   "UCICap_mean","UCICap_low","UCICap_up",
                   "Vac_mean","Vac_low","Vac_up",
                   "Mask_mean","Mask_low","Mask_up"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, step in enumerate(tvals):
            row = {}
            row["Step"] = int(step)
            for e in estados:
                m, lo, up = stats_estados[e]
                row[f"{e}_mean"] = m[i]
                row[f"{e}_low"] = lo[i]
                row[f"{e}_up"] = up[i]
            # UCI
            mOcc, lOcc, uOcc = stats_uci_occ
            mCap, lCap, uCap = stats_uci_cap
            row["UCIOcc_mean"] = mOcc[i]
            row["UCIOcc_low"]  = lOcc[i]
            row["UCIOcc_up"]   = uOcc[i]
            row["UCICap_mean"] = mCap[i]
            row["UCICap_low"]  = lCap[i]
            row["UCICap_up"]   = uCap[i]

            # Vac
            mV, lV, uV = stats_vac
            row["Vac_mean"] = mV[i]*100
            row["Vac_low"]  = lV[i]*100
            row["Vac_up"]   = uV[i]*100

            # Mask
            mM, lM, uM = stats_mask
            row["Mask_mean"] = mM[i]*100
            row["Mask_low"]  = lM[i]*100
            row["Mask_up"]   = uM[i]*100

            writer.writerow(row)

    print(f"CSV exportado: {filename}")

if __name__ == "__main__":
    main()
