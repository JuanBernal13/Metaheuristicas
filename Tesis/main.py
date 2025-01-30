# main.py

import random
import uuid
from collections import defaultdict
import numpy as np
from scipy.stats import beta, gamma as gamma_dist

from services.simulation_service import SimulationService
from models.agent import HealthcareWorker, Patient
from models.states import EpidemicState
from visualization.plot_matplotlib import plot_epidemic, plot_icu, plot_vaccination_mask
from visualization.plot_plotly import plot_spatial, plot_interactive

def sample_parameters():
    """
    Sample epidemiological parameters from probability distributions (extended).
    """

    # Transmission rates for symptomatic
    beta_h = beta.rvs(a=2, b=5)     # HCW base transmission
    beta_g = beta.rvs(a=2, b=5)     # Patients base transmission

    # NEW: transmission rate for Asymptomatic
    beta_asym = 0.5 * (beta_h + beta_g) / 2.0  

    # Rate E->I (or E->A)
    sigma = gamma_dist.rvs(a=2, scale=1/0.2)   # ~5 days

    # Probability that E->A vs E->I
    p_asym = random.uniform(0.3, 0.6)  # 30-60% become asymptomatic carriers

    # Rate A->I
    eta = gamma_dist.rvs(a=2, scale=1/0.25)    # ~4 days

    # Rate to ICU
    alpha = beta.rvs(a=2, b=5)

    # Recovery rates
    gamma_ = gamma_dist.rvs(a=2, scale=1/0.142)  # ~7 days
    gamma_u = gamma_dist.rvs(a=2, scale=1/0.071) # ~14 days

    # Mortalities
    mu_i = beta.rvs(a=2, b=98)   # ~0.02
    mu_u = beta.rvs(a=2, b=32)   # ~0.06
    mu_a = beta.rvs(a=2, b=198)  # smaller (~0.01) for asymptomatic

    mu_nat = 1/(80*365)          # natural mortality

    # Re-susceptibility
    p_resus = beta.rvs(a=2, b=200)

    # Waning of vaccine
    omega_v = beta.rvs(a=2, b=20)  # e.g. 0.05-0.15 typical

    # Infection control measure
    ic_reduction = random.uniform(0.1, 0.5)  # 10-50% reduction in transmissions

    return (beta_h, beta_g, beta_asym, sigma, p_asym, eta, alpha, gamma_, gamma_u,
            mu_i, mu_u, mu_a, mu_nat, p_resus, omega_v, ic_reduction)


def run_simulation(replicate_id, max_steps, initial_patients, initial_workers):
    width, height = 20, 20
    n_icu_rows = 3
    icu_capacity = 5

    service = SimulationService(width=width, height=height,
                                n_icu_rows=n_icu_rows,
                                icu_capacity=icu_capacity)

    # Sample parameters for this replicate
    (beta_h, beta_g, beta_asym, sigma, p_asym, eta, alpha, gamma_, gamma_u,
     mu_i, mu_u, mu_a, mu_nat, p_resus, omega_v, ic_reduction) = sample_parameters()

    # Create HCWs
    for i in range(initial_workers):
        cell = service.get_random_cell()
        unique_id = f"HW-{uuid.uuid4()}"
        vaccinated = random.random() < 0.8
        wears_mask = random.random() < 0.9
        resistant = random.random() < 0.2
        hw = HealthcareWorker(
            unique_id=unique_id,
            cell=cell,
            beta_h=beta_h,
            beta_asym=beta_asym,
            sigma=sigma,
            p_asym=p_asym,
            eta=eta,
            alpha=alpha,
            gamma_=gamma_,
            gamma_u=gamma_u,
            mu_i=mu_i,
            mu_u=mu_u,
            mu_a=mu_a,
            mu_nat=mu_nat,
            p_resus=p_resus,
            omega_v=omega_v,
            ic_reduction=ic_reduction,
            age=random.randint(25, 65),
            use_probabilistic=True,
            vaccinated=vaccinated,
            wears_mask=wears_mask,
            resistant=resistant
        )
        service.add_worker(hw)

    # Create initial patients
    for i in range(initial_patients):
        cell = service.get_random_cell()
        pid = str(uuid.uuid4())
        severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        vaccinated = random.random() < 0.3
        wears_mask = random.random() < 0.7
        resistant = random.random() < 0.3

        p = Patient(
            unique_id=pid,
            cell=cell,
            beta_g=beta_g,
            beta_asym=beta_asym,
            sigma=sigma,
            p_asym=p_asym,
            eta=eta,
            alpha=alpha,
            gamma_=gamma_,
            gamma_u=gamma_u,
            mu_i=mu_i,
            mu_u=mu_u,
            mu_a=mu_a,
            mu_nat=mu_nat,
            p_resus=p_resus,
            omega_v=omega_v,
            ic_reduction=ic_reduction,
            severity=severity,
            age=random.randint(30, 90),
            use_probabilistic=True,
            vaccinated=vaccinated,
            wears_mask=wears_mask,
            resistant=resistant
        )
        service.add_patient(p)

    # Initialize some infected states
    initial_infectious = min(5, len(service.patients))
    for i in range(initial_infectious):
        # Instead of directly I, we can pick A or I
        if random.random() < 0.5:
            service.patients[i].state = EpidemicState.A
        else:
            service.patients[i].state = EpidemicState.I

    # Statistics
    t_vals = []
    state_counts = defaultdict(lambda: defaultdict(list))
    icu_occupied = []
    icu_capacity_total = []
    vaccination_rates = []
    mask_usage_rates = []
    spatial_data = defaultdict(lambda: defaultdict(list))

    for step in range(1, max_steps+1):
        service.step(
            current_step=step,
            beta_h=beta_h,
            beta_g=beta_g,
            beta_asym=beta_asym,
            sigma=sigma,
            p_asym=p_asym,
            eta=eta,
            alpha=alpha,
            gamma_=gamma_,
            gamma_u=gamma_u,
            mu_i=mu_i,
            mu_u=mu_u,
            mu_a=mu_a,
            mu_nat=mu_nat,
            p_resus=p_resus,
            omega_v=omega_v,
            ic_reduction=ic_reduction
        )

        counts = service.get_state_counts()
        for st in EpidemicState:
            state_counts[st][replicate_id].append(counts[st])
        t_vals.append(step)

        occ, cap_total = service.get_icu_occupancy()
        icu_occupied.append(occ)
        icu_capacity_total.append(cap_total)

        vaccination_rates.append(service.get_vaccination_rate())
        total_agents = len(service.workers) + len(service.patients)
        total_masked = sum(1 for w in service.workers if w.wears_mask) \
                       + sum(1 for p in service.patients if p.wears_mask)
        if total_agents > 0:
            mask_usage_rates.append(total_masked / total_agents)
        else:
            mask_usage_rates.append(0)

        # Spatial data
        for ag in (service.workers + service.patients):
            spatial_data[step][ag.state].append((ag.current_cell.x, ag.current_cell.y))

        if step % 50 == 0 or step == max_steps:
            service.logger.info(
                f"Replica {replicate_id} - Step={step} => "
                f"Infectious={service.get_total_infectious()}, "
                f"Deaths={service.deaths_count}, Recoveries={service.recoveries_count}, "
                f"TotalP={len(service.patients)}, UCI={occ}/{cap_total}, "
                f"VaccRate={service.get_vaccination_rate():.2%}, "
                f"MaskUse={mask_usage_rates[-1]:.2%}"
            )

    return {
        't_vals': t_vals,
        'state_counts': state_counts,
        'icu_occupied': icu_occupied,
        'icu_capacity_total': icu_capacity_total,
        'vaccination_rates': vaccination_rates,
        'mask_usage_rates': mask_usage_rates,
        'spatial_data': spatial_data
    }

def main():
    num_replicates = 5
    max_steps = 100
    initial_patients = 50
    initial_workers = 10

    all_state_counts = defaultdict(list)
    all_icu_occupied = []
    all_icu_capacity_total = []
    all_vaccination_rates = []
    all_mask_usage_rates = []
    all_t_vals = []
    all_spatial_data = []

    for replicate in range(1, num_replicates+1):
        replicate_id = f"Rep-{replicate}"
        print(f"Starting {replicate_id}...")
        results = run_simulation(replicate_id, max_steps, initial_patients, initial_workers)

        all_t_vals = results['t_vals']
        for st in EpidemicState:
            all_state_counts[st].append(results['state_counts'][st][replicate_id])
        all_icu_occupied.append(results['icu_occupied'])
        all_icu_capacity_total.append(results['icu_capacity_total'])
        all_vaccination_rates.append(results['vaccination_rates'])
        all_mask_usage_rates.append(results['mask_usage_rates'])
        all_spatial_data.append(results['spatial_data'])

    # Compute summary statistics
    state_stats = {}
    for st in EpidemicState:
        arr = np.array(all_state_counts[st])  # shape: (num_replicates, max_steps)
        mean = arr.mean(axis=0)
        lower = np.percentile(arr, 2.5, axis=0)
        upper = np.percentile(arr, 97.5, axis=0)
        state_stats[st] = {'mean': mean, 'lower': lower, 'upper': upper}

    icu_occ_arr = np.array(all_icu_occupied)
    mean_icu_occ = icu_occ_arr.mean(axis=0)
    lower_icu_occ = np.percentile(icu_occ_arr, 2.5, axis=0)
    upper_icu_occ = np.percentile(icu_occ_arr, 97.5, axis=0)

    icu_cap_arr = np.array(all_icu_capacity_total)
    mean_icu_cap = icu_cap_arr.mean(axis=0)
    lower_icu_cap = icu_cap_arr.min(axis=0)
    upper_icu_cap = icu_cap_arr.max(axis=0)

    vacc_arr = np.array(all_vaccination_rates)
    mean_vacc = vacc_arr.mean(axis=0)
    lower_vacc = np.percentile(vacc_arr, 2.5, axis=0)
    upper_vacc = np.percentile(vacc_arr, 97.5, axis=0)

    mask_arr = np.array(all_mask_usage_rates)
    mean_mask = mask_arr.mean(axis=0)
    lower_mask = np.percentile(mask_arr, 2.5, axis=0)
    upper_mask = np.percentile(mask_arr, 97.5, axis=0)

    # Prepare data for visualization
    state_counts_replicates = {}
    for st in EpidemicState:
        state_counts_replicates[st] = all_state_counts[st]

    # Matplotlib plots
    plot_epidemic(all_t_vals, state_counts_replicates, num_replicates)
    plot_icu(all_t_vals, all_icu_occupied, all_icu_capacity_total, num_replicates)
    plot_vaccination_mask(all_t_vals, all_vaccination_rates, all_mask_usage_rates, num_replicates)

    replicate_to_visualize = 0
    spatial_data = all_spatial_data[replicate_to_visualize]
    steps_to_visualize = [0, max_steps//4, max_steps//2, 3*max_steps//4, max_steps]
    plot_spatial(spatial_data, steps_to_visualize, width=20, height=20)
    plot_interactive(spatial_data, max_steps, 20, 20)

    # Export CSV
    import csv
    with open("resultados_simulacion_klebsiella_enhanced.csv", "w", newline='') as csvfile:
        fieldnames = [
            'Step',
            'S_Mean','S_Lower','S_Upper',
            'V_Mean','V_Lower','V_Upper',
            'E_Mean','E_Lower','E_Upper',
            'A_Mean','A_Lower','A_Upper',
            'I_Mean','I_Lower','I_Upper',
            'U_Mean','U_Lower','U_Upper',
            'R_Mean','R_Lower','R_Upper',
            'D_Mean','D_Lower','D_Upper',
            'ICU_Occ_Mean','ICU_Occ_Lower','ICU_Occ_Upper',
            'ICU_Cap_Mean','ICU_Cap_Lower','ICU_Cap_Upper',
            'VaccRate_Mean','VaccRate_Lower','VaccRate_Upper',
            'MaskUse_Mean','MaskUse_Lower','MaskUse_Upper'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_steps):
            writer.writerow({
                'Step': all_t_vals[i],
                'S_Mean': state_stats[EpidemicState.S]['mean'][i],
                'S_Lower': state_stats[EpidemicState.S]['lower'][i],
                'S_Upper': state_stats[EpidemicState.S]['upper'][i],
                'V_Mean': state_stats[EpidemicState.V]['mean'][i],
                'V_Lower': state_stats[EpidemicState.V]['lower'][i],
                'V_Upper': state_stats[EpidemicState.V]['upper'][i],
                'E_Mean': state_stats[EpidemicState.E]['mean'][i],
                'E_Lower': state_stats[EpidemicState.E]['lower'][i],
                'E_Upper': state_stats[EpidemicState.E]['upper'][i],
                'A_Mean': state_stats[EpidemicState.A]['mean'][i],
                'A_Lower': state_stats[EpidemicState.A]['lower'][i],
                'A_Upper': state_stats[EpidemicState.A]['upper'][i],
                'I_Mean': state_stats[EpidemicState.I]['mean'][i],
                'I_Lower': state_stats[EpidemicState.I]['lower'][i],
                'I_Upper': state_stats[EpidemicState.I]['upper'][i],
                'U_Mean': state_stats[EpidemicState.U]['mean'][i],
                'U_Lower': state_stats[EpidemicState.U]['lower'][i],
                'U_Upper': state_stats[EpidemicState.U]['upper'][i],
                'R_Mean': state_stats[EpidemicState.R]['mean'][i],
                'R_Lower': state_stats[EpidemicState.R]['lower'][i],
                'R_Upper': state_stats[EpidemicState.R]['upper'][i],
                'D_Mean': state_stats[EpidemicState.D]['mean'][i],
                'D_Lower': state_stats[EpidemicState.D]['lower'][i],
                'D_Upper': state_stats[EpidemicState.D]['upper'][i],
                'ICU_Occ_Mean': mean_icu_occ[i],
                'ICU_Occ_Lower': lower_icu_occ[i],
                'ICU_Occ_Upper': upper_icu_occ[i],
                'ICU_Cap_Mean': mean_icu_cap[i],
                'ICU_Cap_Lower': lower_icu_cap[i],
                'ICU_Cap_Upper': upper_icu_cap[i],
                'VaccRate_Mean': mean_vacc[i]*100,
                'VaccRate_Lower': lower_vacc[i]*100,
                'VaccRate_Upper': upper_vacc[i]*100,
                'MaskUse_Mean': mean_mask[i]*100,
                'MaskUse_Lower': lower_mask[i]*100,
                'MaskUse_Upper': upper_mask[i]*100
            })

if __name__ == "__main__":
    main()
