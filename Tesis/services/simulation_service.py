# services/simulation_service.py

import random
import uuid
from collections import defaultdict
from models.cell import Cell
from models.agent import Agent, HealthcareWorker, Patient
from models.states import EpidemicState
from utils.logger import get_logger
import numpy as np
from scipy.stats import beta, gamma as gamma_dist

class SimulationService:
    def __init__(self, width, height, n_icu_rows=2, icu_capacity=3):
        self.width = width
        self.height = height
        self.grid = []
        for x in range(width):
            col = []
            for y in range(height):
                is_icu = (y < n_icu_rows)
                c = Cell(x, y, is_icu=is_icu, icu_capacity=icu_capacity)
                col.append(c)
            self.grid.append(col)

        self.workers = []
        self.patients = []
        self.deaths_count = 0
        self.recoveries_count = 0

        # Arrival rate of new patients
        self.arrival_rate = 0.4  # reduce or increase to see more arrivals

        self.icu_waiting_queue = []
        self.logger = get_logger()

    def add_worker(self, worker):
        self.workers.append(worker)

    def add_patient(self, patient):
        self.patients.append(patient)

    def remove_patient(self, patient):
        if patient in self.patients:
            self.patients.remove(patient)
        cell = patient.current_cell
        if cell:
            cell.remove_agent(patient)

    def move_agent(self, agent):
        if not agent.current_cell:
            return
        cx = agent.current_cell.x
        cy = agent.current_cell.y
        nx = cx + random.randint(-1, 1)
        ny = cy + random.randint(-1, 1)
        nx = max(0, min(nx, self.width - 1))
        ny = max(0, min(ny, self.height - 1))
        new_cell = self.grid[nx][ny]
        agent.set_current_cell(new_cell)
        self.logger.debug(f"{agent.unique_id} moved to cell ({nx}, {ny}).")

    def get_random_cell(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return self.grid[x][y]

    def get_total_infectious(self):
        """
        Infectious includes: I, A, U
        """
        inf_w = sum(1 for w in self.workers if w.state in [EpidemicState.I, EpidemicState.A, EpidemicState.U])
        inf_p = sum(1 for p in self.patients if p.state in [EpidemicState.I, EpidemicState.A, EpidemicState.U])
        return inf_w + inf_p

    def get_asymptomatic_fraction(self):
        """
        Fraction among infectious that are A (Asymptomatic).
        This is a rough approach to weighting transmissions.
        """
        all_infectious = 0
        asym = 0
        for w in self.workers:
            if w.state in [EpidemicState.I, EpidemicState.A, EpidemicState.U]:
                all_infectious += 1
                if w.state == EpidemicState.A:
                    asym += 1
        for p in self.patients:
            if p.state in [EpidemicState.I, EpidemicState.A, EpidemicState.U]:
                all_infectious += 1
                if p.state == EpidemicState.A:
                    asym += 1
        if all_infectious == 0:
            return 0.0
        return asym / all_infectious

    def get_total_population(self):
        """
        All alive (state != D).
        """
        living_w = sum(1 for w in self.workers if w.state != EpidemicState.D)
        living_p = sum(1 for p in self.patients if p.state != EpidemicState.D)
        return living_w + living_p

    def register_death(self, agent: Agent):
        self.deaths_count += 1

    def register_recovery_from_icu(self, agent: Agent):
        self.recoveries_count += 1
        # free bed
        if agent.current_cell:
            agent.current_cell.free_bed()

    def get_state_counts(self):
        """
        Return how many agents are in each state (workers + patients).
        """
        states = {st: 0 for st in EpidemicState}
        for w in self.workers:
            states[w.state] += 1
        for p in self.patients:
            states[p.state] += 1
        return states

    def spawn_patient_if_needed(self, **params):
        """
        With probability arrival_rate, spawn a new patient in state S.
        """
        if random.random() < self.arrival_rate:
            cell = self.get_random_cell()
            pid = str(uuid.uuid4())
            severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            vaccinated = random.random() < 0.3
            wears_mask = random.random() < 0.7
            resistant = random.random() < 0.3

            p = Patient(
                pid, cell,
                beta_g=params['beta_g'],
                beta_asym=params['beta_asym'],
                sigma=params['sigma'],
                p_asym=params['p_asym'],
                eta=params['eta'],
                alpha=params['alpha'],
                gamma_=params['gamma_'],
                gamma_u=params['gamma_u'],
                mu_i=params['mu_i'],
                mu_u=params['mu_u'],
                mu_a=params['mu_a'],
                mu_nat=params['mu_nat'],
                p_resus=params['p_resus'],
                omega_v=params['omega_v'],
                ic_reduction=params['ic_reduction'],
                severity=severity,
                age=None,
                use_probabilistic=True,
                vaccinated=vaccinated,
                wears_mask=wears_mask,
                resistant=resistant
            )
            self.add_patient(p)
            self.logger.debug(f"New patient {pid} arrived at cell ({cell.x}, {cell.y}).")

    def request_icu(self, agent: Agent):
        priority = -getattr(agent, 'severity', 1)
        self.icu_waiting_queue.append((priority, agent))
        self.icu_waiting_queue.sort()
        self.logger.debug(f"{agent.unique_id} requested ICU (priority {priority}).")
        self.assign_icu_beds()

    def assign_icu_beds(self):
        for priority, agent in list(self.icu_waiting_queue):
            # Only assign if still I (or A?), though typically we do I->U
            # We skip if agent already died, recovered, etc.
            if agent.state != EpidemicState.I:
                self.icu_waiting_queue.remove((priority, agent))
                continue
            bed_found = False
            for col in self.grid:
                for cell in col:
                    if cell.is_icu and cell.has_free_bed():
                        cell.occupy_bed()
                        agent.set_current_cell(cell)
                        agent.state = EpidemicState.U
                        self.icu_waiting_queue.remove((priority, agent))
                        bed_found = True
                        self.logger.debug(
                            f"{agent.unique_id} assigned to ICU at ({cell.x},{cell.y}).")
                        break
                if bed_found:
                    break
            if not bed_found:
                # no free bed, remains in queue
                break

    def get_icu_occupancy(self):
        occupied = sum(cell.occupied_beds for col in self.grid for cell in col if cell.is_icu)
        capacity = sum(cell.icu_capacity for col in self.grid for cell in col if cell.is_icu)
        return occupied, capacity

    def get_vaccination_rate(self):
        total_pop = self.get_total_population()
        if total_pop == 0:
            return 0
        total_vax = 0
        for w in self.workers:
            if w.state != EpidemicState.D and w.vaccinated:
                total_vax += 1
        for p in self.patients:
            if p.state != EpidemicState.D and p.vaccinated:
                total_vax += 1
        return total_vax / total_pop

    def step(self, current_step, **params):
        """
        A single simulation step:
          1. Possibly spawn a new patient
          2. Step each agent
          3. Attempt ICU assignments
        """
        self.spawn_patient_if_needed(**params)

        ws = list(self.workers)
        ps = list(self.patients)

        for w in ws:
            w.step(current_step, self)
        for p in ps:
            p.step(current_step, self)

        self.assign_icu_beds()
