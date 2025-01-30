# models/agent.py

import random
import math
import uuid
from models.cell import Cell
from models.states import EpidemicState
import numpy as np

class Agent:
    """
    Base class for any agent in the simulation.
    """
    def __init__(self, unique_id, cell: Cell):
        self.unique_id = unique_id
        self.current_cell = cell
        if cell is not None:
            cell.add_agent(self)

    def set_current_cell(self, new_cell: Cell):
        if self.current_cell:
            self.current_cell.remove_agent(self)
        self.current_cell = new_cell
        if new_cell:
            new_cell.add_agent(self)

    def step(self, current_step, simulation_service):
        pass

    def _prob(self, rate):
        """
        Given a continuous rate 'rate', converts it into a probability 
        of occurrence over one time step (Δt = 1).
        p = 1 - exp(-rate)
        """
        return 1 - math.exp(-rate)


class HealthcareWorker(Agent):
    """
    Healthcare worker with states:
      S -> V -> E -> A -> I -> U -> R -> D
    Adapted for Klebsiella pneumoniae with more complexity.
    """
    def __init__(
        self, unique_id, cell,
        # Infection parameters
        beta_h,        # Transmission rate for symptomatic
        beta_asym,     # Transmission rate for asymptomatic
        sigma,         # Rate E->(A or I)
        p_asym,        # Probability that E->A instead of E->I
        eta,           # Rate A->I
        alpha,         # Rate I->U
        gamma_,        # Rate I->R
        gamma_u,       # Rate U->R
        mu_i,          # Mortality in I
        mu_u,          # Mortality in U
        mu_a,          # Mortality in A (if relevant)
        mu_nat,        # Natural mortality
        p_resus,       # Rate R->S
        omega_v,       # Rate V->S (waning of vaccine immunity)
        ic_reduction,  # Infection control measure: 0..1 (0 = none, 1 = total)
        age=None,
        use_probabilistic=True,
        vaccinated=False,
        wears_mask=True,
        resistant=False
    ):
        super().__init__(unique_id, cell)
        self.state = EpidemicState.S
        self.vaccinated = vaccinated
        self.wears_mask = wears_mask
        self.resistant = resistant

        # Epi rates
        self.beta_h = beta_h
        self.beta_asym = beta_asym
        self.sigma = sigma
        self.p_asym = p_asym
        self.eta = eta
        self.alpha = alpha
        self.gamma = gamma_
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u
        self.mu_a = mu_a
        self.mu_nat = mu_nat
        self.p_resus = p_resus
        self.omega_v = omega_v
        self.ic_reduction = ic_reduction

        # Adjust age if needed
        if age is None:
            age = random.gauss(40, 10)
        self.age = age

        # Scale natural mortality by age
        age_factor = 1.0 + max(0, (self.age - 50) / 50) * 1.5
        self.mu_nat *= age_factor

        self.use_probabilistic = use_probabilistic

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return  # Dead => do nothing

        # (1) Natural death
        p_death_nat = self._prob(self.mu_nat)
        if random.random() < p_death_nat:
            self.state = EpidemicState.D
            simulation_service.register_death(self)
            simulation_service.logger.debug(f"{self.unique_id} died of natural causes.")
            return

        # (2) State transitions
        if self.state == EpidemicState.S:
            # Possibly infected by symptomatic or asymptomatic
            total_inf = simulation_service.get_total_infectious()  # (I + A + U)
            total_pop = simulation_service.get_total_population()

            if total_pop > 0 and total_inf > 0:
                # Effective contact rate
                #   If vaccinated, mask, and infection control => reduce contact
                #   Combine these multiplicatively or additively to scale beta
                beta_eff = self.beta_h
                if self.wears_mask:
                    beta_eff *= 0.7
                if self.vaccinated:
                    beta_eff *= 0.5
                # Infection control factor
                beta_eff *= (1 - self.ic_reduction)

                # Weighted by fraction of asymptomatic vs symptomatic
                # Alternatively, use more refined approach (like separate counts)
                frac_asym = simulation_service.get_asymptomatic_fraction()
                # Weighted average of asymptomatic vs symptomatic rates:
                combined_beta = beta_eff * (1 - frac_asym) + self.beta_asym * frac_asym
                lam = combined_beta * (total_inf / total_pop)

                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} S->E.")

        elif self.state == EpidemicState.V:
            # Check waning immunity: V->S
            p_v_s = self._prob(self.omega_v)
            if random.random() < p_v_s:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} V->S (waning vaccine).")
                return

            # Otherwise, same infection logic as S but presumably lower effective rate
            total_inf = simulation_service.get_total_infectious()
            total_pop = simulation_service.get_total_population()
            if total_pop > 0 and total_inf > 0:
                beta_eff = self.beta_h
                # Already vaccinated => reduce
                beta_eff *= 0.5
                if self.wears_mask:
                    beta_eff *= 0.7
                beta_eff *= (1 - self.ic_reduction)

                frac_asym = simulation_service.get_asymptomatic_fraction()
                combined_beta = beta_eff * (1 - frac_asym) + self.beta_asym * frac_asym
                lam = combined_beta * (total_inf / total_pop)

                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} V->E.")

        elif self.state == EpidemicState.E:
            # E -> A with probability p_asym or E -> I
            p_e_a = self.p_asym
            if random.random() < p_e_a:
                # E->A
                self.state = EpidemicState.A
                simulation_service.logger.debug(f"{self.unique_id} E->A.")
            else:
                # E->I
                p_e_i = self._prob(self.sigma)
                if random.random() < p_e_i:
                    self.state = EpidemicState.I
                    simulation_service.logger.debug(f"{self.unique_id} E->I.")

        elif self.state == EpidemicState.A:
            # Asymptomatic Infectious can progress to I
            p_a_i = self._prob(self.eta)
            if random.random() < p_a_i:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} A->I.")
            else:
                # Optionally, consider a small mortality from A
                p_a_d = self._prob(self.mu_a)
                if random.random() < p_a_d:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    simulation_service.logger.debug(f"{self.unique_id} died in A.")
                else:
                    # Or spontaneously recover from A (some models allow that)
                    # But typically we go A->I->R. It's up to your design.

                    pass  # No direct A->R in this example, unless you add it

        elif self.state == EpidemicState.I:
            # Mortality in I
            p_i_d = self._prob(self.mu_i)
            if random.random() < p_i_d:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} died in I.")
            else:
                # I->U
                p_i_u = self._prob(self.alpha)
                if random.random() < p_i_u:
                    simulation_service.request_icu(self)
                else:
                    # Recovery in I, impacted by antibiotic resistance
                    adjusted_gamma = self.gamma * (0.5 if self.resistant else 1.0)
                    p_i_r = self._prob(adjusted_gamma)
                    if random.random() < p_i_r:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} I->R.")

        elif self.state == EpidemicState.U:
            # Mortality in U
            p_u_d = self._prob(self.mu_u)
            if random.random() < p_u_d:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                self.current_cell.free_bed()
                simulation_service.logger.debug(f"{self.unique_id} died in U.")
            else:
                adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                p_u_r = self._prob(adjusted_gamma_u)
                if random.random() < p_u_r:
                    self.state = EpidemicState.R
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} U->R.")

        elif self.state == EpidemicState.R:
            # Possible re-susceptibilization
            p_r_s = self._prob(self.p_resus)
            if random.random() < p_r_s:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} R->S (re-susceptible).")

        # (3) Movement
        # Healthcare workers might move more often:
        mobility_factor = 0.2  # Adjust as desired or use age-based factor
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)


class Patient(Agent):
    """
    Patient with states:
      S -> V -> E -> A -> I -> U -> R -> D
    Including severity, antibiotic resistance, etc.
    """
    def __init__(
        self, unique_id, cell,
        beta_g,
        beta_asym,
        sigma,
        p_asym,
        eta,
        alpha,
        gamma_,
        gamma_u,
        mu_i,
        mu_u,
        mu_a,
        mu_nat,
        p_resus,
        omega_v,
        ic_reduction,
        severity=1,
        age=None,
        use_probabilistic=True,
        vaccinated=False,
        wears_mask=True,
        resistant=False
    ):
        super().__init__(unique_id, cell)
        self.state = EpidemicState.S
        self.vaccinated = vaccinated
        self.wears_mask = wears_mask
        self.resistant = resistant

        self.beta_g = beta_g
        self.beta_asym = beta_asym
        self.sigma = sigma
        self.p_asym = p_asym
        self.eta = eta
        self.alpha = alpha
        self.gamma = gamma_
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u
        self.mu_a = mu_a
        self.mu_nat = mu_nat
        self.p_resus = p_resus
        self.omega_v = omega_v
        self.ic_reduction = ic_reduction

        if age is None:
            age = random.gauss(65, 15)
        self.age = age

        # Scale natural mortality by age
        age_factor = 1.0 + max(0, (self.age - 60) / 40) * 2.0
        self.mu_nat *= age_factor

        self.use_probabilistic = use_probabilistic
        self.severity = severity  # (1: mild, 2: moderate, 3: severe)
        self.icu_time = 0
        self.max_icu_time = 14  # max days in ICU in this model

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return

        # (1) Natural death
        p_death_nat = self._prob(self.mu_nat)
        if random.random() < p_death_nat:
            self.state = EpidemicState.D
            simulation_service.register_death(self)
            simulation_service.logger.debug(f"{self.unique_id} died of natural causes.")
            return

        # (2) Transitions
        if self.state == EpidemicState.S:
            total_inf = simulation_service.get_total_infectious()
            total_pop = simulation_service.get_total_population()
            if total_pop > 0 and total_inf > 0:
                # Adjust by vaccination, masks, infection control
                beta_eff = self.beta_g
                if self.vaccinated:
                    beta_eff *= 0.5
                if self.wears_mask:
                    beta_eff *= 0.7
                beta_eff *= (1 - self.ic_reduction)

                frac_asym = simulation_service.get_asymptomatic_fraction()
                combined_beta = beta_eff * (1 - frac_asym) + self.beta_asym * frac_asym
                lam = combined_beta * (total_inf / total_pop)

                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} S->E.")

        elif self.state == EpidemicState.V:
            # Waning immunity
            p_v_s = self._prob(self.omega_v)
            if random.random() < p_v_s:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} V->S (waning).")
                return

            total_inf = simulation_service.get_total_infectious()
            total_pop = simulation_service.get_total_population()
            if total_pop > 0 and total_inf > 0:
                beta_eff = self.beta_g
                beta_eff *= 0.5  # vaccination reduces infection rate
                if self.wears_mask:
                    beta_eff *= 0.7
                beta_eff *= (1 - self.ic_reduction)

                frac_asym = simulation_service.get_asymptomatic_fraction()
                combined_beta = beta_eff * (1 - frac_asym) + self.beta_asym * frac_asym
                lam = combined_beta * (total_inf / total_pop)

                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} V->E.")

        elif self.state == EpidemicState.E:
            # E->A or E->I
            if random.random() < self.p_asym:
                self.state = EpidemicState.A
                simulation_service.logger.debug(f"{self.unique_id} E->A.")
            else:
                p_e_i = self._prob(self.sigma)
                if random.random() < p_e_i:
                    self.state = EpidemicState.I
                    simulation_service.logger.debug(f"{self.unique_id} E->I.")

        elif self.state == EpidemicState.A:
            # A->I or (optional) A->R, A->D 
            p_a_i = self._prob(self.eta)
            if random.random() < p_a_i:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} A->I.")
            else:
                # Optional mortality from A
                p_a_d = self._prob(self.mu_a)
                if random.random() < p_a_d:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    simulation_service.logger.debug(f"{self.unique_id} died in A.")

        elif self.state == EpidemicState.I:
            # Mortality
            p_i_d = self._prob(self.mu_i)
            if random.random() < p_i_d:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} died in I.")
            else:
                # Probability to go to ICU
                adjusted_alpha = self.alpha * self.severity * (0.8 if self.vaccinated else 1.0)
                p_i_u = self._prob(adjusted_alpha)
                if random.random() < p_i_u:
                    simulation_service.request_icu(self)
                else:
                    adjusted_gamma = self.gamma * (0.5 if self.resistant else 1.0)
                    p_i_r = self._prob(adjusted_gamma)
                    if random.random() < p_i_r:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} I->R.")

        elif self.state == EpidemicState.U:
            # ICU day-by-day logic
            self.icu_time += 1
            if self.icu_time >= self.max_icu_time:
                # Decide death or recovery
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    simulation_service.logger.debug(
                        f"{self.unique_id} died in U after {self.icu_time} days.")
                else:
                    adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        simulation_service.register_recovery_from_icu(self)
                        simulation_service.logger.debug(
                            f"{self.unique_id} recovered from U after {self.icu_time} days.")
            else:
                # Possibly die or recover earlier
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} died in U.")
                else:
                    adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        self.current_cell.free_bed()
                        simulation_service.logger.debug(f"{self.unique_id} U->R.")

        elif self.state == EpidemicState.R:
            # Possible re-susceptibilization
            p_r_s = self._prob(self.p_resus)
            if random.random() < p_r_s:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} R->S.")
            else:
                # Probability of discharge
                discharge_prob = 0.005
                if random.random() < discharge_prob:
                    simulation_service.remove_patient(self)
                    simulation_service.logger.debug(f"{self.unique_id} discharged from R.")
                    return

        # (3) Movement
        # Typically patients move less:
        mobility_factor = 0.03  # smaller than HCWs
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)
