# models/states.py

from enum import Enum, auto

class EpidemicState(Enum):
    S = auto()  # Susceptible
    V = auto()  # Vaccinated
    E = auto()  # Exposed
    A = auto()  # Asymptomatic Infectious  (NEW)
    I = auto()  # Symptomatic Infectious
    U = auto()  # ICU
    R = auto()  # Recovered
    D = auto()  # Death
