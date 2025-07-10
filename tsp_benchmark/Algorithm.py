from enum import Enum
class Algorithm(Enum):
    """Available algorithms"""
    SIMULATED_ANNEALING = "simulated_annealing"
    QAOA = "qaoa"
    HELD_KARP = "held_karp"