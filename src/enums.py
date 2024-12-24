from enum import Enum


class WavePacket(str, Enum):
    Gaussian = "Gaussian"
    Step = "Step function"
    # Airy = "Airy"
    # Morse = "Morse"
    # Solitary = "Solitary"


class PotentialFunction(str, Enum):
    DoubleWell = "Double-well"
    Quadratic = "Quadratic"
    Quartic = "Quartic"
    Trigonometric = "Trigonometric"
    Uniform = "Uniform"


class BoundaryCondition(str, Enum):
    Cyclic = "Cyclic"
    Dirichlet = "Dirichlet"

class ProbabilityDistribution(str, Enum):
    ParticleDecay = "ParticleDecay"