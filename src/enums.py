"""
Enumerated types for quantum simulation configuration.

This module defines string-based enumerations used throughout the
simulation framework to ensure consistent naming of physical models,
potential functions, boundary conditions, and probability distributions.
"""

from enum import Enum


class WavePacket(str, Enum):
    """Wave packet shape for initial state preparation.

    Attributes:
        Gaussian: Gaussian wave packet (coherent state in position).
        Step: Step function (Heaviside-like) wave packet.
    """

    Gaussian = "Gaussian"
    Step = "Step function"
    # Airy = "Airy"
    # Morse = "Morse"
    # Solitary = "Solitary"


class PotentialFunction(str, Enum):
    """Potential energy function types for quantum systems.

    Attributes:
        DoubleWell: Double-well potential (two minima, barrier in middle).
        Quadratic: Harmonic oscillator potential (V ∝ x²).
        Quartic: Quartic potential (V ∝ x⁴).
        Trigonometric: Periodic potential (V ∝ cos(kx)).
        Uniform: Constant potential (no confinement).
    """

    DoubleWell = "Double-well"
    Quadratic = "Quadratic"
    Quartic = "Quartic"
    Trigonometric = "Trigonometric"
    Uniform = "Uniform"


class BoundaryCondition(str, Enum):
    """Boundary conditions for spatial wavefunctions.

    Attributes:
        Cyclic: Periodic boundary conditions (wavefunction matches at edges).
        Dirichlet: Fixed boundary conditions (wavefunction = 0 at boundaries).
    """

    Cyclic = "Cyclic"
    Dirichlet = "Dirichlet"


class ProbabilityDistribution(str, Enum):
    """Named probability distribution types for stochastic processes.

    Attributes:
        ParticleDecay: Radioactive decay distribution (exponential).
    """

    ParticleDecay = "ParticleDecay"
