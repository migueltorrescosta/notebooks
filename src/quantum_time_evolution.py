"""
Quantum Time Evolution Physics Module.

This module contains the core physics logic for 1D quantum time evolution:
- Hamiltonian construction on a spatial grid
- Eigendecomposition
- Time evolution via spectral decomposition

Physical Model:
- 1D particle with Hamiltonian H = P²/2m + V(x)
- Discretized on a spatial grid with N points
- Uses finite difference method for kinetic term

Units:
- ℏ = 1 (natural units)
- Position in arbitrary units (x)
- Energy in same units as position⁻²
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Any
from src.enums import BoundaryCondition

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from src.validators import (
    validate_orthonormality,
    validate_probability_conservation,
)

# Aliases for backward compatibility
validate_orthonormality = validate_orthonormality
validate_probability_conservation = validate_probability_conservation


# =============================================================================
# Potential Functions
# =============================================================================


def potential_quadratic(x: float, a: float, c: float) -> float:
    """Quadratic potential V(x) = a(x - c)²."""
    return a * (x - c) ** 2


def potential_quartic(x: float, a: float, c: float) -> float:
    """Quartic potential V(x) = a(x - c)⁴."""
    return a * (x - c) ** 4


def potential_trigonometric(
    x: float, amplitude: float, phase: float, k: float, width: float
) -> float:
    """Trigonometric potential V(x) = a*cos(phase + 2πkx/w)."""
    return amplitude * np.cos(phase + k * 2 * np.pi * x / width)


def potential_uniform(x: float, a: float) -> float:
    """Uniform potential V(x) = a*x."""
    return a * x


def potential_double_well(x: float, a: float, b: float, c: float) -> float:
    """Double well potential V(x) = a(x⁴ + 2x²) + b*exp(-cx²)."""
    return a * (x**4 + 2 * x**2) + b * np.exp(-c * x**2)


# =============================================================================
# Initial Wave Packets
# =============================================================================


def gaussian_wave_packet(x: np.ndarray, d: float, x0: float, p: float) -> np.ndarray:
    """Gaussian wave packet ψ(x) = exp(-d(x-x₀)² + ipx).

    Args:
        x: Spatial grid.
        d: Width parameter (positive).
        x0: Center position.
        p: Momentum.

    Returns:
        Complex wavefunction array.
    """
    return np.exp(-d * (x - x0) ** 2 - 1.0j * p * x)


def step_wave_packet(x: np.ndarray, r: float, s: float, p: float) -> np.ndarray:
    """Step wave packet ψ(x) = exp(ipx) * 1_[r,s](x).

    Args:
        x: Spatial grid.
        r: Left boundary.
        s: Right boundary.
        p: Momentum.

    Returns:
        Complex wavefunction array.
    """
    return np.exp(1j * p * x) * (x >= r) * (x <= s)


# =============================================================================
# Hamiltonian Construction
# =============================================================================


def build_1d_hamiltonian(
    spatial_points: int,
    dx: float,
    potential_function: Callable[[float], float],
    boundary_condition: BoundaryCondition,
    mass: float = 1.0,
) -> scipy.sparse.csc_matrix:
    """Build 1D Hamiltonian using finite differences.

    Constructs H = T + V where:
    - T = -∂²/2m (kinetic energy via central difference)
    - V = V(x) (potential energy)

    The Laplacian is approximated as:
    ∂²ψ/∂x² ≈ (ψ(x+dx) - 2ψ(x) + ψ(x-dx))/dx²

    Args:
        spatial_points: Number of grid points N_x.
        dx: Grid spacing.
        potential_function: V(x) function.
        boundary_condition: Dirichlet or Cyclic.
        mass: Particle mass (default 1.0).

    Returns:
        Hamiltonian as sparse matrix.
    """
    n = spatial_points

    # Kinetic term T = -∂²/2m (set mass=1 for simplicity)
    # Using second derivative central difference: d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1})/dx²
    # T = -1/2 * d²/dx² → matrix with -1 on off-diagonals, 2 on diagonal
    hamiltonian = scipy.sparse.eye(n, n, format="lil") * 2

    # Add second derivative (neighbor connections)
    for i in range(n - 1):
        hamiltonian[i, i + 1] = -1
        hamiltonian[i + 1, i] = -1

    # Cyclic boundary conditions
    if boundary_condition == BoundaryCondition.Cyclic:
        hamiltonian[0, n - 1] = -1
        hamiltonian[n - 1, 0] = -1

    hamiltonian = hamiltonian.tocsc() / (2 * mass * dx**2)

    # Add potential term V(x)
    for i in range(n):
        hamiltonian[i, i] = hamiltonian[i, i] + potential_function(float(i * dx))

    return hamiltonian


# =============================================================================
# Eigendecomposition
# =============================================================================


@dataclass
class EnergyLevel:
    """Energy level data."""

    level: int
    energy: float
    wave_function: np.ndarray
    component: float  # Overlap with initial state


def compute_energy_levels(
    hamiltonian: scipy.sparse.csc_matrix,
    initial_state: np.ndarray,
    num_levels: int,
) -> List[EnergyLevel]:
    """Compute lowest energy eigenstates.

    Finds the num_levels lowest eigenvalues and eigenvectors,
    then computes their overlap with the initial state.

    Args:
        hamiltonian: The Hamiltonian matrix.
        initial_state: Initial wavefunction ψ₀.
        num_levels: Number of lowest levels to compute.

    Returns:
        List of EnergyLevel objects.
    """
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
        hamiltonian, k=num_levels, which="SM"
    )

    levels = []
    for level, energy, wf in zip(
        range(num_levels), np.real(eigenvalues), eigenvectors.T
    ):
        component = float(np.vdot(initial_state, wf))
        levels.append(
            EnergyLevel(
                level=level,
                energy=energy,
                wave_function=wf,
                component=component,
            )
        )

    return levels


def normalize_energy_levels(
    levels: List[EnergyLevel],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize energy levels so components are real.

    Transforms each wavefunction so that its component
    with the initial state is a real positive number.

    Args:
        levels: List of energy levels.

    Returns:
        Tuple of (wave_functions_matrix, components, energies).
    """
    # Normalize to ensure |component| = sqrt(probability)
    for el in levels:
        if np.abs(el.component) < 1e-10:
            continue
        # Make component real and positive
        phase = el.component / np.abs(el.component)
        el.wave_function = el.wave_function / phase
        el.component = np.abs(el.component)

    # Stack into matrices for efficient computation
    wave_functions_matrix = np.stack([el.wave_function for el in levels])
    components = np.array([el.component for el in levels])
    energies = np.array([el.energy for el in levels])

    return wave_functions_matrix, components, energies


# =============================================================================
# State Preparation
# =============================================================================


def prepare_initial_state(
    x_grid: np.ndarray,
    state_type: str,
    **kwargs,
) -> np.ndarray:
    """Prepare an initial wavefunction.

    Args:
        x_grid: Spatial grid.
        state_type: Type of state ('gaussian' or 'step').
        **kwargs: Parameters specific to state_type.

    Returns:
        Normalized wavefunction.
    """
    match state_type.lower():
        case "gaussian":
            wf = gaussian_wave_packet(
                x_grid,
                d=kwargs.get("d", 1.0),
                x0=kwargs.get("x0", 0.0),
                p=kwargs.get("p", 0.0),
            )
        case "step":
            wf = step_wave_packet(
                x_grid,
                r=kwargs.get("r", -1.0),
                s=kwargs.get("s", 1.0),
                p=kwargs.get("p", 0.0),
            )
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")

    # Normalize
    norm = np.sqrt(np.sum(np.abs(wf) ** 2))
    if norm > 0:
        wf = wf / norm
    return wf


# =============================================================================
# Time Evolution
# =============================================================================


class TimeEvolver:
    """Efficient time evolver using precomputed energy levels."""

    def __init__(
        self,
        wave_functions: np.ndarray,
        components: np.ndarray,
        energies: np.ndarray,
    ):
        """Initialize evolver with precomputed data.

        Args:
            wave_functions: Matrix of shape (n_levels, n_points).
            components: Overlap with initial state (n_levels,).
            energies: Eigenvalues (n_levels,).
        """
        self.wave_functions = wave_functions
        self.components = components
        self.energies = energies

    def evolve(self, t: float) -> np.ndarray:
        """Evolve to time t.

        ψ(t) = Σ_k c_k * exp(-i E_k t) * φ_k

        Uses vectorized einsum for efficiency.

        Args:
            t: Evolution time.

        Returns:
            Evolved wavefunction.
        """
        # Phase factors: c_k * exp(-i E_k t)
        phases = self.components * np.exp(-1.0j * t * self.energies)

        # Weighted sum of wavefunctions
        wf = np.einsum("k,kx->x", phases, self.wave_functions)

        # Normalize
        norm = np.linalg.norm(wf)
        if norm > 0:
            wf = wf / norm
        return wf

    def evolve_trajectory(
        self,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evolve over multiple times.

        Args:
            times: Array of time points.

        Returns:
            Matrix of shape (n_times, n_points) with |ψ(x,t)|².
        """
        return np.array([np.abs(self.evolve(t)) ** 2 for t in times])


# =============================================================================


# =============================================================================
# Convenience Functions
# =============================================================================


def run_simulation(
    x_min: float,
    x_max: float,
    num_points: int,
    potential_fn: Callable[[float], float],
    initial_state_fn: Callable[[np.ndarray], np.ndarray],
    num_levels: int = 20,
    boundary: str = "Dirichlet",
) -> dict[str, Any]:
    """Run a complete simulation.

    Convenience function that performs all steps:
    1. Build grid and initial state
    2. Build Hamiltonian
    3. Compute energy levels
    4. Normalize
    5. Return data for UI

    Args:
        x_min: Minimum x value.
        x_max: Max x value.
        num_points: Number of grid points.
        potential_fn: Potential V(x).
        initial_state_fn: Function to create initial state.
        num_levels: Number of energy levels.
        boundary: 'Dirichlet' or 'Cyclic'.

    Returns:
        Dictionary with all simulation data.
    """
    # Grid
    dx = (x_max - x_min) / (num_points - 1)
    x_grid = np.linspace(x_min, x_max, num_points)

    # Initial state
    psi0 = initial_state_fn(x_grid)
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2))

    # Hamiltonian
    bc = (
        BoundaryCondition.Cyclic
        if boundary == "Cyclic"
        else BoundaryCondition.Dirichlet
    )
    hamiltonian = build_1d_hamiltonian(num_points, dx, potential_fn, bc)

    # Energy levels
    levels = compute_energy_levels(hamiltonian, psi0, num_levels)

    # Normalize
    wf_matrix, components, energies = normalize_energy_levels(levels)

    # Orthonormality check
    ortho_error = validate_orthonormality(wf_matrix.T)

    return {
        "x_grid": x_grid,
        "dx": dx,
        "initial_state": psi0,
        "hamiltonian": hamiltonian,
        "energy_levels": levels,
        "wave_functions": wf_matrix,
        "components": components,
        "energies": energies,
        "orthonormality_error": ortho_error,
        "total_probability": np.sum(components**2),
    }
