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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from src.utils.enums import BoundaryCondition
from src.utils.validators import (
    validate_orthonormality,
)

# Alias for backward compatibility
# Agent Notes: This re-exports validators from src.utils.validators
# so that calling code can import them from quantum_time_evolution directly.
# If the source functions in validators.py are moved or renamed,
# update this alias to maintain backward compatibility.
validate_orthonormality = validate_orthonormality


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
    x_grid: np.ndarray | None = None,
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
        x_grid: Optional array of spatial positions for potential evaluation.
            If provided, V(x) is evaluated at these points instead of
            arange(spatial_points) * dx. Must have length spatial_points.

    Returns:
        Hamiltonian as sparse matrix.

    """
    n = spatial_points

    # Kinetic term T = -∂²/2m using central differences:
    # d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1})/dx²
    # T = -1/(2m) * d²/dx² → tridiag(-1, 2, -1) / (2*m*dx²)
    T = scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")

    # Cyclic boundary conditions: wrap-around entries at [0, n-1] and [n-1, 0]
    if boundary_condition == BoundaryCondition.Cyclic:
        T = T + scipy.sparse.diags(
            [-1.0, -1.0],
            [-(n - 1), n - 1],
            shape=(n, n),
            format="csc",
        )

    # Use explicit constructor to help mypy understand the type
    T = scipy.sparse.csc_matrix(T / (2 * mass * dx**2))

    # Potential term V(x) as diagonal matrix
    if x_grid is not None:
        assert len(x_grid) == n, (
            f"x_grid length {len(x_grid)} must match spatial_points {n}"
        )
        x_vals = x_grid
    else:
        x_vals = np.arange(n, dtype=float) * dx
    V_diag = np.array([potential_function(float(x)) for x in x_vals])
    V = scipy.sparse.diags(V_diag, 0, shape=(n, n), format="csc")

    return T + V


# =============================================================================
# Eigendecomposition
# =============================================================================


@dataclass
class EnergyLevel:
    """Energy level data."""

    level: int
    energy: float
    wave_function: np.ndarray
    component: complex  # Overlap with initial state (complex)


def compute_energy_levels(
    hamiltonian: scipy.sparse.csc_matrix,
    initial_state: np.ndarray,
    num_levels: int,
) -> list[EnergyLevel]:
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
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        hamiltonian,
        k=num_levels,
        which="SA",
    )

    levels = []
    for level, energy, wf in zip(
        range(num_levels),
        np.real(eigenvalues),
        eigenvectors.T,
        strict=False,
    ):
        # Store complex overlap - normalize_energy_levels will make it real
        component = complex(np.vdot(initial_state, wf))
        levels.append(
            EnergyLevel(
                level=level,
                energy=energy,
                wave_function=wf,
                component=component,
            ),
        )

    return levels


def normalize_energy_levels(
    levels: list[EnergyLevel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Components are now real after normalization (np.abs returns float)
    components = np.real(np.array([el.component for el in levels]))
    energies = np.array([el.energy for el in levels])

    return wave_functions_matrix, components, energies


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
        dt: float = 1.0,
    ):
        """Initialize evolver with precomputed data.

        Args:
            wave_functions: Matrix of shape (n_levels, n_points).
            components: Overlap with initial state (n_levels,).
            energies: Eigenvalues (n_levels,).
            dt: Time step scaling factor applied in the phase evolution
                as exp(-i * dt * E_k * t). Default 1.0.

        """
        self.wave_functions = wave_functions
        self.components = components
        self.energies = energies
        self.dt = dt

    def evolve(self, t: float) -> np.ndarray:
        """Evolve to time t.

        ψ(t) = Σ_k c_k * exp(-i E_k t) * φ_k

        Uses vectorized einsum for efficiency.

        Args:
            t: Evolution time.

        Returns:
            Evolved wavefunction.

        """
        # Phase factors: c_k * exp(-i * dt * E_k * t)
        phases = self.components * np.exp(-1.0j * self.dt * t * self.energies)

        # Weighted sum of wavefunctions
        wf = np.einsum("k,kx->x", phases, self.wave_functions)

        # Normalize
        norm = np.linalg.norm(wf)
        if norm > 0:
            wf = wf / norm
        assert np.isclose(np.linalg.norm(wf), 1.0, rtol=1e-5, atol=1e-8), (
            f"Evolved state at t={t} not normalized: norm={np.linalg.norm(wf)}"
        )
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
    assert np.isclose(np.linalg.norm(psi0), 1.0, rtol=1e-5, atol=1e-8), (
        f"Initial state not normalized after normalization: norm={np.linalg.norm(psi0)}"
    )

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

    # Invariant check: total probability must be ≤ 1 (truncation may reduce it)
    total_prob = np.sum(components**2)
    assert total_prob <= 1.0 + 1e-12, f"Total probability exceeds 1: {total_prob}"
    assert np.isclose(total_prob, 1.0, rtol=1e-2, atol=5e-3), (
        f"Total probability not close to 1 (truncation artifact with {num_levels} levels): {total_prob}"
    )

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
