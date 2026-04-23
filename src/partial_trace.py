"""
Partial Trace Physics Module.

This module contains the core physics logic for partial trace operations:
- Bipartite Hamiltonian construction
- State evolution
- Reduced density matrix calculation

Physical Model:
- Two quantum systems A and B with dimensions N_A and N_B
- Local Hamiltonians H_A, H_B
- Interaction H_int
- Full Hamiltonian H = H_A ⊗ 1 + 1 ⊗ H_B + H_int

Units:
- Dimensionless throughout.
- ℏ = 1.
"""

from dataclasses import dataclass
from typing import Tuple

import functools

import numpy as np
import scipy.linalg

from src.angular_momentum import generate_spin_matrices


# =============================================================================
# Local Hamiltonians
# =============================================================================


def local_hamiltonian(
    dimension: int,
    j: float,
    u: float,
    delta: float,
) -> np.ndarray:
    """Construct local Hamiltonian H = -J J_x + U J_z² + δ J_z.

    Args:
        dimension: System dimension N.
        j: Tunneling strength.
        u: On-site interaction.
        delta: Energy shift.

    Returns:
        Hamiltonian matrix of shape (N, N).
    """
    jx, jz = generate_spin_matrices(dimension)
    return -j * jx + u * jz @ jz + delta * jz


# =============================================================================
# Bipartite Hamiltonian
# =============================================================================


@dataclass
class BipartiteConfig:
    """Configuration for bipartite system."""

    dim_a: int = 2
    dim_b: int = 2
    j_a: float = 1.0
    j_b: float = 0.0
    u_a: float = 0.0
    u_b: float = 0.0
    delta_a: float = 0.0
    delta_b: float = 1.0
    alpha_xx: float = 0.0
    alpha_xz: float = -1.0
    alpha_zx: float = 0.0
    alpha_zz: float = 0.0


def build_bipartite_hamiltonian(
    config: BipartiteConfig,
) -> np.ndarray:
    """Build full bipartite Hamiltonian.

    H = H_A ⊗ 1_B + 1_A ⊗ H_B + H_int

    Args:
        config: System configuration.

    Returns:
        Full Hamiltonian of shape (N_A * N_B, N_A * N_B).
    """
    n_a, n_b = config.dim_a, config.dim_b
    jx_a, jz_a = generate_spin_matrices(n_a)
    jx_b, jz_b = generate_spin_matrices(n_b)

    # Local Hamiltonians
    h_a = -config.j_a * jx_a + config.u_a * jz_a @ jz_a + config.delta_a * jz_a
    h_b = -config.j_b * jx_b + config.u_b * jz_b @ jz_b + config.delta_b * jz_b

    # Interaction
    interaction = functools.reduce(
        lambda x, y: x + y,
        [
            config.alpha_xx * np.kron(jx_a, jx_b),
            config.alpha_xz * np.kron(jx_a, jz_b),
            config.alpha_zx * np.kron(jz_a, jx_b),
            config.alpha_zz * np.kron(jz_a, jz_b),
        ],
    )

    # Full Hamiltonian
    full = functools.reduce(
        lambda x, y: x + y,
        [
            np.kron(h_a, np.eye(n_b) / n_b),  # Tr_B[1/N] = 1/N
            np.kron(np.eye(n_a) / n_a, h_b),
            interaction,
        ],
    )

    return full


def build_bipartite_hamiltonian_components(
    config: BipartiteConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all Hamiltonian components separately.

    Args:
        config: System configuration.

    Returns:
        Tuple of (H_A, H_B, H_int, H_full).
    """
    n_a, n_b = config.dim_a, config.dim_b
    jx_a, jz_a = generate_spin_matrices(n_a)
    jx_b, jz_b = generate_spin_matrices(n_b)

    # Local
    h_a = -config.j_a * jx_a + config.u_a * jz_a @ jz_a + config.delta_a * jz_a
    h_b = -config.j_b * jx_b + config.u_b * jz_b @ jz_b + config.delta_b * jz_b

    # Interaction
    h_int = functools.reduce(
        lambda x, y: x + y,
        [
            config.alpha_xx * np.kron(jx_a, jx_b),
            config.alpha_xz * np.kron(jx_a, jz_b),
            config.alpha_zx * np.kron(jz_a, jx_b),
            config.alpha_zz * np.kron(jz_a, jz_b),
        ],
    )

    # Full
    full = functools.reduce(
        lambda x, y: x + y,
        [
            np.kron(h_a, np.eye(n_b) / n_b),
            np.kron(np.eye(n_a) / n_a, h_b),
            h_int,
        ],
    )

    return h_a, h_b, h_int, full


# =============================================================================
# State Evolution
# =============================================================================


def evolve_state(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time: float,
) -> np.ndarray:
    """Evolve state under Hamiltonian.

    |ψ(t)⟩ = e^{-iHt}|ψ₀⟩

    Args:
        hamiltonian: Hamiltonian matrix.
        initial_state: Initial state vector.
        time: Evolution time.

    Returns:
        Evolved state vector.
    """
    return scipy.linalg.expm(-1j * time * hamiltonian) @ initial_state


def evolve_density_matrix(
    hamiltonian: np.ndarray,
    initial_density: np.ndarray,
    time: float,
) -> np.ndarray:
    """Evolve density matrix under Hamiltonian.

    ρ(t) = e^{-iHt} ρ(0) e^{iHt}

    Args:
        hamiltonian: Hamiltonian.
        initial_density: Initial density matrix.
        time: Evolution time.

    Returns:
        Evolved density matrix.
    """
    unitary = scipy.linalg.expm(-1j * time * hamiltonian)
    return unitary @ initial_density @ unitary.conj().T


# =============================================================================
# Partial Trace
# =============================================================================


def partial_trace_a(
    full_density: np.ndarray,
    dim_a: int,
    dim_b: int,
) -> np.ndarray:
    """Trace out system B to get reduced density of A.

    ρ_A = Tr_B[ρ_AB] = Σ_j (1 ⊗ ⟨j|) ρ_AB (1 ⊗ |j⟩)

    Args:
        full_density: Full density matrix (N_A*N_B, N_A*N_B).
        dim_a: Dimension of system A.
        dim_b: Dimension of system B.

    Returns:
        Reduced density matrix for A (N_A, N_A).
    """
    return np.trace(
        full_density.reshape(dim_a, dim_b, dim_a, dim_b),
        axis1=1,
        axis2=3,
    )


def partial_trace_b(
    full_density: np.ndarray,
    dim_a: int,
    dim_b: int,
) -> np.ndarray:
    """Trace out system A to get reduced density of B.

    Args:
        full_density: Full density matrix.
        dim_a: Dimension of system A.
        dim_b: Dimension of system B.

    Returns:
        Reduced density matrix for B (N_B, N_B).
    """
    return np.trace(
        full_density.reshape(dim_a, dim_b, dim_a, dim_b),
        axis1=0,
        axis2=2,
    )


# =============================================================================
# Combined Operations
# =============================================================================


def compute_reduced_densities(
    config: BipartiteConfig,
    time: float,
    init_state_a: int = 0,
    init_state_b: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all reduced density matrices after evolution.

    Args:
        config: System configuration.
        time: Evolution time.
        init_state_a: Initial state of A.
        init_state_b: Initial state of B.

    Returns:
        Tuple of (ρ_A, ρ_B, ρ_full).
    """
    # Build Hamiltonian
    hamiltonian = build_bipartite_hamiltonian(config)

    # Initial state |init_state_a⟩ ⊗ |init_state_b⟩
    n_a, n_b = config.dim_a, config.dim_b
    psi0 = np.zeros(n_a * n_b, dtype=complex)
    psi0[init_state_a * n_b + init_state_b] = 1.0

    # Evolve
    psi_t = evolve_state(hamiltonian, psi0, time)

    # Density matrix
    rho_full = np.outer(psi_t, psi_t.conj())

    # Reduced densities
    rho_a = partial_trace_a(rho_full, n_a, n_b)
    rho_b = partial_trace_b(rho_full, n_a, n_b)

    return rho_a, rho_b, rho_full


# =============================================================================
# Validation
# =============================================================================


def validate_partial_trace(
    rho_full: np.ndarray,
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Validate partial trace properties.

    Checks:
    1. Tr_A[Tr_B[ρ]] = Tr[ρ] = 1
    2. Tr_B[ρ] has trace = Tr_A[ρ]
    3. Reduced matrices are positive semidefinite

    Args:
        full_density: Full density matrix.
        rho_a: Reduced density for A.
        rho_b: Reduced density for B.
        tolerance: Tolerance.

    Returns:
        True if valid.
    """
    # Check trace = 1
    if not np.isclose(np.trace(rho_full), 1.0, atol=tolerance):
        return False

    # Check equality of traces
    if not np.isclose(
        np.trace(rho_a), np.trace(rho_b), atol=tolerance
    ):
        return False

    # Check Hermitian
    if not np.allclose(rho_full, rho_full.conj().T):
        return False

    # Check positive semidefinite (full)
    eigenvals = np.linalg.eigvalsh(rho_full)
    if not np.all(eigenvals >= -tolerance):
        return False

    return True