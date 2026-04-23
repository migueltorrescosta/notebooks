"""
Heisenberg Model Physics Module.

This module contains the core physics logic for the transverse field Heisenberg model:
- Hamiltonian construction
- Eigendecomposition
- Expectation value calculations

Physical Model:
- 1D spin chain with N spin-1/2 particles
- Hamiltonian: H = H_J + H_U
- H_J = J Σ σ_i^x σ_{i+1}^x (nearest neighbor coupling)
- H_U = (U/2) Σ σ_i^z (transverse field)

Units:
- Dimensionless throughout.
- ℏ = 1, k_B = 1.
"""

from dataclasses import dataclass
from typing import Tuple

import functools

import numpy as np


# =============================================================================
# Pauli Matrices
# =============================================================================


SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
EYE = np.array([[1, 0], [0, 1]], dtype=complex)


# =============================================================================
# Hamiltonian Construction
# =============================================================================


def heisenberg_hamiltonian(
    n_sites: int,
    j: float = 1.0,
    u: float = 1.0,
) -> np.ndarray:
    """Construct the transverse field Heisenberg Hamiltonian.

    H = H_J + H_U where:
    - H_J = J Σ_{i=0}^{N-1} σ_i^x σ_{i+1}^x (exchange coupling)
    - H_U = (U/2) Σ_{i=0}^{N-1} σ_i^z (transverse field)

    Uses periodic boundary conditions (σ_N^x = σ_0^x).

    Args:
        n_sites: Number of spin sites (N).
        j_coupling: Coupling strength J.
        u_field: Transverse field strength U.

    Returns:
        Hamiltonian matrix of shape (2^N, 2^N).

    Raises:
        ValueError: If n_sites < 1 or n_sites > 26.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if n_sites > 26:
        raise ValueError("n_sites must be <= 26 (Hilbert space grows as 2^N)")

    # Build coupling term H_J = J Σ σ_i^x σ_{i+1}^x
    # Optimized: precompute identities once, then modify in-place
    identities = [EYE] * n_sites
    hamiltonian_coupling = np.zeros((2**n_sites, 2**n_sites), dtype=complex)

    for i in range(n_sites):
        op = identities.copy()
        op[i] = SIGMA_X
        op[(i + 1) % n_sites] = SIGMA_X  # Periodic
        hamiltonian_coupling += functools.reduce(np.kron, op)

    # Build field term H_U = (U/2) Σ σ_i^z
    hamiltonian_field = functools.reduce(
        np.kron, [SIGMA_Z for _ in range(n_sites)]
    )

    # Total Hamiltonian
    hamiltonian = j * hamiltonian_field + (0.5 * u) * hamiltonian_coupling

    return hamiltonian


def heisenberg_coupling_term(n_sites: int) -> np.ndarray:
    """Build just the coupling term H_J.

    Args:
        n_sites: Number of spin sites.

    Returns:
        Coupling Hamiltonian.
    """
    identities = [EYE] * n_sites
    hamiltonian = np.zeros((2**n_sites, 2**n_sites), dtype=complex)

    for i in range(n_sites):
        op = identities.copy()
        op[i] = SIGMA_X
        op[(i + 1) % n_sites] = SIGMA_X
        hamiltonian += functools.reduce(np.kron, op)

    return hamiltonian


def heisenberg_field_term(n_sites: int) -> np.ndarray:
    """Build just the field term H_U.

    Args:
        n_sites: Number of spin sites.

    Returns:
        Field Hamiltonian.
    """
    return functools.reduce(np.kron, [SIGMA_Z for _ in range(n_sites)])


# =============================================================================
# Eigendecomposition
# =============================================================================


@dataclass
class HeisenbergEigenstate:
    """Eigenstate of the Heisenberg model."""

    energy: float
    vector: np.ndarray
    site_expectations: np.ndarray  # Shape (n_sites, 2)


def diagonalize_hamiltonian(
    n_sites: int,
    j: float = 1.0,
    u: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the Heisenberg Hamiltonian.

    Args:
        n_sites: Number of sites.
        j: Coupling strength.
        u: Field strength.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    hamiltonian = heisenberg_hamiltonian(n_sites, j, u)
    return np.linalg.eigh(hamiltonian)


def compute_expectation_values(
    n_sites: int,
    vectors: np.ndarray,
) -> np.ndarray:
    """Compute ⟨σ_i^z⟩ for each eigenstate and site.

    For eigenstate |n⟩ = Σ a_i |i⟩, computes:
    ⟨n|σ_i^z|n⟩ = Σ |a_i|² * σ_z(i)

    Args:
        n_sites: Number of spin sites.
        vectors: Eigenvectors as columns (2^N, 2^N).

    Returns:
        Array of shape (n_sites, 2^N, 2) with [site, state, ±1] values.
    """
    dim = 2**n_sites

    # For each site j and each eigenstate, compute expectation
    results = np.zeros((n_sites, dim, 2), dtype=complex)

    for j in range(n_sites):
        for n in range(dim):
            expectation = 0.0
            for basis_state in range(dim):
                # Get the bit representation
                bits = [(basis_state >> (n_sites - 1 - k)) & 1 for k in range(n_sites)]
                # σ_z at site j gives (+1) or (-1)
                sz = 1 if bits[j] == 0 else -1
                # Coefficient squared
                coeff = np.abs(vectors[basis_state, n]) ** 2
                expectation += coeff * sz
            results[j, n, 0 if expectation > 0 else 1] = expectation

    return np.real(results)


# =============================================================================
# Simulation Runner
# =============================================================================


def run_simulation(
    n_sites: int,
    j: float = 1.0,
    u: float = 1.0,
    compute_expectations: bool = True,
) -> dict:
    """Run complete Heisenberg model simulation.

    Args:
        n_sites: Number of sites.
        j: Coupling strength.
        u: Field strength.
        compute_expectations: Whether to compute expectation values.

    Returns:
        Dictionary with hamiltonian, eigenvalues, eigenvectors, etc.
    """
    # Build Hamiltonians
    hamiltonian = heisenberg_hamiltonian(n_sites, j, u)
    h_j = heisenberg_coupling_term(n_sites)
    h_u = heisenberg_field_term(n_sites)

    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    # Expectation values
    expectations = None
    if compute_expectations:
        expectations = compute_expectation_values(n_sites, eigenvectors)

    return {
        "n_sites": n_sites,
        "j": j,
        "u": u,
        "hamiltonian": hamiltonian,
        "h_coupling": h_j,
        "h_field": h_u,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "expectations": expectations,
    }


# =============================================================================
# Validation
# =============================================================================


def validate_hamiltonian_hermitian(hamiltonian: np.ndarray) -> bool:
    """Check Hamiltonian is Hermitian.

    Args:
        hamiltonian: Hamiltonian matrix.

    Returns:
        True if H = H^\dagger.
    """
    return np.allclose(hamiltonian, hamiltonian.conj().T)


def validate_eigenvectors_orthonormal(
    vectors: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Check eigenvectors are orthonormal.

    Args:
        vectors: Eigenvectors as columns.
        tolerance: Tolerance.

    Returns:
        True if V^\dagger V = I.
    """
    overlap = vectors.conj().T @ vectors
    return np.allclose(overlap, np.eye(vectors.shape[1]), atol=tolerance)


def validate_eigendecomposition(
    hamiltonian: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Verify H|v⟩ = E|v⟩ for all eigenpairs.

    Args:
        hamiltonian: Hamiltonian.
        eigenvalues: Eigenvalues.
        eigenvectors: Eigenvectors.
        tolerance: Tolerance.

    Returns:
        True if all equations hold.
    """
    for i in range(len(eigenvalues)):
        Hv = hamiltonian @ eigenvectors[:, i]
        Ev = eigenvalues[i] * eigenvectors[:, i]
        if not np.allclose(Hv, Ev, atol=tolerance):
            return False
    return True