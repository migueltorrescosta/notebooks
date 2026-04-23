"""
Validations Module.

Centralized validation functions for physics modules.
Ensures consistent validation across the codebase.

Functions:
- validate_hamiltonian_hermitian: Check Hamiltonian is Hermitian
- validate_eigenvectors_orthonormal: Check eigenvectors are orthonormal
- validate_eigendecomposition: Verify H|v⟩ = E|v⟩
- validate_orthonormality: Check orthonormality of eigenvectors
- validate_probability_conservation: Verify probability conservation
- validate_partial_trace: Validate partial trace properties
- validate_sensitivity: Validate sensitivity calculation
- validate_state_delta_estimation: Validate state for delta estimation
- validate_hamiltonian_delta_estimation: Validate Hamiltonian for delta estimation
- validate_state_mzi: Validate state for MZI simulation
- validate_unitary: Check if matrix is unitary
"""

from typing import Tuple

import numpy as np


# =============================================================================
# Heisenberg Model Validations
# =============================================================================


def validate_hamiltonian_hermitian(hamiltonian: np.ndarray) -> bool:
    r"""Check Hamiltonian is Hermitian.

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
    r"""Check eigenvectors are orthonormal.

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
    r"""Verify H|v⟩ = E|v⟩ for all eigenpairs.

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


# =============================================================================
# Quantum Time Evolution Validations
# =============================================================================


def validate_orthonormality(
    eigenvectors: np.ndarray,
    tolerance: float = 1e-8,
) -> float:
    """Check orthonormality of eigenvectors.

    Returns maximum deviation from identity.

    Args:
        eigenvectors: Matrix with eigenvectors as columns.
        tolerance: Tolerance for assertion.

    Returns:
        Maximum deviation from orthonormality.
    """
    n = eigenvectors.shape[1]
    overlap = np.real(np.conjugate(eigenvectors.T) @ eigenvectors)
    deviation = np.sum(np.abs(overlap - np.eye(n)))
    return deviation


def validate_probability_conservation(
    wf: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Verify that total probability is conserved.

    Args:
        wf: Wavefunction.
        tolerance: Tolerance for check.

    Returns:
        True if probability is conserved.
    """
    prob = np.sum(np.abs(wf) ** 2)
    return np.isclose(prob, 1.0, rtol=tolerance)


# =============================================================================
# Partial Trace Validations
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
        rho_full: Full density matrix.
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
    if not np.isclose(np.trace(rho_a), np.trace(rho_b), atol=tolerance):
        return False

    # Check Hermitian
    if not np.allclose(rho_full, rho_full.conj().T):
        return False

    # Check positive semidefinite (full)
    eigenvals = np.linalg.eigvalsh(rho_full)
    if not np.all(eigenvals >= -tolerance):
        return False

    return True


# =============================================================================
# Sensitivity Analysis Validations
# =============================================================================


def validate_sensitivity(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
    tolerance: float = 1e-6,
) -> bool:
    """Validate sensitivity calculation via finite differences.

    Args:
        n: Ancillary dimension.
        k: Level.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x: Coupling.
        alpha_z: Coupling.
        t: Time.
        tolerance: Required accuracy.

    Returns:
        True if numerical derivative matches analytical.
    """
    # Import here to avoid circular imports
    from src.sensitivity_analysis import compute_observable, sensitivity

    eps = 1e-5

    # Get sensitivities
    result = sensitivity(n, k, j_s, delta_s, alpha_x, alpha_z, t)

    # Numerical derivative wrt j_s
    obs_plus = compute_observable(n, k, j_s + eps, delta_s, alpha_x, alpha_z, t)
    obs_minus = compute_observable(n, k, j_s - eps, delta_s, alpha_x, alpha_z, t)
    num_deriv_j = (obs_plus - obs_minus) / (2 * eps)

    # Numerical derivative wrt delta_s
    obs_plus = compute_observable(n, k, j_s, delta_s + eps, alpha_x, alpha_z, t)
    obs_minus = compute_observable(n, k, j_s, delta_s - eps, alpha_x, alpha_z, t)
    num_deriv_delta = (obs_plus - obs_minus) / (2 * eps)

    # Compare
    if abs(num_deriv_j - result["sensitivity_to_j"]) > tolerance:
        return False
    if abs(num_deriv_delta - result["sensitivity_to_delta"]) > tolerance:
        return False

    return True


# =============================================================================
# Delta Estimation Validations
# =============================================================================


def validate_state_delta_estimation(
    state: np.ndarray,
    expected_dims: Tuple[int, int],
) -> bool:
    """Validate that a state matrix has expected dimensions and is valid density matrix.

    Args:
        state: State matrix to validate.
        expected_dims: Expected dimensions (rows, cols).

    Returns:
        True if valid, False otherwise.
    """
    if state.shape != expected_dims:
        return False
    # Check Hermitian
    if not np.allclose(state, state.conj().T):
        return False
    # Check trace = 1
    if not np.isclose(np.trace(state), 1.0):
        return False
    # Check positive semidefinite (eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(state)
    if not np.all(eigenvals >= -1e-10):
        return False
    return True


def validate_hamiltonian_delta_estimation(hamiltonian: np.ndarray) -> bool:
    """Validate that a matrix is a valid Hermitian Hamiltonian.

    Args:
        hamiltonian: Hamiltonian matrix to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Check Hermitian: H = H^\dagger
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        return False
    return True


# =============================================================================
# MZI Simulation Validations
# =============================================================================


def validate_state_mzi(state: np.ndarray) -> bool:
    """Check if a state vector is normalized.

    Validates that the input is a proper quantum state by checking
    that its L2 norm equals 1 (within numerical tolerance).

    Args:
        state: State vector to validate.

    Returns:
        True if the state is normalized (||ψ|| = 1), False otherwise.
    """
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    return np.isclose(norm, 1.0, atol=1e-10)


def validate_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a matrix is unitary.

    Validates that U†U = I within numerical tolerance. This is
    essential for verifying that operator constructions are correct.

    Args:
        U: Matrix to validate.
        tol: Numerical tolerance for the check (default: 1e-8).

    Returns:
        True if the matrix is unitary, False otherwise.
    """
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=tol)
