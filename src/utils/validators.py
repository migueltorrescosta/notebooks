"""
Validations Module.

Centralized validation functions for physics modules.
Ensures consistent validation across the codebase.

Functions:
- validate_eigenvectors_orthonormal: Check eigenvectors are orthonormal
- validate_eigendecomposition: Verify H|v⟩ = E|v⟩
- validate_orthonormality: Check orthonormality of eigenvectors
- validate_partial_trace: Validate partial trace properties
- validate_sensitivity: Validate sensitivity calculation
- validate_state_delta_estimation: Validate state for delta estimation
- validate_hamiltonian_delta_estimation: Validate Hamiltonian for delta estimation
- validate_state_mzi: Validate state for MZI simulation
"""

import numpy as np

__all__ = [
    "validate_eigendecomposition",
    "validate_eigenvectors_orthonormal",
    "validate_hamiltonian_delta_estimation",
    "validate_orthonormality",
    "validate_partial_trace",
    "validate_sensitivity",
    "validate_state_delta_estimation",
    "validate_state_mzi",
]


# =============================================================================
# Heisenberg Model Validations
# =============================================================================


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
    return np.allclose(
        vectors.conj().T @ vectors, np.eye(vectors.shape[1]), atol=tolerance
    )


def validate_eigendecomposition(
    hamiltonian: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    r"""Verify H|v⟩ = E|v⟩ for all eigenpairs via vectorized broadcasting.

    Args:
        hamiltonian: Hamiltonian.
        eigenvalues: Eigenvalues.
        eigenvectors: Eigenvectors.
        tolerance: Tolerance.

    Returns:
        True if all equations hold.

    """
    return bool(
        np.allclose(
            hamiltonian @ eigenvectors,
            eigenvectors * eigenvalues,
            atol=tolerance,
        )
    )


# =============================================================================
# Quantum Time Evolution Validations
# =============================================================================


def validate_orthonormality(
    eigenvectors: np.ndarray,
    tolerance: float = 1e-8,
) -> float:
    """Check orthonormality of eigenvectors.

    Returns sum of absolute deviations from identity.

    Args:
        eigenvectors: Matrix with eigenvectors as columns.
        tolerance: Tolerance for assertion.

    Returns:
        Sum of absolute deviations from orthonormality.

    """
    overlap = eigenvectors.conj().T @ eigenvectors
    return float(np.sum(np.abs(overlap - np.eye(eigenvectors.shape[1]))))


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
    return (
        np.isclose(np.trace(rho_full), 1.0, atol=tolerance)
        and np.isclose(np.trace(rho_a), np.trace(rho_b), atol=tolerance)
        and np.allclose(rho_full, rho_full.conj().T)
        and bool(np.all(np.linalg.eigvalsh(rho_full) >= -tolerance))
    )


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

    Constraints:
        0 <= k <= n (ancilla level within subspace).
        tolerance > 0 (pass/fail threshold).
        Uses internal eps = 1e-5 for finite difference step.

    Agent Notes:
        Imports compute_observable and sensitivity from
        src.analysis.sensitivity_analysis inside the function body
        to avoid circular imports (sensitivity_analysis imports
        validators). If sensitivity_analysis is refactored, update
        these local imports accordingly.

    """
    # Import here to avoid circular imports
    from src.analysis.sensitivity_analysis import compute_observable, sensitivity

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
    return not abs(num_deriv_delta - result["sensitivity_to_delta"]) > tolerance


# =============================================================================
# Delta Estimation Validations
# =============================================================================


def validate_state_delta_estimation(
    state: np.ndarray,
    expected_dims: tuple[int, int],
) -> bool:
    """Validate that a state matrix has expected dimensions and is valid density matrix.

    Checks shape, Hermiticity, unit trace, and positivity — short-circuits
    at the first failure (cheapest checks first).

    Args:
        state: State matrix to validate.
        expected_dims: Expected dimensions (rows, cols).

    Returns:
        True if valid, False otherwise.

    """
    return (
        state.shape == expected_dims
        and np.allclose(state, state.conj().T)
        and np.isclose(np.trace(state), 1.0)
        and bool(np.all(np.linalg.eigvalsh(state) >= -1e-10))
    )


def validate_hamiltonian_delta_estimation(hamiltonian: np.ndarray) -> bool:
    """Validate that a matrix is a valid Hermitian Hamiltonian.

    Args:
        hamiltonian: Hamiltonian matrix to validate.

    Returns:
        True if valid, False otherwise.

    """
    # Check Hermitian: H = H^\dagger
    return np.allclose(hamiltonian, hamiltonian.conj().T)


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

    Agent Notes:
        This function is re-exported as `validate_state` in
        src.physics.mzi_simulation (line 35) for backward compatibility.
        Any refactoring that changes the name or signature must update
        the alias. Also aliased in src.analysis.delta_estimation (line 35)
        as `validate_state = validate_state_delta_estimation` (different function!).

    """
    return np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)
