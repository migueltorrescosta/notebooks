"""
Hybrid Oscillator-Spin System for High-Order Squeezing.

Physical Model:
- Oscillator: Bosonic mode in Fock basis |n⟩, n = 0…N (dimension N+1)
- Spin: Two-level system |↓⟩, |↑⟩ (dimension 2)
- Combined: |n⟩ ⊗ |σ⟩ with index = n × 2 + s (dimension 2(N+1))

Hilbert Space:
- Total dimension: 2 × (N+1)
- State ordering: |0,↓⟩, |0,↑⟩, |1,↓⟩, |1,↑⟩, ..., |N,↓⟩, |N,↑⟩

Units:
- Dimensionless throughout (ℏ = 1)
- Squeezing parameters r_n are dimensionless

Conventions:
- Spin operators: σ_x, σ_y, σ_z (Pauli matrices)
- Oscillator operators: a, a† with [a, a†] = 1
- Phase convention: H = Ω/2 (a^n e^{-iθ} + a^†n e^{iθ})

Functions:
- ``hybrid_hamiltonian_n`` — construct n-th order squeezing Hamiltonian
- ``hybrid_ground_state_n`` — lowest-energy eigenstate via exact diagonalisation
- ``hybrid_vacuum_state``, ``hybrid_coherent_state`` — state preparation
- ``adaptive_truncation`` — Fock-basis size from mean-photon safety margin
"""

import numpy as np
import scipy

# =============================================================================
# Spin Operators (Pauli matrices)
# =============================================================================


def spin_operator_x() -> np.ndarray:
    """Return σ_x Pauli matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def spin_operator_y() -> np.ndarray:
    """Return σ_y Pauli matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def spin_operator_z() -> np.ndarray:
    """Return σ_z Pauli matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def spin_operator_phi(phi: float) -> np.ndarray:
    """Return σ_φ = cos φ σ_x + sin φ σ_y."""
    sx = spin_operator_x()
    sy = spin_operator_y()
    return np.cos(phi) * sx + np.sin(phi) * sy


# =============================================================================
# Oscillator Operators in Fock Basis
# =============================================================================


def oscillator_annihilation(N: int) -> np.ndarray:
    """Create annihilation operator a in truncated Fock basis.

    Args:
        N: Maximum photon number (truncation).

    Returns:
        Operator of shape (N+1, N+1) with a|n⟩ = √n |n-1⟩.

    """
    dim = N + 1
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    return a


def oscillator_creation(N: int) -> np.ndarray:
    """Create creation operator a† in truncated Fock basis.

    Args:
        N: Maximum photon number (truncation).

    Returns:
        Operator of shape (N+1, N+1) with a†|n⟩ = √(n+1) |n+1⟩.

    """
    return oscillator_annihilation(N).conj().T


def oscillator_number(N: int) -> np.ndarray:
    """Create number operator a†a in Fock basis.

    Args:
        N: Maximum photon number (truncation).

    Returns:
        Diagonal operator of shape (N+1, N+1).

    """
    dim = N + 1
    n_op = np.zeros((dim, dim), dtype=complex)
    for n in range(dim):
        n_op[n, n] = n
    return n_op


def oscillator_power(a: np.ndarray, n: int) -> np.ndarray:
    """Compute a^n (annihilation operator raised to power n).

    Args:
        a: Annihilation operator of shape (dim, dim).
        n: Power (must be non-negative integer).

    Returns:
        Operator a^n of same shape as a.

    Raises:
        ValueError: If n < 0.

    """
    if n < 0:
        raise ValueError(f"Power must be non-negative, got {n}")
    if n == 0:
        return np.eye(a.shape[0], dtype=complex)
    result = a.copy()
    for _ in range(n - 1):
        result = result @ a
    return result


# =============================================================================
# Hybrid System Operators (Oscillator ⊗ Spin)
# =============================================================================


def hybrid_operator(osc_op: np.ndarray, spin_op: np.ndarray, N: int) -> np.ndarray:
    """Construct operator in hybrid space via Kronecker product.

    Operator = osc_op ⊗ spin_op acting on hybrid Hilbert space.

    Args:
        osc_op: Oscillator operator of shape (N+1, N+1).
        spin_op: Spin operator of shape (2, 2).
        N: Maximum photon number (must match osc_op dimension).

    Returns:
        Hybrid operator of shape (2(N+1), 2(N+1)).

    """
    dim_osc = N + 1
    assert osc_op.shape == (dim_osc, dim_osc), (
        f"osc_op shape {osc_op.shape} != {(dim_osc, dim_osc)}"
    )
    return np.kron(osc_op, spin_op)


def hybrid_hamiltonian_n(
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    phi: float = 0.0,
) -> np.ndarray:
    """Construct n-th order squeezing Hamiltonian for hybrid system.

    H_n = Ω_n/2 [σ_{φ+π/2} ⊗ (a^n e^{-iθ_n} + a^†n e^{iθ_n})] for n=3
    H_n = Ω_n/2 [σ_z ⊗ (a^n e^{-iθ_n} + a^†n e^{iθ_n})] for n=2,4

    Args:
        N: Maximum photon number (truncation).
        n: Squeezing order (2, 3, or 4).
        omega_n: Squeezing rate Ω_n.
        theta_n: Squeezing phase θ_n.
        phi: Base phase for σ_φ (default 0).

    Returns:
        Hamiltonian of shape (2(N+1), 2(N+1)).

    Raises:
        ValueError: If n is not 2, 3, or 4.

    """
    a = oscillator_annihilation(N)
    a_dag = oscillator_creation(N)
    a_n = oscillator_power(a, n)
    a_dag_n = oscillator_power(a_dag, n)

    # Choose spin operator based on n
    if n in {2, 4}:
        # Use σ_z
        spin_op = spin_operator_z()
    elif n == 3:
        # Use σ_{φ+π/2} = cos(φ+π/2)σ_x + sin(φ+π/2)σ_y = -sin(φ)σ_x + cos(φ)σ_y
        phi_shifted = phi + np.pi / 2
        spin_op = spin_operator_phi(phi_shifted)
    else:
        raise ValueError(f"Unsupported order n={n}. Use 2, 3, or 4.")

    # Construct H_n = Ω_n/2 * (spin_op ⊗ (a^n e^{-iθ} + a^†n e^{iθ}))
    osc_term = a_n * np.exp(-1j * theta_n) + a_dag_n * np.exp(1j * theta_n)
    H = (omega_n / 2.0) * hybrid_operator(osc_term, spin_op, N)

    # Ensure Hermitian
    return 0.5 * (H + H.conj().T)


# =============================================================================
# State Preparation
# =============================================================================


def hybrid_ground_state_n(
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    phi: float = 0.0,
) -> np.ndarray:
    """Compute the ground state of the n-th order hybrid Hamiltonian.

    Finds the eigenvector with the smallest eigenvalue of H_n via exact
    diagonalisation. The ground state is the lowest-energy state for the
    given Hamiltonian parameters.

    For n=2 or n=4, H_n couples oscillator number states of the same parity
    (Δn = ±n steps). For n=3, the spin operator is σ_{φ+π/2} which couples
    both spin states.

    Args:
        N: Maximum photon number (truncation). Hilbert space dimension
            is 2(N+1).
        n: Squeezing order (2, 3, or 4).
        omega_n: Squeezing rate Ω_n.
        theta_n: Squeezing phase θ_n.
        phi: Base phase for σ_φ (default 0.0; only used for n=3).

    Returns:
        Ground state vector of shape (2(N+1),), normalised to 1.

    Raises:
        ValueError: If n is not 2, 3, or 4.
        np.linalg.LinAlgError: If the eigendecomposition fails.

    Example:
        >>> N = 10
        >>> gs = hybrid_ground_state_n(N, n=2, omega_n=1.0, theta_n=0.0)
        >>> gs.shape
        (22,)
        >>> np.isclose(np.linalg.norm(gs), 1.0)
        True

    """
    H = hybrid_hamiltonian_n(N, n, omega_n, theta_n, phi)
    _eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Smallest eigenvalue → ground state
    return eigenvectors[:, 0].copy()


def hybrid_vacuum_state(N: int, spin_state: str = "down") -> np.ndarray:
    """Create hybrid vacuum state |0⟩ ⊗ |spin⟩.

    Args:
        N: Maximum photon number (truncation).
        spin_state: Which spin state ("down" for |↓⟩, "up" for |↑⟩).

    Returns:
        State vector of shape (2(N+1),).

    """
    dim_hybrid = 2 * (N + 1)
    state = np.zeros(dim_hybrid, dtype=complex)

    if spin_state == "down":
        state[0] = 1.0  # |0,↓⟩
    elif spin_state == "up":
        state[1] = 1.0  # |0,↑⟩
    else:
        raise ValueError(f"Unknown spin_state: {spin_state}")

    return state


def hybrid_coherent_state(
    N: int,
    alpha: complex,
    spin_state: str = "down",
) -> np.ndarray:
    """Create hybrid coherent state |α⟩ ⊗ |spin⟩.

    Args:
        N: Maximum photon number (truncation).
        alpha: Coherent state amplitude.
        spin_state: Which spin state ("down" or "up").

    Returns:
        State vector of shape (2(N+1),).

    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc

    # Build coherent state in oscillator space
    osc_state = np.zeros(dim_osc, dtype=complex)
    for n in range(dim_osc):
        osc_state[n] = (
            alpha**n
            / np.sqrt(scipy.special.factorial(n))
            * np.exp(-(np.abs(alpha) ** 2) / 2)
        )

    # Embed into hybrid space
    state = np.zeros(dim_hybrid, dtype=complex)
    if spin_state == "down":
        state[::2] = osc_state  # Even indices: |n,↓⟩
    elif spin_state == "up":
        state[1::2] = osc_state  # Odd indices: |n,↑⟩
    else:
        raise ValueError(f"Unknown spin_state: {spin_state}")

    return state


# =============================================================================
# Adaptive Truncation
# =============================================================================


def adaptive_truncation(
    alpha: complex,
    r_n: float,
    n: int,
    N_max: int = 200,
) -> int:
    """Compute adaptive truncation for squeezed state.

    Uses order-dependent safety margin to prevent boundary-induced revivals:
    higher-order operators (a^n) have spectral norm ~N^{n/2}, requiring a
    proportionally larger safety buffer.

    N_osc = min(N_max, ceil(|α|² + n·r_n + (10·n)·sqrt(|α|² + n·r_n + 1)))

    Args:
        alpha: Coherent state amplitude (0 for vacuum).
        r_n: Squeezing parameter.
        n: Squeezing order.
        N_max: Safety upper bound (default 200).

    Returns:
        Truncation N (maximum photon number).

    """
    mean_photon = np.abs(alpha) ** 2 + n * r_n
    safety_factor = 10 * n  # Wider safety margin for higher orders
    N_suggested = int(np.ceil(mean_photon + safety_factor * np.sqrt(mean_photon + 1)))
    return min(N_suggested, N_max)


# =============================================================================
# Expectation Values
# =============================================================================


def hybrid_mean_photon(state: np.ndarray, N: int) -> float:
    """Compute mean photon number ⟨a†a⟩.

    Args:
        state: Hybrid state vector of shape (2(N+1),).
        N: Maximum photon number.

    Returns:
        Mean photon number (real).

    """
    n_op = oscillator_number(N)
    n_hybrid = hybrid_operator(n_op, np.eye(2, dtype=complex), N)
    return np.real(np.vdot(state, n_hybrid @ state))


# =============================================================================
# Validation
# =============================================================================


def validate_hybrid_state(state: np.ndarray, N: int) -> bool:
    """Validate hybrid state vector.

    Checks:
    - Correct dimension: 2(N+1)
    - Normalized: ∑|ψ|² = 1

    Args:
        state: State vector to validate.
        N: Maximum photon number.

    Returns:
        True if valid, False otherwise.

    """
    expected_dim = 2 * (N + 1)
    if state.shape != (expected_dim,):
        return False
    norm = np.sum(np.abs(state) ** 2)
    return np.isclose(norm, 1.0, atol=1e-6)


def validate_hybrid_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix is unitary: U†U = I.

    Args:
        U: Matrix to check.
        tol: Numerical tolerance.

    Returns:
        True if unitary within tolerance.

    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False
    identity = np.eye(U.shape[0], dtype=complex)
    return np.allclose(U.conj().T @ U, identity, atol=tol)
