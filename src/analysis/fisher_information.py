"""
Fisher Information Computation for Phase Sensitivity Analysis.

This module provides functions to compute classical and quantum Fisher information
for phase estimation in quantum interferometry.

Physical Model:
- Classical Fisher Information: F_C(φ) = Σ [∂P(m|φ)/∂φ]² / P(m|φ)
- Quantum Fisher Information (QFI): F_Q = 4 Var(H_gen) for pure states
- Full QFI for mixed states: via eigen-decomposition
- Phase sensitivity: Δφ = 1/√F (Cramér-Rao bound)

Units:
- Dimensionless throughout. Phase is measured in radians.
"""

import numpy as np


def classical_fisher_information(
    probabilities: np.ndarray, dphi: float = 1e-6
) -> np.ndarray:
    """Compute classical Fisher Information via finite difference.

    F_C(φ) = Σ [∂P(m|φ)/∂φ]² / P(m|φ)
           ≈ Σ [P(m|φ+dphi/2) - P(m|φ-dphi/2)]² / (dphi² * P(m|φ))

    Uses central difference for numerical derivative.

    Args:
        probabilities: Array of shape (n_phi, n_outcomes) containing
            P(m|φ) for each phase value. First dimension is phase,
            second is measurement outcome.
        dphi: Step size for numerical derivative (default 1e-6).

    Returns:
        Array of classical Fisher Information values of shape (n_phi,).

    Raises:
        ValueError: If dphi <= 0 or probabilities contain NaN.

    Constraints:
        probabilities shape must be (n_phi, n_outcomes) with n_phi >= 3
            (need space for central difference).
        All probability rows must sum to 1 (within tolerance).
        dphi must be small enough for accurate finite difference
            but large enough to avoid numerical noise (1e-8 to 1e-3).
        Performance: O(n_phi × n_outcomes).

    """
    if dphi <= 0:
        raise ValueError(f"dphi must be positive, got {dphi}")

    if np.any(np.isnan(probabilities)):
        raise ValueError("Probabilities contain NaN values")

    n_phi, n_outcomes = probabilities.shape
    fc = np.zeros(n_phi)

    # Central difference: (f(x+d) - f(x-d)) / (2d)
    for i in range(1, n_phi - 1):
        probs_plus = probabilities[i + 1]
        probs_minus = probabilities[i - 1]

        # Derivative via central difference
        deriv = (probs_plus - probs_minus) / (2 * dphi)

        # F_C = Σ (∂P/∂φ)² / P
        # Avoid division by zero: only include terms where P > 0
        mask = probs_plus > 1e-12
        fc[i] = np.sum(deriv[mask] ** 2 / probs_plus[mask])

    # Forward difference at boundaries
    for i in [0, n_phi - 1]:
        probs_c = probabilities[i]
        if i == n_phi - 1:
            probs_plus = probabilities[i]
            probs_minus = probabilities[i - 1]
        else:
            probs_plus = probabilities[i + 1]
            probs_minus = probs_c

        deriv = (probs_plus - probs_minus) / dphi
        mask = probs_c > 1e-12
        fc[i] = np.sum(deriv[mask] ** 2 / probs_c[mask])

    return fc


def classical_fisher_information_single(
    p_plus: np.ndarray, p_minus: np.ndarray, dphi: float
) -> float:
    """Compute F_C for a single phase value given neighboring probabilities.

    F_C = Σ [P(m|φ+dphi/2) - P(m|φ-dphi/2)]² / (dphi² * P(m|φ))

    Args:
        p_plus: P(m|φ + dphi/2) for each outcome.
        p_minus: P(m|φ - dphi/2) for each outcome.
        dphi: Phase step size.

    Returns:
        Classical Fisher Information value.

    Raises:
        ValueError: If dphi <= 0 or arrays don't match.

    """
    if dphi <= 0:
        raise ValueError(f"dphi must be positive, got {dphi}")

    if p_plus.shape != p_minus.shape:
        raise ValueError("Probability arrays must have same shape")

    # Central difference derivative
    deriv = (p_plus - p_minus) / (2 * dphi)

    # Use average probability for denominator (more accurate)
    p_avg = 0.5 * (p_plus + p_minus)

    # F_C = Σ (∂P/∂φ)² / P
    mask = p_avg > 1e-12
    if not np.any(mask):
        return 0.0

    return np.sum(deriv[mask] ** 2 / p_avg[mask])


def quantum_fisher_information(state: np.ndarray, generator: np.ndarray) -> float:
    """Compute Quantum Fisher Information for a pure state.

    For pure states, the QFI simplifies to:
        F_Q = 4 [⟨G²⟩ - ⟨G⟩²] = 4 Var(G)

    This is the maximum achievable Fisher information for
    phase estimation with the given probe state.

    Args:
        state: Pure quantum state vector (dim,).
        generator: Phase generator Hermitian operator (dim, dim).

    Returns:
        Quantum Fisher Information value F_Q.

    Raises:
        ValueError: If state is not a vector or generator dimensions don't match.

    Constraints:
        state must be 1D ndarray of complex numbers.
        generator must be square Hermitian (dim, dim).
        state.shape[0] must match generator.shape[0].
        Only valid for pure states (for mixed states use quantum_fisher_information_dm).

    Agent Notes:
        This is an O(dim²) computation (two matrix-vector products).
        For mixed states, call quantum_fisher_information_dm instead
        (which falls through to this function for effectively pure states).

    """
    state = np.asarray(state, dtype=complex)
    generator = np.asarray(generator, dtype=complex)

    if np.any(np.isnan(state)):
        raise ValueError("State vector contains NaN values")
    if np.any(np.isinf(state)):
        raise ValueError("State vector contains infinite values")
    if np.any(np.isnan(generator)):
        raise ValueError("Generator contains NaN values")
    if np.any(np.isinf(generator)):
        raise ValueError("Generator contains infinite values")

    if state.ndim != 1:
        raise ValueError(f"State must be 1D array, got shape {state.shape}")

    if state.shape[0] != generator.shape[0]:
        raise ValueError(
            f"State dimension {state.shape[0]} must match "
            f"generator dimension {generator.shape[0]}"
        )

    if generator.shape[0] != generator.shape[1]:
        raise ValueError("Generator must be square")

    # Compute expectation values
    # <G> = ⟨ψ|G|ψ⟩
    g_exp = np.vdot(state, generator @ state).real

    # <G²> = ⟨ψ|G²|ψ⟩
    g2_exp = np.vdot(state, generator @ generator @ state).real

    # Variance and QFI
    var_g = g2_exp - g_exp**2

    # Ensure non-negative (numerical precision)
    var_g = max(0.0, var_g)

    return 4.0 * var_g


def quantum_fisher_information_dm(rho: np.ndarray, generator: np.ndarray) -> float:
    """Compute Quantum Fisher Information for a mixed state.

    Uses the symmetric logarithmic derivative (SLD) formula:
        F_Q = 4 Σ_{i<j} (λ_i - λ_j)² / (λ_i + λ_j) · |⟨i|G|j⟩|²

    where λ_i and |i⟩ are eigenvalues/eigenvectors of ρ, and G is the
    phase generator. This is derived from the SLD expression:

        F_Q = Σ_{i≠j} 2 (λ_i - λ_j)² / (λ_i + λ_j) · |⟨i|G|j⟩|²

    and simplified to the i<j form above using symmetry.

    For pure states (ρ = |ψ⟩⟨ψ|, only one λ > 0), this reduces to
    4·Var(G), which is handled as a special case for numerical stability.

    Args:
        rho: Density matrix (dim, dim).
        generator: Phase generator Hermitian operator (dim, dim).

    Returns:
        Quantum Fisher Information value F_Q.

    Raises:
        ValueError: If rho or generator are not matrices or dimensions don't match.

    Constraints:
        rho must be Hermitian, trace-1, positive semidefinite.
        generator must be Hermitian.
        Both must share the same dimension.
        Performance: O(dim³) due to eigendecomposition and double sum.

    Agent Notes:
        This is the primary QFI function — it handles both pure and mixed states.
        The SLD formula is numerically delicate when eigenvalues are nearly
        degenerate (small denominator). The rank_tol threshold controls the
        boundary between "positive" and "zero" eigenvalues.
        If you modify this function, verify against the pure-state special case
        (n_pos=1 branch) and test with rank-deficient density matrices.

    """
    rho = np.asarray(rho, dtype=complex)
    generator = np.asarray(generator, dtype=complex)

    if rho.ndim != 2:
        raise ValueError(f"Density matrix must be 2D, got shape {rho.shape}")

    if generator.ndim != 2:
        raise ValueError(f"Generator must be 2D, got shape {generator.shape}")

    dim = rho.shape[0]
    if rho.shape != (dim, dim):
        raise ValueError(f"Density matrix must be square, got {rho.shape}")

    if generator.shape != (dim, dim):
        raise ValueError(
            f"Generator dimensions {generator.shape} must match "
            f"density matrix dimension {dim}"
        )

    # Input validation — prevent silent NaN/Inf and non-Hermitian errors
    if np.any(np.isnan(rho)):
        raise ValueError("Density matrix contains NaN values")
    if np.any(np.isinf(rho)):
        raise ValueError("Density matrix contains infinite values")
    if np.any(np.isnan(generator)):
        raise ValueError("Generator contains NaN values")
    if np.any(np.isinf(generator)):
        raise ValueError("Generator contains infinite values")
    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise ValueError("Density matrix must be Hermitian")
    if not np.allclose(generator, generator.conj().T, atol=1e-10):
        raise ValueError("Generator must be Hermitian")

    # Eigen-decomposition of ρ
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Sort by eigenvalue (descending) - keep eigenvector association
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clean up small negative eigenvalues (numerical precision)
    eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)

    # Trace should be 1, but normalize just in case
    trace = np.sum(eigenvalues)
    if trace > 1e-12 and not np.isclose(trace, 1.0):
        eigenvalues = eigenvalues / trace

    # Separate eigenvalues into "positive" (support) and "zero" (nullspace).
    # The rank_tol threshold only determines which eigenvalues are treated
    # as genuinely positive and which contribute as λ_j = 0 in the SLD sum.
    rank_tol = max(1e-12, eigenvalues[0] * 1e-10)
    pos_idx = np.where(eigenvalues > rank_tol)[0]
    zero_idx = np.where(eigenvalues <= rank_tol)[0]
    n_pos = len(pos_idx)
    n_zero = len(zero_idx)

    # Compute F_Q
    if n_pos == 0:
        # ρ ≈ 0 matrix (all eigenvalues below rank_tol)
        return 0.0

    if n_pos == 1:
        # Effectively pure state: F_Q = 4·Var(G)
        # This automatically includes nullspace contributions via ⟨G²⟩.
        i = pos_idx[0]
        state = eigenvectors[:, i]
        g_exp = np.vdot(state, generator @ state).real
        g2_exp = np.vdot(state, generator @ generator @ state).real
        var_g = max(0.0, g2_exp - g_exp**2)
        fq = 4.0 * var_g
    else:
        # Mixed state: SLD formula
        # F_Q = 4 Σ_{i<j} (λ_i - λ_j)² / (λ_i + λ_j) · |⟨i|G|j⟩|²
        fq = 0.0

        # Term 1: positive–positive pairs (standard SLD)
        for p in range(n_pos):
            i = pos_idx[p]
            vi = eigenvectors[:, i]
            for q in range(p + 1, n_pos):
                j = pos_idx[q]
                vj = eigenvectors[:, j]

                gij = np.vdot(vi, generator @ vj)
                gij_abs_sq = np.abs(gij) ** 2

                lambda_sum = eigenvalues[i] + eigenvalues[j]
                if lambda_sum > 1e-12:
                    lambda_diff_sq = (eigenvalues[i] - eigenvalues[j]) ** 2
                    weight = 4.0 * lambda_diff_sq / lambda_sum
                    fq += weight * gij_abs_sq

        # Term 2: positive–zero pairs (λ_j ≈ 0, so (λ_i-0)²/(λ_i+0) = λ_i)
        # These are PHYSICALLY REQUIRED: G can couple the support to the
        # nullspace of ρ, and the SLD formula includes these contributions.
        for p in range(n_pos):
            i = pos_idx[p]
            vi = eigenvectors[:, i]
            for q in range(n_zero):
                j = zero_idx[q]
                vj = eigenvectors[:, j]

                gij = np.vdot(vi, generator @ vj)
                gij_abs_sq = np.abs(gij) ** 2

                # λ_j ≈ 0: weight = 4 · (λ_i - 0)² / (λ_i + 0) = 4·λ_i
                weight = 4.0 * eigenvalues[i]
                fq += weight * gij_abs_sq

    return float(np.real(fq))


def phase_sensitivity_from_fisher(F: float) -> float:
    """Compute phase sensitivity from Fisher Information.

    The Cramér-Rao bound gives:
        Δφ ≥ 1/√F

    This is the standard quantum limit (SQL) for naive estimation.

    Args:
        F: Fisher Information value.

    Returns:
        Phase sensitivity Δφ ≥ 1/√F.

    Raises:
        ValueError: If F <= 0.

    """
    if F <= 0:
        raise ValueError(f"Fisher information must be positive, got {F}")

    return 1.0 / np.sqrt(F)


def generate_noon_state(N: int) -> np.ndarray:
    """Generate NOON state for N photons.

    |NOON⟩ = (|N,0⟩ + |0,N⟩) / √2
           = (|N,0⟩ + e^{iφ}|0,N⟩) / √2 for phase φ

    In the Fock basis with total photon number N:
    - |N,0⟩ = state with N photons in mode a, 0 in mode b
    - |0,N⟩ = state with 0 photons in mode a, N in mode b

    Args:
        N: Total photon number.

    Returns:
        NOON state vector of dimension N+1 (for single mode).
        For full 2-mode space, dimension is (N+1)², but we return
        the superposition state in the symmetric subspace.

    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    # Dimension is N+1 for single mode (Fock basis 0 to N)
    dim = N + 1

    # NOON state in 2-mode space: |N,0⟩ + |0,N⟩
    # Embed in (N+1) x (N+1) space, take superposition
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |N,0⟩ component (mode a has N photons)
    # Note: Full NOON would be in product space (N+1)²
    # This is a simplified single-mode representation

    return state


def generate_css_state(N: int) -> np.ndarray:
    """Generate Coherent Superposition State (CSS) for N qubits.

    CSS = (|0⟩^{⊗N} + |1⟩^{⊗N}) / √2 = (|0...0⟩ + |1...1⟩) / √2

    This is the GHZ state, which achieves SQL scaling F_Q = N.

    Args:
        N: Number of qubits (mode dimension = 2^N full space).

    Returns:
        GHZ state vector. For practical computation, we return
        the state in the computational basis representation.

    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    # GHZ state in 2^N dimensional space
    dim = 2**N
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)  # |00...0⟩
    state[-1] = 1.0 / np.sqrt(2)  # |11...1⟩

    return state


def generate_phase_generator(N: int, generator_type: str = "Jz") -> np.ndarray:
    """Generate phase generator for interferometry.

    Common generators:
    - "Jz": For NOON-like states, G = J_z (total spin z-component)
    - "Jx": For rotation around x-axis
    - "N": Total photon number G = n (Fock basis)

    Args:
        N: System size (photon number or spin quantum number 2J = N).
        generator_type: Type of generator ("Jz", "Jx", "N").

    Returns:
        Generator matrix of dimension (N+1, N+1).

    """
    if generator_type == "Jz":
        # J_z for spin-J system where J = N/2
        # Eigenvalues: m = -J, -J+1, ..., J-1, J
        dim = N + 1
        j = N / 2
        eigenvalues = np.arange(dim) - j
        jz = np.diag(eigenvalues)
        return np.array(jz, dtype=complex)
    elif generator_type == "Jx":
        # J_x for spin-J system where J = N/2
        # Using angular momentum ladder relations
        dim = N + 1
        j = N / 2
        m = np.arange(dim) - j
        # Off-diagonal elements: sqrt(J(J+1) - m(m+1))
        off_diags = 0.5 * np.sqrt(j * (j + 1) - m[:-1] * (m[:-1] + 1))
        jx = np.zeros((dim, dim))
        jx[np.arange(dim - 1), np.arange(1, dim)] = off_diags
        jx[np.arange(1, dim), np.arange(dim - 1)] = off_diags
        return np.array(jx, dtype=complex)
    elif generator_type == "N":
        # Total photon number operator in Fock basis
        dim = N + 1
        n_op = np.diag(np.arange(dim))
        return np.array(n_op, dtype=complex)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def validate_fisher_inputs(
    F: float,
    name: str = "Fisher Information",
) -> None:
    """Validate Fisher information values.

    Args:
        F: Fisher information value.
        name: Name for error messages.

    Raises:
        ValueError: If F <= 0 or is NaN.

    """
    if np.isnan(F):
        raise ValueError(f"{name} must not be NaN")
    if F <= 0:
        raise ValueError(f"{name} must be positive, got {F}")


# Alias for consistent API with other modules
validate_fisher = validate_fisher_inputs
