"""
Lindblad Master Equation Solver for Open Quantum Systems.

This module implements open quantum system dynamics via the Lindblad
master equation, enabling simulation of decoherence effects including:
- One-body loss (single photon loss)
- Two-body loss (pair loss)
- Phase diffusion (dephasing)

Physical Model:
- Master equation: dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
- Lindblad operators:
  - L_1 = √γ₁a (one-body loss)
  - L_2 = √γ₂a² (two-body loss)
  - L_φ = √γ_φ J_z (phase diffusion)

Hilbert Space:
- Bosonic Fock space with truncation N (dimension N+1)
- For two-mode: dimension (N+1)²

Units:
- Dimensionless throughout (ℏ = 1)
- Time in dimensionless units
- Decay rates γ in same dimensionless units

Conventions:
- Phase convention: e^{-iHt} for unitary evolution
- Lindblad form ensures complete positivity
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import scipy
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LindbladConfig:
    """Configuration for Lindblad master equation simulation.

    Attributes:
        N: Maximum photon number for Fock space truncation.
        gamma_1: One-body loss rate (γ₁).
        gamma_2: Two-body loss rate (γ₂).
        gamma_phi: Phase diffusion rate (γ_φ).
        chi: One-axis twisting squeezing strength.
    """

    N: int
    gamma_1: float = 0.0  # one-body loss rate
    gamma_2: float = 0.0  # two-body loss rate
    gamma_phi: float = 0.0  # phase diffusion rate
    chi: float = 0.0  # OAT squeezing strength


# =============================================================================
# Bosonic Operators
# =============================================================================


def create_bosonic_operators(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create annihilation and creation operators in Fock basis.

    Constructs the bosonic operators a and a† acting on a truncated
    Fock space of dimension N+1.

    Args:
        N: Maximum photon number (Fock space truncation).

    Returns:
        Tuple of (a, a†) operators, both shape (N+1, N+1).
    """
    dim = N + 1

    a = np.zeros((dim, dim), dtype=complex)
    a_dag = np.zeros((dim, dim), dtype=complex)

    for n in range(dim):
        # a|n⟩ = √n|n-1⟩
        if n > 0:
            a[n, n - 1] = np.sqrt(n)
            a_dag[n - 1, n] = np.sqrt(n)

    return a, a_dag


def number_operator(N: int) -> np.ndarray:
    """Create number operator n = a†a in Fock basis.

    Args:
        N: Maximum photon number.

    Returns:
        Number operator of shape (N+1, N+1).
    """
    a, a_dag = create_bosonic_operators(N)
    return a_dag @ a


def jz_operator(N: int) -> np.ndarray:
    """Create J_z operator for collective spin in bosonic Fock basis.

    IMPORTANT: This implementation uses BOSONIC FOCK BASIS |n⟩ where n = 0, 1, ..., N.
    Eigenvalues are n - N/2, giving: 0 - N/2, 1 - N/2, ..., N - N/2.
    This differs from the Dicke basis convention (m = N/2, N/2-1, ..., -N/2)
    used in dicke_basis.py and noise_channels.py.

    For bosonic Fock basis |n⟩:
        eigenvalues = [-(N/2), -(N/2)+1, ..., N/2]  # i.e., 0-N/2, 1-N/2, ..., N-N/2

    Args:
        N: Maximum photon number.

    Returns:
        J_z operator of shape (N+1, N+1) in Fock basis.

    Example:
        >>> jz = jz_operator(4)  # N=4, eigenvalues: -2, -1, 0, 1, 2
        >>> jz.diagonal()
        array([-2., -1.,  0.,  1.,  2.])
    """
    n = number_operator(N)
    # J_z = n - N/2 in bosonic Fock basis
    # For N=4: eigenvalues are 0-2, 1-2, 2-2, 3-2, 4-2 = -2, -1, 0, 1, 2
    return n - (N / 2.0)


def create_two_mode_operators(
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create operators for two-mode bosonic system.

    Args:
        N: Maximum photon number per mode.

    Returns:
        Tuple of (a1, a2, a1†, a2†) operators.
    """
    dim = (N + 1) ** 2

    a1 = np.zeros((dim, dim), dtype=complex)
    a2 = np.zeros((dim, dim), dtype=complex)
    a1_dag = np.zeros((dim, dim), dtype=complex)
    a2_dag = np.zeros((dim, dim), dtype=complex)

    for n1 in range(N + 1):
        for n2 in range(N + 1):
            idx = n1 * (N + 1) + n2

            # a1|n1,n2⟩ = √n1|n1-1,n2⟩
            if n1 > 0:
                new_idx = (n1 - 1) * (N + 1) + n2
                a1[idx, new_idx] = np.sqrt(n1)
                a1_dag[new_idx, idx] = np.sqrt(n1)

            # a2|n1,n2⟩ = √n2|n1,n2-1⟩
            if n2 > 0:
                new_idx = n1 * (N + 1) + (n2 - 1)
                a2[idx, new_idx] = np.sqrt(n2)
                a2_dag[new_idx, idx] = np.sqrt(n2)

    return a1, a2, a1_dag, a2_dag


# =============================================================================
# Density Matrix Utilities
# =============================================================================


def ket_to_density(psi: np.ndarray) -> np.ndarray:
    """Convert pure state ket to density matrix.

    Args:
        psi: State vector (pure state).

    Returns:
        Density matrix ρ = |ψ⟩⟨ψ|.
    """
    return np.outer(psi, psi.conj())


def density_to_vector(rho: np.ndarray) -> np.ndarray:
    """Vectorize density matrix to Liouville space.

    Uses column-major ordering: ρ_ij → vector element at index i + j*n.

    Args:
        rho: Density matrix of shape (d, d).

    Returns:
        Vectorized density of shape (d*d,).
    """
    return rho.flatten(order="F")


def vector_to_density(rho_vec: np.ndarray) -> np.ndarray:
    """Convert Liouville vector back to density matrix.

    Args:
        rho_vec: Vectorized density.

    Returns:
        Density matrix of shape (d, d).
    """
    d = int(np.sqrt(rho_vec.shape[0]))
    return rho_vec.reshape((d, d), order="F")


def create_fock_state(n: int, N: int) -> np.ndarray:
    """Create Fock state |n⟩.

    Args:
        n: Photon number.
        N: Maximum photon number (truncation).

    Returns:
        Fock state vector of shape (N+1,).
    """
    if n > N:
        raise ValueError(f"n={n} exceeds truncation N={N}")
    state = np.zeros(N + 1, dtype=complex)
    state[n] = 1.0
    return state


def create_coherent_state(alpha: complex, N: int, truncation: int = 10) -> np.ndarray:
    """Create coherent state |α⟩ in truncated Fock basis.

    Args:
        alpha: Coherent state amplitude.
        N: Maximum photon number for truncation.
        truncation: Additional safety margin (default 10).

    Returns:
        Coherent state vector.
    """
    # Increase truncation if needed
    effective_N = max(N, int(np.abs(alpha) ** 2) + truncation)
    state = np.zeros(effective_N + 1, dtype=complex)

    for n in range(effective_N + 1):
        state[n] = (
            alpha**n
            / np.sqrt(scipy.special.factorial(n))
            * np.exp(-(np.abs(alpha) ** 2) / 2)
        )

    # Truncate to requested N
    if effective_N > N:
        state = state[: N + 1]
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(state) ** 2))
        if norm > 0:
            state = state / norm

    return state


# =============================================================================
# Lindblad Superoperator
# =============================================================================


def lindblad_liouvillian(
    rho: np.ndarray,
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
) -> np.ndarray:
    """Compute dρ/dt from Lindblad master equation.

    Evaluates the Lindblad master equation:
    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Args:
        rho: Density matrix.
        H: Hamiltonian matrix.
        L_ops: List of Lindblad operators L_k.
        gammas: List of decay rates γ_k (same length as L_ops).

    Returns:
        Time derivative of density matrix (dρ/dt).
    """
    # Commutator term: -i[H, rho]
    drho_dt = -1.0j * (H @ rho - rho @ H)

    # Lindblad dissipation terms
    for L, gamma in zip(L_ops, gammas):
        if gamma == 0:
            continue
        L_dag = L.conj().T

        # L rho L†
        L_rho_Ld = L @ rho @ L_dag

        # ½{L†L, rho} = ½(L†L rho + rho L†L)
        LdL = L_dag @ L
        anticomm = LdL @ rho + rho @ LdL

        drho_dt += gamma * (L_rho_Ld - 0.5 * anticomm)

    return drho_dt


def vectorized_liouvillian(
    rho_vec: np.ndarray,
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
) -> np.ndarray:
    """Compute Liouvillian action on vectorized density matrix.

    Uses the vectorized form of the Lindblad equation in
    Liouville space for efficient computation.

    Args:
        rho_vec: Vectorized density matrix.
        H: Hamiltonian.
        L_ops: List of Lindblad operators.
        gammas: List of decay rates.

    Returns:
        Time derivative as vector.
    """
    rho = vector_to_density(rho_vec)
    drho_dt = lindblad_liouvillian(rho, H, L_ops, gammas)
    return density_to_vector(drho_dt)


def build_liouvillian_matrix(
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
) -> np.ndarray:
    """Build the full Liouvillian superoperator matrix.

    Constructs the Lindblad Liouvillian as a matrix acting on
    the vectorized density matrix space.

    L = -i(H ⊗ I - I ⊗ H*) + Σ_k γ_k (L_k ⊗ L_k* - ½(I ⊗ L_k†L_k + L_k†L_k ⊗ I))

    Args:
        H: Hamiltonian matrix.
        L_ops: List of Lindblad operators.
        gammas: List of decay rates.

    Returns:
        Liouvillian matrix of shape (d², d²).
    """
    d = H.shape[0]
    eye = np.eye(d, dtype=complex)

    # Unitary part: -i(H ⊗ I - I ⊗ H*)
    H_kron_I = np.kron(H, eye)
    I_kron_Hstar = np.kron(eye, H.conj())
    L_unitary = -1.0j * (H_kron_I - I_kron_Hstar)

    # Initialize total Liouvillian
    L_total = L_unitary.copy()

    # Dissipative terms
    for L, gamma in zip(L_ops, gammas):
        if gamma == 0:
            continue

        L_dag = L.conj().T
        LdL = L_dag @ L

        # dissipative part: L_k ⊗ L_k* - ½(I ⊗ L_k†L_k + L_k†L_k ⊗ I)
        L_kron_Lstar = np.kron(L, L.conj())
        I_kron_LdL = np.kron(eye, LdL)
        LdL_kron_I = np.kron(LdL, eye)

        L_dissipative = L_kron_Lstar - 0.5 * (I_kron_LdL + LdL_kron_I)

        L_total += gamma * L_dissipative

    return L_total


# =============================================================================
# Time Evolution
# =============================================================================


def evolve_lindblad(
    initial_rho: np.ndarray,
    config: LindbladConfig,
    T: float,
    dt: float,
    method: str = "rk4",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Time-evolve density matrix under Lindblad equation.

    Integrates the Lindblad master equation from t=0 to t=T.

    Args:
        initial_rho: Initial density matrix.
        config: Lindblad configuration.
        T: Final evolution time.
        dt: Timestep for integration.
        method: Integration method ('rk4' for 4th-order Runge-Kutta, 'scipy' for ODE solver).
        seed: Random seed (for reproducibility, currently unused).

    Returns:
        Final density matrix at time T.
    """
    # Build Hamiltonian and Lindblad operators
    N = config.N

    # Use single-mode bosonic operators
    a, a_dag = create_bosonic_operators(N)
    n = a_dag @ a
    jz = jz_operator(N)

    # Build Hamiltonian with optional OAT squeezing
    if config.chi != 0:
        # H = χ J_z² (one-axis twisting)
        H = config.chi * (jz @ jz)
    else:
        # Free Hamiltonian H = n (number states have energy n)
        H = n

    # Build Lindblad operators
    L_ops = []
    gammas = []

    if config.gamma_1 > 0:
        L_ops.append(np.sqrt(config.gamma_1) * a)
        gammas.append(1.0)  # Rate absorbed into operator

    if config.gamma_2 > 0:
        # L_2 = √γ₂ a² for two-body loss
        L_ops.append(np.sqrt(config.gamma_2) * (a @ a))
        gammas.append(1.0)

    if config.gamma_phi > 0:
        # L_phi = √γ_phi * n (number operator) for phase diffusion
        # This causes dephasing between number states while preserving populations
        L_ops.append(np.sqrt(config.gamma_phi) * n)
        gammas.append(1.0)

    # If no dissipation, use simple unitary evolution
    if len(L_ops) == 0:
        # Unitary evolution only
        U = scipy.linalg.expm(-1.0j * H * T)
        return U @ initial_rho @ U.conj().T

    # Choose integration method
    if method == "rk4":
        return _evolve_rk4(initial_rho, H, L_ops, gammas, T, dt)
    elif method == "scipy":
        return _evolve_scipy(initial_rho, H, L_ops, gammas, T)
    else:
        raise ValueError(f"Unknown method: {method}")


def _evolve_rk4(
    initial_rho: np.ndarray,
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
    T: float,
    dt: float,
) -> np.ndarray:
    """4th-order Runge-Kutta integration of Lindblad equation.

    Args:
        initial_rho: Initial density matrix.
        H: Hamiltonian.
        L_ops: Lindblad operators.
        gammas: Decay rates.
        T: Final time.
        dt: Timestep.

    Returns:
        Final density matrix.
    """
    if T <= 0:
        return initial_rho.copy()

    rho = initial_rho.copy()
    num_steps = max(1, int(np.ceil(T / dt)))
    dt = T / num_steps  # Adjust to exactly hit T

    for _ in range(num_steps):
        # RK4 steps
        k1 = lindblad_liouvillian(rho, H, L_ops, gammas)
        k2 = lindblad_liouvillian(rho + 0.5 * dt * k1, H, L_ops, gammas)
        k3 = lindblad_liouvillian(rho + 0.5 * dt * k2, H, L_ops, gammas)
        k4 = lindblad_liouvillian(rho + dt * k3, H, L_ops, gammas)

        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce Hermiticity and normalization for stability
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.trace(rho)
        if trace > 0:
            rho = rho / trace

    return rho


def _evolve_scipy(
    initial_rho: np.ndarray,
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
    T: float,
) -> np.ndarray:
    """Evolve using scipy ODE solver.

    Args:
        initial_rho: Initial density matrix.
        H: Hamiltonian.
        L_ops: Lindblad operators.
        gammas: Decay rates.
        T: Final time.

    Returns:
        Final density matrix.
    """
    # Vectorize initial state
    rho0 = density_to_vector(initial_rho)

    # ODE function
    def rhs(t, rho_vec):
        return vectorized_liouvillian(rho_vec, H, L_ops, gammas)

    # Integrate
    sol = scipy.integrate.solve_ivp(
        rhs,
        (0, T),
        rho0,
        method="RK45",
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    rho_final = sol.y[:, -1]
    return vector_to_density(rho_final)


# =============================================================================
# Steady State
# =============================================================================


def steady_state(
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute steady-state density matrix.

    Finds ρ such that dρ/dt = 0 under the Lindblad equation.
    Uses iterative method starting from maximally mixed state.

    Args:
        H: Hamiltonian.
        L_ops: List of Lindblad operators.
        gammas: List of decay rates.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Steady-state density matrix.
    """
    d = H.shape[0]

    # Start from maximally mixed state
    rho = np.eye(d, dtype=complex) / d

    for _ in range(max_iter):
        drho_dt = lindblad_liouvillian(rho, H, L_ops, gammas)

        # Check convergence
        norm = np.max(np.abs(drho_dt))
        if norm < tol:
            break

        # Simple iteration (can be improved)
        rho = rho + 0.01 * drho_dt

        # Ensure proper density matrix
        rho = 0.5 * (rho + rho.conj().T)  # Hermitize
        rho = rho - np.eye(d) * np.min(np.linalg.eigvalsh(rho).real)  # Make positive
        rho = rho / np.trace(rho)  # Normalize

    return rho


def steady_state_dense(
    H: np.ndarray,
    L_ops: List[np.ndarray],
    gammas: List[float],
) -> np.ndarray:
    """Compute steady state via Liouvillian null space.

    Finds the eigenvector with zero eigenvalue of the Liouvillian.

    Args:
        H: Hamiltonian.
        L_ops: Lindblad operators.
        gammas: Decay rates.

    Returns:
        Steady-state density matrix.
    """
    L_mat = build_liouvillian_matrix(H, L_ops, gammas)

    # Find eigenvector with zero eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(L_mat)

    # Find index of eigenvalue closest to zero
    zero_idx = np.argmin(np.abs(eigenvalues))

    # Get eigenvector
    rho_vec = eigenvectors[:, zero_idx]

    # Reshape to density matrix
    d = H.shape[0]
    rho = rho_vec.reshape((d, d))

    # Ensure physical properties
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)

    return rho


# =============================================================================
# Expectation Values
# =============================================================================


def compute_expectation(rho: np.ndarray, operator: np.ndarray) -> complex:
    """Compute expectation value ⟨O⟩ = Tr(ρ O).

    Args:
        rho: Density matrix.
        operator: Operator O.

    Returns:
        Expectation value.
    """
    return np.trace(rho @ operator)


def compute_photon_number(rho: np.ndarray, N: int) -> float:
    """Compute mean photon number ⟨n⟩.

    Args:
        rho: Density matrix.
        N: Truncation.

    Returns:
        Mean photon number.
    """
    n = number_operator(N)
    return np.real(compute_expectation(rho, n))


def compute_phase_variance(rho: np.ndarray, N: int) -> float:
    """Compute phase variance Δφ² for single mode.

    Uses the Pegg-Barnett phase convention.

    Args:
        rho: Density matrix.
        N: Truncation.

    Returns:
        Phase variance.
    """
    # Phase operator in truncated space
    dim = N + 1
    theta = 2 * np.pi / dim
    phase_ops = np.zeros((dim, dim), dtype=complex)

    for m in range(dim):
        for n in range(dim):
            if m != n:
                phase_ops[m, n] = (1.0j / (m - n)) * (
                    np.exp(1.0j * (m - n) * theta) - 1
                )

    # Compute expectation
    phi = compute_expectation(rho, phase_ops)
    phi_sq = compute_expectation(rho, phase_ops @ phase_ops)

    return np.real(phi_sq - phi**2)


# =============================================================================
# Trajectory Simulation
# =============================================================================


def simulate_trajectory(
    initial_rho: np.ndarray,
    config: LindbladConfig,
    T: float,
    num_times: int,
    method: str = "rk4",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate complete trajectory of density matrix.

    Args:
        initial_rho: Initial density matrix.
        config: Lindblad configuration.
        T: Final time.
        num_times: Number of time points.
        method: Integration method.
        seed: Random seed.

    Returns:
        Tuple of (times, rhos) where:
        - times: Array of time points.
        - rhos: Array of density matrices shape (num_times, d, d).
    """
    times = np.linspace(0, T, num_times)
    dt = times[1] - times[0] if num_times > 1 else T

    rhos = [initial_rho.copy()]

    current_rho = initial_rho.copy()
    for _ in range(num_times - 1):
        current_rho = evolve_lindblad(current_rho, config, dt, dt, method, seed)
        rhos.append(current_rho.copy())

    return times, np.array(rhos)


# =============================================================================
# Validation
# =============================================================================


def validate_density_matrix(
    rho: np.ndarray,
    tolerance: float = 1e-8,
) -> dict:
    """Validate density matrix properties.

    Checks:
    - Hermiticity: ρ = ρ†
    - Trace: Tr[ρ] = 1
    - Positivity: eigenvalues ≥ 0

    Args:
        rho: Density matrix.
        tolerance: Numerical tolerance.

    Returns:
        Dictionary with validation results.
    """
    # Check Hermitian
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tolerance)

    # Check trace
    trace = np.trace(rho)
    is_normalized = np.isclose(trace, 1.0, atol=tolerance)

    # Check positivity
    eigenvalues = np.linalg.eigvalsh(rho)
    is_positive = np.all(eigenvalues >= -tolerance)

    # Maximum deviation from positivity
    min_eigenvalue = np.min(eigenvalues)

    return {
        "is_hermitian": is_hermitian,
        "is_normalized": is_normalized,
        "is_positive": is_positive,
        "trace": np.real(trace),
        "min_eigenvalue": np.real(min_eigenvalue),
    }


# =============================================================================
# Convenience Functions
# =============================================================================


def run_simulation(
    N: int,
    gamma_1: float = 0.0,
    gamma_2: float = 0.0,
    gamma_phi: float = 0.0,
    chi: float = 0.0,
    initial_n: int = 1,
    T: float = 1.0,
    dt: float = 0.01,
    method: str = "rk4",
    seed: Optional[int] = None,
) -> dict:
    """Run complete Lindblad simulation.

    Convenience function that performs all steps.

    Args:
        N: Fock space truncation.
        gamma_1: One-body loss rate.
        gamma_2: Two-body loss rate.
        gamma_phi: Phase diffusion rate.
        chi: OAT squeezing strength.
        initial_n: Initial Fock state |n⟩.
        T: Final evolution time.
        dt: Timestep.
        method: Integration method.
        seed: Random seed.

    Returns:
        Dictionary with simulation results.
    """
    config = LindbladConfig(
        N=N,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_phi=gamma_phi,
        chi=chi,
    )

    # Initial state
    psi = create_fock_state(initial_n, N)
    initial_rho = ket_to_density(psi)

    # Evolve
    final_rho = evolve_lindblad(initial_rho, config, T, dt, method, seed)

    # Validation
    validation = validate_density_matrix(final_rho)

    # Expectation values
    mean_n = compute_photon_number(final_rho, N)

    return {
        "config": config,
        "initial_rho": initial_rho,
        "final_rho": final_rho,
        "mean_photon_number": mean_n,
        "validation": validation,
    }
