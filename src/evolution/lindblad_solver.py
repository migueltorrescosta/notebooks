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

import numpy as np
import qutip
import scipy
import scipy.special

from src.physics.dicke_basis import jz_operator as _jz_operator
from src.utils.enums import OperatorBasis

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


def create_two_mode_operators(
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
# Time Evolution
# =============================================================================


def _build_hamiltonian_and_cops(
    config: LindbladConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build Hamiltonian and collapse operators from config.

    Collapse operators have rates pre-absorbed (c_i = √γ_i · L_i),
    matching the QuTiP convention for mesolve.

    Args:
        config: Lindblad configuration.

    Returns:
        Tuple of (H, c_ops) where H is the Hamiltonian matrix and
        c_ops is the list of collapse operator matrices.

    """
    N = config.N
    a, a_dag = qutip.destroy(N + 1).full(), qutip.create(N + 1).full()
    n = a_dag @ a
    jz = _jz_operator(N, basis=OperatorBasis.FOCK)

    # Build Hamiltonian with optional OAT squeezing
    if config.chi != 0:
        H = config.chi * (jz @ jz)
    else:
        H = n

    # Build collapse operators with rates pre-absorbed
    c_ops: list[np.ndarray] = []

    if config.gamma_1 > 0:
        c_ops.append(np.sqrt(config.gamma_1) * a)

    if config.gamma_2 > 0:
        c_ops.append(np.sqrt(config.gamma_2) * (a @ a))

    if config.gamma_phi > 0:
        c_ops.append(np.sqrt(config.gamma_phi) * jz)

    return H, c_ops


def evolve_lindblad(
    initial_rho: np.ndarray,
    config: LindbladConfig,
    T: float,
    dt: float = 0.01,
    method: str = "rk4",
    seed: int | None = None,
) -> np.ndarray:
    """Time-evolve density matrix under Lindblad equation.

    Delegates to ``qutip.mesolve`` for the integration. The ``dt``
    and ``method`` parameters are accepted for backward compatibility
    but ignored — QuTiP manages adaptive stepping internally.

    Args:
        initial_rho: Initial density matrix.
        config: Lindblad configuration.
        T: Final evolution time.
        dt: Ignored (kept for backward compatibility).
        method: Ignored (kept for backward compatibility).
        seed: Random seed (for reproducibility, currently unused).

    Returns:
        Final density matrix at time T as a numpy array.

    """
    H, c_ops = _build_hamiltonian_and_cops(config)

    # Convert to QuTiP objects
    H_qobj = qutip.Qobj(H)
    rho0_qobj = qutip.Qobj(initial_rho)
    c_ops_qobj = [qutip.Qobj(L) for L in c_ops]

    # Call mesolve — QuTiP handles all integration
    result = qutip.mesolve(H_qobj, rho0_qobj, [0, T], c_ops_qobj, [])

    return result.states[-1].full()


def simulate_trajectory(
    initial_rho: np.ndarray,
    config: LindbladConfig,
    T: float,
    num_times: int,
    method: str = "rk4",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate complete trajectory of density matrix.

    Delegates to ``qutip.mesolve`` with the full time grid. The
    ``method`` and ``seed`` parameters are accepted for backward
    compatibility but ignored.

    Args:
        initial_rho: Initial density matrix.
        config: Lindblad configuration.
        T: Final time.
        num_times: Number of time points.
        method: Ignored (kept for backward compatibility).
        seed: Random seed (for reproducibility, currently unused).

    Returns:
        Tuple of (times, rhos) where:
        - times: Array of time points.
        - rhos: Array of density matrices shape (num_times, d, d).

    """
    H, c_ops = _build_hamiltonian_and_cops(config)
    times = np.linspace(0, T, num_times)

    # Convert to QuTiP objects
    H_qobj = qutip.Qobj(H)
    rho0_qobj = qutip.Qobj(initial_rho)
    c_ops_qobj = [qutip.Qobj(L) for L in c_ops]

    # Single call to mesolve returns states at every time point
    result = qutip.mesolve(H_qobj, rho0_qobj, times, c_ops_qobj, [])

    rhos = np.array([state.full() for state in result.states])
    return times, rhos


# =============================================================================
# Steady State
# =============================================================================


def steady_state(
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute steady-state density matrix.

    Delegates to ``qutip.steadystate``. The ``max_iter`` and ``tol``
    parameters are accepted for backward compatibility but ignored.
    When the Liouvillian has a degenerate null space (e.g., phase
    diffusion without particle loss), falls back to returning the
    maximally mixed state.

    Args:
        H: Hamiltonian matrix.
        L_ops: List of Lindblad operators (bare, without sqrt-rate).
        gammas: List of decay rates (same length as L_ops).
        max_iter: Ignored (kept for backward compatibility).
        tol: Ignored (kept for backward compatibility).

    Returns:
        Steady-state density matrix.

    """
    H_qobj = qutip.Qobj(H)
    c_ops_qobj = [
        qutip.Qobj(np.sqrt(g) * L) for L, g in zip(L_ops, gammas, strict=False) if g > 0
    ]

    if not c_ops_qobj:
        d = H.shape[0]
        return np.eye(d, dtype=complex) / d

    try:
        result = qutip.steadystate(H_qobj, c_ops_qobj, method="direct")
        return result.full()
    except ValueError:
        # Degenerate null space (e.g., phase diffusion only):
        # fall back to maximally mixed state, which is a valid
        # steady state for any purely dephasing Liouvillian.
        d = H.shape[0]
        return np.eye(d, dtype=complex) / d


# =============================================================================
# Expectation Values
# =============================================================================


def compute_photon_number(rho: np.ndarray, N: int) -> float:
    """Compute mean photon number ⟨n⟩.

    Args:
        rho: Density matrix.
        N: Truncation.

    Returns:
        Mean photon number.

    """
    n = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
    return np.real(np.trace(rho @ n))


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
    phi = np.trace(rho @ phase_ops)
    phi_sq = np.trace(rho @ phase_ops @ phase_ops)

    return np.real(phi_sq - phi**2)


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
    trace = np.trace(rho)
    eigenvalues = np.linalg.eigvalsh(rho)

    return {
        "is_hermitian": np.allclose(rho, rho.conj().T, atol=tolerance),
        "is_normalized": np.isclose(trace, 1.0, atol=tolerance),
        "is_positive": np.all(eigenvalues >= -tolerance),
        "trace": np.real(trace),
        "min_eigenvalue": np.real(np.min(eigenvalues)),
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
    seed: int | None = None,
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
    initial_rho = np.outer(psi, psi.conj())

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
