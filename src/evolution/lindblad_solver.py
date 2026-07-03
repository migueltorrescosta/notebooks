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

from src.physics.dicke_basis import jz_operator
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
    jz = jz_operator(N, basis=OperatorBasis.FOCK)

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
    T_decay: float,
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
        T_decay: Final evolution time.
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
    result = qutip.mesolve(H_qobj, rho0_qobj, [0, T_decay], c_ops_qobj, e_ops=[])

    return result.states[-1].full()


def simulate_trajectory(
    initial_rho: np.ndarray,
    config: LindbladConfig,
    T_decay: float,
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
        T_decay: Final time.
        num_times: Number of time points.
        method: Ignored (kept for backward compatibility).
        seed: Random seed (for reproducibility, currently unused).

    Returns:
        Tuple of (times, rhos) where:
        - times: Array of time points.
        - rhos: Array of density matrices shape (num_times, d, d).

    """
    H, c_ops = _build_hamiltonian_and_cops(config)
    times = np.linspace(0, T_decay, num_times)

    # Convert to QuTiP objects
    H_qobj = qutip.Qobj(H)
    rho0_qobj = qutip.Qobj(initial_rho)
    c_ops_qobj = [qutip.Qobj(L) for L in c_ops]

    # Single call to mesolve returns states at every time point
    result = qutip.mesolve(H_qobj, rho0_qobj, times, c_ops_qobj, e_ops=[])

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



# =============================================================================
# Validation
# =============================================================================


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
) -> np.ndarray:
    """Compute dρ/dt from Lindblad master equation.

    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Dimension-agnostic: works for any Hilbert space dimension determined by
    ``rho.shape[0]``.

    Args:
        rho: Density matrix of shape (d, d).
        H: Hamiltonian of shape (d, d).
        L_ops: List of Lindblad jump operators.
        gammas: List of decay rates for each operator.

    Returns:
        Time derivative dρ/dt of shape (d, d).

    """
    drho = -1.0j * (H @ rho - rho @ H)

    for L, gamma in zip(L_ops, gammas, strict=False):
        if gamma == 0:
            continue
        L_dag = L.conj().T
        LdL = L_dag @ L
        L_rho_Ld = L @ rho @ L_dag
        anticomm = LdL @ rho + rho @ LdL
        drho += gamma * (L_rho_Ld - 0.5 * anticomm)

    return drho


def evolve_lindblad_rk4(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
    T_decay: float,
    dt: float,
) -> np.ndarray:
    """4th-order Runge-Kutta integration of the Lindblad master equation.

    Dimension-agnostic: works for any Hilbert space dimension determined by
    ``rho0.shape[0]``. Enforces Hermiticity and trace normalisation at each
    step.

    Args:
        rho0: Initial density matrix of shape (d, d).
        H: Hamiltonian of shape (d, d).
        L_ops: List of Lindblad jump operators.
        gammas: List of decay rates.
        T_decay: Total evolution time.
        dt: Time step.

    Returns:
        Final density matrix of shape (d, d).

    """
    if T_decay <= 0:
        return rho0.copy()

    rho = rho0.copy()
    num_steps = max(1, int(np.ceil(T_decay / dt)))
    dt_eff = T_decay / num_steps

    for _ in range(num_steps):
        k1 = lindblad_rhs(rho, H, L_ops, gammas)
        k2 = lindblad_rhs(rho + 0.5 * dt_eff * k1, H, L_ops, gammas)
        k3 = lindblad_rhs(rho + 0.5 * dt_eff * k2, H, L_ops, gammas)
        k4 = lindblad_rhs(rho + dt_eff * k3, H, L_ops, gammas)

        rho = rho + (dt_eff / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.trace(rho)
        if trace > 0:
            rho = rho / trace

    return rho


def evolve_lindblad_scipy(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
    T_decay: float,
) -> np.ndarray:
    """Evolve Lindblad master equation using scipy.integrate.solve_ivp.

    Fortran-order vectorisation with RK45 and tight tolerances
    (rtol=1e-8, atol=1e-10).

    Args:
        rho0: Initial density matrix of shape (d, d).
        H: Hamiltonian of shape (d, d).
        L_ops: List of Lindblad jump operators.
        gammas: List of decay rates.
        T_decay: Total evolution time.

    Returns:
        Final density matrix of shape (d, d).

    """
    d = rho0.shape[0]
    rho0_vec = rho0.flatten(order="F")

    def _rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = rho_vec.reshape((d, d), order="F")
        drho = lindblad_rhs(rho, H, L_ops, gammas)
        return drho.flatten(order="F")

    sol = scipy.integrate.solve_ivp(
        _rhs,
        (0, T_decay),
        rho0_vec,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    rho_final = sol.y[:, -1].reshape((d, d), order="F")
    rho_final = 0.5 * (rho_final + rho_final.conj().T)
    trace = np.trace(rho_final)
    if trace > 0:
        rho_final = rho_final / trace

    return rho_final


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
# Vectorised Liouvillian
# =============================================================================


def vectorise_rho(rho: np.ndarray) -> np.ndarray:
    """Vectorise a density matrix (column-major stacking).

    ``vec(ρ)`` stacks columns of ρ into an n²-vector, matching the
    convention used by :func:`build_vectorized_liouvillian`.

    Args:
        rho: n × n density matrix.

    Returns:
        n²-vector (Fortran-order flatten).
    """
    return rho.reshape(-1, order="F")


def unvectorise_rho(vec: np.ndarray) -> np.ndarray:
    """Unvectorise a density matrix (column-major unstacking).

    Inverse of :func:`vectorise_rho`.  Reshapes an n²-vector back into
    an n × n matrix.

    Args:
        vec: n²-vector.

    Returns:
        n × n density matrix.
    """
    n = int(np.sqrt(vec.shape[0]))
    return vec.reshape((n, n), order="F")


def build_vectorized_liouvillian(
    H: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    """Construct the Liouvillian superoperator in vectorised form.

    Using column-major vectorisation (``vec(ρ)`` stacks columns):

        ℒ = -i(I ⊗ H - H^T ⊗ I)
            + Σ_k [L_k^* ⊗ L_k - 1/2(I ⊗ L_k^† L_k + (L_k^† L_k)^T ⊗ I)]

    The vectorised density matrix evolves as:

        vec(ρ(t)) = exp(ℒ t) vec(ρ(0))

    Args:
        H: Hamiltonian matrix (n × n Hermitian).
        lindblad_ops: List of Lindblad jump operators (each n × n, with
            rates pre-absorbed: L_k = √γ_k · L_k).

    Returns:
        Liouvillian matrix (n² × n²).

    Raises:
        ValueError: If H is not square or any operator dimension
            mismatches H.
    """
    d = H.shape[0]
    if H.shape != (d, d):
        raise ValueError(f"H must be square, got shape {H.shape}")
    for i, Lk in enumerate(lindblad_ops):
        if Lk.shape != (d, d):
            raise ValueError(
                f"Lindblad op [{i}] has shape {Lk.shape}, expected ({d}, {d})"
            )

    I_d = np.eye(d, dtype=complex)

    # Unitary part: vec(-i[H, ρ]) = -i(I ⊗ H - H^T ⊗ I) vec(ρ)
    L = -1j * (np.kron(I_d, H) - np.kron(H.T, I_d))

    # Dissipative part
    for Lk in lindblad_ops:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        # vec(L_k ρ L_k^†) = (L_k^* ⊗ L_k) vec(ρ)
        L += np.kron(Lk.conj(), Lk)
        # -½ vec({L_k^† L_k, ρ}) = -½(I ⊗ L_k^† L_k + (L_k^† L_k)^T ⊗ I) vec(ρ)
        L -= 0.5 * (np.kron(I_d, Lk_dag_Lk) + np.kron(Lk_dag_Lk.T, I_d))

    return L


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
    T_decay: float = 1.0,
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
        T_decay: Final evolution time.
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
    final_rho = evolve_lindblad(initial_rho, config, T_decay, dt, method, seed)

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
