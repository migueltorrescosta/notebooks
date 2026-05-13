"""
Hybrid Oscillator-Spin Lindblad Solver for High-Order Squeezing.

Solves the Lindblad master equation for a hybrid system consisting of
a bosonic oscillator mode coupled to a spin degree of freedom, with
application to high-order squeezing generation and detection.

Physical Model:
- Oscillator: Bosonic mode in Fock basis |n⟩, n = 0…N (dimension N+1)
- Spin: Two-level system |↓⟩, |↑⟩ (dimension 2)
- Combined: |n⟩ ⊗ |σ⟩ (dimension 2(N+1))
- Hamiltonian: H = H_0 + H_int where H_0 drives the oscillator
  and H_int couples oscillator to spin (basis for squeezing protocol)
- Lindblad operators: L_1 = √γ₁ a (one-body loss), L_φ = √γ_φ n (phase diffusion)

Hilbert Space:
- Total dimension: 2 × (N+1)
- State ordering: |0,↓⟩, |0,↑⟩, |1,↓⟩, |1,↑⟩, ..., |N,↓⟩, |N,↑⟩
- Consistent with src.physics.hybrid_system conventions

Units:
- Dimensionless throughout (ℏ = 1)
- Decay rates γ in same dimensionless units
- Time in dimensionless units

Conventions:
- Master equation: dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
- Phase convention: e^{-iHt} for unitary evolution
- Lindblad form ensures complete positivity
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass
class HybridLindbladConfig:
    """Configuration for hybrid oscillator-spin Lindblad simulation."""

    N: int
    n: int = 2
    omega_n: float = 1.0
    theta_n: float = 0.0
    phi: float = 0.0
    gamma_1: float = 0.0
    gamma_2: float = 0.0
    gamma_phi: float = 0.0
    t_squeeze: float = 1.0


def build_hybrid_hamiltonian(config: HybridLindbladConfig) -> np.ndarray:
    """Build n-th order squeezing Hamiltonian for hybrid system."""
    from .hybrid_system import (
        oscillator_annihilation,
        oscillator_creation,
        oscillator_power,
        spin_operator_phi,
        spin_operator_z,
    )

    N = config.N
    n = config.n
    omega_n = config.omega_n
    theta_n = config.theta_n
    phi = config.phi

    a = oscillator_annihilation(N)
    a_dag = oscillator_creation(N)
    a_n = oscillator_power(a, n)
    a_dag_n = oscillator_power(a_dag, n)

    if n in {2, 4}:
        spin_op = spin_operator_z()
    elif n == 3:
        phi_shifted = phi + np.pi / 2
        spin_op = spin_operator_phi(phi_shifted)
    else:
        raise ValueError(f"Unsupported order n={n}. Use 2, 3, or 4.")

    osc_term = a_n * np.exp(-1j * theta_n) + a_dag_n * np.exp(1j * theta_n)
    H = np.kron(osc_term, spin_op)
    H = (omega_n / 2.0) * H
    return 0.5 * (H + H.conj().T)


def build_hybrid_lindblad_operators(
    config: HybridLindbladConfig,
) -> tuple[list[np.ndarray], list[float]]:
    """Build Lindblad operators for hybrid oscillator-spin system."""
    N = config.N
    dim_osc = N + 1

    a = np.zeros((dim_osc, dim_osc), dtype=complex)
    for n in range(1, dim_osc):
        a[n - 1, n] = np.sqrt(n)
    a2 = a @ a

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I_spin = np.eye(2, dtype=complex)
    I_osc = np.eye(dim_osc, dtype=complex)

    L_ops = []
    gammas = []

    if config.gamma_1 > 0:
        L_1 = np.kron(a, I_spin) * np.sqrt(config.gamma_1)
        L_ops.append(L_1)
        gammas.append(1.0)

    if config.gamma_2 > 0:
        L_2 = np.kron(a2, I_spin) * np.sqrt(config.gamma_2)
        L_ops.append(L_2)
        gammas.append(1.0)

    if config.gamma_phi > 0:
        L_phi = np.kron(I_osc, sigma_z) * np.sqrt(config.gamma_phi / 2)
        L_ops.append(L_phi)
        gammas.append(1.0)

    return L_ops, gammas


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
) -> np.ndarray:
    """Compute dρ/dt from Lindblad master equation."""
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


def evolve_hybrid_lindblad(
    initial_state: np.ndarray,
    config: HybridLindbladConfig,
    T: float,
    dt: float,
    method: str = "rk4",
) -> np.ndarray:
    """Time-evolve hybrid state under Lindblad master equation."""
    H = build_hybrid_hamiltonian(config)
    L_ops, gammas = build_hybrid_lindblad_operators(config)

    if initial_state.ndim == 1:
        rho0 = np.outer(initial_state, initial_state.conj())
    else:
        rho0 = initial_state.copy()

    if len(L_ops) == 0:
        U = scipy.linalg.expm(-1.0j * H * T)
        return U @ rho0 @ U.conj().T

    if method == "rk4":
        return _evolve_rk4_hybrid(rho0, H, L_ops, gammas, T, dt)
    if method == "scipy":
        return _evolve_scipy_hybrid(rho0, H, L_ops, gammas, T)
    raise ValueError(f"Unknown method: {method}")


def _evolve_rk4_hybrid(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
    T: float,
    dt: float,
) -> np.ndarray:
    """4th-order Runge-Kutta integration."""
    if T <= 0:
        return rho0.copy()

    rho = rho0.copy()
    num_steps = max(1, int(np.ceil(T / dt)))
    dt_eff = T / num_steps

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


def _evolve_scipy_hybrid(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    gammas: list[float],
    T: float,
) -> np.ndarray:
    """Evolve using scipy ODE solver."""
    d = rho0.shape[0]
    rho0_vec = rho0.flatten(order="F")

    def rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = rho_vec.reshape((d, d), order="F")
        drho = lindblad_rhs(rho, H, L_ops, gammas)
        return drho.flatten(order="F")

    sol = scipy.integrate.solve_ivp(
        rhs,
        (0, T),
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


def apply_squeezing(
    config: HybridLindbladConfig,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """Apply n-th order squeezing to initial state."""
    from .hybrid_system import hybrid_vacuum_state

    if initial_state is None:
        initial_state = hybrid_vacuum_state(config.N, spin_state="down")

    H = build_hybrid_hamiltonian(config)
    U = scipy.linalg.expm(-1.0j * H * config.t_squeeze)
    return U @ initial_state


def validate_hybrid_density_matrix(
    rho: np.ndarray,
    tolerance: float = 1e-8,
) -> dict:
    """Validate hybrid density matrix properties."""
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tolerance)
    trace = np.trace(rho)
    is_normalized = np.isclose(trace, 1.0, atol=tolerance)
    eigenvalues = np.linalg.eigvalsh(rho)
    is_positive = np.all(eigenvalues >= -tolerance)

    return {
        "is_hermitian": is_hermitian,
        "is_normalized": is_normalized,
        "is_positive": is_positive,
        "trace": np.real(trace),
        "min_eigenvalue": np.real(np.min(eigenvalues)),
    }


def run_hybrid_simulation(
    config: HybridLindbladConfig,
    initial_state: np.ndarray | None = None,
) -> dict:
    """Run complete hybrid squeezing + decoherence simulation."""
    from .hybrid_system import hybrid_vacuum_state

    if initial_state is None:
        initial_state = hybrid_vacuum_state(config.N, spin_state="down")

    squeezed_state = apply_squeezing(config, initial_state)

    final_rho = evolve_hybrid_lindblad(
        squeezed_state,
        config,
        T=config.t_squeeze,
        dt=0.01,
        method="rk4",
    )

    validation = validate_hybrid_density_matrix(final_rho)

    return {
        "config": config,
        "initial_state": initial_state,
        "squeezed_state": squeezed_state,
        "final_state": final_rho,
        "validation": validation,
    }


def run_decoherence_sweep(
    config_base: HybridLindbladConfig,
    gamma_values: np.ndarray,
    gamma_type: str = "gamma_1",
) -> dict:
    """Run decoherence sweep and compute QFI for each gamma value."""
    from .hybrid_mzi import embed_hybrid_in_mzi, mzi_phase_generator
    from .hybrid_system import hybrid_vacuum_state

    psi0 = hybrid_vacuum_state(config_base.N, spin_state="down")
    config_squeeze = HybridLindbladConfig(
        N=config_base.N,
        n=config_base.n,
        omega_n=config_base.omega_n,
        theta_n=config_base.theta_n,
        phi=config_base.phi,
        t_squeeze=config_base.t_squeeze,
        gamma_1=0.0,
        gamma_2=0.0,
        gamma_phi=0.0,
    )
    psi_squeezed = apply_squeezing(config_squeeze, psi0)

    qfi_values = []

    for gamma in gamma_values:
        config_g = HybridLindbladConfig(
            N=config_base.N,
            n=config_base.n,
            omega_n=config_base.omega_n,
            theta_n=config_base.theta_n,
            phi=config_base.phi,
            t_squeeze=0.0,
            gamma_1=gamma if gamma_type == "gamma_1" else 0.0,
            gamma_2=gamma if gamma_type == "gamma_2" else 0.0,
            gamma_phi=gamma if gamma_type == "gamma_phi" else 0.0,
        )

        rho_final = evolve_hybrid_lindblad(
            psi_squeezed,
            config_g,
            T=config_base.t_squeeze,
            dt=0.01,
        )

        rho_embedded = embed_hybrid_in_mzi(rho_final, config_base.N)
        G = mzi_phase_generator(config_base.N)
        fq = _qfi_mixed_state(rho_embedded, G)
        qfi_values.append(fq)

    return {
        "gamma_values": gamma_values,
        "qfi_values": np.array(qfi_values),
        "gamma_type": gamma_type,
    }


def _qfi_mixed_state(rho: np.ndarray, G: np.ndarray) -> float:
    """Compute QFI for mixed state using SLD formulation.

    Delegates to ``quantum_fisher_information_dm`` in the analysis module,
    which provides the correct SLD-based implementation.

    Args:
        rho: Density matrix (dim, dim).
        G: Phase generator Hermitian operator (dim, dim).

    Returns:
        Quantum Fisher Information value F_Q.

    """
    from src.analysis.fisher_information import quantum_fisher_information_dm

    return quantum_fisher_information_dm(rho, G)


def compare_orders_at_gamma(
    N: int,
    omega_n: float,
    t_squeeze: float,
    gamma: float,
    gamma_type: str = "gamma_1",
) -> dict:
    """Compare QFI for n=2, 3, 4 at a given decoherence rate."""
    results = {}

    for n in [2, 3, 4]:
        config = HybridLindbladConfig(
            N=N,
            n=n,
            omega_n=omega_n,
            t_squeeze=t_squeeze,
            gamma_1=gamma if gamma_type == "gamma_1" else 0.0,
            gamma_2=gamma if gamma_type == "gamma_2" else 0.0,
            gamma_phi=gamma if gamma_type == "gamma_phi" else 0.0,
        )

        sim_result = run_hybrid_simulation(config)
        from .hybrid_mzi import embed_hybrid_in_mzi, mzi_phase_generator

        rho_final = sim_result["final_state"]
        rho_embedded = embed_hybrid_in_mzi(rho_final, N)
        G = mzi_phase_generator(N)
        fq = _qfi_mixed_state(rho_embedded, G)
        results[f"n{n}"] = fq

    return results
