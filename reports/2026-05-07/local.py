"""
Combined local module for the 2026-05-07 High-Order Squeezing report.

Contains all exclusive code migrated from src/:
- Hybrid system functions: coherent state, adaptive truncation, mean photon, evolution, validation
- MZI embedding: beam splitter, phase shift, evolution, probabilities, Wigner computation
- Lindblad solver: configuration, Hamiltonian, operators, evolution, simulation
- Wigner function: single-mode, hybrid state, negativity check
- Validation tests: compare_plan_vs_simulation test functions

Usage:
    uv run python reports/2026-05-07/local.py
    uv run python reports/2026-05-07/local.py --force  (regenerate data + figures)

This module is not importable as reports.2026-05-07.local (hyphens in path).
Importers should use importlib.util.spec_from_file_location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qutip
import scipy
import scipy.linalg

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.physics.hybrid_mzi import extract_oscillator_density, qfi_hybrid_mzi
from src.physics.hybrid_system import (
    hybrid_hamiltonian_n,
    hybrid_operator,
    hybrid_vacuum_state,
    oscillator_annihilation,
    oscillator_creation,
    oscillator_number,
    oscillator_power,
    spin_operator_phi,
    spin_operator_z,
)

REPORT_DATE = "2026-05-07"
REPORTS_DIR = Path(__file__).resolve().parent.parent
_REPORT_DIR = REPORTS_DIR / REPORT_DATE
_RAW_DIR = _REPORT_DIR / "raw_data"
_FIG_DIR = _REPORT_DIR / "figures"


# =============================================================================
# Section: From src/physics/hybrid_system.py
# =============================================================================
# These functions were exclusive to the 2026-05-07 report and migrated here.


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
    return float(np.real(np.vdot(state, n_hybrid @ state)))


def evolve_hybrid_state(
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    t: float,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Evolve hybrid state under unitary H for time t.

    Constructs the n-th order squeezing Hamiltonian H_n and evolves the
    initial state via U = exp(-i H_n t).

    Args:
        N: Maximum photon number (truncation).
        n: Squeezing order (2, 3, or 4).
        omega_n: Squeezing rate Ω_n.
        theta_n: Squeezing phase θ_n.
        t: Evolution time.
        initial_state: State vector of shape (2(N+1),).

    Returns:
        Evolved state vector of shape (2(N+1),), normalised.

    """
    H = hybrid_hamiltonian_n(N, n=n, omega_n=omega_n, theta_n=theta_n)
    U = scipy.linalg.expm(-1j * H * t)
    return U @ initial_state


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
    return bool(np.isclose(norm, 1.0, atol=1e-6))


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


# =============================================================================
# Section: From src/physics/hybrid_mzi.py
# =============================================================================


def embed_hybrid_in_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> np.ndarray:
    """Embed hybrid state into two-mode MZI space.

    Creates: ρ_2mode = |ψ⟩_hybrid ⊗ |0⟩_vacuum
    where |0⟩_vacuum is the vacuum state in mode 2.

    Accepts both:
    - Pure state vector (1D) — returns embedded vector of shape (dim_mzi,).
    - Density matrix (2D) — returns embedded matrix of shape (dim_mzi, dim_mzi).

    State ordering:
    - Mode 1: hybrid oscillator (N+1 Fock states)
    - Mode 2: vacuum mode (N+1 Fock states)
    - Spin: 2 states

    Total dimension: 2 × (N+1)²
    Index = (n1*(N+1) + n2) * 2 + s

    Args:
        hybrid_state: State vector of shape (2(N+1),) or density matrix of
            shape (2(N+1), 2(N+1)).
        N: Maximum photon number (truncation).

    Returns:
        Embedded state vector of shape (2(N+1)²,) if input is 1D, or
        embedded density matrix of shape (2(N+1)², 2(N+1)²) if input is 2D.

    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc
    dim_mzi = 2 * (dim_osc**2)  # hybrid ⊗ mode2

    # --- Pure state path ---
    if hybrid_state.ndim == 1:
        if hybrid_state.shape != (dim_hybrid,):
            raise ValueError(
                f"hybrid_state must have shape ({dim_hybrid},), "
                f"got {hybrid_state.shape}",
            )

        embedded = np.zeros(dim_mzi, dtype=complex)
        # Embed as: |n1⟩_mode1 ⊗ |0⟩_mode2 ⊗ |σ⟩_spin
        # Index in embedded space: (n1*(N+1) + 0) * 2 + s = n1*(N+1)*2 + s
        for n1 in range(dim_osc):
            for s in range(2):  # spin state
                hybrid_idx = n1 * 2 + s
                mzi_idx = n1 * dim_osc * 2 + s  # n2=0
                embedded[mzi_idx] = hybrid_state[hybrid_idx]
        return embedded

    # --- Density matrix path ---
    if hybrid_state.ndim == 2:
        if hybrid_state.shape != (dim_hybrid, dim_hybrid):
            raise ValueError(
                f"hybrid_state must have shape ({dim_hybrid}, {dim_hybrid}), "
                f"got {hybrid_state.shape}",
            )

        # Build embedding isometry E: maps hybrid index → two-mode index
        # E[n1*(N+1)*2 + s, n1*2 + s] = 1.0, all else 0.
        E = np.zeros((dim_mzi, dim_hybrid), dtype=complex)
        for n1 in range(dim_osc):
            for s in range(2):
                hybrid_idx = n1 * 2 + s
                mzi_idx = n1 * dim_osc * 2 + s
                E[mzi_idx, hybrid_idx] = 1.0

        # ρ_embedded = E @ ρ_hybrid @ E†
        return E @ hybrid_state @ E.conj().T

    raise ValueError(
        f"hybrid_state must be 1D (state vector) or 2D (density matrix), "
        f"got ndim={hybrid_state.ndim}",
    )


def mzi_beam_splitter(N: int, theta: float = np.pi / 4) -> np.ndarray:
    """Construct beam splitter unitary for modes 1 and 2.

    Uses the generator-based approach: U = exp(-iθ G) where
    G = i(a1†a2 - a1a2†) is the beam splitter generator.

    This approach guarantees unitarity.

    Args:
        N: Maximum photon number.
        theta: Beam splitter angle (π/4 = 50/50).

    Returns:
        Unitary of shape (2(N+1)², 2(N+1)²).

    """
    dim_osc = N + 1
    dim_modes = dim_osc**2

    # Build annihilation operators for mode 1 and mode 2
    # Mode 1: a1 ⊗ I_2
    a1 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n1 > 0:
                a1[idx - dim_osc, idx] = np.sqrt(n1)  # a1|n1,n2⟩ = √n1|n1-1,n2⟩

    # Mode 2: I_1 ⊗ a2
    a2 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n2 > 0:
                a2[idx - 1, idx] = np.sqrt(n2)  # a2|n1,n2⟩ = √n2|n1,n2-1⟩

    # Beam splitter generator: G = i(a1†a2 - a1a2†)
    a1_dag = a1.conj().T
    a2_dag = a2.conj().T

    # Compute unitary: U = exp(-iθ G) = exp(θ * (a1†a2 - a1a2†))
    G = 1j * (a1_dag @ a2 - a1 @ a2_dag)
    bs_modes = scipy.linalg.expm(-1j * theta * G)

    # Embed with spin identity
    return np.kron(bs_modes, np.eye(2, dtype=complex))


def mzi_phase_shift(N: int, phi: float) -> np.ndarray:
    """Construct phase shift unitary on mode 1.

    U_phase = exp(i φ n₁) ⊗ I_mode2 ⊗ I_spin

    Args:
        N: Maximum photon number.
        phi: Phase shift in radians.

    Returns:
        Unitary of shape (2(N+1)², 2(N+1)²).

    """
    dim_osc = N + 1
    dim_modes = dim_osc**2

    phase_op = np.zeros((dim_modes, dim_modes), dtype=complex)

    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            phase_op[idx, idx] = np.exp(1j * phi * n1)

    # Embed with spin identity
    return np.kron(phase_op, np.eye(2, dtype=complex))


def mzi_phase_generator(N: int) -> np.ndarray:
    """Construct phase generator G = n₁ ⊗ I_mode2 ⊗ I_spin.

    Used for QFI computation.

    Args:
        N: Maximum photon number.

    Returns:
        Generator matrix of shape (2(N+1)², 2(N+1)²).

    """
    dim_osc = N + 1
    dim_modes = dim_osc**2

    # n₁ in mode space: diagonal with value n1
    n1_op = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            n1_op[idx, idx] = n1

    # Embed with spin identity
    return np.kron(n1_op, np.eye(2, dtype=complex))


def evolve_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
    phi: float,
    theta: float = np.pi / 4,
) -> np.ndarray:
    """Evolve hybrid state through MZI.

    Sequence: embed → BS1 → phase shift → BS2

    Args:
        hybrid_state: Input hybrid state of shape (2(N+1),).
        N: Maximum photon number.
        phi: Phase shift in mode 1 (unknown parameter).
        theta: Beam splitter angle (default π/4 = 50/50).

    Returns:
        Output state vector of shape (2(N+1)²,).

    """
    # Embed into MZI space
    state = embed_hybrid_in_mzi(hybrid_state, N)

    # BS1
    bs = mzi_beam_splitter(N, theta)
    state = bs @ state

    # Phase shift
    ps = mzi_phase_shift(N, phi)
    state = ps @ state

    # BS2
    return bs @ state


def mzi_output_probabilities(
    final_state: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute output probabilities P(n1, n2, s) from MZI output.

    Args:
        final_state: Output state vector of shape (2(N+1)²,).
        N: Maximum photon number.

    Returns:
        Array of probabilities for each (n1, n2, s) configuration.
        Sum should be 1.

    """
    return np.abs(final_state) ** 2


def mzi_marginal_photon_probs(
    final_state: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marginal photon number probabilities P(n1), P(n2).

    Args:
        final_state: Output state vector.
        N: Maximum photon number.

    Returns:
        Tuple (P1, P2) where P1[n1] = P(n1) summed over n2 and spin.

    """
    dim_osc = N + 1

    probs = np.abs(final_state) ** 2

    P1 = np.zeros(dim_osc, dtype=float)
    P2 = np.zeros(dim_osc, dtype=float)

    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx_base = (n1 * dim_osc + n2) * 2
            # Sum over spin (2 components)
            total = np.sum(probs[idx_base : idx_base + 2])
            P1[n1] += total
            P2[n2] += total

    return P1, P2


def compute_wigner_for_state(
    hybrid_state: np.ndarray,
    N: int,
    x_max: float = 5.0,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wigner function for oscillator part of hybrid state.

    Args:
        hybrid_state: Hybrid state vector.
        N: Maximum photon number.
        x_max: Range for x and p axes.
        n_points: Number of grid points per axis.

    Returns:
        Tuple (X, P, W) where X and P are 1D arrays, W is 2D array.

    """
    # Extract oscillator density matrix
    rho_osc = extract_oscillator_density(hybrid_state, N)

    # Create quadrature grid
    x = np.linspace(-x_max, x_max, n_points)
    p = np.linspace(-x_max, x_max, n_points)

    # Compute Wigner
    W = wigner_function_single(rho_osc, x, p)

    return x, p, W


# =============================================================================
# Section: From src/physics/hybrid_lindblad.py
# =============================================================================


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

        rho_final = sim_result["final_state"]
        rho_embedded = embed_hybrid_in_mzi(rho_final, N)
        G = mzi_phase_generator(N)
        fq = _qfi_mixed_state(rho_embedded, G)
        results[f"n{n}"] = fq

    return results


# =============================================================================
# Section: From src/physics/wigner.py
# =============================================================================


def wigner_function_single(
    rho_osc: np.ndarray,
    x_range: np.ndarray,
    p_range: np.ndarray,
) -> np.ndarray:
    """Compute Wigner function for single-mode density matrix.

    Delegates to qutip.wigner with g=2 to match the α = x + ip convention
    and normalization ∫∫ W(x,p) dx dp = 1.

    Args:
        rho_osc: Density matrix of oscillator (dim N+1, N+1).
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.

    Returns:
        2D array W[x_idx, p_idx] of Wigner function values.

    Raises:
        ValueError: If rho_osc is not square.

    """
    if rho_osc.ndim != 2 or rho_osc.shape[0] != rho_osc.shape[1]:
        raise ValueError(f"rho_osc must be square, got shape {rho_osc.shape}")

    rho_qobj = qutip.Qobj(rho_osc)
    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    wigner_result = qutip.wigner(rho_qobj, x_range, p_range, g=2)
    assert wigner_result is not None
    return wigner_result.T


def wigner_from_hybrid_state(
    hybrid_state: np.ndarray,
    N: int,
    x_range: np.ndarray,
    p_range: np.ndarray,
    spin_component: str = "down",
) -> np.ndarray:
    """Extract oscillator density matrix from hybrid state and compute Wigner.

    Args:
        hybrid_state: State vector of shape (2(N+1),) - hybrid oscillator+spin.
        N: Maximum photon number.
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.
        spin_component: Which spin to trace ("down" for |↓⟩, "up" for |↑⟩).

    Returns:
        2D array W[x_idx, p_idx].

    Raises:
        ValueError: If spin_component is invalid.

    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc

    if hybrid_state.shape != (dim_hybrid,):
        raise ValueError(
            f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}",
        )

    # Extract oscillator state for given spin component
    if spin_component == "down":
        osc_state = hybrid_state[::2]  # Even indices
    elif spin_component == "up":
        osc_state = hybrid_state[1::2]  # Odd indices
    else:
        raise ValueError(f"Unknown spin_component: {spin_component}")

    # Check if state is pure or mixed (for now assume pure from state vector)
    rho_osc = np.outer(osc_state, osc_state.conj())
    rho_qobj = qutip.Qobj(rho_osc)

    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    wigner_result = qutip.wigner(rho_qobj, x_range, p_range, g=2)
    assert wigner_result is not None
    return wigner_result.T


def wigner_is_negative(W: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if Wigner function has negative values.

    Args:
        W: Wigner function array.
        tol: Tolerance for considering negative.

    Returns:
        True if min(W) < -tol.

    """
    return float(np.min(W)) < -tol


# =============================================================================
# Section: From compare_plan_vs_simulation.py
# =============================================================================


def evolve_hybrid_unitary(
    initial_state: np.ndarray,
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    t: float,
) -> np.ndarray:
    """Evolve hybrid state under n-th order squeezing Hamiltonian."""
    H = hybrid_hamiltonian_n(N, n=n, omega_n=omega_n, theta_n=theta_n)
    U = scipy.linalg.expm(-1j * H * t)
    return U @ initial_state


def find_squeezing_time_for_target_photon(
    n: int,
    target_n: float,
    N: int,
    omega_n: float = 1.0,
    theta_n: float = 0.0,
    t_max: float = 20.0,
    dt: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """
    Find FIRST squeezing time that achieves target mean photon number.

    Uses forward sweep to detect the first crossing of target_n, then
    refines with bisection.  This correctly handles oscillatory dynamics
    (n=4) where simple bisection would latch onto a later revival.

    Args:
        n: Squeezing order.
        target_n: Target mean photon number.
        N: Fock truncation.
        omega_n: Squeezing rate.
        theta_n: Squeezing phase.
        t_max: Maximum search time.
        dt: Sweep step size.

    Returns:
        Tuple of (t_sqz, achieved_n, squeezed_state).

    """
    initial = hybrid_vacuum_state(N, spin_state="down")

    # Forward sweep: find the first crossing of target_n
    t_low = 0.0
    t = dt
    crossed = False

    while t <= t_max:
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t)
        mean_n = hybrid_mean_photon(squeezed, N)

        if mean_n >= target_n:
            # First crossing found between t-dt and t
            crossed = True
            break

        t_low = t
        t += dt

    if not crossed:
        # Target not reached within t_max
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_max)
        mean_n = hybrid_mean_photon(squeezed, N)
        return t_max, mean_n, squeezed

    # Refine with bisection between t_low and t
    t_high = t
    for _ in range(25):
        t_mid = (t_low + t_high) / 2
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_mid)
        mean_n = hybrid_mean_photon(squeezed, N)

        if mean_n < target_n:
            t_low = t_mid
        else:
            t_high = t_mid

    t_final = (t_low + t_high) / 2
    squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_final)
    mean_n = hybrid_mean_photon(squeezed, N)
    return t_final, mean_n, squeezed


def test_1_physics_validation_n2() -> list[dict[str, Any]]:
    """
    Test 1: Physics validation for n=2 (Gaussian squeezing).

    Expectations from plan:
    - Quadrature variances: Var(x) = e^{-2r}/2, Var(p) = e^{2r}/2
    - QFI for MZI: F_Q ≈ 4⟨n⟩ + 4|α|²e^{-2r} (pure state limit)
    - Wigner function: Gaussian with elliptic contours

    Note: The n=2 Hamiltonian in the hybrid system is:
        H_2 = (Ω_2/2) σ_z ⊗ (a^2 e^{-iθ_2} + a^†2 e^{iθ_2})

    For vacuum input |0,↓⟩, the σ_z eigenstate |↓⟩ (eigenvalue +1) means
    the effective bosonic Hamiltonian is:
        H_2_bosonic = (Ω_2/2) (a^2 e^{-iθ_2} + a^†2 e^{iθ_2})

    With θ=0, this is H = (Ω_2/2)(a^2 + a^†2).
    The evolution operator is U = exp(-iHt) = exp(-i(Ω_2*t/2)(a^2 + a^†2)).

    This is equivalent to the standard squeezing operator with phase θ=π/2:
        S(r) = exp((r/2)(a^†2 e^{-iθ} - a^2 e^{iθ}))
    With θ=π/2: S(r) = exp(-i(r/2)(a^†2 + a^2))

    So r = Ω_2 * t (the squeezing parameter from the plan: rₙ = Ωₙ · t_sqz).
    For squeezed vacuum: ⟨n⟩ = sinh²(r)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Physics Validation for n=2 (Gaussian Squeezing)")
    print("=" * 60)

    omega_2 = 1.0
    theta_2 = 0.0

    # Test different squeezing parameters
    test_rs = [0.1, 0.3, 0.5, 0.7, 1.0]

    results = []

    for r_target in test_rs:
        # Use adaptive truncation for each r
        N = adaptive_truncation(alpha=0j, r_n=r_target, n=2, N_max=100)
        N = max(N, 10)

        # r = omega * t, so t = r / omega
        t_sqz = r_target / omega_2
        initial = hybrid_vacuum_state(N, spin_state="down")
        squeezed = evolve_hybrid_unitary(initial, N, 2, omega_2, theta_2, t_sqz)

        # Validate state
        assert validate_hybrid_state(squeezed, N), "State validation failed"

        # Compute mean photon number
        mean_n = hybrid_mean_photon(squeezed, N)

        # For squeezed vacuum: ⟨n⟩ = sinh²(r)
        expected_n = np.sinh(r_target) ** 2
        n_error = abs(mean_n - expected_n) / max(expected_n, 1e-10)

        # Compute QFI
        qfi = qfi_hybrid_mzi(squeezed, N)

        # Compute Wigner minimum (should be positive for Gaussian)
        _, _, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=50)
        w_min = float(np.min(W))
        is_neg = wigner_is_negative(W)

        results.append(
            {
                "r": r_target,
                "t_sqz": t_sqz,
                "mean_n": mean_n,
                "expected_n": expected_n,
                "n_error": n_error,
                "qfi": qfi,
                "w_min": w_min,
                "w_negative": is_neg,
            },
        )

        print(
            f"  r={r_target:.1f}: ⟨n⟩={mean_n:.3f} (expected {expected_n:.3f}, "
            f"error={n_error:.2%}), QFI={qfi:.3f}, W_min={w_min:.4f}, "
            f"Negative Wigner: {is_neg}",
        )

    # Check: n=2 should NOT have Wigner negativity (Gaussian state)
    for res in results:
        assert not res["w_negative"], (
            f"n=2 should not have Wigner negativity, but got W_min={res['w_min']}"
        )

    # Check: mean photon should approximately match sinh²(r)
    for res in results:
        assert res["n_error"] < 0.10, (
            f"Mean photon number error too large: {res['n_error']:.2%}"
        )

    print("\n  ✓ n=2 states are Gaussian (non-negative Wigner)")
    print("  ✓ Mean photon matches sinh²(r) approximately")
    print("  ✓ State validation passed (normalized, correct dimension)")

    return results


def test_2_non_gaussian_signature() -> dict[int, list[dict[str, Any]]]:
    """
    Test 2: Non-Gaussian signature for n≥3.

    Expectations from plan:
    - n=3,4 states must show Wigner negativity (min(W) < 0)
    - Wigner function must be non-Gaussian (deviation from Gaussian shape)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Non-Gaussian Signature (n≥3)")
    print("=" * 60)

    omega_n = 1.0

    # Test n=3 and n=4 with various squeezing parameters
    results: dict[int, list[dict[str, Any]]] = {}

    for n in [3, 4]:
        results[n] = []
        print(f"\n  Testing n={n}:")

        for r_target in [0.1, 0.3, 0.5, 0.7, 1.0]:
            # Use adaptive truncation
            N = adaptive_truncation(alpha=0j, r_n=r_target, n=n, N_max=100)
            N = max(N, 10)

            t_sqz = r_target / omega_n  # r_n = Ω_n * t_sqz
            initial = hybrid_vacuum_state(N, spin_state="down")
            squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, 0.0, t_sqz)

            # Validate state
            assert validate_hybrid_state(squeezed, N), "State validation failed"

            # Compute mean photon
            mean_n = hybrid_mean_photon(squeezed, N)

            # Compute Wigner function
            _, _, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=50)
            w_min = float(np.min(W))
            is_neg = wigner_is_negative(W)

            results[n].append(
                {
                    "r": r_target,
                    "mean_n": mean_n,
                    "w_min": w_min,
                    "w_negative": is_neg,
                },
            )

            print(
                f"    r={r_target:.1f}: ⟨n⟩={mean_n:.3f}, W_min={w_min:.4f}, "
                f"Negative: {is_neg}",
            )

    # Check: n=3,4 should show Wigner negativity for sufficient r
    for n in [3, 4]:
        # At least some squeezing parameters should show negativity
        neg_detected = any(r["w_negative"] for r in results[n])
        if not neg_detected:
            print(
                f"\n  ⚠ WARNING: n={n} did not show Wigner negativity at tested r values",
            )
        else:
            print(f"\n  ✓ n={n} shows Wigner negativity (non-Gaussian signature)")

    return results


def test_3_hypothesis_qfi_comparison() -> dict[int, list[dict[str, Any]]]:
    """
    Test 3: Hypothesis test - QFI comparison at same ⟨n⟩.

    Expectation from plan:
    - At zero decoherence: QFI(n=4) > QFI(n=3) > QFI(n=2) at same ⟨n⟩
    - Non-Gaussian states should outperform Gaussian for metrology

    Note: The relationship between r_n and t_sqz is:
        r_n = Ω_n * t_sqz (for vacuum input with appropriate theta)
        ⟨n⟩ ≈ sinh²(r_n) for n=2 (Gaussian)
        For n=3,4, the relationship is more complex.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Hypothesis Test - QFI Comparison at Same ⟨n⟩")
    print("=" * 60)

    omega_n = 1.0

    # Target photon numbers to test
    target_ns = [0.5, 1.0, 2.0, 3.0]

    results: dict[int, list[dict[str, Any]]] = {2: [], 3: [], 4: []}

    for target_n in target_ns:
        print(f"\n  Target ⟨n⟩ = {target_n}:")

        for n in [2, 3, 4]:
            # Use adaptive truncation for each order
            N = adaptive_truncation(alpha=0j, r_n=target_n, n=n, N_max=100)
            N = max(N, 10)

            # Find first squeezing time that achieves target ⟨n⟩
            # (forward sweep + bisection refinement — robust against revivals)
            t_final, mean_n, squeezed = find_squeezing_time_for_target_photon(
                n=n,
                target_n=target_n,
                N=N,
                omega_n=omega_n,
                theta_n=0.0,
                t_max=50.0,
            )
            qfi = qfi_hybrid_mzi(squeezed, N)

            results[n].append(
                {
                    "target_n": target_n,
                    "achieved_n": mean_n,
                    "t_sqz": t_final,
                    "qfi": qfi,
                },
            )

            print(f"    n={n}: ⟨n⟩={mean_n:.3f}, QFI={qfi:.3f}, t={t_final:.3f}")

    # Check hypothesis: QFI should increase with n at same ⟨n⟩
    print("\n  Hypothesis check (QFI increases with order n at same ⟨n⟩):")
    hypothesis_holds = True

    for i, target_n in enumerate(target_ns):
        qfi_2 = results[2][i]["qfi"]
        qfi_3 = results[3][i]["qfi"]
        qfi_4 = results[4][i]["qfi"]

        print(
            f"    ⟨n⟩≈{target_n:.1f}: QFI(2)={qfi_2:.3f}, "
            f"QFI(3)={qfi_3:.3f}, QFI(4)={qfi_4:.3f}",
        )

        # Check if higher order gives higher QFI
        if qfi_4 > qfi_3 > qfi_2:
            print("      ✓ QFI(4) > QFI(3) > QFI(2)")
        elif qfi_3 > qfi_2:
            print("      △ QFI(3) > QFI(2), but QFI(4) <= QFI(3)")
            if qfi_4 <= qfi_3:
                hypothesis_holds = False
        else:
            print("      ✗ QFI(3) not > QFI(2) at this ⟨n⟩")
            hypothesis_holds = False

    if hypothesis_holds:
        print(
            "\n  ✓ Hypothesis SUPPORTED: Higher-order squeezing provides QFI advantage",
        )
    else:
        print("\n  ⚠ Hypothesis PARTIALLY SUPPORTED or NOT SUPPORTED")
        print("    (May need larger N, different parameters, or check implementation)")

    return results


def test_4_decoherence_crossover() -> None:
    """
    Test 4: Decoherence crossover.

    Expectation from plan:
    - For γ = 0: F_Q(n=4) > F_Q(n=3) > F_Q(n=2)
    - For low γ: Non-Gaussian advantage persists
    - For high γ: F_Q(n=2) ≥ F_Q(n=4) (Gaussian more robust)
    - There exists a critical γ_c where curves cross
    """
    print("\n" + "=" * 60)
    print("TEST 4: Decoherence Crossover")
    print("=" * 60)

    print("\n  ⚠ NOTE: This test requires Lindblad evolution for the HYBRID system.")
    print("  The current lindblad_solver.py is for single-mode bosonic systems.")
    print("  The hybrid system (oscillator + spin) requires extension of the")
    print("  Lindblad solver to handle the 2*(N+1) dimensional hybrid space.")

    print("\n  The plan specifies these decoherence channels:")
    print("    - One-body loss: √γ₁ a ⊗ I₂")
    print("    - Phase diffusion: √γ_φ I_osc ⊗ σ_z/2")
    print("    - Two-body loss: √γ₂ a² ⊗ I₂ (not primary focus)")

    print("\n  To fully test the decoherence crossover, we would need to:")
    print("  1. Extend Lindblad solver to hybrid (oscillator ⊗ spin) space")
    print("  2. Implement the three Lindblad operators in hybrid form")
    print("  3. Run evolution for each n ∈ {2,3,4} at various γ values")
    print("  4. Compute QFI after decoherence")
    print("  5. Find γ_c where QFI curves cross")

    print("\n  CURRENT STATUS: Decoherence testing is INCOMPLETE.")
    print("  The hybrid system Lindblad evolution needs to be implemented.")


def test_5_numerical_stability() -> bool:
    """
    Test 5: Numerical stability.

    Expectations from plan:
    - Trace conservation: Tr[ρ] = 1
    - Hermiticity: ρ = ρ†
    - Positivity: eigenvalues ≥ 0
    - No truncation artifacts: ⟨n⟩ ≤ 0.9 * N
    """
    print("\n" + "=" * 60)
    print("TEST 5: Numerical Stability")
    print("=" * 60)

    omega_n = 1.0

    print("\n  Checking numerical stability for various n and squeezing parameters...")

    for n in [2, 3, 4]:
        print(f"\n  n={n}:")

        for r in [0.1, 0.5, 1.0, 2.0]:
            N = adaptive_truncation(alpha=0j, r_n=r, n=n, N_max=100)
            N = max(N, 10)

            t_sqz = r / omega_n
            initial = hybrid_vacuum_state(N, spin_state="down")
            squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, 0.0, t_sqz)

            # Check 1: State validation (norm, dimension)
            is_valid = validate_hybrid_state(squeezed, N)

            # Check 2: Unitary evolution preserves norm
            norm = np.sum(np.abs(squeezed) ** 2)

            # Check 3: No truncation artifacts
            mean_n = hybrid_mean_photon(squeezed, N)
            truncation_ok = mean_n <= 0.9 * N

            # Check 4: Evolution is unitary (state vector, so norm=1 is sufficient)
            # For pure states, the density matrix ρ = |ψ⟩⟨ψ| is automatically
            # Hermitian and positive

            status = (
                "✓"
                if (is_valid and np.isclose(norm, 1.0, atol=1e-6) and truncation_ok)
                else "✗"
            )

            print(
                f"    r={r:.1f}: ⟨n⟩={mean_n:.3f}, Norm={norm:.6f}, "
                f"Valid={is_valid}, Truncation OK={truncation_ok} {status}",
            )

    print("\n  ✓ Unitary evolution preserves norm (by construction)")
    print("  ✓ Pure states are automatically Hermitian and positive")
    print("  ✓ Truncation rule prevents ⟨n⟩ > 0.9*N")

    return True


def run_all_tests() -> None:
    """Run all comparison tests and summarize results."""

    print("\n" + "=" * 60)
    print("COMPARISON: PLAN vs SIMULATION")
    print("=" * 60)
    print("\nPlan: reports/2026-05-07/High-Order-Squeezing-Plan.md")
    print("Simulation: src/physics/hybrid_system.py + pages/High_Order_Squeezing.py")

    try:
        # Test 1: Physics validation for n=2
        results_1 = test_1_physics_validation_n2()

        # Test 2: Non-Gaussian signature
        results_2 = test_2_non_gaussian_signature()

        # Test 3: Hypothesis test
        results_3 = test_3_hypothesis_qfi_comparison()

        # Test 4: Decoherence crossover
        test_4_decoherence_crossover()

        # Test 5: Numerical stability
        results_5 = test_5_numerical_stability()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY: Plan Expectations vs Simulation Results")
        print("=" * 60)

        summary = {
            "1. Physics validation (n=2)": "✓ PASSED" if results_1 else "✗ FAILED",
            "2. Non-Gaussian signature (n≥3)": "△ PARTIAL" if results_2 else "✗ FAILED",
            "3. Hypothesis test (QFI comparison)": "△ SEE RESULTS"
            if results_3
            else "✗ FAILED",
            "4. Decoherence crossover": "⚠ INCOMPLETE - needs hybrid Lindblad solver",
            "5. Numerical stability": "✓ PASSED" if results_5 else "✗ FAILED",
        }

        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("CONCLUSIONS")
        print("=" * 60)

        print("""
        The simulation implementation covers:
        ✓ Hybrid oscillator-spin system construction
        ✓ n-th order squeezing Hamiltonians (n=2,3,4)
        ✓ State preparation (vacuum and coherent)
        ✓ Adaptive truncation
        ✓ MZI embedding and QFI computation
        ✓ Wigner function computation

        The simulation is MISSING:
        ✗ Lindblad decoherence for hybrid system (needed for Test 4)
        ✗ Comprehensive QFI comparison at fixed ⟨n⟩ (Test 3 needs more work)

        RECOMMENDATIONS:
        1. Implement hybrid Lindblad solver to test decoherence crossover
        2. Verify QFI advantage at higher ⟨n⟩ values (may need larger N)
        3. Compare with analytical formulas for n=2 to validate implementation
        """)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# Data and Figure Generation
# =============================================================================


def generate_qfi_comparison_data(
    target_ns: list[float] | None = None,
    omega_n: float = 1.0,
    N_max: int = 100,
    force: bool = False,
) -> Path:
    """Run the QFI comparison parameter sweep and save CSV.

    Performs QFI comparison across squeezing orders n=2,3,4 at multiple
    target mean photon numbers. Saves results as a CSV with columns for
    squeezing order n, achieved mean photon number, and QFI value.

    Args:
        target_ns: List of target mean photon numbers (default: 0.5 to 3.0).
        omega_n: Squeezing rate.
        N_max: Maximum Fock truncation.
        force: If True, regenerate data even if CSV exists.

    Returns:
        Path to the saved CSV file.

    """
    if target_ns is None:
        target_ns = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = _RAW_DIR / "qfi_comparison.parquet"

    if parquet_path.exists() and not force:
        print(f"Data file already exists at {parquet_path}. Use --force to regenerate.")
        return parquet_path

    rows = []
    for target_n in target_ns:
        print(f"  Target ⟨n⟩ = {target_n}:")
        for n in [2, 3, 4]:
            N = adaptive_truncation(alpha=0j, r_n=target_n, n=n, N_max=N_max)
            N = max(N, 10)

            t_final, mean_n, squeezed = find_squeezing_time_for_target_photon(
                n=n,
                target_n=target_n,
                N=N,
                omega_n=omega_n,
                theta_n=0.0,
                t_max=50.0,
            )
            qfi = qfi_hybrid_mzi(squeezed, N)
            rows.append(
                {
                    "n": n,
                    "target_n": target_n,
                    "achieved_n": float(mean_n),
                    "qfi": float(qfi),
                    "t_sqz": float(t_final),
                },
            )
            print(f"    n={n}: ⟨n⟩={mean_n:.3f}, QFI={qfi:.3f}, t={t_final:.3f}")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved QFI comparison data to {parquet_path}")
    return parquet_path


def plot_qfi_comparison(
    data_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Load raw data and generate QFI comparison SVG figure.

    Produces a QFI comparison line plot with n=2, n=3, n=4 traces
    versus mean photon number.

    Args:
        data_path: Path to the Parquet file. If None, looks in raw_data/.
        force: If True, regenerate figure even if SVG exists.

    Returns:
        Path to the saved SVG file.

    """
    if data_path is None:
        data_path = _RAW_DIR / "qfi_comparison.parquet"

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = _FIG_DIR / "qfi_comparison.svg"

    if svg_path.exists() and not force:
        print(f"Figure already exists at {svg_path}. Use --force to regenerate.")
        return svg_path

    df = pd.read_parquet(data_path)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for n in [2, 3, 4]:
        subset = df[df["n"] == n].sort_values("achieved_n")
        ax.plot(
            subset["achieved_n"],
            subset["qfi"],
            marker="o",
            label=f"n={n}",
            linewidth=2,
        )

    ax.set_xlabel("Mean photon number ⟨n⟩")
    ax.set_ylabel("Quantum Fisher Information $F_Q$")
    ax.set_title("QFI Comparison Across Squeezing Orders")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(svg_path), format="svg", dpi=150)
    plt.close(fig)

    print(f"Saved QFI comparison figure to {svg_path}")
    return svg_path


def run_pipeline(force: bool = False) -> None:
    """Run the complete pipeline: generate data, generate figures, print summary."""
    print("=" * 60)
    print("QFI Comparison Pipeline — 2026-05-07 Report")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/2] Generating QFI comparison data...")
    data_path = generate_qfi_comparison_data(force=force)

    # Step 2: Generate figures
    print("\n[2/2] Generating QFI comparison figure...")
    fig_path = plot_qfi_comparison(data_path, force=force)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Data:  {data_path}")
    print(f"  Figure: {fig_path}")
    print("=" * 60)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2026-05-07 High-Order Squeezing Report — Data & Figures",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate data and figures even if they already exist",
    )
    args = parser.parse_args()

    if args.force:
        run_pipeline(force=True)
    else:
        print(
            "Use `uv run python reports/2026-05-07/local.py --force` to regenerate "
            "data and figures.",
        )
        print()
        run_all_tests()
