"""
Single-Particle MZI Simulation: Sensitivity Scaling with Holding Time.

Implements the exact analytical model for a single-particle (spin-1/2
equivalent) Mach-Zehnder interferometer where the parameter ω is encoded
via H = ω J_z during a holding time t_hold.

Physical Model:
- Hilbert space: two-mode bosonic Fock space truncated at max_photons = 1.
  Only two basis states are physical: |1,0⟩ and |0,1⟩ (dimension 2).
- Beam splitter: 50:50, U_BS = exp(-i(π/4)(a_0^† a_1 + a_1^† a_0)).
- Holding: U_hold(t_hold) = exp(-i ω t_hold J_z), with J_z = (n_1 - n_2)/2.
- State: |1,0⟩ → U_BS → U_hold → U_BS → measurement of J_z.

Analytical result:
- ⟨J_z⟩ = -(1/2) cos(ω t_hold)
- Var(J_z) = (1/4) sin²(ω t_hold)
- ∂⟨J_z⟩/∂ω = (t_hold/2) sin(ω t_hold)
- Δω = 1/t_hold  (independent of ω, away from sin(ω t_hold) = 0)

Units: Dimensionless throughout.
Conventions: J_z = (n_1 - n_2)/2, beam-splitter generator = a_0^† a_1 + a_1^† a_0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy

from src.physics.beam_splitter import bs_fock
from src.physics.mzi_simulation import prepare_input_state
from src.physics.mzi_states import compute_jz_variance, two_mode_jz_operator


def build_holding_unitary(omega: float, t_hold: float, jz: np.ndarray) -> np.ndarray:
    """Build holding unitary U_hold = exp(-i ω t_hold J_z).

    Args:
        omega: True value of the rate parameter ω.
        t_hold: Holding time t_hold.
        jz: J_z operator.

    Returns:
        Unitary matrix of same dimension as jz.

    """
    return scipy.linalg.expm(-1j * omega * t_hold * jz)


def evolve_single_particle_mzi(
    omega: float,
    t_hold: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    input_state: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve a single-particle state through the MZI circuit.

    Circuit: |ψ_in⟩ → U_BS → U_hold(t_hold) → U_BS → |ψ_out⟩

    Args:
        omega: True value of the rate parameter ω.
        t_hold: Holding time t_hold.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        input_state: Input state (default: |1,0⟩).

    Returns:
        Final state vector after the full MZI circuit.

    """
    if input_state is None:
        input_state = prepare_input_state("single_photon", max_photons=1, mode=0)

    u_hold = build_holding_unitary(omega, t_hold, jz)
    psi = u_bs @ input_state
    psi = u_hold @ psi
    return u_bs @ psi


def compute_analytical_derivative(t_hold: float, omega: float) -> float:
    """Compute ∂⟨J_z⟩/∂ω analytically.

    ∂⟨J_z⟩/∂ω = (t_hold/2) · sin(ω t_hold)

    Args:
        t_hold: Holding time t_hold.
        omega: True value of ω.

    Returns:
        Analytical derivative value.

    """
    return float(0.5 * t_hold * np.sin(omega * t_hold))


def compute_numerical_derivative(
    omega: float,
    t_hold: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    fd_step: float = 1e-6,
) -> float:
    """Compute ∂⟨J_z⟩/∂ω via central finite differences.

    ∂⟨J_z⟩/∂ω ≈ [⟨J_z⟩(ω + δ) - ⟨J_z⟩(ω - δ)] / (2δ)

    Args:
        omega: True value of ω (center point).
        t_hold: Holding time t_hold.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Numerical derivative value.

    """
    psi_plus = evolve_single_particle_mzi(omega + fd_step, t_hold, u_bs, jz)
    psi_minus = evolve_single_particle_mzi(omega - fd_step, t_hold, u_bs, jz)

    jz_plus = float(np.real(np.conj(psi_plus) @ jz @ psi_plus))
    jz_minus = float(np.real(np.conj(psi_minus) @ jz @ psi_minus))

    return float((jz_plus - jz_minus) / (2.0 * fd_step))


def compute_delta_omega_from_propagation(
    t_hold: float,
    omega: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    use_numerical: bool = False,
    fd_step: float = 1e-6,
) -> tuple[float, float, float, float, float]:
    """Compute sensitivity Δω via error propagation for a single t_hold.

    Δω = √Var(J_z) / |∂⟨J_z⟩/∂ω|

    Args:
        t_hold: Holding time t_hold.
        omega: True value of ω.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        use_numerical: If True, use finite-difference derivative.
        fd_step: Finite-difference step size (ignored if use_numerical=False).

    Returns:
        Tuple (delta_omega, jz_mean, jz_var, d_jz_domega, is_fringe_extremum).

    """
    psi = evolve_single_particle_mzi(omega, t_hold, u_bs, jz)
    jz_mean = float(np.real(np.conj(psi) @ jz @ psi))
    jz_var = compute_jz_variance(psi, max_photons=1)

    if use_numerical:
        d_jz = compute_numerical_derivative(omega, t_hold, u_bs, jz, fd_step)
    else:
        d_jz = compute_analytical_derivative(t_hold, omega)

    # Detect fringe extremum where denominator vanishes
    abs_sin = abs(np.sin(omega * t_hold))
    is_fringe = abs_sin < 1e-6

    denom = abs(d_jz)
    delta_omega = np.inf if denom < 1e-15 else np.sqrt(jz_var) / denom

    return delta_omega, float(jz_mean), float(jz_var), float(d_jz), bool(is_fringe)


def compute_sensitivity_sweep(
    omega: float = 1.0,
    t_hold_min: float = 0.1,
    t_hold_max: float = 100.0,
    n_points: int = 50,
    delta_fd: float = 1e-6,
) -> pd.DataFrame:
    """Sweep over t_hold and compute sensitivity from both analytical and numerical derivatives.

    Args:
        omega: True value of ω (radians per unit time).
        t_hold_min: Minimum holding time.
        t_hold_max: Maximum holding time.
        n_points: Number of log-spaced t_hold points.
        delta_fd: Finite-difference step size.

    Returns:
        DataFrame with columns:
            t_hold, omega, jz_mean, jz_var,
            d_jz_analytical, d_jz_numerical,
            delta_omega_analytical, delta_omega_numerical,
            delta_omega_theory (1/t_hold),
            is_fringe_extremum, abs_sin

    """
    # Build operators once
    u_bs = bs_fock(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)

    t_hold_values = np.logspace(np.log10(t_hold_min), np.log10(t_hold_max), n_points)

    rows = []
    for t_hold in t_hold_values:
        # Analytical derivative
        dt_a, jz_mean, jz_var, d_jz_a, is_fringe = compute_delta_omega_from_propagation(
            t_hold,
            omega,
            u_bs,
            jz,
            use_numerical=False,
        )

        # Numerical derivative
        dt_n, _, _, d_jz_n, _ = compute_delta_omega_from_propagation(
            t_hold,
            omega,
            u_bs,
            jz,
            use_numerical=True,
            fd_step=delta_fd,
        )

        abs_sin_val = abs(np.sin(omega * t_hold))

        rows.append(
            {
                "t_hold": t_hold,
                "omega": omega,
                "jz_mean": jz_mean,
                "jz_var": jz_var,
                "d_jz_analytical": d_jz_a,
                "d_jz_numerical": d_jz_n,
                "delta_omega_analytical": dt_a,
                "delta_omega_numerical": dt_n,
                "delta_omega_theory": 1.0 / t_hold,
                "is_fringe_extremum": is_fringe,
                "abs_sin": abs_sin_val,
            },
        )

    return pd.DataFrame(rows)


def run_validation(omega: float = 1.0, t_hold: float = 1.0) -> dict:
    """Run all validation checks for the single-particle MZI simulation.

    Args:
        omega: True value of ω.
        t_hold: Holding time t_hold.

    Returns:
        Dictionary with validation results:
            - state_normalized: bool
            - bs_unitary: bool
            - delta_omega_matches_theory: bool
            - derivative_match: bool
            - derivative_relative_diff: float

    """
    u_bs = bs_fock(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)

    # State normalization
    psi = evolve_single_particle_mzi(omega, t_hold, u_bs, jz)
    norm = float(np.linalg.norm(psi))
    state_normalized = bool(np.isclose(norm, 1.0))

    # BS unitarity
    bs_unitary = bool(np.allclose(u_bs @ u_bs.conj().T, np.eye(4)))

    # Analytical Δω
    dt_a, _, _, _, _ = compute_delta_omega_from_propagation(
        t_hold,
        omega,
        u_bs,
        jz,
        use_numerical=False,
    )
    theory = 1.0 / t_hold
    delta_omega_matches = bool(np.isclose(dt_a, theory))

    # Derivative match
    d_analytical = compute_analytical_derivative(t_hold, omega)
    d_numerical = compute_numerical_derivative(omega, t_hold, u_bs, jz)

    denom = max(abs(d_analytical), 1e-15)
    rel_diff = abs(d_analytical - d_numerical) / denom
    derivative_match = bool(np.isclose(d_analytical, d_numerical, rtol=1e-6))

    return {
        "state_normalized": state_normalized,
        "norm": norm,
        "bs_unitary": bs_unitary,
        "delta_omega_matches_theory": delta_omega_matches,
        "delta_omega_analytical": float(dt_a),
        "delta_omega_theory": float(theory),
        "derivative_match": derivative_match,
        "derivative_relative_diff": float(rel_diff),
        "d_jz_analytical": float(d_analytical),
        "d_jz_numerical": float(d_numerical),
    }
