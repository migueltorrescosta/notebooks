"""
Single-Particle MZI: Sensitivity Scaling with Holding Time.

Implements the exact analytical model for a single-particle (spin-1/2
equivalent) Mach-Zehnder interferometer where the parameter θ is encoded
via H = θ J_z during a holding time T_H.

Physical Model:
================
- Hilbert space: two-mode bosonic Fock space truncated at max_photons = 1.
  Only two basis states are physical: |1,0⟩ and |0,1⟩ (dimension 2).
- Beam splitter: 50:50, U_BS = exp(-i(π/4)(a₀†a₁ + a₁†a₀)).
- Holding: U_hold(T_H) = exp(-i θ T_H J_z), with J_z = (n₁ - n₂)/2.
- State: |1,0⟩ → U_BS → U_hold → U_BS → measurement of J_z.

Analytical result:
==================
⟨J_z⟩ = -(1/2) cos(θ T_H)
Var(J_z) = (1/4) sin²(θ T_H)
∂⟨J_z⟩/∂θ = (T_H/2) sin(θ T_H)
⇒ Δθ = 1/T_H  (independent of θ, away from sin(θ T_H) = 0)

Units: Dimensionless throughout.
Conventions: J_z = (n₁ - n₂)/2, beam-splitter generator = a₀†a₁ + a₁†a₀.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import scipy


# =============================================================================
# Core Operators
# =============================================================================


def build_jz_operator() -> np.ndarray:
    """Build J_z for the single-particle (max_photons=1) Fock space.

    J_z = (n₁ - n₂)/2, diagonal in the Fock basis.

    Only the |1,0⟩ and |0,1⟩ states are physical; the |0,0⟩ and |1,1⟩
    basis states are included for completeness but not populated.

    Returns:
        4×4 diagonal J_z operator matrix.
    """
    dim = 4  # (max_photons + 1)^2 = 4
    jz = np.zeros((dim, dim), dtype=complex)
    # Basis ordering: |n1, n2⟩ → index = n1 * 2 + n2 (since max_photons + 1 = 2)
    for n1 in range(2):
        for n2 in range(2):
            idx = n1 * 2 + n2
            jz[idx, idx] = (n1 - n2) / 2.0
    return jz


def build_beam_splitter() -> np.ndarray:
    """Build the 50:50 beam-splitter unitary.

    Uses the generator H_BS = a₀†a₁ + a₁†a₀ so that

        U_BS = exp(-i(π/4)(a₀†a₁ + a₁†a₀)).

    In the {|1,0⟩, |0,1⟩} subspace this evaluates to

        U_BS = (1/√2) [[1, -i], [-i, 1]].

    Returns:
        4×4 unitary matrix (acts on the full 4D space, but only the
        2D |1,0⟩/|0,1⟩ subspace is physically relevant).
    """
    dim = 4  # (max_photons + 1)^2 = 4
    # Build H_BS = a0†a1 + a1†a0
    h_bs = np.zeros((dim, dim), dtype=complex)
    # a0†a1: a0†a1 |n0, n1⟩ = √(n0+1)√(n1) |n0+1, n1-1⟩
    for n1 in range(2):
        for n2 in range(2):
            idx = n1 * 2 + n2
            # a0†a1: n0+1, n1-1
            if n1 < 1 and n2 > 0:
                idx_target = (n1 + 1) * 2 + (n2 - 1)
                h_bs[idx_target, idx] = np.sqrt(n1 + 1) * np.sqrt(n2)
            # a1†a0: n0-1, n1+1
            if n1 > 0 and n2 < 1:
                idx_target = (n1 - 1) * 2 + (n2 + 1)
                h_bs[idx_target, idx] = np.sqrt(n1) * np.sqrt(n2 + 1)

    u_bs = scipy.linalg.expm(-1j * (np.pi / 4.0) * h_bs)
    return u_bs


def build_holding_unitary(theta: float, t_h: float, jz: np.ndarray) -> np.ndarray:
    """Build holding unitary U_hold = exp(-i θ T_H J_z).

    Args:
        theta: True value of the rate parameter θ.
        t_h: Holding time T_H.
        jz: J_z operator.

    Returns:
        Unitary matrix of same dimension as jz.
    """
    return scipy.linalg.expm(-1j * theta * t_h * jz)


# =============================================================================
# State Preparation
# =============================================================================


def fock_state(n0: int, n1: int) -> np.ndarray:
    """Create a Fock state |n₀, n₁⟩ for the 2-mode space (max_photons=1).

    Args:
        n0: Photons in mode 0 (0 or 1).
        n1: Photons in mode 1 (0 or 1).

    Returns:
        4-element state vector.

    Raises:
        ValueError: If n0 or n1 is not 0 or 1.
    """
    if n0 not in (0, 1) or n1 not in (0, 1):
        raise ValueError(f"Photon numbers must be 0 or 1, got ({n0}, {n1})")
    state = np.zeros(4, dtype=complex)
    idx = n0 * 2 + n1
    state[idx] = 1.0
    return state


# =============================================================================
# MZI Evolution
# =============================================================================


def evolve_single_particle_mzi(
    theta: float,
    t_h: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    input_state: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve a single-particle state through the MZI circuit.

    Circuit: |ψ_in⟩ → U_BS → U_hold(T_H) → U_BS → |ψ_out⟩

    Args:
        theta: True value of the rate parameter θ.
        t_h: Holding time T_H.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        input_state: Input state (default: |1,0⟩).

    Returns:
        Final state vector after the full MZI circuit.
    """
    if input_state is None:
        input_state = fock_state(1, 0)

    u_hold = build_holding_unitary(theta, t_h, jz)
    psi = u_bs @ input_state
    psi = u_hold @ psi
    psi = u_bs @ psi
    return psi


# =============================================================================
# Observables
# =============================================================================


def compute_expectation_jz(state: np.ndarray, jz: np.ndarray) -> float:
    """Compute ⟨J_z⟩ = ⟨ψ|J_z|ψ⟩.

    Args:
        state: Pure state vector.
        jz: J_z operator.

    Returns:
        Expectation value (real).
    """
    return float(np.real(np.conj(state) @ jz @ state))


def compute_variance_jz(state: np.ndarray, jz: np.ndarray) -> float:
    """Compute Var(J_z) = ⟨J_z²⟩ - ⟨J_z⟩².

    Args:
        state: Pure state vector.
        jz: J_z operator.

    Returns:
        Variance (non-negative real).
    """
    mean = np.conj(state) @ jz @ state
    jz_sq = jz @ jz
    mean_sq = np.conj(state) @ jz_sq @ state
    var = np.real(mean_sq - mean**2)
    # Guard against tiny negative values from numerical error
    return max(0.0, var)


def compute_analytical_derivative(t_h: float, theta: float) -> float:
    """Compute ∂⟨J_z⟩/∂θ analytically.

    ∂⟨J_z⟩/∂θ = (T_H/2) · sin(θ T_H)

    Args:
        t_h: Holding time T_H.
        theta: True value of θ.

    Returns:
        Analytical derivative value.
    """
    return float(0.5 * t_h * np.sin(theta * t_h))


def compute_numerical_derivative(
    theta: float,
    t_h: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    delta: float = 1e-6,
) -> float:
    """Compute ∂⟨J_z⟩/∂θ via central finite differences.

    ∂⟨J_z⟩/∂θ ≈ [⟨J_z⟩(θ + δ) - ⟨J_z⟩(θ - δ)] / (2δ)

    Args:
        theta: True value of θ (center point).
        t_h: Holding time T_H.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        delta: Finite-difference step size (default 1e-6).

    Returns:
        Numerical derivative value.
    """
    psi_plus = evolve_single_particle_mzi(theta + delta, t_h, u_bs, jz)
    psi_minus = evolve_single_particle_mzi(theta - delta, t_h, u_bs, jz)

    jz_plus = compute_expectation_jz(psi_plus, jz)
    jz_minus = compute_expectation_jz(psi_minus, jz)

    return float((jz_plus - jz_minus) / (2.0 * delta))


def compute_delta_theta_from_propagation(
    t_h: float,
    theta: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    use_numerical: bool = False,
    delta: float = 1e-6,
) -> Tuple[float, float, float, float, float]:
    """Compute sensitivity Δθ via error propagation for a single T_H.

    Δθ = √Var(J_z) / |∂⟨J_z⟩/∂θ|

    Args:
        t_h: Holding time T_H.
        theta: True value of θ.
        u_bs: Beam-splitter unitary.
        jz: J_z operator.
        use_numerical: If True, use finite-difference derivative.
        delta: Finite-difference step size (ignored if use_numerical=False).

    Returns:
        Tuple (delta_theta, jz_mean, jz_var, d_jz_dtheta, is_fringe_extremum).
    """
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    jz_mean = compute_expectation_jz(psi, jz)
    jz_var = compute_variance_jz(psi, jz)

    if use_numerical:
        d_jz = compute_numerical_derivative(theta, t_h, u_bs, jz, delta)
    else:
        d_jz = compute_analytical_derivative(t_h, theta)

    # Detect fringe extremum where denominator vanishes
    abs_sin = abs(np.sin(theta * t_h))
    is_fringe = abs_sin < 1e-6

    denom = abs(d_jz)
    if denom < 1e-15:
        delta_theta = np.inf
    else:
        delta_theta = np.sqrt(jz_var) / denom

    return delta_theta, float(jz_mean), float(jz_var), float(d_jz), bool(is_fringe)


# =============================================================================
# Parameter Sweep
# =============================================================================


def compute_sensitivity_sweep(
    theta: float = 1.0,
    t_h_min: float = 0.1,
    t_h_max: float = 100.0,
    n_points: int = 50,
    delta_fd: float = 1e-6,
) -> pd.DataFrame:
    """Sweep over T_H and compute sensitivity from both analytical and numerical derivatives.

    Args:
        theta: True value of θ (radians per unit time).
        t_h_min: Minimum holding time.
        t_h_max: Maximum holding time.
        n_points: Number of log-spaced T_H points.
        delta_fd: Finite-difference step size.

    Returns:
        DataFrame with columns:
            T_H, theta, jz_mean, jz_var,
            d_jz_analytical, d_jz_numerical,
            delta_theta_analytical, delta_theta_numerical,
            delta_theta_theory (1/T_H),
            is_fringe_extremum, abs_sin
    """
    # Build operators once
    u_bs = build_beam_splitter()
    jz = build_jz_operator()

    t_h_values = np.logspace(np.log10(t_h_min), np.log10(t_h_max), n_points)

    rows = []
    for t_h in t_h_values:
        # Analytical derivative
        dt_a, jz_mean, jz_var, d_jz_a, is_fringe = compute_delta_theta_from_propagation(
            t_h, theta, u_bs, jz, use_numerical=False
        )

        # Numerical derivative
        dt_n, _, _, d_jz_n, _ = compute_delta_theta_from_propagation(
            t_h, theta, u_bs, jz, use_numerical=True, delta=delta_fd
        )

        abs_sin_val = abs(np.sin(theta * t_h))

        rows.append(
            {
                "T_H": t_h,
                "theta": theta,
                "jz_mean": jz_mean,
                "jz_var": jz_var,
                "d_jz_analytical": d_jz_a,
                "d_jz_numerical": d_jz_n,
                "delta_theta_analytical": dt_a,
                "delta_theta_numerical": dt_n,
                "delta_theta_theory": 1.0 / t_h,
                "is_fringe_extremum": is_fringe,
                "abs_sin": abs_sin_val,
            }
        )

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# Scaling Exponent Fit
# =============================================================================


def fit_scaling_exponent(
    df: pd.DataFrame,
    column: str = "delta_theta_analytical",
    exclude_fringe: bool = True,
) -> Tuple[float, float, pd.DataFrame]:
    """Fit scaling exponent α from log-log linear regression.

    Fits log(Δθ) = α · log(T_H) + const via least squares.

    Args:
        df: DataFrame with columns "T_H" and column (default delta_theta).
        column: Column name for Δθ values to fit.
        exclude_fringe: If True, exclude points near fringe extrema.

    Returns:
        Tuple (alpha, r_squared, fit_df) where fit_df is a copy of df
        with a "valid_for_fit" column added.
    """
    fit_df = df.copy()
    fit_df["valid_for_fit"] = True

    if exclude_fringe:
        fit_df["valid_for_fit"] = ~fit_df["is_fringe_extremum"]

    valid = fit_df[fit_df["valid_for_fit"]].copy()
    # Also exclude any infinite values
    valid = valid[np.isfinite(valid[column])]

    if len(valid) < 3:
        return np.nan, np.nan, fit_df

    log_t = np.log10(valid["T_H"].values)
    log_dt = np.log10(valid[column].values)

    # Linear fit: log10(Δθ) = α · log10(T_H) + c
    coeffs = np.polyfit(log_t, log_dt, 1)
    alpha = coeffs[0]

    # R²
    log_dt_pred = np.polyval(coeffs, log_t)
    residuals = log_dt - log_dt_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_dt - np.mean(log_dt)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(alpha), float(r_squared), fit_df


# =============================================================================
# Validation
# =============================================================================


def run_validation(theta: float = 1.0, t_h: float = 1.0) -> dict:
    """Run all validation checks for the single-particle MZI simulation.

    Args:
        theta: True value of θ.
        t_h: Holding time T_H.

    Returns:
        Dictionary with validation results:
            - state_normalized: bool
            - bs_unitary: bool
            - delta_theta_matches_theory: bool
            - derivative_match: bool
            - derivative_relative_diff: float
    """
    u_bs = build_beam_splitter()
    jz = build_jz_operator()

    # State normalization
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    norm = np.linalg.norm(psi)
    state_normalized = bool(np.isclose(norm, 1.0))

    # BS unitarity
    bs_unitary = bool(np.allclose(u_bs @ u_bs.conj().T, np.eye(4)))

    # Analytical Δθ
    dt_a, _, _, _, _ = compute_delta_theta_from_propagation(
        t_h, theta, u_bs, jz, use_numerical=False
    )
    theory = 1.0 / t_h
    delta_theta_matches = bool(np.isclose(dt_a, theory))

    # Derivative match
    d_analytical = compute_analytical_derivative(t_h, theta)
    d_numerical = compute_numerical_derivative(theta, t_h, u_bs, jz)

    denom = max(abs(d_analytical), 1e-15)
    rel_diff = abs(d_analytical - d_numerical) / denom
    derivative_match = bool(np.isclose(d_analytical, d_numerical, rtol=1e-6))

    return {
        "state_normalized": state_normalized,
        "norm": float(norm),
        "bs_unitary": bs_unitary,
        "delta_theta_matches_theory": delta_theta_matches,
        "delta_theta_analytical": float(dt_a),
        "delta_theta_theory": float(theory),
        "derivative_match": derivative_match,
        "derivative_relative_diff": float(rel_diff),
        "d_jz_analytical": float(d_analytical),
        "d_jz_numerical": float(d_numerical),
    }
