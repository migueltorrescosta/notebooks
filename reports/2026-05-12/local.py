"""
Local module for the 2026-05-12 reports.

Contains all code exclusive to these reports:
- Single-Particle MZI Holding-Time Scaling (core simulation functions,
  data generation, and figure creation)
- Ancilla-Assisted Metrology Optimization (`validate_hold_unitarity`)

Usage:
    uv run python reports/2026-05-12/local.py --force
    uv run python reports/2026-05-12/local.py --experiment single-particle --force

This module is **not** importable as ``reports.2026-05-12.local`` (the directory
name contains hyphens).  Importers should use ``importlib.util.spec_from_file_location``
or add this directory to ``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for shared-module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  # noqa: E402
import scipy  # noqa: E402
import seaborn as sns  # noqa: E402

from src.analysis.ancilla_optimization import (  # noqa: E402
    build_two_qubit_operators,
    hold_unitary,
    random_search_alpha,
    run_theta_scan,
    scan_alpha_single_parameter,
)
from src.physics.mzi_states import two_mode_jz_operator  # noqa: E402

sns.set_theme(style="whitegrid")

REPORT_DATE = "2026-05-12"
REPORTS_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# Single-Particle MZI: Sensitivity Scaling with Holding Time
# =============================================================================
#
# Implements the exact analytical model for a single-particle (spin-1/2
# equivalent) Mach-Zehnder interferometer where the parameter θ is encoded
# via H = θ J_z during a holding time T_H.
#
# Physical Model:
# - Hilbert space: two-mode bosonic Fock space truncated at max_photons = 1.
#   Only two basis states are physical: |1,0⟩ and |0,1⟩ (dimension 2).
# - Beam splitter: 50:50, U_BS = exp(-i(π/4)(a₀†a₁ + a₁†a₀)).
# - Holding: U_hold(T_H) = exp(-i θ T_H J_z), with J_z = (n₁ - n₂)/2.
# - State: |1,0⟩ → U_BS → U_hold → U_BS → measurement of J_z.
#
# Analytical result:
# ⟨J_z⟩ = -(1/2) cos(θ T_H)
# Var(J_z) = (1/4) sin²(θ T_H)
# ∂⟨J_z⟩/∂θ = (T_H/2) sin(θ T_H)
# ⇒ Δθ = 1/T_H  (independent of θ, away from sin(θ T_H) = 0)
#
# Units: Dimensionless throughout.
# Conventions: J_z = (n₁ - n₂)/2, beam-splitter generator = a₀†a₁ + a₁†a₀.

# =============================================================================
# Core Operators
# =============================================================================


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

    return scipy.linalg.expm(-1j * (np.pi / 4.0) * h_bs)


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
    return u_bs @ psi


# =============================================================================
# Observables
# =============================================================================


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

    jz_plus = float(np.real(np.conj(psi_plus) @ jz @ psi_plus))
    jz_minus = float(np.real(np.conj(psi_minus) @ jz @ psi_minus))

    return float((jz_plus - jz_minus) / (2.0 * delta))


def compute_delta_theta_from_propagation(
    t_h: float,
    theta: float,
    u_bs: np.ndarray,
    jz: np.ndarray,
    use_numerical: bool = False,
    delta: float = 1e-6,
) -> tuple[float, float, float, float, float]:
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
    jz_mean = float(np.real(np.conj(psi) @ jz @ psi))
    jz_var = compute_variance_jz(psi, jz)

    if use_numerical:
        d_jz = compute_numerical_derivative(theta, t_h, u_bs, jz, delta)
    else:
        d_jz = compute_analytical_derivative(t_h, theta)

    # Detect fringe extremum where denominator vanishes
    abs_sin = abs(np.sin(theta * t_h))
    is_fringe = abs_sin < 1e-6

    denom = abs(d_jz)
    delta_theta = np.inf if denom < 1e-15 else np.sqrt(jz_var) / denom

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
    jz = two_mode_jz_operator(1)

    t_h_values = np.logspace(np.log10(t_h_min), np.log10(t_h_max), n_points)

    rows = []
    for t_h in t_h_values:
        # Analytical derivative
        dt_a, jz_mean, jz_var, d_jz_a, is_fringe = compute_delta_theta_from_propagation(
            t_h,
            theta,
            u_bs,
            jz,
            use_numerical=False,
        )

        # Numerical derivative
        dt_n, _, _, d_jz_n, _ = compute_delta_theta_from_propagation(
            t_h,
            theta,
            u_bs,
            jz,
            use_numerical=True,
            delta=delta_fd,
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
            },
        )

    return pd.DataFrame(rows)


# =============================================================================
# Scaling Exponent Fit
# =============================================================================


def fit_scaling_exponent(
    df: pd.DataFrame,
    column: str = "delta_theta_analytical",
    exclude_fringe: bool = True,
) -> tuple[float, float, pd.DataFrame]:
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
    jz = two_mode_jz_operator(1)

    # State normalization
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    norm = np.linalg.norm(psi)
    state_normalized = bool(np.isclose(norm, 1.0))

    # BS unitarity
    bs_unitary = bool(np.allclose(u_bs @ u_bs.conj().T, np.eye(4)))

    # Analytical Δθ
    dt_a, _, _, _, _ = compute_delta_theta_from_propagation(
        t_h,
        theta,
        u_bs,
        jz,
        use_numerical=False,
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


# =============================================================================
# Ancilla Report: validate_hold_unitarity (exclusive function)
# =============================================================================


def validate_hold_unitarity(
    T_H: float = 1.0,
    theta: float = 1.0,
    alpha: tuple[float, float, float, float] = (0.1, 0.0, 0.0, 0.0),
) -> bool:
    """Validate the hold unitary.

    Args:
        T_H: Holding time.
        theta: Phase rate.
        alpha: Interaction coefficients.

    Returns:
        True if unitary.

    """
    ops = build_two_qubit_operators()
    U = hold_unitary(T_H, theta, alpha, ops)
    I_4 = np.eye(4, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), "Hold must be unitary"
    return True


# =============================================================================
# Raw Data Generation
# =============================================================================


def generate_single_particle_raw_data(force: bool = False) -> Path:
    """Run the single-particle sensitivity sweep and save raw data CSV.

    Uses the report's standard parameters:
        theta = 1.0
        T_H range: 0.1 to 100, 500 log-spaced points
        Also runs multi-theta sweep at [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    Args:
        force: Re-run even if CSV exists.

    Returns:
        Path to the saved CSV directory.

    """
    raw_dir = REPORTS_DIR / REPORT_DATE / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Standard sweep at theta = 1.0
    csv_path = raw_dir / f"{REPORT_DATE}-single-particle-sweep.csv"
    if not csv_path.exists() or force:
        print(f"  Generating {csv_path.name} ...")
        df = compute_sensitivity_sweep(theta=1.0, n_points=500)
        df.to_csv(csv_path, index=False, float_format="%.10g")
    else:
        print(f"  {csv_path.name} exists (use --force to regenerate)")

    # Multi-theta sweep
    multi_csv = raw_dir / f"{REPORT_DATE}-single-particle-multi-theta-sweep.csv"
    if not multi_csv.exists() or force:
        print(f"  Generating {multi_csv.name} ...")
        theta_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rows = []
        for theta in theta_values:
            df_t = compute_sensitivity_sweep(theta=theta, n_points=500)
            rows.append(df_t)
        multi_df = pd.concat(rows, ignore_index=True)
        multi_df.to_csv(multi_csv, index=False, float_format="%.10g")
    else:
        print(f"  {multi_csv.name} exists (use --force to regenerate)")

    return raw_dir


def generate_ancilla_raw_data(force: bool = False) -> Path:
    """Run ancilla-metrology experiments and save raw data CSVs.

    Generates:
        1. Theta-scan with Nelder–Mead optimisation
        2. Alpha single-coefficient grid scan
        3. Alpha 4D random search

    Args:
        force: Re-run even if CSVs exist.

    Returns:
        Path to the saved CSV directory.

    """
    raw_dir = REPORTS_DIR / REPORT_DATE / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Theta scan (moderate settings for automated generation;
    #    use higher n_restarts/maxiter for report-quality results)
    theta_csv = raw_dir / f"{REPORT_DATE}-ancilla-theta-scan.csv"
    if not theta_csv.exists() or force:
        print(f"  Generating {theta_csv.name} (this may take several minutes) ...")
        theta_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        scan_result = run_theta_scan(
            theta_values=theta_values,
            n_restarts=3,
            maxiter=500,
        )
        scan_result.save_csv(theta_csv)
    else:
        print(f"  {theta_csv.name} exists (use --force to regenerate)")

    # 2. Alpha single-coefficient scan
    alpha_csv = raw_dir / f"{REPORT_DATE}-ancilla-alpha-scan.csv"
    if not alpha_csv.exists() or force:
        print(f"  Generating {alpha_csv.name} ...")
        alpha_names = ["xx", "xz", "zx", "zz"]
        scan_rows = []
        for name in alpha_names:
            result = scan_alpha_single_parameter(
                alpha_name=name,
                alpha_min=-2.0,
                alpha_max=2.0,
                n_points=21,
            )
            df = result.to_dataframe()
            df["coefficient"] = name
            scan_rows.append(df)
        pd.concat(scan_rows, ignore_index=True).to_csv(
            alpha_csv, index=False, float_format="%.10g"
        )
    else:
        print(f"  {alpha_csv.name} exists (use --force to regenerate)")

    # 3. Alpha 4D random search
    random_csv = raw_dir / f"{REPORT_DATE}-ancilla-random-search.csv"
    if not random_csv.exists() or force:
        print(f"  Generating {random_csv.name} ...")
        result = random_search_alpha(n_samples=200)
        result.save_csv(random_csv)
    else:
        print(f"  {random_csv.name} exists (use --force to regenerate)")

    return raw_dir


# =============================================================================
# Figure Generation
# =============================================================================


def generate_single_particle_figures(force: bool = False) -> Path:
    """Generate figures for the single-particle MZI report.

    Creates:
        1. log-log Δθ vs T_H with analytical and numerical traces
        2. ⟨J_z⟩ vs T_H oscillatory signal
        3. ∂⟨J_z⟩/∂θ analytical vs numerical comparison

    Args:
        force: Re-generate even if SVGs exist.

    Returns:
        Path to the saved figures directory.

    """
    fig_dir = REPORTS_DIR / REPORT_DATE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run the standard sweep
    df = compute_sensitivity_sweep(theta=1.0, n_points=500)
    _, _, df_fit = fit_scaling_exponent(df)

    t_h_min = float(df["T_H"].min())
    t_h_max = float(df["T_H"].max())

    # ── Figure 1: log-log Δθ vs T_H ──
    fig1_path = fig_dir / f"{REPORT_DATE}-single-particle-scaling.svg"
    if not fig1_path.exists() or force:
        print(f"  Generating {fig1_path.name} ...")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Theory reference: 1/T_H
        t_ref = np.array([t_h_min, t_h_max])
        dt_ref = 1.0 / t_ref
        ax.loglog(
            t_ref,
            dt_ref,
            "--",
            color="gray",
            linewidth=2,
            label=r"$1/T_H$ (theory, $\alpha=-1$)",
        )

        # Clean (non-fringe) points
        clean = df[~df["is_fringe_extremum"]]
        ax.loglog(
            clean["T_H"],
            clean["delta_theta_analytical"],
            "o",
            color="#1f77b4",
            markersize=4,
            label=r"$\Delta\theta$ (analytical)",
        )
        ax.loglog(
            clean["T_H"],
            clean["delta_theta_numerical"],
            "x",
            color="#ff7f0e",
            markersize=4,
            label=r"$\Delta\theta$ (numerical)",
        )

        # Fringe points
        fringe = df[df["is_fringe_extremum"]]
        if not fringe.empty:
            ax.loglog(
                fringe["T_H"],
                fringe["delta_theta_analytical"],
                "r*",
                markersize=8,
                alpha=0.6,
                label="Fringe extremum (excluded)",
            )

        ax.set_xlabel(r"$T_H$")
        ax.set_ylabel(r"$\Delta\theta$")
        ax.set_title("Single-Particle MZI: Sensitivity Scaling with $T_H$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig1_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    # ── Figure 2: ⟨J_z⟩ vs T_H ──
    fig2_path = fig_dir / f"{REPORT_DATE}-single-particle-jz-mean.svg"
    if not fig2_path.exists() or force:
        print(f"  Generating {fig2_path.name} ...")
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.semilogx(
            df["T_H"],
            df["jz_mean"],
            "-o",
            color="#2ca02c",
            markersize=3,
            label=r"$\langle J_z \rangle$",
        )

        t_dense = np.logspace(np.log10(t_h_min), np.log10(t_h_max), 500)
        ax.semilogx(
            t_dense,
            -0.5 * np.cos(1.0 * t_dense),
            ":",
            color="gray",
            linewidth=1,
            label=r"$-\frac{1}{2}\cos(\theta T_H)$",
        )

        ax.set_xlabel(r"$T_H$")
        ax.set_ylabel(r"$\langle J_z \rangle$")
        ax.set_title("Single-Particle MZI: Signal Oscillation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig2_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    # ── Figure 3: ∂⟨J_z⟩/∂θ comparison ──
    fig3_path = fig_dir / f"{REPORT_DATE}-single-particle-derivative.svg"
    if not fig3_path.exists() or force:
        print(f"  Generating {fig3_path.name} ...")
        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.loglog(
            df["T_H"],
            df["d_jz_analytical"],
            "-",
            color="#1f77b4",
            linewidth=2,
            label=r"Analytical: $\frac{T_H}{2}\sin(\theta T_H)$",
        )
        ax1.loglog(
            df["T_H"],
            df["d_jz_numerical"],
            "x",
            color="#ff7f0e",
            markersize=3,
            label=r"Numerical ($\delta = 10^{-6}$)",
        )
        ax1.set_xlabel(r"$T_H$")
        ax1.set_ylabel(r"$\partial\langle J_z \rangle / \partial\theta$")
        ax1.set_title("Derivative Comparison: Analytical vs Numerical")
        ax1.legend(loc="upper left")

        # Relative difference on second y-axis
        rel_diff = np.abs(
            (df["d_jz_analytical"] - df["d_jz_numerical"])
            / np.maximum(np.abs(df["d_jz_analytical"]), 1e-15)
        )
        ax2 = ax1.twinx()
        ax2.loglog(
            df["T_H"],
            rel_diff,
            "--",
            color="red",
            linewidth=1,
            alpha=0.7,
            label="Relative diff",
        )
        ax2.set_ylabel("Relative difference")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(fig3_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    return fig_dir


def generate_ancilla_figures(force: bool = False) -> Path:
    """Generate figures for the ancilla-assisted metrology report.

    Creates:
        1. Δθ vs θ (best-per-theta from Nelder–Mead)

    Args:
        force: Re-generate even if SVGs exist.

    Returns:
        Path to the saved figures directory.

    """
    fig_dir = REPORTS_DIR / REPORT_DATE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Sensitivity vs θ
    fig_path = fig_dir / f"{REPORT_DATE}-ancilla-theta-scan.svg"
    if not fig_path.exists() or force:
        print(f"  Generating {fig_path.name} ...")
        theta_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        scan_result = run_theta_scan(
            theta_values=theta_values,
            n_restarts=3,
            maxiter=500,
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        # Best per θ
        ax.semilogy(
            scan_result.theta_values,
            scan_result.best_per_theta,
            "-o",
            color="firebrick",
            linewidth=2,
            markersize=8,
            label="Best Δθ (Nelder–Mead)",
        )

        # SQL reference (1/T_H with optimal T_H from each θ)
        sql_vals = []
        for theta in theta_values:
            results = scan_result.all_results.get(theta, [])
            if results:
                best = min(results, key=lambda r: r.delta_theta_opt)
                t_h_star = best.params_opt[6]
                sql_vals.append(1.0 / t_h_star if t_h_star > 0 else float("inf"))
            else:
                sql_vals.append(float("inf"))
        ax.semilogy(
            theta_values,
            sql_vals,
            "--",
            color="gray",
            linewidth=2,
            label="SQL (1/T_H*)",
        )

        ax.set_xlabel(r"True $\theta$")
        ax.set_ylabel(r"$\Delta\theta$")
        ax.set_title("Ancilla-Assisted Metrology: Sensitivity vs θ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    return fig_dir


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs for 2026-05-12 reports",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing CSVs)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["single-particle", "ancilla"],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist.
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "single-particle": (
            generate_single_particle_raw_data,
            generate_single_particle_figures,
        ),
        "ancilla": (
            generate_ancilla_raw_data,
            generate_ancilla_figures,
        ),
    }

    if args.experiment:
        raw_fn, fig_fn = experiments[args.experiment]
        print(f"\n=== {args.experiment} ===")
        print("  Raw data ...")
        raw_fn(force=args.force)
        print("  Figures ...")
        fig_fn(force=args.force)
    else:
        for name, (raw_fn, fig_fn) in experiments.items():
            print(f"\n=== {name} ===")
            print("  Raw data ...")
            raw_fn(force=args.force)
            print("  Figures ...")
            fig_fn(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
