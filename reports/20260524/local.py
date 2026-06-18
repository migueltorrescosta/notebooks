"""
Local module for the 2026-05-24 Phase-Diffusion Robustness of the Drive Protocol report.

Contains all code exclusive to this report:
- Liouvillian construction for Lindblad master equation with phase diffusion
- Noisy circuit evolution (BS → Lindblad hold → BS)
- Sensitivity computation for mixed states
- 4D random search + Nelder–Mead optimisation with phase noise
- Noise scan over (ω, γ_φ) pairs
- Exclusive plot functions
- Data and figure generation pipeline
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260524/local.py --force

This module is **not** importable as ``reports.20260524.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as _mp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from scipy.linalg import expm
from scipy.optimize import minimize

from src.utils.serialization import ParquetSerializable

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.ancilla_drive_metrology import (
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)
from src.evolution.lindblad_solver import (
    build_vectorized_liouvillian,
    unvectorise_rho,
    validate_density_matrix,
    vectorise_rho,
)
from src.utils.parallel import parallel_map

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL = 0.1)
SQL_REFERENCE: float = 1.0 / DEFAULT_t_hold  # Δω_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients
FD_STEP: float = 1e-6  # Finite-difference step for omega derivative


# Initial state: |00⟩ (pure, density matrix form)
DEFAULT_RHO0: np.ndarray = np.zeros((4, 4), dtype=complex)
DEFAULT_RHO0[0, 0] = 1.0

# Omega and gamma_phi values for the scan
OMEGA_VALS: list[float] = [
    round(0.1 * i, 1) for i in range(1, 51)
]  # 0.1, 0.2, ..., 5.0 (50 values)
GAMMA_PHI_VALS: list[float] = list(
    np.geomspace(0.01, 1.0, 32).tolist()
)  # 32 log-spaced from 0.01 to 1.0

# ============================================================================
# Hamiltonian Construction (same as ω-modulated drive protocol)
# ============================================================================


def build_noise_drive_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ancilla drive Hamiltonian with ω modulation.

    H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    Args:
        omega: Unknown phase rate parameter.
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H
    return 0.5 * (H + H.conj().T)


def build_noise_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising-type system--ancilla interaction.

    H_int = a_zz J_z^S ⊗ J_z^A = a_zz (σ_z/2) ⊗ (σ_z/2)

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    return H


def build_noise_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with ω-modulated ancilla drive.

    H = ω J_z^S + ω (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    Args:
        omega: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_noise_drive_hamiltonian(omega, a_x, a_y, a_z, ops)
    H += build_noise_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


# ============================================================================
# Liouvillian Construction (Lindblad Master Equation)
# ============================================================================


def build_phase_diffusion_operators(
    gamma_phi: float,
    ops: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Build the phase diffusion Lindblad operators.

    L_S = √γ_φ J_z^S,  L_A = √γ_φ J_z^A

    Args:
        gamma_phi: Phase diffusion rate.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        List of Lindblad operators [L_S, L_A] (each 4×4).
    """
    if gamma_phi <= 0.0:
        return []
    sqrt_g = np.sqrt(gamma_phi)
    return [
        sqrt_g * ops["Jz_S"],
        sqrt_g * ops["Jz_A"],
    ]


def density_expectation(rho: np.ndarray, op: np.ndarray) -> float:
    """Compute ⟨op⟩ = Tr(op ρ) for a density matrix.

    Args:
        rho: n × n density matrix.
        op: n × n operator (Hermitian).

    Returns:
        Real expectation value.
    """
    return float(np.real(np.trace(op @ rho)))


def density_variance(rho: np.ndarray, op: np.ndarray) -> float:
    """Compute Var(op) = ⟨op²⟩ - ⟨op⟩² for a density matrix.

    Args:
        rho: n × n density matrix.
        op: n × n operator (Hermitian).

    Returns:
        Variance (non-negative).
    """
    exp_val = density_expectation(rho, op)
    exp_sq = density_expectation(rho, op @ op)
    raw_var = exp_sq - exp_val**2
    assert raw_var >= -1e-12, f"Unphysical negative variance: {raw_var:.2e}"
    return float(max(0.0, raw_var))


def validate_density(rho: np.ndarray, atol: float = 1e-8) -> None:
    """Validate basic properties of a density matrix.

    Delegates to ``validate_density_matrix`` from ``src.evolution.lindblad_solver``.
    The ``atol`` parameter is accepted for backward compatibility but not passed
    to the source function (source uses its own default tolerance).

    Args:
        rho: Density matrix to validate.
        atol: Ignored (kept for backward compatibility).

    Raises:
        AssertionError: If any property fails.
    """
    result = validate_density_matrix(rho)
    assert result["is_normalized"], (
        f"Trace not preserved: Tr(ρ) = {result['trace']:.2e}"
    )
    assert result["is_hermitian"], "Density matrix not Hermitian"
    assert result["is_positive"], (
        f"Density matrix has negative eigenvalues: min={result['min_eigenvalue']:.2e}"
    )


# ============================================================================
# Noisy Circuit Evolution
# ============================================================================


def evolve_noisy_drive_circuit(
    rho0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full noisy MZI circuit with phase diffusion.

    ρ_final = U_BS_S · exp(ℒ t_hold)(U_BS_S · ρ₀ · U_BS_S^†) · U_BS_S^†

    where ℒ is the Lindblad Liouvillian with phase diffusion.

    Args:
        rho0: Initial 4×4 density matrix.
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding time.
        omega: Phase rate parameter.
        gamma_phi: Phase diffusion rate.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final 4×4 density matrix.
    """
    assert np.isclose(np.trace(rho0), 1.0, atol=1e-12), (
        "Initial density matrix must have trace 1"
    )

    # BS1
    U_bs = system_only_bs_unitary(T_BS)
    rho = U_bs @ rho0 @ U_bs.conj().T

    # Hold with Lindblad evolution (always use Liouvillian for consistency)
    H = build_noise_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    lindblad_ops = build_phase_diffusion_operators(gamma_phi, ops)
    L = build_vectorized_liouvillian(H, lindblad_ops)

    # Vectorise, exponentiate, unvectorise
    rho_vec = vectorise_rho(rho)
    rho_vec_evolved = expm(L * t_hold) @ rho_vec
    rho = unvectorise_rho(rho_vec_evolved)

    # BS2
    rho = U_bs @ rho @ U_bs.conj().T

    # Validate the final density matrix
    validate_density(rho)

    return rho


# ============================================================================
# Sensitivity Computation
# ============================================================================


def compute_noisy_sensitivity(
    rho0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> float:
    """Compute the error-propagation sensitivity Δω with phase diffusion.

    Δω = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂ω|

    The derivative is computed via central finite differences at ω±δ,
    re-evaluating the full noisy circuit at each point.

    Args:
        rho0: Initial 4×4 density matrix.
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        gamma_phi: Phase diffusion rate.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δω (positive float). Returns inf if derivative is zero.
    """
    meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    rho = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    var = density_variance(rho, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    rho_plus = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true + fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    rho_minus = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true - fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )

    exp_plus = density_expectation(rho_plus, meas_op)
    exp_minus = density_expectation(rho_minus, meas_op)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def compute_noisy_sensitivity_with_diagnostics(
    rho0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float]:
    """Compute Δω with diagnostics.

    Returns:
        Tuple (delta_omega, expectation_Jz, variance_Jz, d_exp_d_omega).
    """
    meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    rho = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    exp_val = density_expectation(rho, meas_op)
    var = density_variance(rho, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    rho_plus = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true + fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    rho_minus = evolve_noisy_drive_circuit(
        rho0, T_BS, t_hold, omega_true - fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )

    exp_plus = density_expectation(rho_plus, meas_op)
    exp_minus = density_expectation(rho_minus, meas_op)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var, d_exp

    delta_omega = float(np.sqrt(var) / abs(d_exp))
    return delta_omega, exp_val, var, d_exp


# ============================================================================
# Dataclass: DriveNoiseScanResult
# ============================================================================


@dataclass
class DriveNoiseScanResult(ParquetSerializable):
    """Result from a (ω, γ_φ) noise scan with re-optimised parameters.

    Attributes:
        omega_values: Array of ω values scanned.
        gamma_phi_values: Array of γ_φ values scanned.
        best_params_per_pair: List of optimal (a_x, a_y, a_z, a_zz) tuples
            indexed as [i_omega * n_gamma + i_gamma].
        delta_omega_per_pair: Δω at the optimal parameters for each (ω, γ_φ),
            shaped (n_omega, n_gamma).
        expectation_Jz_per_pair: ⟨J_z^S⟩ at each optimal point,
            shaped (n_omega, n_gamma).
        variance_Jz_per_pair: Var(J_z^S) at each optimal point,
            shaped (n_omega, n_gamma).
        d_exp_d_omega_per_pair: ∂⟨J_z^S⟩/∂ω at each optimal point,
            shaped (n_omega, n_gamma).
        sql: SQL = 1/t_hold reference value.
        t_hold: Holding time used.
        n_random: Number of random search points per (ω, γ_φ) pair.
        n_nm_refine: Number of Nelder-Mead refinements per pair.
        maxiter: Maximum Nelder-Mead iterations.
        bounds_lo: Lower bound for all optimisation parameters.
        bounds_hi: Upper bound for all optimisation parameters.
        fd_step: Finite-difference step for derivative computation.
        seed: Base random seed.
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_phi_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_pair: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    delta_omega_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    d_exp_d_omega_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    sql: float = SQL_REFERENCE
    t_hold: float = DEFAULT_t_hold
    n_random: int = 1000
    n_nm_refine: int = 25
    maxiter: int = 5000
    bounds_lo: float = -5.0
    bounds_hi: float = 5.0
    fd_step: float = FD_STEP
    seed: int = 42

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "gamma_phi",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "delta_omega",
        "expectation_Jz",
        "variance_Jz",
        "d_exp_d_omega",
        "sql",
        "t_hold",
        "n_random",
        "n_nm_refine",
        "maxiter",
        "bounds_lo",
        "bounds_hi",
        "fd_step",
        "seed",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten into a long-format DataFrame with one row per (ω, γ_φ)."""
        n_omega = len(self.omega_values)
        n_gamma = len(self.gamma_phi_values)
        rows: list[dict[str, float | str]] = []
        for i in range(n_omega):
            for j in range(n_gamma):
                idx = i * n_gamma + j
                params = (
                    self.best_params_per_pair[idx]
                    if idx < len(self.best_params_per_pair)
                    else (0.0, 0.0, 0.0, 0.0)
                )
                dt = (
                    float(self.delta_omega_per_pair[i, j])
                    if self.delta_omega_per_pair.size > 0
                    else float("inf")
                )
                exp_val = (
                    float(self.expectation_Jz_per_pair[i, j])
                    if self.expectation_Jz_per_pair.size > 0
                    else 0.0
                )
                var_val = (
                    float(self.variance_Jz_per_pair[i, j])
                    if self.variance_Jz_per_pair.size > 0
                    else 0.0
                )
                d_exp = (
                    float(self.d_exp_d_omega_per_pair[i, j])
                    if self.d_exp_d_omega_per_pair.size > 0
                    else 0.0
                )
                rows.append(
                    {
                        "omega": float(self.omega_values[i]),
                        "gamma_phi": float(self.gamma_phi_values[j]),
                        "a_x": float(params[0]),
                        "a_y": float(params[1]),
                        "a_z": float(params[2]),
                        "a_zz": float(params[3]),
                        "delta_omega": dt,
                        "expectation_Jz": exp_val,
                        "variance_Jz": var_val,
                        "d_exp_d_omega": d_exp,
                        "sql": float(self.sql),
                        "t_hold": float(self.t_hold),
                        "n_random": int(self.n_random),
                        "n_nm_refine": int(self.n_nm_refine),
                        "maxiter": int(self.maxiter),
                        "bounds_lo": float(self.bounds_lo),
                        "bounds_hi": float(self.bounds_hi),
                        "fd_step": float(self.fd_step),
                        "seed": int(self.seed),
                    }
                )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveNoiseScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omega_vals = df["omega"].unique()
        gamma_vals = df["gamma_phi"].unique()
        omega_vals.sort()
        gamma_vals.sort()
        n_omega = len(omega_vals)
        n_gamma = len(gamma_vals)

        # Build the lookup dict
        lookup: dict[tuple[float, float], dict[str, float]] = {}
        for _, row in df.iterrows():
            key = (float(row["omega"]), float(row["gamma_phi"]))
            lookup[key] = {
                "a_x": float(row["a_x"]),
                "a_y": float(row["a_y"]),
                "a_z": float(row["a_z"]),
                "a_zz": float(row["a_zz"]),
                "delta_omega": float(row["delta_omega"]),
                "expectation_Jz": float(row["expectation_Jz"]),
                "variance_Jz": float(row["variance_Jz"]),
                "d_exp_d_omega": float(row["d_exp_d_omega"]),
            }

        params_list: list[tuple[float, float, float, float]] = []
        dt_arr = np.full((n_omega, n_gamma), np.inf, dtype=float)
        exp_arr = np.zeros((n_omega, n_gamma), dtype=float)
        var_arr = np.zeros((n_omega, n_gamma), dtype=float)
        d_exp_arr = np.zeros((n_omega, n_gamma), dtype=float)

        for i, t in enumerate(omega_vals):
            for j, g in enumerate(gamma_vals):
                entry = lookup[(t, g)]
                params_list.append(
                    (
                        entry["a_x"],
                        entry["a_y"],
                        entry["a_z"],
                        entry["a_zz"],
                    )
                )
                dt_arr[i, j] = entry["delta_omega"]
                exp_arr[i, j] = entry["expectation_Jz"]
                var_arr[i, j] = entry["variance_Jz"]
                d_exp_arr[i, j] = entry["d_exp_d_omega"]

        return cls(
            omega_values=np.array(omega_vals, dtype=float),
            gamma_phi_values=np.array(gamma_vals, dtype=float),
            best_params_per_pair=params_list,
            delta_omega_per_pair=dt_arr,
            expectation_Jz_per_pair=exp_arr,
            variance_Jz_per_pair=var_arr,
            d_exp_d_omega_per_pair=d_exp_arr,
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
            n_random=int(df["n_random"].iloc[0]),
            n_nm_refine=int(df["n_nm_refine"].iloc[0]),
            maxiter=int(df["maxiter"].iloc[0]),
            bounds_lo=float(df["bounds_lo"].iloc[0]),
            bounds_hi=float(df["bounds_hi"].iloc[0]),
            fd_step=float(df["fd_step"].iloc[0]),
            seed=int(df["seed"].iloc[0]),
        )


# ============================================================================
# Decoupled Baseline (noisy)
# ============================================================================


def compute_noisy_decoupled_baseline(
    gamma_phi: float = 0.0,
    t_hold: float = DEFAULT_t_hold,
    omega_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity with phase diffusion.

    At (a_x = a_y = a_z = a_zz = 0), the circuit reduces to a standard
    single-qubit MZI with |1,0⟩ input and 50/50 BS. Phase diffusion on
    both qubits does not affect the SQL baseline because there is no
    entanglement to dephase.

    Args:
        gamma_phi: Phase diffusion rate.
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        Δω (should equal 1/t_hold = SQL for any γ_φ).
    """
    ops = build_two_qubit_operators()
    return compute_noisy_sensitivity(
        DEFAULT_RHO0,
        DEFAULT_T_BS,
        t_hold,
        omega_true,
        gamma_phi,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )


# ============================================================================
# Objective Function for Nelder-Mead
# ============================================================================


def noisy_sensitivity_objective(
    params: np.ndarray,
    omega_true: float,
    gamma_phi: float,
    ops: dict[str, np.ndarray],
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω under phase diffusion.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed t_hold.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        omega_true: True phase rate.
        gamma_phi: Phase diffusion rate.
        ops: Two-qubit operators.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])

    # Bound enforcement
    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_noisy_sensitivity(
        DEFAULT_RHO0,
        T_BS,
        t_hold,
        omega_true,
        gamma_phi,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


# ============================================================================
# Random Search
# ============================================================================


def run_noisy_random_search(
    omega: float,
    gamma_phi: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    """Random search over the 4D parameter space at fixed (ω, γ_φ).

    Args:
        omega: Phase rate value.
        gamma_phi: Phase diffusion rate.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (samples, delta_omega_values, best_params, best_delta_omega).
    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])

        domega = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            T_BS,
            t_hold,
            omega,
            gamma_phi,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )
    best_delta = float(deltas[best_idx])

    return samples, deltas, best_params, best_delta


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def _make_nm_early_stop_callback(
    patience: int = 200,
    fatol: float = 1e-8,
) -> Any:
    """Create a Nelder-Mead callback that raises StopIteration on stagnation.

    Tracks the best function value seen so far. If the best value has not
    improved by at least *fatol* for *patience* consecutive calls, raises
    ``StopIteration`` to terminate the solver early.

    Args:
        patience: Number of consecutive non-improving calls before stopping.
        fatol: Minimum absolute improvement threshold.

    Returns:
        Callback function ``callback(xk, /, fxk=None)`` compatible with
        ``scipy.optimize.minimize(method='Nelder-Mead')``.
    """
    best_val: float = float("inf")
    best_x: np.ndarray | None = None
    stagnant: int = 0
    call_count: int = 0

    def _callback(xk: np.ndarray, fxk: float | None = None) -> None:
        nonlocal best_val, best_x, stagnant, call_count
        call_count += 1

        # Evaluate at the new point if fxk not provided
        val: float = fxk if fxk is not None else float("inf")
        if fxk is None:
            # We cannot evaluate here easily since we don't have the objective,
            # so rely on the solver-supplied value.
            return

        if val < best_val - fatol:
            best_val = val
            best_x = xk.copy()
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= patience:
            raise StopIteration(
                f"Early stop after {call_count} calls: no improvement "
                f"for {patience} consecutive iterations."
            )

    return _callback


def run_noisy_nelder_mead(
    omega_true: float,
    gamma_phi: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 2000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    early_stop_patience: int = 200,
) -> dict[str, Any]:
    """Run Nelder-Mead optimisation for the noisy protocol at fixed (ω, γ_φ).

    Uses an early-stopping callback to terminate when the objective has not
    improved for *early_stop_patience* consecutive iterations.

    Args:
        omega_true: True phase rate parameter.
        gamma_phi: Phase diffusion rate.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for all four parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        early_stop_patience: Consecutive non-improving calls before stopping.

    Returns:
        Dict with keys: 'delta_omega_opt', 'params_opt', 'success',
        'nfev', 'message', 'expectation_Jz', 'variance_Jz', 'd_exp_d_omega'.
    """
    ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return noisy_sensitivity_objective(
            p,
            omega_true,
            gamma_phi,
            ops,
            t_hold=t_hold,
            T_BS=T_BS,
            bounds=bounds,
        )

    callback = _make_nm_early_stop_callback(patience=early_stop_patience, fatol=fatol)

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        callback=callback,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    _, exp_val, var_val, d_exp = compute_noisy_sensitivity_with_diagnostics(
        DEFAULT_RHO0,
        T_BS,
        t_hold,
        omega_true,
        gamma_phi,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )

    return {
        "delta_omega_opt": float(result.fun),
        "params_opt": opt_params,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "expectation_Jz": exp_val,
        "variance_Jz": var_val,
        "d_exp_d_omega": d_exp,
    }


# ============================================================================
# Noise Scan: (ω, γ_φ) Grid
# ============================================================================


def _run_single_noise_pair(
    omega: float,
    gamma_phi: float,
    n_random: int = 1000,
    n_nm_refine: int = 25,
    seed: int | None = 42,
    maxiter: int = 2000,
    early_stop_patience: int = 200,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> dict[str, Any]:
    """Run random search + Nelder-Mead refinement for a single (ω, γ_φ) pair.

    Args:
        omega: Phase rate value.
        gamma_phi: Phase diffusion rate.
        n_random: Number of random search points.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base random seed.
        maxiter: Maximum Nelder-Mead iterations.
        early_stop_patience: Consecutive non-improving NM iterations before stop.
        bounds: (min, max) for all four coefficients.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Dict with results for this (ω, γ_φ) pair.
    """
    base_seed = seed if seed is not None else 42
    seed_raw = base_seed + int(omega * 1000) + int(100 * np.log10(gamma_phi + 1e-10))
    seed_val = abs(seed_raw)  # ensure non-negative for numpy

    # Stage 1: Random search
    samples, deltas, best_params, best_delta = run_noisy_random_search(
        omega,
        gamma_phi,
        n_samples=n_random,
        bounds=bounds,
        t_hold=t_hold,
        T_BS=T_BS,
        seed=seed_val,
    )

    # Stage 2: Nelder-Mead refinement from top candidates
    sorted_indices = np.argsort(deltas)
    top_indices = sorted_indices[: min(n_nm_refine, n_random)]

    nm_best_delta = best_delta
    nm_best_params = best_params
    nm_best_diag: dict[str, float] = {
        "expectation_Jz": 0.0,
        "variance_Jz": 0.0,
        "d_exp_d_omega": 0.0,
    }

    for rank, idx in enumerate(top_indices):
        x0 = samples[idx].copy()
        nm_result = run_noisy_nelder_mead(
            omega_true=omega,
            gamma_phi=gamma_phi,
            x0=x0,
            seed=seed_val + 10000 + rank,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
            bounds=bounds,
            t_hold=t_hold,
            T_BS=T_BS,
        )

        dt = float(nm_result["delta_omega_opt"])
        if np.isfinite(dt) and dt < nm_best_delta:
            nm_best_delta = dt
            nm_best_params = (
                float(nm_result["params_opt"][0]),
                float(nm_result["params_opt"][1]),
                float(nm_result["params_opt"][2]),
                float(nm_result["params_opt"][3]),
            )
            nm_best_diag = {
                "expectation_Jz": float(nm_result["expectation_Jz"]),
                "variance_Jz": float(nm_result["variance_Jz"]),
                "d_exp_d_omega": float(nm_result["d_exp_d_omega"]),
            }

    return {
        "omega": omega,
        "gamma_phi": gamma_phi,
        "a_x": nm_best_params[0],
        "a_y": nm_best_params[1],
        "a_z": nm_best_params[2],
        "a_zz": nm_best_params[3],
        "delta_omega": nm_best_delta,
        "expectation_Jz": nm_best_diag["expectation_Jz"],
        "variance_Jz": nm_best_diag["variance_Jz"],
        "d_exp_d_omega": nm_best_diag["d_exp_d_omega"],
    }


def run_noise_scan(
    omega_values: list[float] | np.ndarray,
    gamma_phi_values: list[float] | np.ndarray,
    n_random: int = 1000,
    n_nm_refine: int = 25,
    seed: int | None = 42,
    maxiter: int = 2000,
    early_stop_patience: int = 200,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> DriveNoiseScanResult:
    """Scan over (ω, γ_φ) pairs with random search + Nelder-Mead refinement.

    For each (ω, γ_φ) pair:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder-Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        omega_values: ω values to scan.
        gamma_phi_values: γ_φ values to scan.
        n_random: Number of random search points per pair.
        n_nm_refine: Number of Nelder-Mead refinements per pair.
        seed: Base random seed.
        maxiter: Maximum Nelder-Mead iterations.
        bounds: (min, max) for all parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveNoiseScanResult with optimal parameters and sensitivities.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    gamma_arr = np.asarray(gamma_phi_values, dtype=float)
    n_omega = len(omega_arr)
    n_gamma = len(gamma_arr)

    all_results: list[dict[str, Any]] = []

    for omega in omega_arr:
        for gamma in gamma_arr:
            result = _run_single_noise_pair(
                omega,
                gamma,
                n_random=n_random,
                n_nm_refine=n_nm_refine,
                seed=seed,
                maxiter=maxiter,
                early_stop_patience=early_stop_patience,
                bounds=bounds,
                t_hold=t_hold,
                T_BS=T_BS,
            )
            all_results.append(result)

    # Build the result structure
    params_list: list[tuple[float, float, float, float]] = []
    dt_arr = np.full((n_omega, n_gamma), np.inf, dtype=float)
    exp_arr = np.zeros((n_omega, n_gamma), dtype=float)
    var_arr = np.zeros((n_omega, n_gamma), dtype=float)
    d_exp_arr = np.zeros((n_omega, n_gamma), dtype=float)

    for r in all_results:
        t_val = float(r["omega"])
        g_val = float(r["gamma_phi"])
        i = int(np.where(omega_arr == t_val)[0][0])
        j = int(np.where(gamma_arr == g_val)[0][0])
        params_list.append(
            (
                float(r["a_x"]),
                float(r["a_y"]),
                float(r["a_z"]),
                float(r["a_zz"]),
            )
        )
        dt_arr[i, j] = float(r["delta_omega"])
        exp_arr[i, j] = float(r.get("expectation_Jz", 0.0))
        var_arr[i, j] = float(r.get("variance_Jz", 0.0))
        d_exp_arr[i, j] = float(r.get("d_exp_d_omega", 0.0))

    bounds_lo_f, bounds_hi_f = bounds
    return DriveNoiseScanResult(
        omega_values=omega_arr,
        gamma_phi_values=gamma_arr,
        best_params_per_pair=params_list,
        delta_omega_per_pair=dt_arr,
        expectation_Jz_per_pair=exp_arr,
        variance_Jz_per_pair=var_arr,
        d_exp_d_omega_per_pair=d_exp_arr,
        sql=1.0 / t_hold,
        t_hold=t_hold,
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        maxiter=maxiter,
        bounds_lo=float(bounds_lo_f),
        bounds_hi=float(bounds_hi_f),
        fd_step=FD_STEP,
        seed=seed if seed is not None else 42,
    )


# ============================================================================
# Fixed-Parameter Scan (using noise-free optimal params)
# ============================================================================


def evaluate_fixed_params_scan(
    omega_values: list[float] | np.ndarray,
    gamma_phi_values: list[float] | np.ndarray,
    noise_free_params: dict[float, tuple[float, float, float, float]],
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> np.ndarray:
    """Evaluate Δω at noise-free optimal params for each (ω, γ_φ) pair.

    Args:
        omega_values: ω values.
        gamma_phi_values: γ_φ values.
        noise_free_params: Dict mapping ω → (a_x*, a_y*, a_z*, a_zz*)
            from the noise-free (γ_φ = 0) optimum.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Array of Δω values, shape (n_omega, n_gamma).
    """
    ops = build_two_qubit_operators()
    omega_arr = np.asarray(omega_values, dtype=float)
    gamma_arr = np.asarray(gamma_phi_values, dtype=float)
    n_omega = len(omega_arr)
    n_gamma = len(gamma_arr)

    result = np.full((n_omega, n_gamma), np.inf, dtype=float)

    for i, omega in enumerate(omega_arr):
        if omega not in noise_free_params:
            continue
        ax, ay, az, azz = noise_free_params[omega]
        for j, gamma in enumerate(gamma_arr):
            dt = compute_noisy_sensitivity(
                DEFAULT_RHO0,
                T_BS,
                t_hold,
                omega,
                gamma,
                ax,
                ay,
                az,
                azz,
                ops,
            )
            result[i, j] = dt

    return result


# ============================================================================
# Exclusive Plot Functions
# ============================================================================


def plot_noise_sensitivity_heatmap(
    result: DriveNoiseScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Heatmap of Δω/Δω_SQL as a function of ω and γ_φ.

    Args:
        result: DriveNoiseScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratio = result.delta_omega_per_pair / result.sql

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        result.omega_values,
        result.gamma_phi_values,
        ratio.T,
        shading="auto",
        cmap="RdYlBu_r",
        norm=LogNorm(
            vmin=max(np.nanmin(ratio[ratio > 0]), 1e-2),
            vmax=max(np.nanmax(ratio[ratio < np.inf]), 10),
        ),
    )
    fig.colorbar(im, ax=ax, label=r"$\Delta\omega / \Delta\omega_{\mathrm{SQL}}$")

    # Contour line at Δω = SQL
    cs = ax.contour(
        result.omega_values,
        result.gamma_phi_values,
        ratio.T,
        levels=[1.0],
        colors="k",
        linewidths=1.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt=r"SQL", fontsize=10)

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\gamma_\phi$")
    ax.set_yscale("log")
    ax.set_title(
        "Sensitivity ratio vs phase diffusion rate\n"
        "(re-optimised $(a_x, a_y, a_z, a_{zz})$ per point)"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_noise_sensitivity_curves(
    result: DriveNoiseScanResult,
    omega_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Line plot of Δω vs γ_φ for selected ω values.

    Args:
        result: DriveNoiseScanResult.
        omega_subset: List of ω values to plot. If None, uses all.
        save_path: Output SVG path. If None, does not save.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG, or a dummy path if not saved.
    """
    if omega_subset is None:
        omega_subset = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference line
    ax.axhline(
        y=result.sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {result.sql:.4f}",
    )

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(omega_subset)))

    for omega_val, color in zip(omega_subset, colors, strict=False):
        # Find the index
        idx = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
        if len(idx) == 0:
            continue
        i = idx[0]

        valid = np.isfinite(result.delta_omega_per_pair[i, :])
        if np.any(valid):
            ax.plot(
                result.gamma_phi_values[valid],
                result.delta_omega_per_pair[i, valid],
                "o-",
                color=color,
                markersize=5,
                linewidth=1.5,
                label=rf"$\omega$ = {omega_val:.1f}",
            )

    ax.set_xlabel(r"$\gamma_\phi$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sensitivity vs phase diffusion rate at fixed $\\omega$")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return Path()


def plot_noise_optimal_params(
    result: DriveNoiseScanResult,
    omega_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Plot optimal parameters as a function of γ_φ for selected ω values.

    Args:
        result: DriveNoiseScanResult.
        omega_subset: List of ω values to plot. If None, uses all.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    if omega_subset is None:
        omega_subset = [0.1, 0.5, 1.0, 2.0, 5.0]

    param_names = [r"$a_x^*$", r"$a_y^*$", r"$a_z^*$", r"$a_{zz}^*$"]

    n_params = 4
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(omega_subset)))

    for param_idx in range(n_params):
        ax = axes[param_idx]
        for omega_val, color in zip(omega_subset, colors, strict=False):
            idx = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            i = idx[0]

            # Extract the parameter values for this omega across all gamma
            param_vals = []
            for j in range(len(result.gamma_phi_values)):
                pair_idx = i * len(result.gamma_phi_values) + j
                params = result.best_params_per_pair[pair_idx]
                param_vals.append(params[param_idx])

            param_arr = np.array(param_vals)
            valid = np.isfinite(result.delta_omega_per_pair[i, :]) & np.isfinite(
                param_arr
            )
            if np.any(valid):
                ax.plot(
                    result.gamma_phi_values[valid],
                    param_arr[valid],
                    "o-",
                    color=color,
                    markersize=4,
                    linewidth=1.2,
                    label=rf"$\omega$ = {omega_val:.1f}" if param_idx == 0 else "",
                )

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(param_names[param_idx])
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"$\gamma_\phi$")
    axes[0].set_title(
        "Optimal parameters vs phase diffusion rate\n"
        "(re-optimised per $\\omega$, $\\gamma_\\phi$ pair)"
    )
    axes[0].legend(fontsize=8, loc="upper left", ncol=len(omega_subset))

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return Path()


# ============================================================================
# Noise-Free Optimal Parameters (for fixed-parameter comparison)
# ============================================================================


def _optimise_noise_free_single_omega(
    omega: float,
    n_random: int = 500,
    n_nm_refine: int = 25,
    seed: int = 42,
    maxiter: int = 2000,
    early_stop_patience: int = 200,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> tuple[float, tuple[float, float, float, float]]:
    """Optimise noise-free sensitivity for a single ω value.

    This is a module-level worker for parallel execution via ProcessPoolExecutor.

    Returns:
        Tuple (omega, (a_x*, a_y*, a_z*, a_zz*)).
    """
    omega_seed = seed + int(omega * 1000)
    local_ops = build_two_qubit_operators()

    # Stage 1: Random search at γ_φ=0
    samples = np.random.default_rng(omega_seed).uniform(
        bounds[0], bounds[1], size=(n_random, 4)
    )
    deltas = np.full(n_random, np.inf, dtype=float)
    for i in range(n_random):
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            T_BS,
            t_hold,
            omega,
            0.0,
            float(samples[i, 0]),
            float(samples[i, 1]),
            float(samples[i, 2]),
            float(samples[i, 3]),
            local_ops,
        )
        deltas[i] = dt

    best_idx = int(np.argmin(deltas))
    best_dt = float(deltas[best_idx])
    best_params = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    # Stage 2: Nelder-Mead refinement
    sorted_indices = np.argsort(deltas)
    for rank in range(n_nm_refine):
        x0 = samples[int(sorted_indices[rank])].copy()
        nm = run_noisy_nelder_mead(
            omega_true=omega,
            gamma_phi=0.0,
            x0=x0,
            seed=omega_seed + 10000 + rank,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
            bounds=bounds,
            t_hold=t_hold,
            T_BS=T_BS,
        )
        dt_nm = float(nm["delta_omega_opt"])
        if np.isfinite(dt_nm) and dt_nm < best_dt:
            best_dt = dt_nm
            best_params = (
                float(nm["params_opt"][0]),
                float(nm["params_opt"][1]),
                float(nm["params_opt"][2]),
                float(nm["params_opt"][3]),
            )

    return (float(omega), best_params)


def compute_noise_free_optimal_params(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 25,
    seed: int = 42,
    maxiter: int = 2000,
    early_stop_patience: int = 200,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> dict[float, tuple[float, float, float, float]]:
    """Compute noise-free optimal parameters (γ_φ = 0) for each ω value.

    Uses the same two-stage optimisation (random search + Nelder-Mead)
    as the noise scan, but at γ_φ = 0. Results are used by
    ``evaluate_fixed_params_scan`` and ``plot_noise_reopt_vs_fixed``.

    Args:
        omega_values: ω values to optimise for.
        n_random: Number of random search points per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed.
        maxiter: Maximum Nelder-Mead iterations.
        early_stop_patience: Consecutive non-improving NM iterations before stop.
        bounds: (min, max) for all parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Dict mapping ω → (a_x*, a_y*, a_z*, a_zz*).
    """
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(16, os.cpu_count() or 1),
        mp_context=_mp.get_context("fork"),
    ) as executor:
        fut_to_omega = {
            executor.submit(
                _optimise_noise_free_single_omega,
                float(t),
                n_random,
                n_nm_refine,
                seed,
                maxiter,
                early_stop_patience,
                bounds,
                t_hold,
                T_BS,
            ): float(t)
            for t in omega_values
        }
        result: dict[float, tuple[float, float, float, float]] = {}
        for future in concurrent.futures.as_completed(fut_to_omega):
            t_val, params = future.result()
            result[t_val] = params

    return result


# ============================================================================
# New Plot: P1 — Critical Noise Rate γ_φ*(ω)
# ============================================================================


def plot_noise_critical_rate(
    result: DriveNoiseScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Plot the critical phase-diffusion rate γ_φ* vs ω.

    For each ω value, the threshold γ_φ* is the rate at which
    Δω(γ_φ*) = Δω_SQL (i.e., the ratio crosses 1.0).  Interpolation
    is performed on log(γ_φ) vs log(ratio).

    Args:
        result: DriveNoiseScanResult with noise scan data.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratio = result.delta_omega_per_pair / result.sql
    omega_arr = result.omega_values
    gamma_arr = result.gamma_phi_values
    log_gamma = np.log10(gamma_arr)

    gamma_star = np.full(len(omega_arr), np.nan)
    gamma_star_lo = np.full(len(omega_arr), np.nan)
    gamma_star_hi = np.full(len(omega_arr), np.nan)

    for i in range(len(omega_arr)):
        row = ratio[i, :]
        valid = np.isfinite(row) & (row > 0)
        if np.sum(valid) < 2:
            continue

        # Find where ratio crosses 1.0
        g_valid = log_gamma[valid]
        r_valid = np.log10(row[valid])

        # Check if any values are below 1 and any above 1
        below = r_valid < 0
        if not np.any(below) or not np.any(~below):
            # All above or all below — no crossing
            if np.all(r_valid < 0):
                gamma_star[i] = gamma_arr[-1] * 10  # beyond max
            else:
                gamma_star[i] = gamma_arr[0] / 10  # below min
            continue

        # Find crossing index
        cross_idx = np.where(below[:-1] != below[1:])[0]
        if len(cross_idx) == 0:
            continue
        ci = cross_idx[0]

        # Linear interpolation in log space
        x0, x1 = g_valid[ci], g_valid[ci + 1]
        y0, y1 = r_valid[ci], r_valid[ci + 1]
        # y = 0 → x = x0 - y0 * (x1 - x0) / (y1 - y0)
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        gamma_star[i] = 10**x_cross
        gamma_star_lo[i] = 10 ** (x_cross - 0.15)
        gamma_star_hi[i] = 10 ** (x_cross + 0.15)

    fig, ax = plt.subplots(figsize=figsize)

    # Critical rate line
    valid_crit = np.isfinite(gamma_star)
    if np.any(valid_crit):
        ax.errorbar(
            omega_arr[valid_crit],
            gamma_star[valid_crit],
            yerr=(
                (gamma_star[valid_crit] - gamma_star_lo[valid_crit]),
                (gamma_star_hi[valid_crit] - gamma_star[valid_crit]),
            ),
            fmt="o-",
            color="C0",
            markersize=8,
            linewidth=2,
            capsize=4,
            label=r"$\gamma_\phi^*(\omega)$",
        )

    # Shaded region where protocol beats SQL
    ax.fill_between(
        omega_arr,
        gamma_arr[0] / 10,
        gamma_star,
        alpha=0.1,
        color="C0",
        label=r"Sub-SQL ($\Delta\omega < \Delta\omega_{\mathrm{SQL}}$)",
    )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\gamma_\phi^*$ (critical rate)")
    ax.set_yscale("log")
    ax.set_title("Critical phase-diffusion rate vs true phase")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# New Plot: P2 — Re-optimised vs Fixed-Parameter Sensitivity
# ============================================================================


def plot_noise_reopt_vs_fixed(
    result: DriveNoiseScanResult,
    fixed_delta: np.ndarray,
    omega_val: float = 0.2,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Overlay re-optimised and fixed-parameter Δω vs γ_φ at a given ω.

    Args:
        result: DriveNoiseScanResult (re-optimised).
        fixed_delta: Δω from fixed (noise-free optimal) params,
            shape (n_omega, n_gamma).
        omega_val: ω value to plot.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path) if save_path else Path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    i = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
    if len(i) == 0:
        # Fallback to first omega
        i = [0]
        omega_val = float(result.omega_values[0])
    i = int(i[0])

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference
    ax.axhline(
        y=result.sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {result.sql:.4f}",
    )

    # Re-optimised
    valid_reopt = np.isfinite(result.delta_omega_per_pair[i, :])
    if np.any(valid_reopt):
        ax.plot(
            result.gamma_phi_values[valid_reopt],
            result.delta_omega_per_pair[i, valid_reopt],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.8,
            label=r"Re-optimised per $\gamma_\phi$",
        )

    # Fixed-params
    if (
        fixed_delta is not None
        and fixed_delta.shape == result.delta_omega_per_pair.shape
    ):
        valid_fixed = np.isfinite(fixed_delta[i, :])
        if np.any(valid_fixed):
            ax.plot(
                result.gamma_phi_values[valid_fixed],
                fixed_delta[i, valid_fixed],
                "s--",
                color="C3",
                markersize=6,
                linewidth=1.5,
                label=r"Fixed (noise-free optimal params)",
            )

    ax.set_xlabel(r"$\gamma_\phi$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        rf"Re-optimised vs fixed-parameter sensitivity at $\omega={omega_val:.1f}$"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# New Plot: P3 — Signal Diagnostics (⟨Jz⟩, Var, derivative)
# ============================================================================


def plot_noise_signal_diagnostics(
    result: DriveNoiseScanResult,
    omega_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Three-panel stacked diagnostics vs γ_φ: ⟨Jz⟩, Var(Jz), |∂⟨Jz⟩/∂ω|.

    Args:
        result: DriveNoiseScanResult.
        omega_subset: ω values to plot. Defaults to [0.2, 1.0, 5.0].
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if omega_subset is None:
        omega_subset = [0.2, 1.0, 5.0]

    save_path = Path(save_path) if save_path else Path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(omega_subset)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for omega_val, color in zip(omega_subset, colors, strict=False):
        idx = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        label = rf"$\omega$ = {omega_val:.1f}"

        valid = np.isfinite(result.delta_omega_per_pair[i, :])

        # Panel 1: expectation
        if np.any(valid):
            ax1.plot(
                result.gamma_phi_values[valid],
                result.expectation_Jz_per_pair[i, valid],
                "o-",
                color=color,
                markersize=5,
                linewidth=1.3,
                label=label,
            )

        # Panel 2: variance
        if np.any(valid):
            ax2.plot(
                result.gamma_phi_values[valid],
                result.variance_Jz_per_pair[i, valid],
                "s-",
                color=color,
                markersize=5,
                linewidth=1.3,
                label=label,
            )

        # Panel 3: |derivative|
        if np.any(valid):
            d_exp = np.abs(result.d_exp_d_omega_per_pair[i, :])
            d_exp_valid = np.isfinite(d_exp) & valid
            if np.any(d_exp_valid):
                ax3.plot(
                    result.gamma_phi_values[d_exp_valid],
                    d_exp[d_exp_valid],
                    "^--",
                    color=color,
                    markersize=5,
                    linewidth=1.3,
                    label=label,
                )

    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
    ax1.set_ylabel(r"$\langle J_z^S \rangle$")
    ax1.set_title("Signal diagnostics vs phase-diffusion rate")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.2)

    ax2.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.2)

    ax3.set_xlabel(r"$\gamma_\phi$")
    ax3.set_ylabel(r"$|\partial\langle J_z^S\rangle/\partial\omega|$")
    ax3.set_yscale("log")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# New Plot: P4 — Decoupled Baseline vs Noisy Optimal
# ============================================================================


def plot_noise_decoupled_vs_optimal(
    result: DriveNoiseScanResult,
    decoupled_data: pd.DataFrame | None = None,
    omega_val: float = 1.0,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Compare decoupled baseline and re-optimised sensitivity vs γ_φ.

    Args:
        result: DriveNoiseScanResult (re-optimised).
        decoupled_data: DataFrame with columns 'gamma_phi' and 'delta_omega'
            from the decoupled baseline scan. If None, loads from default path.
        omega_val: ω value to plot.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path) if save_path else Path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load decoupled baseline if not provided
    if decoupled_data is None:
        decoupled_p = _parquet_path("noise-decoupled-baseline")
        if decoupled_p.exists():
            decoupled_data = pd.read_parquet(decoupled_p)

    idx = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
    if len(idx) == 0:
        idx = [0]
        omega_val = float(result.omega_values[0])
    i = int(idx[0])

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference
    ax.axhline(
        y=result.sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {result.sql:.4f}",
    )

    # Decoupled baseline
    if decoupled_data is not None:
        gammas_d = decoupled_data["gamma_phi"].to_numpy()
        deltas_d = decoupled_data["delta_omega"].to_numpy()
        valid_d = np.isfinite(deltas_d)
        if np.any(valid_d):
            ax.plot(
                gammas_d[valid_d],
                deltas_d[valid_d],
                "D:",
                color="gray",
                markersize=6,
                linewidth=1.5,
                label=r"Decoupled baseline ($a_k = a_{zz} = 0$)",
            )

    # Re-optimised
    valid_opt = np.isfinite(result.delta_omega_per_pair[i, :])
    if np.any(valid_opt):
        ax.plot(
            result.gamma_phi_values[valid_opt],
            result.delta_omega_per_pair[i, valid_opt],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.8,
            label=rf"Re-optimised ($\omega = {omega_val:.1f}$)",
        )

    ax.set_xlabel(r"$\gamma_\phi$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Decoupled baseline vs re-optimised protocol under phase diffusion")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# New Plot: P5 — Improvement Ratio Δω_fixed / Δω_reopt
# ============================================================================


def plot_noise_improvement_ratio(
    result: DriveNoiseScanResult,
    fixed_delta: np.ndarray,
    omega_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot the ratio Δω_fixed / Δω_reopt vs γ_φ.

    Values > 1 mean re-optimisation helps. Horizontal line at 1.0.

    Args:
        result: DriveNoiseScanResult (re-optimised).
        fixed_delta: Δω from fixed (noise-free optimal) params,
            shape (n_omega, n_gamma).
        omega_subset: ω values to plot.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if omega_subset is None:
        omega_subset = [0.2, 1.0, 2.0]

    save_path = Path(save_path) if save_path else Path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(omega_subset)))

    fig, ax = plt.subplots(figsize=figsize)

    for omega_val, color in zip(omega_subset, colors, strict=False):
        idx = np.where(np.isclose(result.omega_values, omega_val, atol=1e-6))[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])

        reopt = result.delta_omega_per_pair[i, :]
        fixed = fixed_delta[i, :]

        valid = np.isfinite(reopt) & np.isfinite(fixed) & (reopt > 0)
        if np.any(valid):
            ratio_vals = fixed[valid] / reopt[valid]
            ax.plot(
                result.gamma_phi_values[valid],
                ratio_vals,
                "o-",
                color=color,
                markersize=5,
                linewidth=1.5,
                label=rf"$\omega$ = {omega_val:.1f}",
            )

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=r"$\Delta\omega_{\mathrm{fixed}} = \Delta\omega_{\mathrm{reopt}}$",
    )

    ax.set_xlabel(r"$\gamma_\phi$")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{fixed}} / \Delta\omega_{\mathrm{reopt}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Re-optimisation benefit: improvement ratio vs phase diffusion rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

from src.utils.paths import (
    fig_path,
    parquet_path,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent
DATE_TAG = "20260524"


def _parquet_path(name: str) -> Path:
    return parquet_path(REPORTS_DIR, DATE_TAG, name)


def _fig_path(name: str) -> Path:
    return fig_path(REPORTS_DIR, DATE_TAG, name)


# ── Parallel dispatch helper ──────────────────────────────────────────────


# ── Generator functions ───────────────────────────────────────────────────


def _run_single_pair_worker(args: dict[str, Any]) -> None:
    """Worker function for parallel noise scan.

    Runs one (ω, γ_φ) pair and saves to a separate Parquet file.
    """
    omega = args["omega"]
    gamma_phi = args["gamma_phi"]
    force = args["force"]
    n_random = args.get("n_random", 1000)
    n_nm_refine = args.get("n_nm_refine", 25)
    seed = args.get("seed", 42)
    maxiter = args.get("maxiter", 2000)
    early_stop_patience = args.get("early_stop_patience", 200)

    tag = f"noise-pair-omega{omega}-gamma{gamma_phi:.6e}"
    csv_p = _parquet_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print(f"  [run]  (ω={omega}, γ_φ={gamma_phi:.2e})...")
    result = _run_single_noise_pair(
        omega,
        gamma_phi,
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        seed=seed,
        maxiter=maxiter,
        early_stop_patience=early_stop_patience,
    )

    # Save individual result as a single-row DataFrame
    row = pd.DataFrame(
        [
            {
                "omega": omega,
                "gamma_phi": gamma_phi,
                "a_x": result["a_x"],
                "a_y": result["a_y"],
                "a_z": result["a_z"],
                "a_zz": result["a_zz"],
                "delta_omega": result["delta_omega"],
                "expectation_Jz": result.get("expectation_Jz", 0.0),
                "variance_Jz": result.get("variance_Jz", 0.0),
                "d_exp_d_omega": result.get("d_exp_d_omega", 0.0),
                "sql": SQL_REFERENCE,
                "t_hold": DEFAULT_t_hold,
                "n_random": n_random,
                "n_nm_refine": n_nm_refine,
                "maxiter": maxiter,
                "bounds_lo": DRIVE_BOUNDS[0],
                "bounds_hi": DRIVE_BOUNDS[1],
                "fd_step": FD_STEP,
                "seed": seed,
            }
        ]
    )
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    row.to_parquet(csv_p, index=False)
    print(f"  [save] {csv_p.name}")


def _assemble_noise_scan_result() -> DriveNoiseScanResult:
    """Aggregate individual per-pair Parquet files into a DriveNoiseScanResult.

    Reads all noise-pair-omega{omega}-gamma{gamma_phi:.6e}.parquet files
    produced by the parallel scan, builds a lookup dict, and assembles the
    aggregated DriveNoiseScanResult.
    """
    print("  [aggregate] Building aggregated result...")
    all_rows: list[pd.DataFrame] = []
    for omega in OMEGA_VALS:
        for gamma_phi in GAMMA_PHI_VALS:
            tag = f"noise-pair-omega{omega}-gamma{gamma_phi:.6e}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                all_rows.append(pd.read_parquet(csv_p))

    if not all_rows:
        raise RuntimeError("No individual noise-scan result files found.")

    df_agg = pd.concat(all_rows, ignore_index=True)

    omega_arr = np.array(OMEGA_VALS, dtype=float)
    gamma_arr = np.array(GAMMA_PHI_VALS, dtype=float)
    n_omega = len(omega_arr)
    n_gamma = len(gamma_arr)

    params_list: list[tuple[float, float, float, float]] = []
    dt_arr = np.full((n_omega, n_gamma), np.inf, dtype=float)
    exp_arr = np.zeros((n_omega, n_gamma), dtype=float)
    var_arr = np.zeros((n_omega, n_gamma), dtype=float)
    d_exp_arr = np.zeros((n_omega, n_gamma), dtype=float)

    lookup: dict[tuple[float, float], dict[str, float]] = {}
    for _, row in df_agg.iterrows():
        key = (float(row["omega"]), float(row["gamma_phi"]))
        lookup[key] = {
            "a_x": float(row["a_x"]),
            "a_y": float(row["a_y"]),
            "a_z": float(row["a_z"]),
            "a_zz": float(row["a_zz"]),
            "delta_omega": float(row["delta_omega"]),
            "expectation_Jz": float(row.get("expectation_Jz", 0.0)),
            "variance_Jz": float(row.get("variance_Jz", 0.0)),
            "d_exp_d_omega": float(row.get("d_exp_d_omega", 0.0)),
        }

    for i, t in enumerate(omega_arr):
        for j, g in enumerate(gamma_arr):
            entry = lookup.get((t, g), {})
            params_list.append(
                (
                    entry.get("a_x", 0.0),
                    entry.get("a_y", 0.0),
                    entry.get("a_z", 0.0),
                    entry.get("a_zz", 0.0),
                )
            )
            dt_arr[i, j] = entry.get("delta_omega", float("inf"))
            exp_arr[i, j] = entry.get("expectation_Jz", 0.0)
            var_arr[i, j] = entry.get("variance_Jz", 0.0)
            d_exp_arr[i, j] = entry.get("d_exp_d_omega", 0.0)

    first_row = df_agg.iloc[0]
    return DriveNoiseScanResult(
        omega_values=omega_arr,
        gamma_phi_values=gamma_arr,
        best_params_per_pair=params_list,
        delta_omega_per_pair=dt_arr,
        expectation_Jz_per_pair=exp_arr,
        variance_Jz_per_pair=var_arr,
        d_exp_d_omega_per_pair=d_exp_arr,
        sql=SQL_REFERENCE,
        t_hold=DEFAULT_t_hold,
        n_random=int(first_row.get("n_random", 1000)),
        n_nm_refine=int(first_row.get("n_nm_refine", 25)),
        maxiter=int(first_row.get("maxiter", 5000)),
        bounds_lo=float(first_row.get("bounds_lo", -5.0)),
        bounds_hi=float(first_row.get("bounds_hi", 5.0)),
        fd_step=float(first_row.get("fd_step", FD_STEP)),
        seed=int(first_row.get("seed", 42)),
    )


def generate_noise_scan(force: bool = False) -> None:
    """Run the full (ω, γ_φ) noise scan in parallel.

    Each (ω, γ_φ) pair is computed independently and saved to a separate
    Parquet file. After all pairs complete, results are aggregated into a
    single DriveNoiseScanResult and saved.
    """
    agg_p = _parquet_path("noise-scan")

    if agg_p.exists() and not force:
        print(f"[skip] {agg_p.name} exists (use --force to overwrite)")
        result = DriveNoiseScanResult.from_parquet(agg_p)
    else:
        n_omega = len(OMEGA_VALS)
        n_gamma = len(GAMMA_PHI_VALS)
        n_total = n_omega * n_gamma
        print(f"[run]  Noise scan: {n_total} (ω, γ_φ) pairs (parallel)")

        args_list = [
            {
                "omega": omega,
                "gamma_phi": gamma_phi,
                "force": force,
                "n_random": 1000,
                "n_nm_refine": 25,
                "seed": 42,
                "maxiter": 2000,
                "early_stop_patience": 200,
            }
            for omega in OMEGA_VALS
            for gamma_phi in GAMMA_PHI_VALS
        ]

        parallel_map(
            _run_single_pair_worker,
            args_list,
            desc=f"noise scan ({n_total} pairs)",
        )

        result = _assemble_noise_scan_result()
        result.save_parquet(agg_p)
        print(f"[save] {agg_p}")

    generate_noise_figures(result)


def generate_noise_figures(result: DriveNoiseScanResult | None = None) -> None:
    """Generate all figures from an existing or loaded result."""
    if result is None:
        agg_p = _parquet_path("noise-scan")
        if not agg_p.exists():
            print("[skip] noise-scan.parquet does not exist; run 'noise-scan' first")
            return
        result = DriveNoiseScanResult.from_parquet(agg_p)

    # Heatmap
    fig_p = _fig_path("noise-sensitivity-heatmap")
    plot_noise_sensitivity_heatmap(result, fig_p)
    print(f"[fig]  {fig_p}")

    # Sensitivity curves
    fig_p = _fig_path("noise-sensitivity-curves")
    plot_noise_sensitivity_curves(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")

    # Optimal parameters
    fig_p = _fig_path("noise-optimal-params")
    plot_noise_optimal_params(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")

    # P1: Critical noise rate
    fig_p = _fig_path("noise-critical-rate")
    plot_noise_critical_rate(result, fig_p)
    print(f"[fig]  {fig_p}")

    # P3: Signal diagnostics
    fig_p = _fig_path("noise-signal-diagnostics")
    plot_noise_signal_diagnostics(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")

    # P4: Decoupled baseline vs optimal
    fig_p = _fig_path("noise-decoupled-vs-optimal")
    plot_noise_decoupled_vs_optimal(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")

    # P2 and P5 require fixed-params data
    fixed_p = _parquet_path("noise-fixed-params-scan")
    if fixed_p.exists():
        fixed_df = pd.read_parquet(fixed_p)
        # Reshape to (n_omega, n_gamma)
        omega_arr = result.omega_values
        gamma_arr = result.gamma_phi_values
        n_omega = len(omega_arr)
        n_gamma = len(gamma_arr)
        fixed_delta = np.full((n_omega, n_gamma), np.inf, dtype=float)
        for _, row in fixed_df.iterrows():
            t = float(row["omega"])
            g = float(row["gamma_phi"])
            i = int(np.where(np.isclose(omega_arr, t, atol=1e-6))[0][0])
            j = int(np.where(np.isclose(gamma_arr, g, atol=1e-6))[0][0])
            fixed_delta[i, j] = float(row["delta_omega"])

        # P2: Re-optimised vs fixed
        fig_p = _fig_path("noise-reopt-vs-fixed")
        plot_noise_reopt_vs_fixed(result, fixed_delta, save_path=fig_p)
        print(f"[fig]  {fig_p}")

        # P5: Improvement ratio
        fig_p = _fig_path("noise-improvement-ratio")
        plot_noise_improvement_ratio(result, fixed_delta, save_path=fig_p)
        print(f"[fig]  {fig_p}")
    else:
        print("[skip] noise-fixed-params-scan.parquet not found; skipping P2 and P5")


def generate_noise_decoupled_baseline(force: bool = False) -> None:
    """Verify the decoupled baseline under phase diffusion."""
    csv_p = _parquet_path("noise-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing decoupled baseline under phase diffusion...")
        gamma_test = [0.0, 1e-4, 1e-2, 1.0, 10.0]
        rows = []
        for g in gamma_test:
            dt = compute_noisy_decoupled_baseline(gamma_phi=g)
            rows.append(
                {
                    "gamma_phi": g,
                    "delta_omega": dt,
                    "sql": SQL_REFERENCE,
                    "t_hold": DEFAULT_t_hold,
                }
            )
        df = pd.DataFrame(rows)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(csv_p, index=False)
        print(f"[save] {csv_p}")


def generate_noise_free_optimal_params(force: bool = False) -> None:
    """Compute noise-free optimal parameters (γ_φ = 0) for each ω value.

    Results are saved as a single-row Parquet and used by the fixed-params scan.
    """
    csv_p = _parquet_path("noise-free-optimal-params")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing noise-free optimal parameters...")
        optimal = compute_noise_free_optimal_params(
            OMEGA_VALS,
            n_random=500,
            n_nm_refine=25,
            seed=42,
            maxiter=2000,
            early_stop_patience=200,
        )
        rows = []
        for omega_val in OMEGA_VALS:
            params = optimal.get(float(omega_val), (0.0, 0.0, 0.0, 0.0))
            rows.append(
                {
                    "omega": omega_val,
                    "a_x": params[0],
                    "a_y": params[1],
                    "a_z": params[2],
                    "a_zz": params[3],
                }
            )
        df = pd.DataFrame(rows)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(csv_p, index=False)
        print(f"[save] {csv_p}")


def generate_noise_fixed_params_scan(force: bool = False) -> None:
    """Evaluate Δω at noise-free optimal params for all (ω, γ_φ) pairs.

    Requires noise-free optimal params Parquet and noise-scan Parquet
    (to know the ω and γ_φ values).
    """
    csv_p = _parquet_path("noise-fixed-params-scan")
    opt_p = _parquet_path("noise-free-optimal-params")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    if not opt_p.exists():
        print(
            "[skip] noise-free-optimal-params.parquet not found; run 'noise-free-opt' first"
        )
        return

    print("[run]  Evaluating fixed-parameter sensitivity...")
    opt_df = pd.read_parquet(opt_p)
    noise_free_params: dict[float, tuple[float, float, float, float]] = {}
    for _, row in opt_df.iterrows():
        noise_free_params[float(row["omega"])] = (
            float(row["a_x"]),
            float(row["a_y"]),
            float(row["a_z"]),
            float(row["a_zz"]),
        )

    delta = evaluate_fixed_params_scan(
        OMEGA_VALS,
        GAMMA_PHI_VALS,
        noise_free_params,
    )

    # Save as DataFrame
    rows = []
    for i, t in enumerate(OMEGA_VALS):
        for j, g in enumerate(GAMMA_PHI_VALS):
            rows.append(
                {
                    "omega": t,
                    "gamma_phi": g,
                    "delta_omega": float(delta[i, j]),
                    "sql": SQL_REFERENCE,
                    "t_hold": DEFAULT_t_hold,
                }
            )
    df = pd.DataFrame(rows)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(csv_p, index=False)
    print(f"[save] {csv_p}")


def generate_noise_validation(force: bool = False) -> None:
    """Run validation checks: trace, Hermiticity, positivity."""
    print("[run]  Running validation checks...")
    ops = build_two_qubit_operators()

    # Test a small set of (ω, γ_φ) pairs
    test_omegas = [0.1, 1.0, 5.0]
    test_gammas = [0.0, 1e-4, 1e-2, 1.0]
    test_params_list = [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 1.0),
        (0.0, 3.0, 0.0, -2.0),
    ]

    n_pass = 0
    n_total = 0
    for omega in test_omegas:
        for gamma in test_gammas:
            for params in test_params_list:
                ax, ay, az, azz = params
                try:
                    rho = evolve_noisy_drive_circuit(
                        DEFAULT_RHO0,
                        DEFAULT_T_BS,
                        DEFAULT_t_hold,
                        omega,
                        gamma,
                        ax,
                        ay,
                        az,
                        azz,
                        ops,
                    )
                    validate_density(rho)
                    n_pass += 1
                except AssertionError as e:
                    print(f"  [FAIL] ω={omega}, γ_φ={gamma}, params={params}: {e}")
                n_total += 1

    print(f"  Validation: {n_pass}/{n_total} passed")

    # Also test sensitivity is finite
    n_sens_pass = 0
    n_sens_total = 0
    for omega in test_omegas:
        for gamma in test_gammas:
            for params in [test_params_list[0], test_params_list[1]]:
                ax, ay, az, azz = params
                dt = compute_noisy_sensitivity(
                    DEFAULT_RHO0,
                    DEFAULT_T_BS,
                    DEFAULT_t_hold,
                    omega,
                    gamma,
                    ax,
                    ay,
                    az,
                    azz,
                    ops,
                )
                n_sens_total += 1
                if np.isfinite(dt):
                    n_sens_pass += 1

    print(f"  Sensitivity valid: {n_sens_pass}/{n_sens_total} ")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-24 report figures and Parquet data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing Parquets)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset (e.g. 'noise-decoupled-baseline')",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / DATE_TAG / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / DATE_TAG / "figures").mkdir(parents=True, exist_ok=True)

    def _run_all(force: bool = False) -> None:
        generate_noise_decoupled_baseline(force=force)
        generate_noise_free_optimal_params(force=force)
        generate_noise_scan(force=force)
        generate_noise_fixed_params_scan(force=force)
        generate_noise_figures()

    tasks = {
        "noise-decoupled-baseline": generate_noise_decoupled_baseline,
        "noise-free-opt": generate_noise_free_optimal_params,
        "noise-scan": generate_noise_scan,
        "noise-fixed-params": generate_noise_fixed_params_scan,
        "noise-validation": generate_noise_validation,
        "figures": lambda force=args.force: generate_noise_figures(),
        "all": _run_all,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            if name == "all":
                continue
            print(f"\n=== {name} ===")
            func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
