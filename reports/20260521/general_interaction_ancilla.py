"""
Local module for the 2026-05-21 General-Interaction Ancilla Metrology report.

Contains all code exclusive to this report:
- Core physics simulation (general 4-parameter interaction Hamiltonian,
  symmetric phase encoding on both qubits, system-only beam splitters,
  sensitivity computation with ancilla trace-out)
- Multi-start L-BFGS-B optimisation over (alpha_xx, alpha_xz, alpha_zx, alpha_zz)
- Decoupled baseline verification
- Exclusive plot functions
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260521/general_interaction_ancilla.py --force

This module is **not** importable as ``reports.20260521.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from src.utils.parallel import parallel_map

# Force single-threaded BLAS before any heavy numerical imports.
# This avoids in-process deadlocks when forking and keeps thread contention low.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "VECLIB_MAXIMUM_THREADS" not in os.environ:
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
if "NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deltalake import DeltaTable, write_deltalake
from deltalake._internal import CommitFailedError
from scipy.linalg import expm
from scipy.optimize import minimize

from src.analysis.ancilla_drive_metrology import (
    system_only_bs_unitary,
)
from src.analysis.ancilla_drive_results import (
    DriveDecoupledBaselineResult,
)
from src.analysis.ancilla_optimization import (
    build_interaction_hamiltonian,
    build_two_qubit_operators,
)

# Shared primitives
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
)
from src.utils.constants import I_4
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_t_hold  # Δω_SQL = 0.1
ALPHA_BOUNDS: tuple[float, float] = (-20.0, 20.0)  # Range for all α coefficients
N_BFGS_STARTS: int = 100  # Number of L-BFGS-B random starts per ω
OMEGA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]


# ============================================================================
# Hamiltonian Construction
# ============================================================================


def build_general_hold_hamiltonian(
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the full holding Hamiltonian with symmetric phase encoding.

    H = ω (J_z^S + J_z^A) + H_int(α)

    where:
        H_int = α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
              + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A

    Both system and ancilla experience the same unknown phase ω.

    Args:
        omega: Unknown phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * (ops["Jz_S"] + ops["Jz_A"])
    H += build_interaction_hamiltonian(alpha)
    return 0.5 * (H + H.conj().T)


def general_hold_unitary(
    t_hold: float,
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the general-interaction protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω (J_z^S + J_z^A) + H_int(α).

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_general_hold_hamiltonian(omega, alpha, ops)
    U = expm(-1j * t_hold * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Hold unitary not unitary for t_hold={t_hold}, ω={omega}, α={alpha}"
    )
    return U


# ============================================================================
# Circuit Evolution
# ============================================================================


def evolve_general_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full general-interaction MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = general_hold_unitary(t_hold, omega, alpha, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_reduced_variance(psi: np.ndarray) -> float:
    """Compute Var(J_z^S) via partial trace over the ancilla.

    For a two-qubit state |ψ⟩ with computational basis ordering |00⟩, |01⟩,
    |10⟩, |11⟩, the reduced density matrix of the system is:
        ρ_S = Tr_A(|ψ⟩⟨ψ|)

    Then Var(J_z^S) = Tr(ρ_S (J_z^S)^2) - (Tr(ρ_S J_z^S))^2
                    = 1/4 - ⟨J_z^S⟩²

    Args:
        psi: 4-vector state (pure).

    Returns:
        Variance of J_z^S after tracing the ancilla.
    """
    # Reshape into 2×2 matrix: rows = system, columns = ancilla
    psi_mat = psi.reshape(2, 2)  # shape (2, 2)

    # Reduced density matrix of system: ρ_S = psi @ psi^† traced over ancilla
    rho_S = psi_mat @ psi_mat.conj().T  # shape (2, 2)

    # Check trace preservation
    trace = float(np.real(np.trace(rho_S)))
    assert np.isclose(trace, 1.0, atol=1e-12), f"Reduced trace = {trace} != 1"

    # J_z^S = σ_z/2  (for a single qubit)
    Jz_S_sys = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)

    exp_val = float(np.real(np.trace(rho_S @ Jz_S_sys)))
    exp_sq = float(np.real(np.trace(rho_S @ (Jz_S_sys @ Jz_S_sys))))
    var_val = exp_sq - exp_val**2

    # Clamp negative variance to zero (numerical round-off)
    if var_val < 0 and var_val > -1e-12:
        var_val = 0.0

    assert var_val >= -1e-12, f"Unphysical negative variance: {var_val:.2e}"
    return float(max(0.0, var_val))


def compute_reduced_expectation(psi: np.ndarray) -> float:
    """Compute ⟨J_z^S⟩ via partial trace over the ancilla.

    Args:
        psi: 4-vector state (pure).

    Returns:
        Expectation value ⟨J_z^S⟩.
    """
    psi_mat = psi.reshape(2, 2)
    rho_S = psi_mat @ psi_mat.conj().T
    Jz_sys = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
    return float(np.real(np.trace(rho_S @ Jz_sys)))


def compute_general_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> float:
    """Compute the error-propagation sensitivity Δω.

    Δω = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂ω|

    Uses the reduced density matrix (trace out ancilla) for the variance,
    and central finite differences for the derivative of the expectation.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δω (positive float). Returns inf if derivative is zero.
    """
    # Evaluate at omega_true
    psi = evolve_general_circuit(psi0, T_BS, t_hold, omega_true, alpha, ops)
    var = compute_reduced_variance(psi)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_general_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        alpha,
        ops,
    )
    psi_minus = evolve_general_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        alpha,
        ops,
    )

    exp_plus = compute_reduced_expectation(psi_plus)
    exp_minus = compute_reduced_expectation(psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def compute_general_sensitivity_with_diagnostics(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> tuple[float, float, float, float]:
    """Compute Δω and return diagnostics.

    Returns:
        Tuple (delta_omega, expectation_Jz, variance_Jz, d_exp_d_omega).
    """
    # Evaluate at omega_true
    psi = evolve_general_circuit(psi0, T_BS, t_hold, omega_true, alpha, ops)
    var = compute_reduced_variance(psi)
    exp_val = compute_reduced_expectation(psi)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_general_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        alpha,
        ops,
    )
    psi_minus = evolve_general_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        alpha,
        ops,
    )

    exp_plus = compute_reduced_expectation(psi_plus)
    exp_minus = compute_reduced_expectation(psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var, d_exp

    delta_omega = float(np.sqrt(var) / abs(d_exp))
    return delta_omega, exp_val, var, d_exp


# ============================================================================
# L-BFGS-B Objective
# ============================================================================


def _general_sensitivity_objective(
    alpha_params: np.ndarray,
    omega_true: float,
    ops: dict[str, np.ndarray],
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    psi0: np.ndarray = DEFAULT_PSI0,
    fd_step: float = 1e-6,
) -> float:
    """Objective function for L-BFGS-B optimisation.

    f(α) = Δω(α; ω_true)  (to be minimised)

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed t_hold.

    Args:
        alpha_params: 4-element array [α_xx, α_xz, α_zx, α_zz].
        omega_true: True phase rate parameter.
        ops: Two-qubit operators.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        psi0: Initial state vector.
        fd_step: Finite-difference step.

    Returns:
        Δω (positive float). Returns inf if fringe extremum.
    """
    alpha: tuple[float, float, float, float] = (
        float(alpha_params[0]),
        float(alpha_params[1]),
        float(alpha_params[2]),
        float(alpha_params[3]),
    )
    return compute_general_sensitivity(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        alpha,
        ops,
        fd_step,
    )


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class GeneralBFGSOptimizationResult(ParquetSerializable):
    """Result from a multi-start L-BFGS-B optimisation at a single ω.

    Attributes:
        omega_value: ω at which the optimisation was performed.
        alpha_opt: Optimal (α_xx, α_xz, α_zx, α_zz) found.
        delta_omega_opt: Minimal Δω found.
        sql: SQL = 1/t_hold reference value.
        expectation_Jz: ⟨J_z^S⟩ at the optimal point.
        variance_Jz: Var(J_z^S) at the optimal point.
        d_exp_d_omega: ∂⟨J_z^S⟩/∂ω at the optimal point.
        n_starts: Number of random starts used.
        n_converged: Number of starts that converged successfully.
    """

    omega_value: float
    alpha_opt: tuple[float, float, float, float]
    delta_omega_opt: float
    sql: float = 0.1
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_exp_d_omega: float = 0.0
    n_starts: int = N_BFGS_STARTS
    n_converged: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega_value",
        "alpha_xx_opt",
        "alpha_xz_opt",
        "alpha_zx_opt",
        "alpha_zz_opt",
        "delta_omega_opt",
        "sql",
        "expectation_Jz",
        "variance_Jz",
        "d_exp_d_omega",
        "n_starts",
        "n_converged",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with all metadata."""
        return pd.DataFrame(
            {
                "omega_value": [self.omega_value],
                "alpha_xx_opt": [self.alpha_opt[0]],
                "alpha_xz_opt": [self.alpha_opt[1]],
                "alpha_zx_opt": [self.alpha_opt[2]],
                "alpha_zz_opt": [self.alpha_opt[3]],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [
                    self.delta_omega_opt / self.sql
                    if np.isfinite(self.delta_omega_opt) and self.sql > 0
                    else float("inf")
                ],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "d_exp_d_omega": [self.d_exp_d_omega],
                "n_starts": [self.n_starts],
                "n_converged": [self.n_converged],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> GeneralBFGSOptimizationResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_value=float(df["omega_value"].iloc[0]),
            alpha_opt=(
                float(df["alpha_xx_opt"].iloc[0]),
                float(df["alpha_xz_opt"].iloc[0]),
                float(df["alpha_zx_opt"].iloc[0]),
                float(df["alpha_zz_opt"].iloc[0]),
            ),
            delta_omega_opt=float(df["delta_omega_opt"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            d_exp_d_omega=float(df["d_exp_d_omega"].iloc[0]),
            n_starts=int(df["n_starts"].iloc[0]),
            n_converged=int(df["n_converged"].iloc[0]),
        )


@dataclass
class GeneralOmegaScanResult(ParquetSerializable):
    """Results of a ω scan over L-BFGS-B-optimised sensitivities.

    Attributes:
        omega_values: Array of ω values scanned.
        alpha_xx_opt_per_omega: Optimal α_xx for each ω value.
        alpha_xz_opt_per_omega: Optimal α_xz for each ω value.
        alpha_zx_opt_per_omega: Optimal α_zx for each ω value.
        alpha_zz_opt_per_omega: Optimal α_zz for each ω value.
        delta_omega_opt_per_omega: Optimal Δω for each ω value.
        sql_values: SQL = 1/t_hold for each ω.
        expectation_Jz_per_omega: ⟨J_z^S⟩ at each optimal point.
        variance_Jz_per_omega: Var(J_z^S) at each optimal point.
        d_exp_d_omega_per_omega: ∂⟨J_z^S⟩/∂ω at each optimal point.
        n_converged_per_omega: Number of converged starts per ω.
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xx_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xz_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zx_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zz_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    d_exp_d_omega_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    n_converged_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "alpha_xx_opt",
        "alpha_xz_opt",
        "alpha_zx_opt",
        "alpha_zz_opt",
        "best_delta_omega",
        "sql",
    ]

    @staticmethod
    def _safe_get(arr: np.ndarray, i: int, default: float) -> float:
        """Get array element with bounds checking."""
        if i < len(arr):
            return float(arr[i])
        return default

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        n = len(self.omega_values)
        for i in range(n):
            best = self._safe_get(self.delta_omega_opt_per_omega, i, float("inf"))
            sql = self._safe_get(self.sql_values, i, 0.1)
            rows.append(
                {
                    "omega": float(self.omega_values[i]),
                    "alpha_xx_opt": self._safe_get(
                        self.alpha_xx_opt_per_omega, i, float("nan")
                    ),
                    "alpha_xz_opt": self._safe_get(
                        self.alpha_xz_opt_per_omega, i, float("nan")
                    ),
                    "alpha_zx_opt": self._safe_get(
                        self.alpha_zx_opt_per_omega, i, float("nan")
                    ),
                    "alpha_zz_opt": self._safe_get(
                        self.alpha_zz_opt_per_omega, i, float("nan")
                    ),
                    "best_delta_omega": best,
                    "sql": sql,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "expectation_Jz": self._safe_get(
                        self.expectation_Jz_per_omega, i, 0.0
                    ),
                    "variance_Jz": self._safe_get(self.variance_Jz_per_omega, i, 0.0),
                    "d_exp_d_omega": self._safe_get(
                        self.d_exp_d_omega_per_omega, i, 0.0
                    ),
                    "n_converged": int(
                        self._safe_get(self.n_converged_per_omega, i, 0.0)
                    ),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> GeneralOmegaScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omegas = df["omega"].to_numpy(dtype=float)
        a_xx = df["alpha_xx_opt"].to_numpy(dtype=float)
        a_xz = df["alpha_xz_opt"].to_numpy(dtype=float)
        a_zx = df["alpha_zx_opt"].to_numpy(dtype=float)
        a_zz = df["alpha_zz_opt"].to_numpy(dtype=float)
        best = df["best_delta_omega"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = (
            df["expectation_Jz"].to_numpy(dtype=float)
            if "expectation_Jz" in df.columns
            else np.zeros_like(omegas)
        )
        vars_ = (
            df["variance_Jz"].to_numpy(dtype=float)
            if "variance_Jz" in df.columns
            else np.zeros_like(omegas)
        )
        d_exp = (
            df["d_exp_d_omega"].to_numpy(dtype=float)
            if "d_exp_d_omega" in df.columns
            else np.zeros_like(omegas)
        )
        n_conv = (
            df["n_converged"].to_numpy(dtype=float)
            if "n_converged" in df.columns
            else np.zeros_like(omegas)
        )
        return cls(
            omega_values=omegas,
            alpha_xx_opt_per_omega=a_xx,
            alpha_xz_opt_per_omega=a_xz,
            alpha_zx_opt_per_omega=a_zx,
            alpha_zz_opt_per_omega=a_zz,
            delta_omega_opt_per_omega=best,
            sql_values=sql,
            expectation_Jz_per_omega=exps,
            variance_Jz_per_omega=vars_,
            d_exp_d_omega_per_omega=d_exp,
            n_converged_per_omega=n_conv,
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_general_decoupled_baseline(
    t_hold: float = DEFAULT_t_hold,
    omega_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω.

    At α = (0, 0, 0, 0), the circuit reduces to a standard single-qubit MZI
    with |1,0⟩ input and 50/50 BS on the system, giving Δω = 1/t_hold.
    The ancilla evolves independently under ω J_z^A and is traced out,
    contributing nothing.

    Args:
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    domega = compute_general_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        t_hold,
        omega_true,
        alpha,
        ops,
    )
    return DriveDecoupledBaselineResult(
        t_hold_value=t_hold,
        delta_omega=domega,
        sql=1.0 / t_hold,
    )


# ============================================================================
# L-BFGS-B Multi-Start Optimisation
# ============================================================================


def run_general_bfgs_optimization(
    omega_true: float,
    alpha_bounds: tuple[float, float] = ALPHA_BOUNDS,
    n_starts: int = N_BFGS_STARTS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> GeneralBFGSOptimizationResult:
    """Run multi-start L-BFGS-B optimisation at a fixed ω.

    For each start:
    1. Generate random initial α ∈ [alpha_bounds[0], alpha_bounds[1]]^4.
    2. Run L-BFGS-B with bounded optimisation.
    3. Select the run with lowest Δω.

    Args:
        omega_true: True phase rate parameter.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        seed: Base random seed (incremented per start).
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        GeneralBFGSOptimizationResult with best parameters found.
    """
    ops = build_two_qubit_operators()
    lo, hi = alpha_bounds
    base_seed = seed if seed is not None else 42
    bounds_ls = [(lo, hi)] * 4

    best_delta = float("inf")
    best_alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    n_converged = 0

    for start in range(n_starts):
        rng = np.random.default_rng(base_seed + int(omega_true * 1000) + start)
        x0 = rng.uniform(lo, hi, size=4)

        result = minimize(
            _general_sensitivity_objective,
            x0,
            args=(omega_true, ops, t_hold, T_BS, DEFAULT_PSI0, fd_step),
            method="L-BFGS-B",
            bounds=bounds_ls,
            options={
                "maxiter": maxiter,
                "gtol": gtol,
                "ftol": 1e-12,
            },
        )

        if result.success:
            n_converged += 1

        delta_val = float(result.fun)
        if np.isfinite(delta_val) and delta_val < best_delta:
            best_delta = delta_val
            best_alpha = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
                float(result.x[3]),
            )

    # Compute diagnostics at the optimal point
    _, exp_val, var_val, d_exp = compute_general_sensitivity_with_diagnostics(
        DEFAULT_PSI0,
        T_BS,
        t_hold,
        omega_true,
        best_alpha,
        ops,
        fd_step,
    )

    return GeneralBFGSOptimizationResult(
        omega_value=omega_true,
        alpha_opt=best_alpha,
        delta_omega_opt=best_delta,
        sql=1.0 / t_hold,
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        d_exp_d_omega=d_exp,
        n_starts=n_starts,
        n_converged=n_converged,
    )


# ============================================================================
# ω Scan
# ============================================================================


def run_general_omega_scan(
    omega_values: list[float] | np.ndarray,
    alpha_bounds: tuple[float, float] = ALPHA_BOUNDS,
    n_starts: int = N_BFGS_STARTS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> GeneralOmegaScanResult:
    """Scan over ω values with multi-start L-BFGS-B optimisation at each ω.

    For each ω, run `n_starts` random-start L-BFGS-B optimisations and
    record the optimal α and Δω.

    Args:
        omega_values: ω values to scan.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts per ω.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        seed: Base random seed.
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        GeneralOmegaScanResult with optimal parameters and sensitivities.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    n_omega = len(omega_arr)

    a_xx_opts = np.full(n_omega, np.nan, dtype=float)
    a_xz_opts = np.full(n_omega, np.nan, dtype=float)
    a_zx_opts = np.full(n_omega, np.nan, dtype=float)
    a_zz_opts = np.full(n_omega, np.nan, dtype=float)
    best_deltas = np.full(n_omega, np.inf, dtype=float)
    sql_vals = np.full(n_omega, 1.0 / t_hold, dtype=float)
    exp_vals = np.zeros(n_omega, dtype=float)
    var_vals = np.zeros(n_omega, dtype=float)
    d_exp_vals = np.zeros(n_omega, dtype=float)
    n_conv = np.zeros(n_omega, dtype=float)

    for i, omega in enumerate(omega_arr):
        result = run_general_bfgs_optimization(
            omega_true=omega,
            alpha_bounds=alpha_bounds,
            n_starts=n_starts,
            t_hold=t_hold,
            T_BS=T_BS,
            fd_step=fd_step,
            seed=seed,
            maxiter=maxiter,
            gtol=gtol,
        )
        a_xx_opts[i] = result.alpha_opt[0]
        a_xz_opts[i] = result.alpha_opt[1]
        a_zx_opts[i] = result.alpha_opt[2]
        a_zz_opts[i] = result.alpha_opt[3]
        best_deltas[i] = result.delta_omega_opt
        exp_vals[i] = result.expectation_Jz
        var_vals[i] = result.variance_Jz
        d_exp_vals[i] = result.d_exp_d_omega
        n_conv[i] = result.n_converged

    return GeneralOmegaScanResult(
        omega_values=omega_arr,
        alpha_xx_opt_per_omega=a_xx_opts,
        alpha_xz_opt_per_omega=a_xz_opts,
        alpha_zx_opt_per_omega=a_zx_opts,
        alpha_zz_opt_per_omega=a_zz_opts,
        delta_omega_opt_per_omega=best_deltas,
        sql_values=sql_vals,
        expectation_Jz_per_omega=exp_vals,
        variance_Jz_per_omega=var_vals,
        d_exp_d_omega_per_omega=d_exp_vals,
        n_converged_per_omega=n_conv,
    )


# ============================================================================
# Exclusive Plot Functions
# ============================================================================


def plot_general_omega_scan(
    result: GeneralOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δω vs ω with SQL reference and optimal α as secondary axis.

    Args:
        result: GeneralOmegaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    omega = result.omega_values
    sql_ref = float(result.sql_values[0]) if len(result.sql_values) > 0 else 0.1
    best_deltas = result.delta_omega_opt_per_omega

    # SQL reference line
    ax1.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_ref:.4f}",
    )

    # Δω vs ω
    valid = np.isfinite(best_deltas)
    if np.any(valid):
        ax1.plot(
            omega[valid],
            best_deltas[valid],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.8,
            label=r"$\Delta\omega_{\mathrm{opt}}$",
        )
        # Annotate best point
        best_idx = int(np.argmin(best_deltas[valid]))
        best_omega = float(omega[valid][best_idx])
        best_val = float(best_deltas[valid][best_idx])
        best_ratio = best_val / sql_ref if sql_ref > 0 else float("inf")
        ax1.annotate(
            rf"Best: $\Delta\omega$={best_val:.5f} ({best_ratio:.3f}$\times$SQL)"
            rf" at $\omega$={best_omega:.2f}",
            xy=(best_omega, best_val),
            xytext=(best_omega + 0.8, best_val + 0.02),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(
        "General-Interaction Sensitivity vs $\\omega$:\n"
        "Optimal $\\Delta\\omega$ via L-BFGS-B over "
        "$(\\alpha_{xx}, \\alpha_{xz}, \\alpha_{zx}, \\alpha_{zz})$"
    )

    # Secondary axis: all four optimal α parameters
    ax2 = ax1.twinx()
    for label, arr, color, marker in [
        (r"$\alpha_{xx}^*$", result.alpha_xx_opt_per_omega, "C1", "s"),
        (r"$\alpha_{xz}^*$", result.alpha_xz_opt_per_omega, "C2", "d"),
        (r"$\alpha_{zx}^*$", result.alpha_zx_opt_per_omega, "C3", "^"),
        (r"$\alpha_{zz}^*$", result.alpha_zz_opt_per_omega, "C4", "v"),
    ]:
        valid_a = np.isfinite(arr)
        if np.any(valid_a):
            ax2.plot(
                omega[valid_a],
                arr[valid_a],
                marker + "-",
                color=color,
                markersize=5,
                linewidth=1.0,
                alpha=0.6,
                label=label,
            )
    ax2.set_ylabel(r"$\alpha_{ij}^*$")
    ax2.tick_params(axis="y")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_general_convergence(
    result: GeneralOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> Path:
    """Plot convergence metrics: converged starts and SQL ratio vs ω.

    Args:
        result: GeneralOmegaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    omega = result.omega_values
    sql_ref = float(result.sql_values[0]) if len(result.sql_values) > 0 else 0.1

    # Left panel: number of converged starts
    valid_n = result.n_converged_per_omega > 0
    if np.any(valid_n):
        ax1.plot(
            omega[valid_n],
            result.n_converged_per_omega[valid_n],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.5,
        )
    ax1.axhline(
        y=N_BFGS_STARTS,
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label=f"Total starts = {N_BFGS_STARTS}",
    )
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel("Converged starts")
    ax1.set_title("L-BFGS-B Convergence vs $\\omega$")
    ax1.legend(fontsize=9)

    # Right panel: Δω / SQL ratio
    valid_d = np.isfinite(result.delta_omega_opt_per_omega)
    if np.any(valid_d):
        ratios = result.delta_omega_opt_per_omega / sql_ref
        ax2.plot(
            omega[valid_d],
            ratios[valid_d],
            "o-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="SQL = 1.0",
    )
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega / \mathrm{SQL}$")
    ax2.set_title("Sensitivity Ratio vs $\\omega$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
DATE_TAG = "20260521"
BFGS_TABLE_DIR = str(REPORTS_DIR / DATE_TAG / "raw_data" / "bfgs-results")


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, DATE_TAG)


def _upsert_bfgs_result(result: GeneralBFGSOptimizationResult) -> None:
    """Append one row to the Delta table (concurrent-writer safe)."""
    row = result.to_dataframe()
    Path(BFGS_TABLE_DIR).parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            write_deltalake(BFGS_TABLE_DIR, row, mode="append")
            return
        except (OSError, ValueError, CommitFailedError):
            if attempt < 4:
                import time

                time.sleep(0.05 * (attempt + 1))
            else:
                raise


# ── Generator functions ───────────────────────────────────────────────────


def _run_single_bfgs(omega: float, force: bool) -> None:
    """Run L-BFGS-B optimisation for a single ω value, upserting to Delta table."""
    print(f"  [run]  Computing L-BFGS-B at ω={omega} ({N_BFGS_STARTS} starts)...")
    result = run_general_bfgs_optimization(omega_true=omega)
    _upsert_bfgs_result(result)


def generate_bfgs_omega_scan(force: bool = False) -> None:
    """L-BFGS-B optimisation at all ω values (parallel) with Delta Lake storage."""
    if force:
        import shutil

        shutil.rmtree(BFGS_TABLE_DIR, ignore_errors=True)

    n = len(OMEGA_VALS)
    print(f"[run]  L-BFGS-B scans at {n} ω values (parallel)")

    # Wrap _run_single_bfgs to fix the force argument
    from functools import partial as _partial

    _worker = _partial(_run_single_bfgs, force=force)
    parallel_map(_worker, OMEGA_VALS, desc="L-BFGS-B optimisation per ω")

    # Compact small files into fewer larger ones for efficient reads
    compact_info = DeltaTable(BFGS_TABLE_DIR).optimize.compact()
    print(
        f"  [compact] {compact_info['numFilesRemoved']} files → "
        f"{compact_info['numFilesAdded']} file"
    )

    # Vacuum tombstoned files from disk
    dt = DeltaTable(BFGS_TABLE_DIR)
    dt.alter.set_table_properties(
        {"delta.deletedFileRetentionDuration": "interval 0 days"}
    )
    n_vacuumed = len(dt.vacuum(retention_hours=0, dry_run=False))
    print(f"  [vacuum] {n_vacuumed} tombstoned files removed")

    # Single read for all rows, sorted by omega_value
    df = (
        DeltaTable(BFGS_TABLE_DIR)
        .to_pandas()
        .sort_values("omega_value")
        .reset_index(drop=True)
    )

    agg_result = GeneralOmegaScanResult(
        omega_values=df["omega_value"].to_numpy(dtype=float),
        alpha_xx_opt_per_omega=df["alpha_xx_opt"].to_numpy(dtype=float),
        alpha_xz_opt_per_omega=df["alpha_xz_opt"].to_numpy(dtype=float),
        alpha_zx_opt_per_omega=df["alpha_zx_opt"].to_numpy(dtype=float),
        alpha_zz_opt_per_omega=df["alpha_zz_opt"].to_numpy(dtype=float),
        delta_omega_opt_per_omega=df["delta_omega_opt"].to_numpy(dtype=float),
        sql_values=df["sql"].to_numpy(dtype=float),
        expectation_Jz_per_omega=df["expectation_Jz"].to_numpy(dtype=float),
        variance_Jz_per_omega=df["variance_Jz"].to_numpy(dtype=float),
        d_exp_d_omega_per_omega=df["d_exp_d_omega"].to_numpy(dtype=float),
        n_converged_per_omega=df["n_converged"].to_numpy(dtype=float),
    )

    agg_csv_p = _parquet_path("omega-scan")
    agg_result.save_parquet(agg_csv_p)
    print(f"[save] {agg_csv_p}")

    agg_fig_p = _fig_path("omega-scan")
    plot_general_omega_scan(agg_result, agg_fig_p)
    print(f"[fig]  {agg_fig_p}")

    conv_fig_p = _fig_path("convergence")
    plot_general_convergence(agg_result, conv_fig_p)
    print(f"[fig]  {conv_fig_p}")


def generate_figures(force: bool = False) -> None:
    """Generate all figures from existing Parquets."""
    # ω-scan figure
    agg_csv_p = _parquet_path("omega-scan")
    if agg_csv_p.exists():
        agg_result = GeneralOmegaScanResult.from_parquet(agg_csv_p)
        plot_general_omega_scan(agg_result, _fig_path("omega-scan"))
        print(f"[fig]  {_fig_path('omega-scan')}")
        plot_general_convergence(agg_result, _fig_path("convergence"))
        print(f"[fig]  {_fig_path('convergence')}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-21 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / DATE_TAG / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / DATE_TAG / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "decoupled-baseline": lambda force=False: generate_decoupled_baseline(
            force=force,
            parquet_path=_parquet_path("decoupled-baseline"),
            compute_fn=compute_general_decoupled_baseline,
            result_cls=DriveDecoupledBaselineResult,
            label="decoupled baseline",
        ),
        "bfgs-omega-scan": generate_bfgs_omega_scan,
        "figures": generate_figures,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
