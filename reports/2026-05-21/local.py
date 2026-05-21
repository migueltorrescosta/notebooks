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
    uv run python reports/2026-05-21/local.py --force

This module is **not** importable as ``reports.2026-05-21.local`` (the directory
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize

# Ensure project root is on sys.path for shared-module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# Shared primitives
from src.analysis.ancilla_drive_metrology import (  # noqa: E402
    DriveDecoupledBaselineResult,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (  # noqa: E402
    build_interaction_hamiltonian,
    build_two_qubit_operators,
)

sns.set_theme(style="whitegrid")

I_4 = np.eye(4, dtype=complex)

# ============================================================================
# Physical constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_H: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_T_H  # Δθ_SQL = 0.1
ALPHA_BOUNDS: tuple[float, float] = (-20.0, 20.0)  # Range for all α coefficients
N_BFGS_STARTS: int = 100  # Number of L-BFGS-B random starts per θ
THETA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]


# ============================================================================
# Hamiltonian Construction
# ============================================================================


def build_general_hold_hamiltonian(
    theta: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the full holding Hamiltonian with symmetric phase encoding.

    H = θ (J_z^S + J_z^A) + H_int(α)

    where:
        H_int = α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
              + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A

    Both system and ancilla experience the same unknown phase θ.

    Args:
        theta: Unknown phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = theta * (ops["Jz_S"] + ops["Jz_A"])
    H += build_interaction_hamiltonian(alpha)
    return 0.5 * (H + H.conj().T)


def general_hold_unitary(
    T_H: float,
    theta: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the general-interaction protocol.

    U_hold(T_H) = exp(-i T_H H)
    where H = θ (J_z^S + J_z^A) + H_int(α).

    Args:
        T_H: Holding-time strength.
        theta: True phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_general_hold_hamiltonian(theta, alpha, ops)
    U = expm(-1j * T_H * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Hold unitary not unitary for T_H={T_H}, θ={theta}, α={alpha}"
    )
    return U


# ============================================================================
# Circuit Evolution
# ============================================================================


def evolve_general_circuit(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full general-interaction MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(T_H) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        T_H: Holding-time strength.
        theta: Phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = general_hold_unitary(T_H, theta, alpha, ops) @ psi
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
    T_H: float,
    theta_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> float:
    """Compute the error-propagation sensitivity Δθ.

    Δθ = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂θ|

    Uses the reduced density matrix (trace out ancilla) for the variance,
    and central finite differences for the derivative of the expectation.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        T_H: Holding-time strength.
        theta_true: True phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δθ (positive float). Returns inf if derivative is zero.
    """
    # Evaluate at theta_true
    psi = evolve_general_circuit(psi0, T_BS, T_H, theta_true, alpha, ops)
    var = compute_reduced_variance(psi)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    psi_plus = evolve_general_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        alpha,
        ops,
    )
    psi_minus = evolve_general_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
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
    T_H: float,
    theta_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> tuple[float, float, float, float]:
    """Compute Δθ and return diagnostics.

    Returns:
        Tuple (delta_theta, expectation_Jz, variance_Jz, d_exp_d_theta).
    """
    # Evaluate at theta_true
    psi = evolve_general_circuit(psi0, T_BS, T_H, theta_true, alpha, ops)
    var = compute_reduced_variance(psi)
    exp_val = compute_reduced_expectation(psi)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    psi_plus = evolve_general_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        alpha,
        ops,
    )
    psi_minus = evolve_general_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
        alpha,
        ops,
    )

    exp_plus = compute_reduced_expectation(psi_plus)
    exp_minus = compute_reduced_expectation(psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var, d_exp

    delta_theta = float(np.sqrt(var) / abs(d_exp))
    return delta_theta, exp_val, var, d_exp


# ============================================================================
# L-BFGS-B Objective
# ============================================================================


def _general_sensitivity_objective(
    alpha_params: np.ndarray,
    theta_true: float,
    ops: dict[str, np.ndarray],
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    psi0: np.ndarray = DEFAULT_PSI0,
    fd_step: float = 1e-6,
) -> float:
    """Objective function for L-BFGS-B optimisation.

    f(α) = Δθ(α; θ_true)  (to be minimised)

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed T_H.

    Args:
        alpha_params: 4-element array [α_xx, α_xz, α_zx, α_zz].
        theta_true: True phase rate parameter.
        ops: Two-qubit operators.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        psi0: Initial state vector.
        fd_step: Finite-difference step.

    Returns:
        Δθ (positive float). Returns inf if fringe extremum.
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
        T_H,
        theta_true,
        alpha,
        ops,
        fd_step,
    )


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class GeneralBFGSOptimizationResult:
    """Result from a multi-start L-BFGS-B optimisation at a single θ.

    Attributes:
        theta_value: θ at which the optimisation was performed.
        alpha_opt: Optimal (α_xx, α_xz, α_zx, α_zz) found.
        delta_theta_opt: Minimal Δθ found.
        sql: SQL = 1/T_H reference value.
        expectation_Jz: ⟨J_z^S⟩ at the optimal point.
        variance_Jz: Var(J_z^S) at the optimal point.
        d_exp_d_theta: ∂⟨J_z^S⟩/∂θ at the optimal point.
        n_starts: Number of random starts used.
        n_converged: Number of starts that converged successfully.
    """

    theta_value: float
    alpha_opt: tuple[float, float, float, float]
    delta_theta_opt: float
    sql: float = 0.1
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_exp_d_theta: float = 0.0
    n_starts: int = N_BFGS_STARTS
    n_converged: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with all metadata."""
        return pd.DataFrame(
            {
                "theta_value": [self.theta_value],
                "alpha_xx_opt": [self.alpha_opt[0]],
                "alpha_xz_opt": [self.alpha_opt[1]],
                "alpha_zx_opt": [self.alpha_opt[2]],
                "alpha_zz_opt": [self.alpha_opt[3]],
                "delta_theta_opt": [self.delta_theta_opt],
                "sql": [self.sql],
                "ratio": [
                    self.delta_theta_opt / self.sql
                    if np.isfinite(self.delta_theta_opt) and self.sql > 0
                    else float("inf")
                ],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "d_exp_d_theta": [self.d_exp_d_theta],
                "n_starts": [self.n_starts],
                "n_converged": [self.n_converged],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> GeneralBFGSOptimizationResult:
        df = pd.read_parquet(path)
        required = {
            "theta_value",
            "alpha_xx_opt",
            "alpha_xz_opt",
            "alpha_zx_opt",
            "alpha_zz_opt",
            "delta_theta_opt",
            "sql",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        return cls(
            theta_value=float(df["theta_value"].iloc[0]),
            alpha_opt=(
                float(df["alpha_xx_opt"].iloc[0]),
                float(df["alpha_xz_opt"].iloc[0]),
                float(df["alpha_zx_opt"].iloc[0]),
                float(df["alpha_zz_opt"].iloc[0]),
            ),
            delta_theta_opt=float(df["delta_theta_opt"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0])
            if "expectation_Jz" in df.columns
            else 0.0,
            variance_Jz=float(df["variance_Jz"].iloc[0])
            if "variance_Jz" in df.columns
            else 0.0,
            d_exp_d_theta=float(df["d_exp_d_theta"].iloc[0])
            if "d_exp_d_theta" in df.columns
            else 0.0,
            n_starts=int(df["n_starts"].iloc[0])
            if "n_starts" in df.columns
            else N_BFGS_STARTS,
            n_converged=int(df["n_converged"].iloc[0])
            if "n_converged" in df.columns
            else 0,
        )


@dataclass
class GeneralThetaScanResult:
    """Results of a θ scan over L-BFGS-B-optimised sensitivities.

    Attributes:
        theta_values: Array of θ values scanned.
        alpha_xx_opt_per_theta: Optimal α_xx for each θ value.
        alpha_xz_opt_per_theta: Optimal α_xz for each θ value.
        alpha_zx_opt_per_theta: Optimal α_zx for each θ value.
        alpha_zz_opt_per_theta: Optimal α_zz for each θ value.
        delta_theta_opt_per_theta: Optimal Δθ for each θ value.
        sql_values: SQL = 1/T_H for each θ.
        expectation_Jz_per_theta: ⟨J_z^S⟩ at each optimal point.
        variance_Jz_per_theta: Var(J_z^S) at each optimal point.
        d_exp_d_theta_per_theta: ∂⟨J_z^S⟩/∂θ at each optimal point.
        n_converged_per_theta: Number of converged starts per θ.
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xx_opt_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xz_opt_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zx_opt_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zz_opt_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_theta_opt_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    d_exp_d_theta_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    n_converged_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for i in range(len(self.theta_values)):
            theta = float(self.theta_values[i])
            sql = float(self.sql_values[i]) if i < len(self.sql_values) else 0.1
            best = (
                float(self.delta_theta_opt_per_theta[i])
                if i < len(self.delta_theta_opt_per_theta)
                else float("inf")
            )
            a_xx = (
                float(self.alpha_xx_opt_per_theta[i])
                if i < len(self.alpha_xx_opt_per_theta)
                else float("nan")
            )
            a_xz = (
                float(self.alpha_xz_opt_per_theta[i])
                if i < len(self.alpha_xz_opt_per_theta)
                else float("nan")
            )
            a_zx = (
                float(self.alpha_zx_opt_per_theta[i])
                if i < len(self.alpha_zx_opt_per_theta)
                else float("nan")
            )
            a_zz = (
                float(self.alpha_zz_opt_per_theta[i])
                if i < len(self.alpha_zz_opt_per_theta)
                else float("nan")
            )
            exp_jz = (
                float(self.expectation_Jz_per_theta[i])
                if i < len(self.expectation_Jz_per_theta)
                else 0.0
            )
            var_jz = (
                float(self.variance_Jz_per_theta[i])
                if i < len(self.variance_Jz_per_theta)
                else 0.0
            )
            d_exp = (
                float(self.d_exp_d_theta_per_theta[i])
                if i < len(self.d_exp_d_theta_per_theta)
                else 0.0
            )
            n_conv = (
                int(self.n_converged_per_theta[i])
                if i < len(self.n_converged_per_theta)
                else 0
            )
            rows.append(
                {
                    "theta": theta,
                    "alpha_xx_opt": a_xx,
                    "alpha_xz_opt": a_xz,
                    "alpha_zx_opt": a_zx,
                    "alpha_zz_opt": a_zz,
                    "best_delta_theta": best,
                    "sql": sql,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "expectation_Jz": exp_jz,
                    "variance_Jz": var_jz,
                    "d_exp_d_theta": d_exp,
                    "n_converged": n_conv,
                }
            )
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> GeneralThetaScanResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "alpha_xx_opt",
            "alpha_xz_opt",
            "alpha_zx_opt",
            "alpha_zz_opt",
            "best_delta_theta",
            "sql",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        thetas = df["theta"].to_numpy(dtype=float)
        a_xx = df["alpha_xx_opt"].to_numpy(dtype=float)
        a_xz = df["alpha_xz_opt"].to_numpy(dtype=float)
        a_zx = df["alpha_zx_opt"].to_numpy(dtype=float)
        a_zz = df["alpha_zz_opt"].to_numpy(dtype=float)
        best = df["best_delta_theta"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = (
            df["expectation_Jz"].to_numpy(dtype=float)
            if "expectation_Jz" in df.columns
            else np.zeros_like(thetas)
        )
        vars_ = (
            df["variance_Jz"].to_numpy(dtype=float)
            if "variance_Jz" in df.columns
            else np.zeros_like(thetas)
        )
        d_exp = (
            df["d_exp_d_theta"].to_numpy(dtype=float)
            if "d_exp_d_theta" in df.columns
            else np.zeros_like(thetas)
        )
        n_conv = (
            df["n_converged"].to_numpy(dtype=float)
            if "n_converged" in df.columns
            else np.zeros_like(thetas)
        )
        return cls(
            theta_values=thetas,
            alpha_xx_opt_per_theta=a_xx,
            alpha_xz_opt_per_theta=a_xz,
            alpha_zx_opt_per_theta=a_zx,
            alpha_zz_opt_per_theta=a_zz,
            delta_theta_opt_per_theta=best,
            sql_values=sql,
            expectation_Jz_per_theta=exps,
            variance_Jz_per_theta=vars_,
            d_exp_d_theta_per_theta=d_exp,
            n_converged_per_theta=n_conv,
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_general_decoupled_baseline(
    T_H: float = DEFAULT_T_H,
    theta_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δθ.

    At α = (0, 0, 0, 0), the circuit reduces to a standard single-qubit MZI
    with |1,0⟩ input and 50/50 BS on the system, giving Δθ = 1/T_H.
    The ancilla evolves independently under θ J_z^A and is traced out,
    contributing nothing.

    Args:
        T_H: Holding-time strength.
        theta_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    dtheta = compute_general_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        T_H,
        theta_true,
        alpha,
        ops,
    )
    return DriveDecoupledBaselineResult(
        T_H_value=T_H,
        delta_theta=dtheta,
        sql=1.0 / T_H,
    )


# ============================================================================
# L-BFGS-B Multi-Start Optimisation
# ============================================================================


def run_general_bfgs_optimization(
    theta_true: float,
    alpha_bounds: tuple[float, float] = ALPHA_BOUNDS,
    n_starts: int = N_BFGS_STARTS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> GeneralBFGSOptimizationResult:
    """Run multi-start L-BFGS-B optimisation at a fixed θ.

    For each start:
    1. Generate random initial α ∈ [alpha_bounds[0], alpha_bounds[1]]^4.
    2. Run L-BFGS-B with bounded optimisation.
    3. Select the run with lowest Δθ.

    Args:
        theta_true: True phase rate parameter.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts.
        T_H: Holding time.
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
        rng = np.random.default_rng(base_seed + int(theta_true * 1000) + start)
        x0 = rng.uniform(lo, hi, size=4)

        result = minimize(
            _general_sensitivity_objective,
            x0,
            args=(theta_true, ops, T_H, T_BS, DEFAULT_PSI0, fd_step),
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
        T_H,
        theta_true,
        best_alpha,
        ops,
        fd_step,
    )

    return GeneralBFGSOptimizationResult(
        theta_value=theta_true,
        alpha_opt=best_alpha,
        delta_theta_opt=best_delta,
        sql=1.0 / T_H,
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        d_exp_d_theta=d_exp,
        n_starts=n_starts,
        n_converged=n_converged,
    )


# ============================================================================
# θ Scan
# ============================================================================


def run_general_theta_scan(
    theta_values: list[float] | np.ndarray,
    alpha_bounds: tuple[float, float] = ALPHA_BOUNDS,
    n_starts: int = N_BFGS_STARTS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> GeneralThetaScanResult:
    """Scan over θ values with multi-start L-BFGS-B optimisation at each θ.

    For each θ, run `n_starts` random-start L-BFGS-B optimisations and
    record the optimal α and Δθ.

    Args:
        theta_values: θ values to scan.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts per θ.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        seed: Base random seed.
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        GeneralThetaScanResult with optimal parameters and sensitivities.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    n_theta = len(theta_arr)

    a_xx_opts = np.full(n_theta, np.nan, dtype=float)
    a_xz_opts = np.full(n_theta, np.nan, dtype=float)
    a_zx_opts = np.full(n_theta, np.nan, dtype=float)
    a_zz_opts = np.full(n_theta, np.nan, dtype=float)
    best_deltas = np.full(n_theta, np.inf, dtype=float)
    sql_vals = np.full(n_theta, 1.0 / T_H, dtype=float)
    exp_vals = np.zeros(n_theta, dtype=float)
    var_vals = np.zeros(n_theta, dtype=float)
    d_exp_vals = np.zeros(n_theta, dtype=float)
    n_conv = np.zeros(n_theta, dtype=float)

    for i, theta in enumerate(theta_arr):
        result = run_general_bfgs_optimization(
            theta_true=theta,
            alpha_bounds=alpha_bounds,
            n_starts=n_starts,
            T_H=T_H,
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
        best_deltas[i] = result.delta_theta_opt
        exp_vals[i] = result.expectation_Jz
        var_vals[i] = result.variance_Jz
        d_exp_vals[i] = result.d_exp_d_theta
        n_conv[i] = result.n_converged

    return GeneralThetaScanResult(
        theta_values=theta_arr,
        alpha_xx_opt_per_theta=a_xx_opts,
        alpha_xz_opt_per_theta=a_xz_opts,
        alpha_zx_opt_per_theta=a_zx_opts,
        alpha_zz_opt_per_theta=a_zz_opts,
        delta_theta_opt_per_theta=best_deltas,
        sql_values=sql_vals,
        expectation_Jz_per_theta=exp_vals,
        variance_Jz_per_theta=var_vals,
        d_exp_d_theta_per_theta=d_exp_vals,
        n_converged_per_theta=n_conv,
    )


# ============================================================================
# Exclusive Plot Functions
# ============================================================================


def plot_general_theta_scan(
    result: GeneralThetaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δθ vs θ with SQL reference and optimal α as secondary axis.

    Args:
        result: GeneralThetaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    theta = result.theta_values
    sql_ref = float(result.sql_values[0]) if len(result.sql_values) > 0 else 0.1
    best_deltas = result.delta_theta_opt_per_theta

    # SQL reference line
    ax1.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_ref:.4f}",
    )

    # Δθ vs θ
    valid = np.isfinite(best_deltas)
    if np.any(valid):
        ax1.plot(
            theta[valid],
            best_deltas[valid],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.8,
            label=r"$\Delta\theta_{\mathrm{opt}}$",
        )
        # Annotate best point
        best_idx = int(np.argmin(best_deltas[valid]))
        best_theta = float(theta[valid][best_idx])
        best_val = float(best_deltas[valid][best_idx])
        best_ratio = best_val / sql_ref if sql_ref > 0 else float("inf")
        ax1.annotate(
            rf"Best: $\Delta\theta$={best_val:.5f} ({best_ratio:.3f}$\times$SQL)"
            rf" at $\theta$={best_theta:.2f}",
            xy=(best_theta, best_val),
            xytext=(best_theta + 0.8, best_val + 0.02),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title(
        "General-Interaction Sensitivity vs $\\theta$:\n"
        "Optimal $\\Delta\\theta$ via L-BFGS-B over "
        "$(\\alpha_{xx}, \\alpha_{xz}, \\alpha_{zx}, \\alpha_{zz})$"
    )

    # Secondary axis: all four optimal α parameters
    ax2 = ax1.twinx()
    for label, arr, color, marker in [
        (r"$\alpha_{xx}^*$", result.alpha_xx_opt_per_theta, "C1", "s"),
        (r"$\alpha_{xz}^*$", result.alpha_xz_opt_per_theta, "C2", "d"),
        (r"$\alpha_{zx}^*$", result.alpha_zx_opt_per_theta, "C3", "^"),
        (r"$\alpha_{zz}^*$", result.alpha_zz_opt_per_theta, "C4", "v"),
    ]:
        valid_a = np.isfinite(arr)
        if np.any(valid_a):
            ax2.plot(
                theta[valid_a],
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
    result: GeneralThetaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> Path:
    """Plot convergence metrics: converged starts and SQL ratio vs θ.

    Args:
        result: GeneralThetaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    theta = result.theta_values
    sql_ref = float(result.sql_values[0]) if len(result.sql_values) > 0 else 0.1

    # Left panel: number of converged starts
    valid_n = result.n_converged_per_theta > 0
    if np.any(valid_n):
        ax1.plot(
            theta[valid_n],
            result.n_converged_per_theta[valid_n],
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
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("Converged starts")
    ax1.set_title("L-BFGS-B Convergence vs $\\theta$")
    ax1.legend(fontsize=9)

    # Right panel: Δθ / SQL ratio
    valid_d = np.isfinite(result.delta_theta_opt_per_theta)
    if np.any(valid_d):
        ratios = result.delta_theta_opt_per_theta / sql_ref
        ax2.plot(
            theta[valid_d],
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
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\Delta\theta / \mathrm{SQL}$")
    ax2.set_title("Sensitivity Ratio vs $\\theta$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = PROJECT_ROOT / "reports"
DATE_TAG = "2026-05-21"


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / DATE_TAG / "raw_data" / f"{DATE_TAG}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / DATE_TAG / "figures" / f"{DATE_TAG}-{name}.svg"


# ── Parallel dispatch helper ──────────────────────────────────────────────


def _parallel_map(
    worker_fn,
    items,
    desc: str = "Processing",
    max_workers: int | None = None,
) -> None:
    """Run *worker_fn(item)* for each *item* in parallel via process pool.

    Each worker is a top-level function that performs its own file I/O.
    Results are implicitly persisted to disk by the worker.

    Args:
        worker_fn: Callable taking a single item argument.
        items: Iterable of items (typically θ values).
        desc: Short description for progress logging.
        max_workers: Number of subprocess workers (default: CPU count).
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)
    item_list = list(items)
    print(f"  [parallel] {desc}: {len(item_list)} items, {max_workers} workers")

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_item = {executor.submit(worker_fn, item): item for item in item_list}
        for future in concurrent.futures.as_completed(fut_to_item):
            item = fut_to_item[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item}: {exc}")
                raise


# ── Generator functions ───────────────────────────────────────────────────


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline verification."""
    csv_p = _parquet_path("decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline...")
        result = compute_general_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # No text-only figure — see theta-scan and convergence plots for visual results.


def _run_single_bfgs(theta: float, force: bool) -> None:
    """Run L-BFGS-B optimisation for a single θ value."""
    tag = f"bfgs-theta{theta}"
    csv_p = _parquet_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = GeneralBFGSOptimizationResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing L-BFGS-B at θ={theta} ({N_BFGS_STARTS} starts)...")
        result = run_general_bfgs_optimization(theta_true=theta)
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    # No text-only figure — see theta-scan and convergence plots for visual results.


def generate_bfgs_theta_scan(force: bool = False) -> None:
    """L-BFGS-B optimisation at all θ values (parallel)."""
    n = len(THETA_VALS)
    print(f"[run]  L-BFGS-B scans at {n} θ values (parallel)")

    # Wrap _run_single_bfgs to fix the force argument
    from functools import partial as _partial

    _worker = _partial(_run_single_bfgs, force=force)
    _parallel_map(_worker, THETA_VALS, desc="L-BFGS-B optimisation per θ")

    # Aggregate results into a single θ-scan Parquet
    agg_csv_p = _parquet_path("theta-scan")
    agg_fig_p = _fig_path("theta-scan")
    conv_fig_p = _fig_path("convergence")

    # Check that all per-θ Parquets exist
    theta_arr = np.array(THETA_VALS, dtype=float)
    n_theta = len(theta_arr)

    a_xx_opts = np.full(n_theta, np.nan, dtype=float)
    a_xz_opts = np.full(n_theta, np.nan, dtype=float)
    a_zx_opts = np.full(n_theta, np.nan, dtype=float)
    a_zz_opts = np.full(n_theta, np.nan, dtype=float)
    best_deltas = np.full(n_theta, np.inf, dtype=float)
    sql_vals = np.full(n_theta, 0.1, dtype=float)
    exp_vals = np.zeros(n_theta, dtype=float)
    var_vals = np.zeros(n_theta, dtype=float)
    d_exp_vals = np.zeros(n_theta, dtype=float)
    n_conv = np.zeros(n_theta, dtype=float)

    for i, theta in enumerate(theta_arr):
        tag = f"bfgs-theta{theta}"
        csv_p = _parquet_path(tag)
        if csv_p.exists():
            result = GeneralBFGSOptimizationResult.from_parquet(csv_p)
            a_xx_opts[i] = result.alpha_opt[0]
            a_xz_opts[i] = result.alpha_opt[1]
            a_zx_opts[i] = result.alpha_opt[2]
            a_zz_opts[i] = result.alpha_opt[3]
            best_deltas[i] = result.delta_theta_opt
            exp_vals[i] = result.expectation_Jz
            var_vals[i] = result.variance_Jz
            d_exp_vals[i] = result.d_exp_d_theta
            n_conv[i] = result.n_converged

    agg_result = GeneralThetaScanResult(
        theta_values=theta_arr,
        alpha_xx_opt_per_theta=a_xx_opts,
        alpha_xz_opt_per_theta=a_xz_opts,
        alpha_zx_opt_per_theta=a_zx_opts,
        alpha_zz_opt_per_theta=a_zz_opts,
        delta_theta_opt_per_theta=best_deltas,
        sql_values=sql_vals,
        expectation_Jz_per_theta=exp_vals,
        variance_Jz_per_theta=var_vals,
        d_exp_d_theta_per_theta=d_exp_vals,
        n_converged_per_theta=n_conv,
    )
    agg_result.save_parquet(agg_csv_p)
    print(f"[save] {agg_csv_p}")

    plot_general_theta_scan(agg_result, agg_fig_p)
    print(f"[fig]  {agg_fig_p}")

    plot_general_convergence(agg_result, conv_fig_p)
    print(f"[fig]  {conv_fig_p}")


def generate_figures(force: bool = False) -> None:
    """Generate all figures from existing Parquets."""
    # θ-scan figure
    agg_csv_p = _parquet_path("theta-scan")
    if agg_csv_p.exists():
        agg_result = GeneralThetaScanResult.from_parquet(agg_csv_p)
        plot_general_theta_scan(agg_result, _fig_path("theta-scan"))
        print(f"[fig]  {_fig_path('theta-scan')}")
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
        "decoupled-baseline": generate_decoupled_baseline,
        "bfgs-theta-scan": generate_bfgs_theta_scan,
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
