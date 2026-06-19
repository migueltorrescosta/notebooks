"""
Local module for the 2026-05-20 XX-Coupling Ancilla Metrology report.

Contains all code exclusive to this report:
- Core physics simulation (XX coupling Hamiltonian, circuit evolution,
  sensitivity computation with ancilla trace-out)
- 1D α_xx grid scan and ω-optimisation
- Exclusive plot functions
- Data and figure generation pipeline (``generate_xx_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260520/local.py --force

This module is **not** importable as ``reports.20260520.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm

from src.utils.parallel import parallel_map
from src.utils.paths import fig_path, parquet_path

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# Shared primitives
from src.analysis.ancilla_drive_metrology import (
    DriveDecoupledBaselineResult,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)
from src.utils.constants import I_4
from src.utils.serialization import ParquetSerializable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_t_hold  # Δω_SQL = 0.1
AXX_BOUNDS: tuple[float, float] = (0.0, 20.0)  # Range for α_xx
N_GRID_POINTS: int = 2001  # Grid points for α_xx scan


# ============================================================================
# Operator/Hamiltonian Construction
# ============================================================================


def build_xx_interaction(alpha_xx: float, ops: dict[str, np.ndarray]) -> np.ndarray:
    """Build the XX interaction Hamiltonian.

    H_int = α_xx J_x^S ⊗ J_x^A

    Args:
        alpha_xx: XX coupling strength.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if alpha_xx != 0.0:
        # J_x^S ⊗ J_x^A = (J_x ⊗ I_2) @ (I_2 ⊗ J_x)
        H += alpha_xx * (ops["Jx_S"] @ ops["Jx_A"])
    return 0.5 * (H + H.conj().T)


def build_xx_hold_hamiltonian(
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with symmetric phase encoding
    and XX coupling.

    H = H_S + H_A + H_int
      = ω J_z^S + ω J_z^A + α_xx J_x^S ⊗ J_x^A
      = ω (J_z^S + J_z^A) + α_xx J_x^S ⊗ J_x^A

    Both system and ancilla experience the same unknown phase ω.

    Args:
        omega: Unknown phase rate parameter.
        alpha_xx: XX coupling strength.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * (ops["Jz_S"] + ops["Jz_A"])
    H += build_xx_interaction(alpha_xx, ops)
    return 0.5 * (H + H.conj().T)


def xx_hold_unitary(
    t_hold: float,
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the XX-coupling protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω(J_z^S + J_z^A) + α_xx J_x^S ⊗ J_x^A.

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        alpha_xx: XX coupling strength.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_xx_hold_hamiltonian(omega, alpha_xx, ops)
    U = expm(-1j * t_hold * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"XX hold unitary not unitary for t_hold={t_hold}, ω={omega}, α_xx={alpha_xx}"
    )
    return U


# ============================================================================
# Circuit Evolution and Sensitivity
# ============================================================================


def evolve_xx_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full XX-coupling MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        alpha_xx: XX coupling strength.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = xx_hold_unitary(t_hold, omega, alpha_xx, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_reduced_variance(psi: np.ndarray, meas_op: np.ndarray) -> float:
    """Compute Var(J_z^S) via partial trace over the ancilla.

    For a two-qubit state |ψ⟩ with computational basis ordering |00⟩, |01⟩,
    |10⟩, |11⟩, the reduced density matrix of the system is:
        ρ_S = Tr_A(|ψ⟩⟨ψ|)

    Since (J_z^S)^2 = (1/4) I_2, we compute:
        Var(J_z^S) = Tr(ρ_S (J_z^S)^2) - (Tr(ρ_S J_z^S))^2
                  = 1/4 - ⟨J_z^S⟩²

    However, for full rigor we compute via the reduced density matrix.

    Args:
        psi: 4-vector state (pure).
        meas_op: Measurement operator (e.g., J_z^S).

    Returns:
        Variance of the measurement operator after tracing the ancilla.
    """
    # Reshape into 2×2 matrix: rows = system, columns = ancilla
    psi_mat = psi.reshape(2, 2)  # shape (2, 2)

    # Reduced density matrix of system: ρ_S = psi @ psi^† traced over ancilla
    rho_S = psi_mat @ psi_mat.conj().T  # shape (2, 2)

    # Check trace preservation
    trace = float(np.real(np.trace(rho_S)))
    assert np.isclose(trace, 1.0, atol=1e-12), f"Reduced trace = {trace} != 1"

    # J_z^S = σ_z/2
    Jz_S_sys = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)

    exp_val = float(np.real(np.trace(rho_S @ Jz_S_sys)))
    exp_sq = float(np.real(np.trace(rho_S @ (Jz_S_sys @ Jz_S_sys))))
    var_val = exp_sq - exp_val**2

    # Clamp negative variance to zero (numerical round-off)
    if var_val < 0 and var_val > -1e-12:
        var_val = 0.0

    assert var_val >= -1e-12, f"Unphysical negative variance: {var_val:.2e}"
    return float(max(0.0, var_val))


def compute_xx_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    alpha_xx: float,
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
        alpha_xx: XX coupling strength.
        ops: Two-qubit operators.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δω (positive float). Returns inf if derivative is zero.
    """
    meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_xx_circuit(psi0, T_BS, t_hold, omega_true, alpha_xx, ops)
    var = compute_reduced_variance(psi, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_xx_circuit(
        psi0, T_BS, t_hold, omega_true + fd_step, alpha_xx, ops
    )
    psi_minus = evolve_xx_circuit(
        psi0, T_BS, t_hold, omega_true - fd_step, alpha_xx, ops
    )

    # Expectation from reduced state
    def _reduced_exp(psi_state: np.ndarray) -> float:
        psi_m = psi_state.reshape(2, 2)
        rho = psi_m @ psi_m.conj().T
        Jz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
        return float(np.real(np.trace(rho @ Jz)))

    exp_plus = _reduced_exp(psi_plus)
    exp_minus = _reduced_exp(psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class XXGridScanResult(ParquetSerializable):
    """Result from a 1D α_xx grid scan at a fixed ω.

    Attributes:
        alpha_xx_values: Array of α_xx values scanned.
        delta_omega_values: Δω at each α_xx value.
        omega_value: ω at which the scan was performed.
        alpha_xx_opt: α_xx value giving minimal Δω.
        delta_omega_opt: Minimal Δω found.
        sql: SQL = 1/t_hold reference value.
        expectation_Jz: ⟨J_z^S⟩ at the optimal point.
        variance_Jz: Var(J_z^S) at the optimal point.
    """

    alpha_xx_values: np.ndarray
    delta_omega_values: np.ndarray
    omega_value: float
    alpha_xx_opt: float
    delta_omega_opt: float
    sql: float = 0.1
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "alpha_xx",
        "delta_omega",
        "omega_value",
        "sql",
        "alpha_xx_opt",
        "delta_omega_opt",
        "expectation_Jz",
        "variance_Jz",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "alpha_xx": self.alpha_xx_values,
                "delta_omega": self.delta_omega_values,
                "omega_value": [self.omega_value] * len(self.alpha_xx_values),
                "sql": [self.sql] * len(self.alpha_xx_values),
                "alpha_xx_opt": [self.alpha_xx_opt] * len(self.alpha_xx_values),
                "delta_omega_opt": [self.delta_omega_opt] * len(self.alpha_xx_values),
                "expectation_Jz": [self.expectation_Jz] * len(self.alpha_xx_values),
                "variance_Jz": [self.variance_Jz] * len(self.alpha_xx_values),
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> XXGridScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        alpha_xx_vals = df["alpha_xx"].to_numpy(dtype=float)
        delta_vals = df["delta_omega"].to_numpy(dtype=float)
        # Find the unique metadata from the first row
        return cls(
            alpha_xx_values=alpha_xx_vals,
            delta_omega_values=delta_vals,
            omega_value=float(df["omega_value"].iloc[0]),
            alpha_xx_opt=float(df["alpha_xx_opt"].iloc[0]),
            delta_omega_opt=float(df["delta_omega_opt"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
        )


@dataclass
class XXOmegaScanResult(ParquetSerializable):
    """Results of a ω scan over α_xx-optimised sensitivities.

    Attributes:
        omega_values: Array of ω values scanned.
        alpha_xx_opt_per_omega: Optimal α_xx for each ω value.
        delta_omega_opt_per_omega: Optimal Δω for each ω value.
        sql_values: SQL = 1/t_hold for each ω.
        expectation_Jz_per_omega: ⟨J_z^S⟩ at each optimal point.
        variance_Jz_per_omega: Var(J_z^S) at each optimal point.
        count_below_sql_per_omega: Number of α_xx grid points below SQL at each ω.
        total_points_per_omega: Total α_xx grid points at each ω.
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xx_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    count_below_sql_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    total_points_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "alpha_xx_opt",
        "best_delta_omega",
        "sql",
        "ratio",
        "expectation_Jz",
        "variance_Jz",
        "count_below_sql",
        "total_points",
        "fraction_below_sql",
    ]

    @staticmethod
    def _safe_get(arr: np.ndarray, i: int, default: float) -> float:
        """Get array element with bounds checking."""
        if i < len(arr):
            return float(arr[i])
        return default

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for i in range(len(self.omega_values)):
            best = self._safe_get(self.delta_omega_opt_per_omega, i, float("inf"))
            sql = self._safe_get(self.sql_values, i, 0.1)
            count_below = self._safe_get(self.count_below_sql_per_omega, i, 0.0)
            total = self._safe_get(self.total_points_per_omega, i, 0.0)
            rows.append(
                {
                    "omega": float(self.omega_values[i]),
                    "alpha_xx_opt": self._safe_get(
                        self.alpha_xx_opt_per_omega, i, float("nan")
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
                    "count_below_sql": int(count_below),
                    "total_points": int(total),
                    "fraction_below_sql": count_below / total if total > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> XXOmegaScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omegas = df["omega"].to_numpy(dtype=float)
        alpha_opts = df["alpha_xx_opt"].to_numpy(dtype=float)
        best = df["best_delta_omega"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = df["expectation_Jz"].to_numpy(dtype=float)
        vars_ = df["variance_Jz"].to_numpy(dtype=float)
        count_below = df["count_below_sql"].to_numpy(dtype=float)
        total = df["total_points"].to_numpy(dtype=float)
        return cls(
            omega_values=omegas,
            alpha_xx_opt_per_omega=alpha_opts,
            delta_omega_opt_per_omega=best,
            sql_values=sql,
            expectation_Jz_per_omega=exps,
            variance_Jz_per_omega=vars_,
            count_below_sql_per_omega=count_below,
            total_points_per_omega=total,
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_xx_decoupled_baseline(
    t_hold: float = DEFAULT_t_hold,
    omega_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω.

    At α_xx = 0, the circuit reduces to a standard single-qubit MZI
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
    domega = compute_xx_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        t_hold,
        omega_true,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        t_hold_value=t_hold,
        delta_omega=domega,
        sql=1.0 / t_hold,
    )


# ============================================================================
# 1D α_xx Grid Scan
# ============================================================================


def xx_grid_scan(
    omega: float,
    alpha_xx_range: tuple[float, float] = AXX_BOUNDS,
    n_points: int = N_GRID_POINTS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> XXGridScanResult:
    """Run a 1D grid scan over α_xx at fixed ω.

    Evaluates Δω on a dense grid of α_xx values and records the minimum.

    Args:
        omega: Phase rate value.
        alpha_xx_range: (min, max) for α_xx.
        n_points: Number of grid points.
        t_hold: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).

    Returns:
        XXGridScanResult with the sensitivity curve.
    """
    ops = build_two_qubit_operators()
    alpha_vals = np.linspace(alpha_xx_range[0], alpha_xx_range[1], n_points)
    delta_vals = np.full(n_points, np.inf, dtype=float)

    for i, a_val in enumerate(alpha_vals):
        domega = compute_xx_sensitivity(
            DEFAULT_PSI0,
            T_BS,
            t_hold,
            omega,
            a_val,
            ops,
        )
        delta_vals[i] = domega

    # Find the finite minimum
    finite_mask = np.isfinite(delta_vals)
    if np.any(finite_mask):
        best_idx = int(np.argmin(delta_vals[finite_mask]))
        # Need to map back to original indices
        finite_indices = np.where(finite_mask)[0]
        best_orig_idx = finite_indices[best_idx]
        alpha_opt = float(alpha_vals[best_orig_idx])
        delta_opt = float(delta_vals[best_orig_idx])
    else:
        alpha_opt = float("nan")
        delta_opt = float("inf")

    # Get expectation and variance at the optimal point
    exp_val = 0.0
    var_val = 0.0
    if np.isfinite(alpha_opt):
        psi = evolve_xx_circuit(DEFAULT_PSI0, T_BS, t_hold, omega, alpha_opt, ops)
        meas_op = ops["Jz_S"]
        var_val = compute_reduced_variance(psi, meas_op)
        # Also compute expectation using the reduced state
        psi_m = psi.reshape(2, 2)
        rho = psi_m @ psi_m.conj().T
        Jz_sys = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
        exp_val = float(np.real(np.trace(rho @ Jz_sys)))

    return XXGridScanResult(
        alpha_xx_values=alpha_vals,
        delta_omega_values=delta_vals,
        omega_value=omega,
        alpha_xx_opt=alpha_opt,
        delta_omega_opt=delta_opt,
        sql=1.0 / t_hold,
        expectation_Jz=exp_val,
        variance_Jz=var_val,
    )


# ============================================================================
# ω Scan: Grid scan at each ω
# ============================================================================


def run_xx_omega_scan(
    omega_values: list[float] | np.ndarray,
    alpha_xx_range: tuple[float, float] = AXX_BOUNDS,
    n_points: int = N_GRID_POINTS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> XXOmegaScanResult:
    """Scan over ω values with full α_xx grid scan at each ω.

    For each ω:
    1. Run a dense 1D α_xx grid scan.
    2. Record the optimal α_xx and Δω.
    3. Record how many grid points fall below SQL.

    Args:
        omega_values: ω values to scan.
        alpha_xx_range: (min, max) for α_xx.
        n_points: Number of α_xx grid points per ω.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        XXOmegaScanResult with optimal parameters and sensitivities.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    n_omega = len(omega_arr)

    alpha_opts = np.full(n_omega, np.nan, dtype=float)
    best_deltas = np.full(n_omega, np.inf, dtype=float)
    sql_vals = np.full(n_omega, 1.0 / t_hold, dtype=float)
    exp_vals = np.zeros(n_omega, dtype=float)
    var_vals = np.zeros(n_omega, dtype=float)
    count_below = np.zeros(n_omega, dtype=float)
    total_pts = np.full(n_omega, n_points, dtype=float)

    for i, omega in enumerate(omega_arr):
        result = xx_grid_scan(
            omega=omega,
            alpha_xx_range=alpha_xx_range,
            n_points=n_points,
            t_hold=t_hold,
            T_BS=T_BS,
        )
        alpha_opts[i] = result.alpha_xx_opt
        best_deltas[i] = result.delta_omega_opt
        exp_vals[i] = result.expectation_Jz
        var_vals[i] = result.variance_Jz
        count_below[i] = float(np.sum(result.delta_omega_values < result.sql))

    return XXOmegaScanResult(
        omega_values=omega_arr,
        alpha_xx_opt_per_omega=alpha_opts,
        delta_omega_opt_per_omega=best_deltas,
        sql_values=sql_vals,
        expectation_Jz_per_omega=exp_vals,
        variance_Jz_per_omega=var_vals,
        count_below_sql_per_omega=count_below,
        total_points_per_omega=total_pts,
    )


# ============================================================================
# Exclusive Plot Functions
# ============================================================================


def plot_xx_omega_scan(
    result: XXOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δω vs ω with optimal α_xx as a secondary axis or annotation.

    Args:
        result: XXOmegaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    omega = result.omega_values
    sql_vals = result.sql_values
    best_deltas = result.delta_omega_opt_per_omega

    # SQL reference line
    sql_ref = float(sql_vals[0]) if len(sql_vals) > 0 else 0.1
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
        "XX-Coupling Sensitivity vs $\\omega$:\n"
        "Optimal $\\Delta\\omega$ over $\\alpha_{xx} \\in [0, 20]$"
    )

    # Secondary axis: optimal α_xx
    ax2 = ax1.twinx()
    alpha_opts_valid = result.alpha_xx_opt_per_omega
    valid_alpha = np.isfinite(alpha_opts_valid)
    if np.any(valid_alpha):
        ax2.plot(
            omega[valid_alpha],
            alpha_opts_valid[valid_alpha],
            "s--",
            color="C1",
            markersize=5,
            linewidth=1.2,
            alpha=0.7,
            label=r"$\alpha_{xx}^*$",
        )
    ax2.set_ylabel(r"$\alpha_{xx}^*$", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_xx_optimal_params(
    result: XXOmegaScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> Path:
    """Plot optimal α_xx and the fraction below SQL vs ω.

    Args:
        result: XXOmegaScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    omega = result.omega_values

    # Left panel: optimal α_xx vs ω
    valid_alpha = np.isfinite(result.alpha_xx_opt_per_omega)
    if np.any(valid_alpha):
        ax1.plot(
            omega[valid_alpha],
            result.alpha_xx_opt_per_omega[valid_alpha],
            "s-",
            color="C1",
            markersize=6,
            linewidth=1.5,
        )
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\alpha_{xx}^*$")
    ax1.set_title(r"Optimal $\alpha_{xx}$ vs $\omega$")

    # Right panel: fraction below SQL vs ω
    valid_frac = result.total_points_per_omega > 0
    if np.any(valid_frac):
        fractions = result.count_below_sql_per_omega / result.total_points_per_omega
        ax2.plot(
            omega[valid_frac],
            fractions[valid_frac],
            "o-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel("Fraction below SQL")
    ax2.set_title("Fraction of $\\alpha_{xx}$ grid below SQL")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_xx_grid_scan_example(
    result: XXGridScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 5),
) -> Path:
    """Plot the Δω vs α_xx curve for a single ω value.

    Args:
        result: XXGridScanResult for a single ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    alpha = result.alpha_xx_values
    delta = result.delta_omega_values

    sql_ref = result.sql

    # SQL reference
    ax.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_ref:.4f}",
    )

    # Finite points
    finite_mask = np.isfinite(delta)
    if np.any(finite_mask):
        ax.plot(
            alpha[finite_mask],
            delta[finite_mask],
            "-",
            color="C0",
            linewidth=1.2,
            label=rf"$\Delta\omega$ at $\omega={result.omega_value}$",
        )

    # Mark optimum
    if np.isfinite(result.alpha_xx_opt) and np.isfinite(result.delta_omega_opt):
        ax.plot(
            result.alpha_xx_opt,
            result.delta_omega_opt,
            "D",
            color="red",
            markersize=8,
            label=rf"Optimum: $\alpha_{{xx}}^*={result.alpha_xx_opt:.3f}$, "
            rf"$\Delta\omega={result.delta_omega_opt:.5f}$",
        )

    ax.set_xlabel(r"$\alpha_{xx}$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(rf"$\Delta\omega$ vs $\alpha_{{xx}}$ at $\omega={result.omega_value}$")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
XX_DATE = "20260520"
XX_OMEGA_VALS = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
XX_N_GRID = 2001


def _parquet_path(name: str) -> Path:
    return parquet_path(REPORTS_DIR, XX_DATE, name)


def _fig_path(name: str) -> Path:
    return fig_path(REPORTS_DIR, XX_DATE, name)


# ── Generator functions ───────────────────────────────────────────────────


def generate_xx_decoupled_baseline(force: bool = False) -> None:
    """XX-coupling decoupled baseline verification."""
    csv_p = _parquet_path("xx-decoupled-baseline")
    fig_p = _fig_path("xx-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing XX-coupling decoupled baseline...")
        result = compute_xx_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Use the shared plot function from the main codebase
    try:
        from src.visualization.ancilla_drive_plots import plot_drive_decoupled_baseline

        plot_drive_decoupled_baseline(result, fig_p)
        print(f"[fig]  {fig_p}")
    except ImportError:
        # Fallback: create a simple text figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        text = (
            f"Decoupled Baseline (α_xx = 0)\n"
            f"Δω = {result.delta_omega:.10f}\n"
            f"SQL = {result.sql:.10f}\n"
            f"Ratio = {result.delta_omega / result.sql:.6f}"
        )
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_p, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[fig]  {fig_p}")


def _run_xx_grid_scan(omega: float, force: bool) -> None:
    """Run an XX-coupling grid scan for a single ω value."""
    tag = f"xx-grid-scan-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = XXGridScanResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing XX grid scan at ω={omega} ({XX_N_GRID} points)...")
        result = xx_grid_scan(
            omega=omega,
            n_points=XX_N_GRID,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    plot_xx_grid_scan_example(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_xx_grid_scans(force: bool = False) -> None:
    """XX-coupling grid scans at all ω values (parallel)."""
    n = len(XX_OMEGA_VALS)
    print(f"[run]  XX grid scans at {n} ω values (parallel)")
    worker = partial(_run_xx_grid_scan, force=force)
    parallel_map(worker, XX_OMEGA_VALS, desc="grid scans")


def generate_xx_omega_scan(force: bool = False) -> None:
    """XX-coupling ω-scan with α_xx optimisation at each ω."""
    csv_p = _parquet_path("xx-omega-scan")
    fig_p = _fig_path("xx-omega-scan")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = XXOmegaScanResult.from_parquet(csv_p)
    else:
        n = len(XX_OMEGA_VALS)
        print(f"[run]  Computing XX ω-scan for {n} ω values ({XX_N_GRID} pts each)...")
        result = run_xx_omega_scan(
            omega_values=XX_OMEGA_VALS,
            n_points=XX_N_GRID,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_xx_omega_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_xx_optimal_params(force: bool = False) -> None:
    """Optimal parameter evolution (α_xx and fraction below SQL) vs ω."""
    csv_p = _parquet_path("xx-omega-scan")
    fig_p = _fig_path("xx-optimal-params")

    if not csv_p.exists():
        print("[skip] xx-omega-scan.parquet does not exist; run 'xx-omega-scan' first")
        return

    result = XXOmegaScanResult.from_parquet(csv_p)
    plot_xx_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-20 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'xx-decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / XX_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / XX_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "xx-decoupled-baseline": generate_xx_decoupled_baseline,
        "xx-grid-scans": generate_xx_grid_scans,
        "xx-omega-scan": generate_xx_omega_scan,
        "xx-optimal-params": generate_xx_optimal_params,
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
