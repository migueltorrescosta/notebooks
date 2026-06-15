"""
Local module for the 2026-05-22 Multi-Particle XX-Coupling Dual-MZI report.

Contains all code exclusive to this report:
- Multi-particle operator construction (Dicke basis, N up to 20)
- Dual MZI circuit: BS(S)⊗BS(A) → Hold → BS(S)⊗BS(A) → Tr_A → measure J_z^S
- Sensitivity via error propagation with central finite differences
- α_xx optimisation (coarse grid + bounded 1D refinement) per (ω, N) pair
- Full 2D sweep over ω ∈ [0.1, 5.0] and N ∈ [1, 20]
- Scaling analysis (log-log fit Δω ∝ N^α)
- Exclusive plot functions for heatmaps and scaling curves
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260522/local.py --force

This module is **not** importable as ``reports.20260522.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize_scalar

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,
    fit_scaling_exponents,
)
from src.physics.dicke_basis import jz_operator
from src.physics.multi_mzi import (
    build_hold_hamiltonian,  # noqa: F401 — re-exported for tests
    compute_reduced_expectation_and_variance,
    dual_bs_unitary,  # noqa: F401 — re-exported for tests
    embed_combined_operators,
    evolve_circuit,
    hold_unitary_dicke,  # noqa: F401 — re-exported for tests
    single_bs_unitary,  # noqa: F401 — re-exported for tests
)
from src.utils.enums import OperatorBasis

if TYPE_CHECKING:
    from collections.abc import Callable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL reference)
AXX_BOUNDS: tuple[float, float] = (0.0, 20.0)  # α_xx optimisation range
N_COARSE_GRID: int = 101  # Coarse grid points for α_xx scan
FD_STEP: float = 1e-6  # Central finite-difference step

# ω and N sweep ranges (from report)
OMEGA_MIN: float = 0.1
OMEGA_MAX: float = 5.0
OMEGA_STEP: float = 0.1
N_MIN: int = 1
N_MAX: int = 20


# ============================================================================
# Circuit Evolution
# ============================================================================


def initial_state(N: int) -> np.ndarray:
    """Create the initial product state |J,J⟩_S ⊗ |J,J⟩_A.

    This is the first computational basis vector of length (N+1)².

    Args:
        N: Particle number per subsystem.

    Returns:
        Normalised (N+1)²-vector.
    """
    dim = (N + 1) ** 2
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    return psi


# ============================================================================
# Ancilla Trace-Out and Reduced Variance
# ============================================================================


# _reduced_expectation replaced by compute_reduced_expectation_and_variance from src


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_sensitivity(
    N: int,
    psi0: np.ndarray,
    omega_true: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
    T_BS: float = DEFAULT_T_BS,
    t_hold: float = DEFAULT_t_hold,
) -> tuple[float, float, float, float]:
    """Compute the error-propagation sensitivity Δω.

    Δω = √Var(J_z^S) / |∂⟨J_z^S⟩/∂ω|

    Also returns ⟨J_z^S⟩, Var(J_z^S), and ∂⟨J_z^S⟩/∂ω at omega_true.

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector.
        omega_true: True phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        meas_op: (N+1)×(N+1) measurement operator (default = J_z^S single).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_circuit(N, psi0, omega_true, alpha_xx, ops, T_BS, t_hold)
    # Use single-subsystem J_z for the reduced expectation/variance
    Jz_single = jz_operator(N, basis=OperatorBasis.DICKE)
    exp_val, var_val = compute_reduced_expectation_and_variance(psi, N, Jz_single)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_circuit(
        N, psi0, omega_true + fd_step, alpha_xx, ops, T_BS, t_hold
    )
    psi_minus = evolve_circuit(
        N, psi0, omega_true - fd_step, alpha_xx, ops, T_BS, t_hold
    )

    exp_plus, _ = compute_reduced_expectation_and_variance(psi_plus, N, Jz_single)
    exp_minus, _ = compute_reduced_expectation_and_variance(psi_minus, N, Jz_single)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_omega = float(np.sqrt(var_val) / abs(d_exp))
    return delta_omega, exp_val, var_val, d_exp


# ============================================================================
# α_xx Optimisation (Coarse Grid + Bounded 1D Refinement)
# ============================================================================


def optimise_alpha_xx(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    axx_bounds: tuple[float, float] = AXX_BOUNDS,
    n_coarse: int = N_COARSE_GRID,
    T_BS: float = DEFAULT_T_BS,
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
) -> dict[str, float]:
    """Optimise Δω over α_xx for a given (ω, N) pair.

    Two-stage approach:
    Stage 1: Evaluate Δω on a coarse grid of n_coarse points.
    Stage 2: Bounded 1D refinement via scipy.optimize.minimize_scalar,
             seeded at the best grid point.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        axx_bounds: (min, max) for α_xx.
        n_coarse: Number of coarse grid points.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Dict with keys:
            'alpha_xx_opt': optimal α_xx value.
            'delta_omega_opt': minimal Δω.
            'expectation_Jz': ⟨J_z^S⟩ at optimum.
            'variance_Jz': Var(J_z^S) at optimum.
            'd_expectation': ∂⟨J_z^S⟩/∂ω at optimum.
            'sql': SQL = 1/(√N * t_hold) reference.
    """
    if psi0 is None:
        psi0 = initial_state(N)

    sql = 1.0 / (np.sqrt(N) * t_hold)

    # Stage 1: coarse grid
    alpha_vals = np.linspace(axx_bounds[0], axx_bounds[1], n_coarse)
    delta_vals = np.full(n_coarse, np.inf, dtype=float)

    for i, a_val in enumerate(alpha_vals):
        dt, _, _, _ = compute_sensitivity(N, psi0, omega, a_val, ops, fd_step=fd_step)
        delta_vals[i] = dt

    # Find best grid point
    finite_mask = np.isfinite(delta_vals)
    if not np.any(finite_mask):
        return {
            "alpha_xx_opt": float("nan"),
            "delta_omega_opt": float("inf"),
            "expectation_Jz": 0.0,
            "variance_Jz": 0.0,
            "d_expectation": 0.0,
            "sql": sql,
        }

    finite_indices = np.where(finite_mask)[0]
    best_idx = finite_indices[int(np.argmin(delta_vals[finite_mask]))]
    seed_alpha = float(alpha_vals[best_idx])

    # Stage 2: bounded 1D refinement
    def _objective(a: float) -> float:
        dt, _, _, _ = compute_sensitivity(N, psi0, omega, a, ops, fd_step=fd_step)
        return dt if np.isfinite(dt) else 1e10

    result = minimize_scalar(
        _objective,
        bounds=axx_bounds,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 100},
    )

    alpha_opt = float(result.x)
    delta_opt = float(result.fun)

    # Re-evaluate at optimum for expectation and variance
    dt_opt, exp_opt, var_opt, d_exp = compute_sensitivity(
        N, psi0, omega, alpha_opt, ops, fd_step=fd_step
    )

    # If the refine result is worse than the grid, prefer the grid result
    if delta_opt > delta_vals[best_idx]:
        alpha_opt = seed_alpha
        dt_opt, exp_opt, var_opt, d_exp = compute_sensitivity(
            N, psi0, omega, alpha_opt, ops, fd_step=fd_step
        )
        delta_opt = dt_opt

    return {
        "alpha_xx_opt": alpha_opt,
        "delta_omega_opt": delta_opt,
        "expectation_Jz": exp_opt,
        "variance_Jz": var_opt,
        "d_expectation": d_exp,
        "sql": sql,
    }


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class DualMZISweepResult:
    """Full 2D sweep over ω and N with α_xx optimisation per point.

    All array fields have the same length (n_omega × n_N), stored in
    row-major order (ω varies fastest, then N).

    Attributes:
        omega_values: ω values for each point.
        N_values: N values for each point.
        alpha_xx_opt: Optimal α_xx at each point.
        delta_omega_opt: Minimal Δω at each point.
        sql_values: SQL = 1/(√N t_hold) at each point.
        ratio: Δω_opt / SQL at each point.
        expectation_Jz: ⟨J_z^S⟩ at optimum.
        variance_Jz: Var(J_z^S) at optimum.
        d_expectation: ∂⟨J_z^S⟩/∂ω at optimum.
        t_hold: Holding time (scalar).
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    alpha_xx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    d_expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    t_hold: float = DEFAULT_t_hold

    def __post_init__(self) -> None:
        # Ensure int dtype for N_values
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "N": self.N_values,
                "t_hold": np.full(len(self.omega_values), self.t_hold),
                "alpha_xx_opt": self.alpha_xx_opt,
                "delta_omega_opt": self.delta_omega_opt,
                "sql": self.sql_values,
                "ratio": self.ratio,
                "expectation_Jz": self.expectation_Jz,
                "variance_Jz": self.variance_Jz,
                "d_expectation": self.d_expectation,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DualMZISweepResult:
        df = pd.read_parquet(path)
        required = {
            "omega",
            "N",
            "t_hold",
            "alpha_xx_opt",
            "delta_omega_opt",
            "sql",
            "ratio",
            "expectation_Jz",
            "variance_Jz",
            "d_expectation",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )

        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            N_values=df["N"].to_numpy(dtype=int),
            alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
            delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
            ratio=df["ratio"].to_numpy(dtype=float),
            expectation_Jz=df["expectation_Jz"].to_numpy(dtype=float),
            variance_Jz=df["variance_Jz"].to_numpy(dtype=float),
            d_expectation=df["d_expectation"].to_numpy(dtype=float),
            t_hold=float(df["t_hold"].iloc[0]),
        )

    @property
    def n_points(self) -> int:
        return len(self.omega_values)

    @property
    def n_omega_unique(self) -> int:
        return len(np.unique(self.omega_values))

    @property
    def n_N_unique(self) -> int:
        return len(np.unique(self.N_values))


# ============================================================================
# Sweep Execution
# ============================================================================


def run_sweep(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    t_hold: float = DEFAULT_t_hold,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DualMZISweepResult:
    """Run the full 2D sweep over ω and N with α_xx optimisation.

    Args:
        omega_values: ω values to sweep (default: 0.1 to 5.0 step 0.1).
        N_values: N values to sweep (default: 1 to 20 inclusive).
        t_hold: Holding time.
        progress_callback: Optional callback (current, total).

    Returns:
        DualMZISweepResult with all optimised points.
    """
    if omega_values is None:
        omega_values = np.arange(OMEGA_MIN, OMEGA_MAX + 1e-9, OMEGA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N

    omegas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    alpha_opts = np.full(total, np.nan, dtype=float)
    delta_opts = np.full(total, np.inf, dtype=float)
    sqls = np.zeros(total, dtype=float)
    ratios = np.full(total, np.inf, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)

    idx = 0
    for N in N_values:
        ops = embed_combined_operators(N)
        psi0 = initial_state(N)
        for omega in omega_values:
            omegas[idx] = omega
            Ns[idx] = N

            opt_result = optimise_alpha_xx(
                N=N,
                omega=omega,
                ops=ops,
                psi0=psi0,
                t_hold=t_hold,
            )
            alpha_opts[idx] = opt_result["alpha_xx_opt"]
            delta_opts[idx] = opt_result["delta_omega_opt"]
            sqls[idx] = opt_result["sql"]
            ratios[idx] = (
                opt_result["delta_omega_opt"] / opt_result["sql"]
                if np.isfinite(opt_result["delta_omega_opt"]) and opt_result["sql"] > 0
                else float("inf")
            )
            exps[idx] = opt_result["expectation_Jz"]
            vars_[idx] = opt_result["variance_Jz"]
            d_exps[idx] = opt_result["d_expectation"]

            idx += 1
            if progress_callback is not None:
                progress_callback(idx, total)

    return DualMZISweepResult(
        omega_values=omegas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        delta_omega_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        t_hold=t_hold,
    )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def compute_decoupled_baseline(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
) -> DualMZISweepResult:
    """Verify the decoupled baseline (α_xx = 0) for all (ω, N) pairs.

    At α_xx = 0, the sensitivity should equal SQL = 1/(√N t_hold).

    Args:
        omega_values: ω values (default: sweep range).
        N_values: N values (default: 1 to 20).
        t_hold: Holding time.

    Returns:
        DualMZISweepResult with α_xx=0 results.
    """
    if omega_values is None:
        omega_values = np.arange(OMEGA_MIN, OMEGA_MAX + 1e-9, OMEGA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N

    omegas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    sqls = np.zeros(total, dtype=float)
    delta_opts = np.zeros(total, dtype=float)
    alpha_opts = np.zeros(total, dtype=float)  # all zero
    ratios = np.zeros(total, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)

    idx = 0
    for N in N_values:
        ops = embed_combined_operators(N)
        psi0 = initial_state(N)
        for omega in omega_values:
            omegas[idx] = omega
            Ns[idx] = N
            sql = 1.0 / (np.sqrt(N) * t_hold)
            sqls[idx] = sql

            dt, exp_val, var_val, d_exp_val = compute_sensitivity(
                N, psi0, omega, 0.0, ops, t_hold=t_hold, fd_step=fd_step
            )
            delta_opts[idx] = dt
            ratios[idx] = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
            exps[idx] = exp_val
            vars_[idx] = var_val
            d_exps[idx] = d_exp_val

            idx += 1

    return DualMZISweepResult(
        omega_values=omegas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        delta_omega_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        t_hold=t_hold,
    )


# ============================================================================
# Scaling Analysis
# ============================================================================


# ============================================================================
# Plot Functions
# ============================================================================


def plot_ratio_heatmap(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of Δω_opt / SQL ratio across (ω, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(sweep.omega_values)
    N_vals = np.unique(sweep.N_values)
    ratio_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                ratio_map[j, i] = float(sweep.ratio[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(0.0, float(np.nanmin(ratio_map)))
    vmax = max(2.0, float(np.nanmax(ratio_map[ratio_map < 10])))

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        ratio_map,
        shading="nearest",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(
        im, ax=ax, label=r"$\Delta\omega_{\mathrm{opt}} / \Delta\omega_{\mathrm{SQL}}$"
    )
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title("Sensitivity Ratio: Dual-MZI XX Coupling\n(lower = better; 1.0 = SQL)")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alpha_opt_heatmap(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal α_xx across (ω, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(sweep.omega_values)
    N_vals = np.unique(sweep.N_values)
    alpha_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                alpha_map[j, i] = float(sweep.alpha_xx_opt[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(alpha_map)) if np.any(np.isfinite(alpha_map)) else 20.0

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        alpha_map,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=r"$\alpha_{xx}^*$")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(r"Optimal $\alpha_{xx}$: Dual-MZI XX Coupling")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    omega_fixed: float | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δω_opt vs N at a fixed ω, with SQL and HL reference lines.

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        omega_fixed: ω value to plot. If None, uses the first ω.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if omega_fixed is None:
        omega_fixed = float(np.unique(sweep.omega_values)[0])

    mask = np.isclose(sweep.omega_values, omega_fixed)
    N_vals = sweep.N_values[mask].astype(float)
    delta_vals = sweep.delta_omega_opt[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference: Δω = 1/(√N t_hold)
    N_dense = np.logspace(np.log10(1), np.log10(20), 100)
    sql_dense = 1.0 / (np.sqrt(N_dense) * sweep.t_hold)
    hl_dense = 1.0 / (N_dense * sweep.t_hold)

    ax.loglog(N_dense, sql_dense, "--", color="gray", alpha=0.7, label="SQL")
    ax.loglog(N_dense, hl_dense, ":", color="gray", alpha=0.5, label="HL")

    # Data
    finite_mask = np.isfinite(delta_vals) & (delta_vals > 0)
    if np.any(finite_mask):
        ax.loglog(
            N_vals[finite_mask],
            delta_vals[finite_mask],
            "o-",
            color="C0",
            markersize=8,
            linewidth=1.8,
            label=rf"$\Delta\omega_{{\mathrm{{opt}}}}(\omega={omega_fixed:.2f})$",
        )

    ax.set_xlabel(r"$N$ (particles per subsystem)")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(f"N-Scaling at $\\omega={omega_fixed:.2f}$:\nDual-MZI XX Coupling")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_omega_dependence(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    N_fixed: int | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δω_opt vs ω at fixed N, with SQL reference line.

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        N_fixed: N value to plot. If None, uses the first N.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if N_fixed is None:
        N_fixed = int(np.unique(sweep.N_values)[0])

    mask = sweep.N_values == N_fixed
    omega_vals = sweep.omega_values[mask]
    delta_vals = sweep.delta_omega_opt[mask]
    sql_vals = sweep.sql_values[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference (flat line for fixed N)
    sql_val = 1.0 / (np.sqrt(N_fixed) * sweep.t_hold) if len(sql_vals) > 0 else 0.1
    ax.axhline(
        y=sql_val,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_val:.5f}",
    )

    finite_mask = np.isfinite(delta_vals)
    if np.any(finite_mask):
        ax.plot(
            omega_vals[finite_mask],
            delta_vals[finite_mask],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.5,
            label=rf"$\Delta\omega_{{\mathrm{{opt}}}}(N={N_fixed})$",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(f"$\\omega$-Dependence at $N={N_fixed}$:\nDual-MZI XX Coupling")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling_exponents(
    scaling: ScalingAnalysisResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot the scaling exponent α vs ω from log-log fits.

    Args:
        scaling: Scaling analysis result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: exponent vs ω
    valid_exp = np.isfinite(scaling.exponents)
    if np.any(valid_exp):
        ax1.plot(
            scaling.omega_values[valid_exp],
            scaling.exponents[valid_exp],
            "o-",
            color="C1",
            markersize=6,
            linewidth=1.5,
        )
    ax1.axhline(
        y=-0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="SQL (α = −0.5)",
    )
    ax1.axhline(
        y=-1.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="HL (α = −1.0)",
    )
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"Scaling exponent $\alpha$")
    ax1.set_title("Exponent $\\alpha$ from\n$\\Delta\\omega = C N^{\\alpha}$")
    ax1.legend(fontsize=9)

    # Right: R² vs ω
    valid_r2 = np.isfinite(scaling.r_squared)
    if np.any(valid_r2):
        ax2.plot(
            scaling.omega_values[valid_r2],
            scaling.r_squared[valid_r2],
            "s-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_title("Goodness of Fit")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260522"
OMEGA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
N_VALS: list[int] = list(range(1, 21))


def parquet_path(name: str) -> Path:
    """Return path to a raw_data Parquet file for this report."""
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def fig_path(name: str) -> Path:
    """Return path to a figures SVG file for this report."""
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# Backward-compatible aliases
_parquet_path = parquet_path
_fig_path = fig_path


# ── Generator Functions ────────────────────────────────────────────────


def generate_sweep(force: bool = False) -> None:
    """Run the full ω × N sweep with α_xx optimisation."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_ratio = _fig_path("dual-mzi-ratio-heatmap")
    fig_alpha = _fig_path("dual-mzi-alpha-opt-heatmap")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZISweepResult.from_parquet(csv_p)
    else:
        print(
            "[run]  Computing dual-MZI ω×N sweep "
            f"({len(OMEGA_VALS)}×{len(N_VALS)} = {len(OMEGA_VALS) * len(N_VALS)} points)..."
        )
        omega_arr = np.array(OMEGA_VALS, dtype=float)
        N_arr = np.array(N_VALS, dtype=int)
        result = run_sweep(omega_values=omega_arr, N_values=N_arr)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio)
    print(f"[fig]  {fig_ratio}")
    plot_alpha_opt_heatmap(result, fig_alpha)
    print(f"[fig]  {fig_alpha}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline (α_xx = 0) verification."""
    csv_p = _parquet_path("dual-mzi-decoupled-baseline")
    fig_p = _fig_path("dual-mzi-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZISweepResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline (α_xx = 0)...")
        N_arr = np.array(N_VALS, dtype=int)
        # Use a subset for speed (every 5th ω, all N)
        omega_subset = np.array([v for i, v in enumerate(OMEGA_VALS) if i % 5 == 0])
        result = compute_decoupled_baseline(
            omega_values=omega_subset,
            N_values=N_arr,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Create a verification figure: heatmap of |ratio - 1| on log scale
    from matplotlib.colors import LogNorm

    omega_vals = np.unique(result.omega_values)
    N_vals = np.unique(result.N_values)
    dev_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(result.omega_values, omega) & (result.N_values == N_val)
            if np.any(mask):
                r = float(result.ratio[mask][0])
                dev_map[j, i] = abs(r - 1.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = dev_map[np.isfinite(dev_map)]
    if len(finite) > 0:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin, vmax = 1e-15, 1.0

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        dev_map,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=max(vmin, 1e-16), vmax=vmax),
    )
    fig.colorbar(im, ax=ax, label=r"$|\Delta\omega/\Delta\omega_{\mathrm{SQL}} - 1|$")

    max_dev = float(np.max(finite)) if len(finite) > 0 else 0.0
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(
        f"Decoupled Baseline Verification ($\\alpha_{{xx}} = 0$, $t_hold = {result.t_hold}$)\n"
        f"Max $|\\Delta\\omega/\\mathrm{{SQL}} - 1| = {max_dev:.2e}$, "
        f"points checked: {len(finite)}"
    )

    fig.tight_layout()
    fig.savefig(fig_p, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {fig_p}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling analysis from the sweep data."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_n3 = _fig_path("dual-mzi-n-scaling-omega0.3")
    fig_n1 = _fig_path("dual-mzi-n-scaling-omega1.0")
    fig_n3p = _fig_path("dual-mzi-n-scaling-omega3.0")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    # Plot at three representative ω values
    for omega_val, fig_p in [(0.3, fig_n3), (1.0, fig_n1), (3.0, fig_n3p)]:
        plot_n_scaling(result, fig_p, omega_fixed=omega_val)
        print(f"[fig]  {fig_p}")


def generate_omega_dependence(force: bool = False) -> None:
    """ω-dependence plots at fixed N values."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_n1 = _fig_path("dual-mzi-omega-N1")
    fig_n5 = _fig_path("dual-mzi-omega-N5")
    fig_n20 = _fig_path("dual-mzi-omega-N20")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    for N_fixed, fig_p in [(1, fig_n1), (5, fig_n5), (20, fig_n20)]:
        plot_omega_dependence(result, fig_p, N_fixed=N_fixed)
        print(f"[fig]  {fig_p}")


def generate_scaling_analysis(force: bool = False) -> None:
    """Scaling analysis (exponents) from the sweep data."""
    csv_p = _parquet_path("dual-mzi-sweep")
    scaling_csv = _parquet_path("dual-mzi-scaling")
    fig_p = _fig_path("dual-mzi-scaling-exponents")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    if scaling_csv.exists() and not force:
        scaling = ScalingAnalysisResult.from_parquet(scaling_csv)
        print(f"[skip] {scaling_csv.name} exists")
    else:
        print("[run]  Fitting scaling exponents...")
        scaling = fit_scaling_exponents(
            result.omega_values,
            result.N_values,
            result.delta_omega_opt,
        )
        scaling.save_parquet(scaling_csv)
        print(f"[save] {scaling_csv}")

    plot_scaling_exponents(scaling, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-22 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'sweep'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Callable[..., None]] = {
        "decoupled-baseline": generate_decoupled_baseline,
        "sweep": generate_sweep,
        "n-scaling": generate_n_scaling,
        "omega-dependence": generate_omega_dependence,
        "scaling-analysis": generate_scaling_analysis,
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
