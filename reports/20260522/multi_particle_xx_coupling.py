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
    uv run python reports/20260522/multi_particle_xx_coupling.py --force

This module is importable via ``importlib.import_module("reports.20260522.multi_particle_xx_coupling")``.
"""

from __future__ import annotations

__all__ = [
    "ScalingAnalysisResult",
    "build_hold_hamiltonian",
    "compute_reduced_expectation_and_variance",
    "dual_bs_unitary",
    "evolve_circuit",
    "fit_scaling_exponents",
    "hold_unitary_dicke",
    "single_bs_unitary",
]

import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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

from src.analysis.coupling_sweep import resolve_sweep_defaults, run_sweep_base
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
    plot_decoupled_baseline_heatmap,
)
from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,
    fit_scaling_exponents,
    generate_scaling_analysis,
)
from src.analysis.n_scaling_sweep import (
    generate_n_scaling_plots,
)
from src.physics.multi_mzi import (
    build_hold_hamiltonian,
    compute_multi_particle_sensitivity,
    compute_reduced_expectation_and_variance,
    dual_bs_unitary,
    embed_combined_operators,
    evolve_circuit,
    hold_unitary_dicke,
    single_bs_unitary,
)
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable
from src.visualization.coupling_heatmaps import (
    plot_alpha_opt_heatmap,
    plot_ratio_heatmap,
)

# Re-export shared compute_sensitivity for backward compatibility with tests.
# The shared function (compute_multi_particle_sensitivity) takes the same
# signature: (N, psi0, omega_true, alpha_xx, ops, meas_op, fd_step, T_BS, t_hold).
compute_sensitivity = compute_multi_particle_sensitivity

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
class DualMZISweepResult(ParquetSerializable):
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

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
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
    ]

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

    @classmethod
    def from_parquet(cls, path: str | Path) -> DualMZISweepResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)

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
    omega_values, N_values = resolve_sweep_defaults(
        omega_values,
        N_values,
        default_omega_range=(OMEGA_MIN, OMEGA_MAX, OMEGA_STEP),
        default_N_range=(N_MIN, N_MAX),
    )

    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, float]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        r = optimise_alpha_xx(
            N=N, omega=omega, ops=_cache["ops"], psi0=_cache["psi0"], t_hold=t_hold
        )
        return {
            "delta_omega": r["delta_omega_opt"],
            "sql": r["sql"],
            "expectation_Jz": r["expectation_Jz"],
            "variance_Jz": r["variance_Jz"],
            "d_expectation": r["d_expectation"],
            "alpha_xx_opt": r["alpha_xx_opt"],
        }

    data = run_sweep_base(
        omega_values, N_values, per_point, progress_callback=progress_callback
    )
    return DualMZISweepResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        alpha_xx_opt=data["alpha_xx_opt"],
        delta_omega_opt=data["delta_opts"],
        sql_values=data["sqls"],
        ratio=data["ratio"],
        expectation_Jz=data["expectation_Jz"],
        variance_Jz=data["variance_Jz"],
        d_expectation=data["d_expectation"],
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
    omega_values, N_values = resolve_sweep_defaults(
        omega_values,
        N_values,
        default_omega_range=(OMEGA_MIN, OMEGA_MAX, OMEGA_STEP),
        default_N_range=(N_MIN, N_MAX),
    )

    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, float]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        dt, exp_val, var_val, d_exp_val = compute_sensitivity(
            N, _cache["psi0"], omega, 0.0, _cache["ops"], t_hold=t_hold, fd_step=fd_step
        )
        sql = 1.0 / (np.sqrt(N) * t_hold)
        return {
            "delta_omega": dt,
            "sql": sql,
            "expectation_Jz": exp_val,
            "variance_Jz": var_val,
            "d_expectation": d_exp_val,
            "alpha_xx_opt": 0.0,
        }

    data = run_sweep_base(omega_values, N_values, per_point)
    return DualMZISweepResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        alpha_xx_opt=data["alpha_xx_opt"],
        delta_omega_opt=data["delta_opts"],
        sql_values=data["sqls"],
        ratio=data["ratio"],
        expectation_Jz=data["expectation_Jz"],
        variance_Jz=data["variance_Jz"],
        d_expectation=data["d_expectation"],
        t_hold=t_hold,
    )


# ============================================================================
# Scaling Analysis
# ============================================================================


# ============================================================================
# Plot Functions
# ============================================================================


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


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260522"
OMEGA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
N_VALS: list[int] = list(range(1, 21))


parquet_path, fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)

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

    plot_ratio_heatmap(
        result.omega_values,
        result.N_values,
        result.ratio,
        fig_ratio,
        title="Sensitivity Ratio: Dual-MZI XX Coupling\n(lower = better; 1.0 = SQL)",
    )
    print(f"[fig]  {fig_ratio}")
    plot_alpha_opt_heatmap(
        result.omega_values,
        result.N_values,
        result.alpha_xx_opt,
        fig_alpha,
        vmin=0.0,
        cbar_label=r"$\alpha_{xx}^*$",
        title=r"Optimal $\alpha_{xx}$: Dual-MZI XX Coupling",
    )
    print(f"[fig]  {fig_alpha}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling analysis from the sweep data."""
    omega_fig_pairs = [
        (0.3, _fig_path("dual-mzi-n-scaling-omega0.3")),
        (1.0, _fig_path("dual-mzi-n-scaling-omega1.0")),
        (3.0, _fig_path("dual-mzi-n-scaling-omega3.0")),
    ]
    generate_n_scaling_plots(
        force=force,
        parquet_path=_parquet_path("dual-mzi-sweep"),
        result_cls=DualMZISweepResult,
        omega_fig_pairs=omega_fig_pairs,
        label="dual-mzi n-scaling",
    )


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

    tasks: dict[str, Callable[..., Any]] = {
        "decoupled-baseline": lambda force=False: generate_decoupled_baseline(
            force=force,
            parquet_path=_parquet_path("dual-mzi-decoupled-baseline"),
            fig_path=_fig_path("dual-mzi-decoupled-baseline"),
            compute_fn=compute_decoupled_baseline,
            compute_kwargs={
                "omega_values": np.array(
                    [v for i, v in enumerate(OMEGA_VALS) if i % 5 == 0]
                ),
                "N_values": np.array(N_VALS, dtype=int),
            },
            result_cls=DualMZISweepResult,
            plot_fn=lambda r, p: plot_decoupled_baseline_heatmap(
                r,
                p,
                title_prefix=(r"Decoupled Baseline Verification ($\alpha_{xx} = 0$"),
            ),
            label="decoupled baseline (α_xx = 0)",
        ),
        "sweep": generate_sweep,
        "n-scaling": generate_n_scaling,
        "omega-dependence": generate_omega_dependence,
        "scaling-analysis": partial(
            generate_scaling_analysis,
            parquet_path=_parquet_path("dual-mzi-sweep"),
            scaling_path=_parquet_path("dual-mzi-scaling"),
            fig_path=_fig_path("dual-mzi-scaling-exponents"),
            result_cls=DualMZISweepResult,
            label="dual-mzi scaling analysis",
        ),
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
