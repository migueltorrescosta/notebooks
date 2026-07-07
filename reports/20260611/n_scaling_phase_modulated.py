"""
Local module for the 2026-06-11 N-Scaling of Phase-Modulated Ancilla Drive report.

Extends the ω-modulated ancilla drive mechanism (20260519) to N > 1 system
particles (J_S = N/2) while keeping the ancilla at J_A = 1/2.

Tests whether the 4.91× SQL-violation ratio at N=1 improves with N or saturates.

Operator construction:
- System: (N+1)-dimensional Dicke basis for N particles.
- Ancilla: 2-dimensional single-qubit space.
- Full space: 2(N+1) dimensions with basis ordering:
    {|m_S⟩_S ⊗ |0⟩_A, ... , |m_S⟩_S ⊗ |1⟩_A}  (m_S descending from +J_S to -J_S)

Circuit: BS_S → Hold → BS_S → measure J_z^S.

Usage:
    uv run python reports/20260611/n_scaling_phase_modulated.py --force
"""

from __future__ import annotations

__all__ = [
    "NScalingResult",
    "NScalingScanResult",
    "compute_n_particle_decoupled_baseline",
    "plot_n_scaling_optimal_params",
    "plot_n_scaling_ratio",
    "plot_n_scaling_sensitivity",
]

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd

from src.analysis.decoupled_baseline import (
    build_decoupled_baseline_df,
    generate_decoupled_baseline,
)
from src.analysis.n_scaling_result import (
    NScalingResult,
    NScalingScanResult,
)
from src.analysis.n_scaling_sweep import run_n_scaling_scan
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    build_nm_result,
    build_rs_result,
    run_two_phase_pipeline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_decoupled_baseline,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)
from src.utils.paths import report_path_fn
from src.visualization.scaling_plots import (
    plot_n_scaling_optimal_params,
    plot_n_scaling_ratio,
    plot_n_scaling_sensitivity,
)

# Local copies of shared constants (originally in src/physics/n_particle_drive.py)
T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Parameter bounds

# ω values for the scan
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values for the scaling scan (1 to 20)
N_VALS: list[int] = list(range(1, 21))

# Random search parameters
N_RANDOM: int = 500
N_NM_REFINE: int = 50
NM_MAXITER: int = 5000


# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260611"


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# N-Scaling Scan (Single (N, ω) Worker)
# ============================================================================


def _make_n_particle_objective(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Build the raw (unpenalised) Δω objective for a given (N, ω) pair."""

    def _raw_objective(p: np.ndarray) -> float:
        return compute_n_particle_sensitivity(
            N,
            psi0,
            T_BS,
            T_HOLD,
            omega,
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            ops,
            FD_STEP,
        )

    return _raw_objective


def run_single_n_omega(
    N: int,
    omega: float,
    n_random: int = N_RANDOM,
    n_nm_refine: int = N_NM_REFINE,
    seed: int | None = 42,
) -> NScalingResult:
    """Run the full optimisation pipeline for a single (N, ω) pair.

    1. 4D random search (n_random samples).
    2. Nelder-Mead refinement from top n_nm_refine points.
    3. Return the best result.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        n_random: Number of random search samples.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base random seed (incremented per call).

    Returns:
        NScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    raw_obj = _make_n_particle_objective(N, omega, ops, psi0)

    best_nm, _ = run_two_phase_pipeline(
        random_search_fn=lambda n_samples, seed, **kw: build_rs_result(
            raw_obj,
            n_samples,
            seed,
            omega=omega,
            sql=sql_reference(N),
            t_hold=T_HOLD,
        ),
        nm_fn=lambda x0, seed, **kw: build_nm_result(
            raw_obj,
            x0,
            omega=omega,
            ops=ops,
            psi0=psi0,
            evolve_fn=lambda psi, ax, ay, az, azz, _ops: evolve_n_particle_circuit(
                N,
                psi,
                T_BS,
                T_HOLD,
                omega,
                ax,
                ay,
                az,
                azz,
                _ops,
            ),
            t_hold=T_HOLD,
            maxiter=NM_MAXITER,
        ),
        config=TwoPhaseConfig(
            n_random=n_random, n_nm_refine=n_nm_refine, seed=base_seed
        ),
    )

    sql_val = sql_reference(N)
    return NScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql_val,
        ratio=sql_val / best_nm.delta_omega_opt
        if np.isfinite(best_nm.delta_omega_opt) and best_nm.delta_omega_opt > 0
        else float("nan"),
        a_x_opt=float(best_nm.params_opt[0]),
        a_y_opt=float(best_nm.params_opt[1]),
        a_z_opt=float(best_nm.params_opt[2]),
        a_zz_opt=float(best_nm.params_opt[3]),
        expectation_Jz=best_nm.expectation_Jz,
        variance_Jz=best_nm.variance_Jz,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Data Generation Pipeline
# ============================================================================


def generate_n_scaling_scan(force: bool = False) -> None:
    """Full N-scaling scan: 20 N values × 5 ω values = 100 optimisation runs.

    Each (N, ω) pair runs:
      1. 4D random search with 500 points.
      2. Nelder-Mead refinement from top 50 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-N-value to allow resumption if interrupted.

    Delegates to :func:`src.analysis.n_scaling_sweep.run_n_scaling_scan`.

    Args:
        force: Re-run even if Parquet exists.
    """
    run_n_scaling_scan(
        force=force,
        run_single_n_omega=run_single_n_omega,
        n_values=N_VALS,
        omega_values=OMEGA_VALS,
        parquet_path=_parquet_path("n-scaling-scan"),
        checkpoint_dir=REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints",
        fig_ratio_path=_fig_path("n-scaling-ratio"),
        fig_sensitivity_path=_fig_path("n-scaling-sensitivity"),
        fig_params_path=_fig_path("n-scaling-optimal-params"),
        t_hold=T_HOLD,
    )


def generate_n1_consistency(force: bool = False) -> None:
    """Verify N=1 consistency with the 20260519 report.

    At N=1, ω=0.2, the pipeline should find Δω ≈ 0.02036 (R ≈ 4.91).
    """
    csv_p = _parquet_path("n1-consistency")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        df = pd.read_parquet(csv_p)
    else:
        print("[run] N=1 consistency check at ω=0.2...")
        result = run_single_n_omega(N=1, omega=0.2)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")
        df = result.to_dataframe()

    delta = float(df["delta_omega_opt"].iloc[0])
    ratio = float(df["ratio"].iloc[0])
    print(f"  N=1, ω=0.2: Δω = {delta:.6f}, R = {ratio:.3f}")
    print("  Expected:   Δω ≈ 0.02036, R ≈ 4.91")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="N-Scaling of Phase-Modulated Ancilla Drive (20260611)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations even if Parquet files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only a specific generator (e.g., 'decoupled-baseline')",
    )
    args = parser.parse_args()

    force = args.force

    generators: dict[str, tuple[str, str | Callable[..., None], bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            lambda force=False: generate_decoupled_baseline(
                force=force,
                parquet_path=_parquet_path("decoupled-baseline"),
                compute_fn=build_decoupled_baseline_df,
                label="decoupled baseline",
            ),
            False,
        ),
        "n1-consistency": (
            "N=1 Consistency Check",
            "generate_n1_consistency",
            False,
        ),
        "n-scaling-scan": (
            "N-Scaling Full Scan",
            "generate_n_scaling_scan",
            True,
        ),
    }

    if args.only:
        if args.only not in generators:
            print(f"Unknown generator: {args.only}")
            print(f"Available: {list(generators.keys())}")
            sys.exit(1)
        gen_list = [args.only]
    else:
        gen_list = list(generators.keys())

    # Ensure raw_data and figures directories exist
    for d in ["raw_data", "figures"]:
        (REPORTS_DIR / REPORT_DATE / d).mkdir(parents=True, exist_ok=True)

    for key in gen_list:
        name, func_or_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = (
            globals()[func_or_name] if isinstance(func_or_name, str) else func_or_name
        )
        func(force=force)


if __name__ == "__main__":
    import sys

    main()
