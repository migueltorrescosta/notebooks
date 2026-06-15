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
    uv run python reports/20260611/local.py --force
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analysis.ancilla_drive_metrology import (
    DriveNelderMeadResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.n_scaling_result import NScalingResult, NScalingScanResult
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_decoupled_baseline,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)
from src.utils.parallel import parallel_map as _parallel_map
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


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, float], bool]:
    """Verify the decoupled baseline for all (N, ω) pairs.

    At zero drive and zero interaction, the sensitivity must equal
    Δω = 1/(√N × T_HOLD) to machine precision.

    Args:
        N_values: List of N values (default: 1 to 20).
        omega_values: List of ω values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping (N, ω) → PASS/FAIL (True/False).
    """
    if N_values is None:
        N_values = N_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N)
        for omega in omega_values:
            delta = compute_n_particle_decoupled_baseline(N, omega)
            results[(N, omega)] = bool(np.isclose(delta, sql_ref, rtol=rtol))
    return results


# ============================================================================
# 4D Random Search
# ============================================================================


def n_particle_random_search(
    N: int,
    omega: float,
    n_samples: int = N_RANDOM,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])
        domega = compute_n_particle_sensitivity(
            N,
            psi0,
            T_BS,
            T_HOLD,
            omega,
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

    return DriveRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N),
        t_hold=T_HOLD,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def n_particle_sensitivity_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω in the N-particle protocol.

    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        N: Number of system particles.
        omega_true: True phase rate.
        ops: N-particle operators.
        psi0: Initial state vector.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
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

    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_n_particle_nelder_mead(
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder-Mead optimisation for the N-particle ω-modulated protocol.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        ops: N-particle operators (built if None).
        psi0: Initial state (built if None).
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for all four parameters.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
    """
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return n_particle_sensitivity_objective(
            p,
            N,
            omega_true,
            ops,
            psi0,
            bounds=bounds,
        )

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,  # type: ignore[arg-type]
        options={  # type: ignore[call-overload]
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# N-Scaling Scan (Single (N, ω) Worker)
# ============================================================================


def run_single_n_omega(
    N: int,
    omega: float,
    seed: int | None = 42,
) -> NScalingResult:
    """Run the full optimisation pipeline for a single (N, ω) pair.

    1. 4D random search (500 samples).
    2. Nelder-Mead refinement from top 50 points.
    3. Return the best result.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        seed: Base random seed (incremented per call).

    Returns:
        NScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)

    # Stage 1: Random search
    rs_result = n_particle_random_search(
        N,
        omega,
        n_samples=N_RANDOM,
        seed=base_seed,
    )

    # Sort by Δω, take top N_NM_REFINE
    sorted_indices = np.argsort(rs_result.delta_omega_values)
    top_indices = sorted_indices[:N_NM_REFINE]

    # Stage 2: Nelder-Mead refinement from each top point
    nm_results: list[DriveNelderMeadResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        nm = run_n_particle_nelder_mead(
            N=N,
            omega_true=omega,
            ops=ops,
            psi0=psi0,
            x0=x0,
            seed=base_seed + int(omega * 1000) + 10000 + rank,
        )
        nm_results.append(nm)

    nm_results.sort(key=lambda r: r.delta_omega_opt)
    best_nm = nm_results[0]

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


def _run_single_n_omega_for_parallel(
    args: tuple[int, float],
) -> dict[str, int | float | str]:
    """Worker for parallel N-scaling scan.

    Args:
        args: Tuple (N, omega).

    Returns:
        Dict of result data.
    """
    N, omega = args
    print(f"  [run] N={N}, ω={omega}")
    result = run_single_n_omega(N, omega)
    return {
        "N": result.N,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "a_zz_opt": result.a_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
    }


def generate_n_scaling_scan(force: bool = False) -> None:
    """Full N-scaling scan: 20 N values × 5 ω values = 100 optimisation runs.

    Each (N, ω) pair runs:
      1. 4D random search with 500 points.
      2. Nelder-Mead refinement from top 50 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-N-value to allow resumption if interrupted.

    Args:
        force: Re-run even if Parquet exists.
    """
    csv_p = _parquet_path("n-scaling-scan")
    checkpoint_dir = REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints"
    fig_ratio_p = _fig_path("n-scaling-ratio")
    fig_sensitivity_p = _fig_path("n-scaling-sensitivity")
    fig_params_p = _fig_path("n-scaling-optimal-params")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        summary = NScalingScanResult.from_parquet(csv_p)
    else:
        if force:
            # Remove existing checkpoints and final file
            csv_p.unlink(missing_ok=True)
            if checkpoint_dir.exists():
                import shutil

                shutil.rmtree(checkpoint_dir)

        # Load existing checkpoints if present
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed: set[tuple[int, float]] = set()
        checkpoint_results: list[NScalingResult] = []
        for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
            try:
                df_ckpt = pd.read_parquet(ckpt_file)
                for _, row in df_ckpt.iterrows():
                    n_val = int(row["N"])
                    w_val = float(row["omega"])
                    delta = float(row["delta_omega_opt"])
                    if np.isfinite(delta):
                        completed.add((n_val, w_val))
                        checkpoint_results.append(
                            NScalingResult(
                                N=n_val,
                                omega=w_val,
                                delta_omega_opt=delta,
                                sql=float(row["sql"]),
                                ratio=float(row["ratio"]),
                                a_x_opt=float(row["a_x_opt"]),
                                a_y_opt=float(row["a_y_opt"]),
                                a_z_opt=float(row["a_z_opt"]),
                                a_zz_opt=float(row["a_zz_opt"]),
                                expectation_Jz=float(row.get("expectation_Jz", 0.0)),
                                variance_Jz=float(row.get("variance_Jz", 0.0)),
                                success=bool(int(row.get("success", 0))),
                                nfev=int(row.get("nfev", 0)),
                            ),
                        )
            except Exception as exc:
                print(f"  [warn] Could not load checkpoint {ckpt_file}: {exc}")

        # Process N values sequentially (with ω parallelism within each N)
        items_to_run = [
            (N, omega)
            for N in N_VALS
            for omega in OMEGA_VALS
            if (N, omega) not in completed
        ]
        if items_to_run:
            print(f"[run] N-scaling scan: {len(items_to_run)} remaining (N, ω) pairs")
            print(f"  (batch by N value, {min(32, os.cpu_count() or 1)} workers)")

            # Group by N and process each group in parallel
            by_N: dict[int, list[tuple[int, float]]] = {}
            for N, omega in items_to_run:
                by_N.setdefault(N, []).append((N, omega))

            for N in sorted(by_N):
                omega_items = by_N[N]
                n_ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
                if n_ckpt.exists():
                    print(f"  [ckpt] N={N} already done, skipping")
                    continue
                print(f"  [batch] N={N}: {len(omega_items)} ω values (parallel)")
                batch_results = _parallel_map(
                    _run_single_n_omega_for_parallel,
                    omega_items,
                    desc=f"N={N} scan",
                )
                # Save checkpoint
                ckpt_list: list[NScalingResult] = []
                for rdict in batch_results:
                    delta = rdict["delta_omega_opt"]
                    if not np.isfinite(delta):
                        print(
                            f"    [skip] N={rdict['N']}, ω={rdict['omega']}: Δω={delta}"
                        )
                        continue
                    ckpt_list.append(
                        NScalingResult(
                            N=rdict["N"],
                            omega=rdict["omega"],
                            delta_omega_opt=rdict["delta_omega_opt"],
                            sql=rdict["sql"],
                            ratio=rdict["ratio"],
                            a_x_opt=rdict["a_x_opt"],
                            a_y_opt=rdict["a_y_opt"],
                            a_z_opt=rdict["a_z_opt"],
                            a_zz_opt=rdict["a_zz_opt"],
                            expectation_Jz=rdict["expectation_Jz"],
                            variance_Jz=rdict["variance_Jz"],
                            success=bool(rdict["success"]),
                            nfev=rdict["nfev"],
                        ),
                    )
                if ckpt_list:
                    ckpt_scan = NScalingScanResult(results=ckpt_list)
                    ckpt_scan.save_parquet(n_ckpt)
                    checkpoint_results.extend(ckpt_list)
                    print(f"    [ckpt] saved {n_ckpt.name}")
        else:
            print("  [skip] all pairs already completed in checkpoints")

        # Merge all checkpoint results and save final file
        summary = NScalingScanResult(results=checkpoint_results)
        summary.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Generate figures
    df = summary.to_dataframe()
    plot_n_scaling_ratio(df, fig_ratio_p)
    print(f"[fig]  {fig_ratio_p}")
    plot_n_scaling_sensitivity(df, fig_sensitivity_p, t_hold=T_HOLD)
    print(f"[fig]  {fig_sensitivity_p}")
    plot_n_scaling_optimal_params(df, fig_params_p)
    print(f"[fig]  {fig_params_p}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Verify the decoupled baseline for all N and ω values."""
    csv_p = _parquet_path("decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] Computing decoupled baseline for all (N, ω)...")
    verifications = verify_decoupled_baseline()
    results_list: list[dict[str, float | int | str]] = []
    for (N, omega), passed in verifications.items():
        sql_ref = sql_reference(N)
        delta = compute_n_particle_decoupled_baseline(N, omega)
        results_list.append(
            {
                "N": N,
                "omega": omega,
                "delta_omega": delta,
                "sql": sql_ref,
                "ratio": delta / sql_ref if sql_ref > 0 else float("nan"),
                "pass": str(passed),
            },
        )
    df = pd.DataFrame(results_list)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(csv_p, index=False)
    print(f"[save] {csv_p}")


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

    generators: dict[str, tuple[str, str, bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            "generate_decoupled_baseline",
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
        name, func_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = globals()[func_name]
        func(force=force)


if __name__ == "__main__":
    import sys

    main()
