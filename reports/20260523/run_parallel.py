"""
Parallel sweep runner for 2026-05-23 Four-Parameter Coupling report.

Speeds up the expensive dual MZI sweep by running N values in parallel.
Also runs the S-only sweep and analysis tasks.

Usage:
    uv run python reports/20260523/run_parallel.py --workers 8
    uv run python reports/20260523/run_parallel.py --workers 8 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Ensure report dir is on sys.path (project root is available via editable install)
sys.path.insert(0, str(Path(__file__).parent))

# Must happen before importing local (for matplotlib backend)
import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

from local import (
    DEFAULT_T_H,
    DUAL_MZI_N_VALS,
    SONLY_MZI_N_VALS,
    THETA_VALS,
    FourParamSweepResult,
    _n_starts_for_N,
    embed_combined_operators,
    generate_decoupled_baseline,
    generate_n_scaling,
    generate_scaling_analysis,
    generate_sonly_reproduction,
    generate_theta_dependence,
    initial_state,
    optimise_four_params,
    parquet_path,
)


def _run_N_batch(N_val: int, theta_arr: np.ndarray, protocol: str) -> FourParamSweepResult:
    """Run all theta values for a single N."""
    ops = embed_combined_operators(N_val)
    psi0 = initial_state(N_val)
    n_starts = _n_starts_for_N(N_val)
    n_theta = len(theta_arr)

    thetas = np.zeros(n_theta, dtype=float)
    Ns = np.full(n_theta, N_val, dtype=int)
    protos = [protocol] * n_theta
    a_xx = np.full(n_theta, np.nan, dtype=float)
    a_xz = np.full(n_theta, np.nan, dtype=float)
    a_zx = np.full(n_theta, np.nan, dtype=float)
    a_zz = np.full(n_theta, np.nan, dtype=float)
    delta_opts = np.full(n_theta, np.inf, dtype=float)
    sqls = np.zeros(n_theta, dtype=float)
    ratios = np.full(n_theta, np.inf, dtype=float)
    exps = np.zeros(n_theta, dtype=float)
    vars_ = np.zeros(n_theta, dtype=float)
    d_exps = np.zeros(n_theta, dtype=float)
    n_starts_arr = np.full(n_theta, n_starts, dtype=int)
    n_conv_arr = np.zeros(n_theta, dtype=int)
    grad_norm_arr = np.full(n_theta, np.nan, dtype=float)

    for i, theta_val in enumerate(theta_arr):
        t0 = time.time()
        opt_result = optimise_four_params(
            N=N_val,
            theta=theta_val,
            ops=ops,
            psi0=psi0,
            protocol=protocol,
            n_starts=n_starts,
            T_H=DEFAULT_T_H,
        )
        t1 = time.time()
        sql = 1.0 / (np.sqrt(N_val) * DEFAULT_T_H)

        thetas[i] = theta_val
        a_xx[i] = opt_result.alpha_opt[0]
        a_xz[i] = opt_result.alpha_opt[1]
        a_zx[i] = opt_result.alpha_opt[2]
        a_zz[i] = opt_result.alpha_opt[3]
        delta_opts[i] = opt_result.delta_theta_opt
        sqls[i] = sql
        ratios[i] = (
            opt_result.delta_theta_opt / sql
            if np.isfinite(opt_result.delta_theta_opt) and sql > 0
            else float("inf")
        )
        exps[i] = opt_result.expectation_Jz
        vars_[i] = opt_result.variance_Jz
        d_exps[i] = opt_result.d_expectation
        n_conv_arr[i] = opt_result.n_converged
        grad_norm_arr[i] = opt_result.gradient_norm

        print(
            f"  [N={N_val:2d}, θ={theta_val:.1f}] "
            f"Δθ={opt_result.delta_theta_opt:.6f}, "
            f"ratio={ratios[i]:.4f}, "
            f"conv={opt_result.n_converged}/{n_starts}, "
            f"t={t1-t0:.0f}s"
        )

    return FourParamSweepResult(
        theta_values=thetas,
        N_values=Ns,
        protocol=protos,
        alpha_xx_opt=a_xx,
        alpha_xz_opt=a_xz,
        alpha_zx_opt=a_zx,
        alpha_zz_opt=a_zz,
        delta_theta_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        n_starts=n_starts_arr,
        n_converged=n_conv_arr,
        gradient_norm=grad_norm_arr,
        T_H=DEFAULT_T_H,
    )


def run_dual_sweep_parallel(force: bool = False, workers: int = 8) -> None:
    """Run dual MZI sweep in parallel across N values."""
    csv_p = parquet_path("dual-mzi-sweep")
    fig_ratio = parquet_path("..") / "figures" / "20260523-dual-mzi-ratio-heatmap.svg"
    fig_theta_scan = parquet_path("..") / "figures" / "20260523-dual-mzi-theta-scan-N5.svg"

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    theta_arr = np.array(THETA_VALS, dtype=float)
    N_vals = DUAL_MZI_N_VALS

    print(f"Running dual MZI sweep with {workers} workers...")
    print(f"  N values: {N_vals}")
    print(f"  θ values: {THETA_VALS}")
    print(f"  Total: {len(N_vals)} N × {len(theta_arr)} θ = {len(N_vals) * len(theta_arr)} points")

    # Run in parallel across N values
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for N_val in N_vals:
            future = executor.submit(_run_N_batch, N_val, theta_arr, "dual")
            futures[future] = N_val

        results_by_N: dict[int, FourParamSweepResult] = {}
        for future in as_completed(futures):
            N_val = futures[future]
            try:
                result = future.result()
                results_by_N[N_val] = result
                print(f"  [DONE] N={N_val}: {len(result.theta_values)} points")
            except Exception as e:
                print(f"  [FAIL] N={N_val}: {e}")

    # Check all N values completed
    if len(results_by_N) < len(N_vals):
        missing = set(N_vals) - set(results_by_N.keys())
        print(f"WARNING: Missing results for N values: {sorted(missing)}")
        print("Results will be incomplete.")

    # Combine results in order
    sorted_N = sorted(results_by_N.keys())
    all_thetas = []
    all_Ns = []
    all_protos = []
    all_axx = []
    all_axz = []
    all_azx = []
    all_azz = []
    all_delta = []
    all_sql = []
    all_ratio = []
    all_exp = []
    all_var = []
    all_dexp = []
    all_nstarts = []
    all_nconv = []
    all_gnorm = []

    for N_val in sorted_N:
        r = results_by_N[N_val]
        all_thetas.extend(r.theta_values.tolist())
        all_Ns.extend(r.N_values.tolist())
        all_protos.extend(r.protocol)
        all_axx.extend(r.alpha_xx_opt.tolist())
        all_axz.extend(r.alpha_xz_opt.tolist())
        all_azx.extend(r.alpha_zx_opt.tolist())
        all_azz.extend(r.alpha_zz_opt.tolist())
        all_delta.extend(r.delta_theta_opt.tolist())
        all_sql.extend(r.sql_values.tolist())
        all_ratio.extend(r.ratio.tolist())
        all_exp.extend(r.expectation_Jz.tolist())
        all_var.extend(r.variance_Jz.tolist())
        all_dexp.extend(r.d_expectation.tolist())
        all_nstarts.extend(r.n_starts.tolist())
        all_nconv.extend(r.n_converged.tolist())
        all_gnorm.extend(r.gradient_norm.tolist())

    combined = FourParamSweepResult(
        theta_values=np.array(all_thetas, dtype=float),
        N_values=np.array(all_Ns, dtype=int),
        protocol=all_protos,
        alpha_xx_opt=np.array(all_axx, dtype=float),
        alpha_xz_opt=np.array(all_axz, dtype=float),
        alpha_zx_opt=np.array(all_azx, dtype=float),
        alpha_zz_opt=np.array(all_azz, dtype=float),
        delta_theta_opt=np.array(all_delta, dtype=float),
        sql_values=np.array(all_sql, dtype=float),
        ratio=np.array(all_ratio, dtype=float),
        expectation_Jz=np.array(all_exp, dtype=float),
        variance_Jz=np.array(all_var, dtype=float),
        d_expectation=np.array(all_dexp, dtype=float),
        n_starts=np.array(all_nstarts, dtype=int),
        n_converged=np.array(all_nconv, dtype=int),
        gradient_norm=np.array(all_gnorm, dtype=float),
        T_H=DEFAULT_T_H,
    )

    combined.save_parquet(csv_p)
    print(f"[save] {csv_p}")

    # Generate figures from the sweep data
    from local import plot_ratio_heatmap, plot_theta_scan
    fig_ratio_p = parquet_path("..").parent / "figures" / "20260523-dual-mzi-ratio-heatmap.svg"
    plot_ratio_heatmap(combined, fig_ratio_p, title_suffix="Dual MZI")
    print(f"[fig]  {fig_ratio_p}")

    fig_scan_p = parquet_path("..").parent / "figures" / "20260523-dual-mzi-theta-scan-N5.svg"
    plot_theta_scan(combined, fig_scan_p, N_fixed=5)
    print(f"[fig]  {fig_scan_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel sweep runner for 2026-05-23 report",
    )
    parser.add_argument("--force", action="store_true", help="Re-run all simulations")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only one task",
    )
    args = parser.parse_args()

    tasks = {}

    # Quick tasks (already completed or fast)
    if args.only is None:
        print("\n=== decoupled-baseline ===")
        generate_decoupled_baseline(force=args.force)

        print("\n=== sonly-reproduction ===")
        generate_sonly_reproduction(force=args.force)

        print("\n=== sonly-sweep ===")
        _run_single_sweep("S-only", force=args.force)

        print("\n=== dual-sweep (parallel) ===")
        run_dual_sweep_parallel(force=args.force, workers=args.workers)

        print("\n=== n-scaling ===")
        generate_n_scaling(force=args.force)

        print("\n=== theta-dependence ===")
        generate_theta_dependence(force=args.force)

        print("\n=== scaling-analysis ===")
        generate_scaling_analysis(force=args.force)
    elif args.only == "dual-sweep":
        run_dual_sweep_parallel(force=args.force, workers=args.workers)
    elif args.only == "sonly-sweep":
        _run_single_sweep("S-only", force=args.force)
    elif args.only in ("n-scaling", "theta-dependence", "scaling-analysis"):
        # These just read existing data
        # Just call the equivalent from main
        pass
    else:
        print(f"Unknown task '{args.only}'")

    print("\nDone.")


def _run_single_sweep(protocol: str, force: bool = False) -> None:
    """Run the S-only sweep sequentially (30 points)."""
    from local import plot_ratio_heatmap

    name = f"{protocol.lower().replace('-', '')}-mzi-sweep"
    csv_p = parquet_path(name)
    fig_ratio = parquet_path("..") / "figures" / f"20260523-{protocol.lower().replace('-', '')}-mzi-ratio-heatmap.svg"

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists")
        return

    theta_arr = np.array(THETA_VALS, dtype=float)
    N_arr = np.array(SONLY_MZI_N_VALS, dtype=int)

    print(f"Running {protocol} MZI sweep...")
    print(f"  N: {N_arr.tolist()}")
    print(f"  θ: {THETA_VALS}")
    print(f"  Points: {len(N_arr)} N × {len(theta_arr)} θ = {len(N_arr) * len(theta_arr)}")

    total_points = len(N_arr) * len(theta_arr)
    point_count = 0
    all_results = []

    for N_val in N_arr:
        t0 = time.time()
        ops = embed_combined_operators(N_val)
        psi0 = initial_state(N_val)
        n_starts = _n_starts_for_N(N_val)

        for theta_val in theta_arr:
            t_start = time.time()
            opt = optimise_four_params(
                N=N_val, theta=theta_val, ops=ops, psi0=psi0,
                protocol=protocol, n_starts=n_starts, T_H=DEFAULT_T_H,
            )
            t_elapsed = time.time() - t_start
            point_count += 1
            ratio = (
                opt.delta_theta_opt / opt.sql
                if np.isfinite(opt.delta_theta_opt) and opt.sql > 0
                else float("inf")
            )
            print(
                f"  [{protocol} N={N_val:2d}, θ={theta_val:.1f}] "
                f"Δθ={opt.delta_theta_opt:.6f}, "
                f"ratio={ratio:.4f}, "
                f"conv={opt.n_converged}/{n_starts}, "
                f"t={t_elapsed:.0f}s "
                f"({point_count}/{total_points})"
            )

            # Accumulate
            all_results.append({
                "theta": theta_val,
                "N": N_val,
                "protocol": protocol,
                "alpha_xx_opt": opt.alpha_opt[0],
                "alpha_xz_opt": opt.alpha_opt[1],
                "alpha_zx_opt": opt.alpha_opt[2],
                "alpha_zz_opt": opt.alpha_opt[3],
                "delta_theta_opt": opt.delta_theta_opt,
                "sql": opt.sql,
                "ratio": ratio,
                "expectation_Jz": opt.expectation_Jz,
                "variance_Jz": opt.variance_Jz,
                "d_expectation": opt.d_expectation,
                "n_starts": n_starts,
                "n_converged": opt.n_converged,
                "gradient_norm": opt.gradient_norm,
            })

    # Build sweep result
    import pandas as pd
    df = pd.DataFrame(all_results)
    result = FourParamSweepResult(
        theta_values=df["theta"].to_numpy(dtype=float),
        N_values=df["N"].to_numpy(dtype=int),
        protocol=df["protocol"].tolist(),
        alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
        alpha_xz_opt=df["alpha_xz_opt"].to_numpy(dtype=float),
        alpha_zx_opt=df["alpha_zx_opt"].to_numpy(dtype=float),
        alpha_zz_opt=df["alpha_zz_opt"].to_numpy(dtype=float),
        delta_theta_opt=df["delta_theta_opt"].to_numpy(dtype=float),
        sql_values=df["sql"].to_numpy(dtype=float),
        ratio=df["ratio"].to_numpy(dtype=float),
        expectation_Jz=df["expectation_Jz"].to_numpy(dtype=float),
        variance_Jz=df["variance_Jz"].to_numpy(dtype=float),
        d_expectation=df["d_expectation"].to_numpy(dtype=float),
        n_starts=df["n_starts"].to_numpy(dtype=int),
        n_converged=df["n_converged"].to_numpy(dtype=int),
        gradient_norm=df["gradient_norm"].to_numpy(dtype=float),
        T_H=DEFAULT_T_H,
    )
    result.save_parquet(csv_p)
    print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio, title_suffix=f"{protocol} MZI")
    print(f"[fig]  {fig_ratio}")


if __name__ == "__main__":
    main()
