"""
Incremental sweep runner for the dual MZI and S-only sweeps.
Saves results after each N value, allowing partial results if interrupted.

Usage:
    uv run python reports/20260523/sweep_runner.py dual-sweep
    uv run python reports/20260523/sweep_runner.py sonly-sweep
    uv run python reports/20260523/sweep_runner.py all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

from local import (
    DUAL_MZI_N_VALS,
    OMEGA_VALS,
    SONLY_MZI_N_VALS,
    DEFAULT_T_hold,
    FourParamSweepResult,
    _n_starts_for_N,
    embed_combined_operators,
    generate_decoupled_baseline,
    generate_sonly_reproduction,
    initial_state,
    optimise_four_params,
    parquet_path,
    plot_omega_scan,
    plot_ratio_heatmap,
)


def run_dual_N(N_val: int, omega_arr: np.ndarray, force: bool = False) -> Path | None:
    """Run all omega for a single N, save to individual parquet."""
    indv_path = parquet_path(f"dual-mzi-N{N_val:02d}")
    if indv_path.exists() and not force:
        print(f"  [skip] N={N_val}: {indv_path.name} exists")
        return indv_path

    n_starts = _n_starts_for_N(N_val)
    ops = embed_combined_operators(N_val)
    psi0 = initial_state(N_val)

    omegas = np.zeros(len(omega_arr), dtype=float)
    Ns = np.full(len(omega_arr), N_val, dtype=int)
    protos = ["dual"] * len(omega_arr)
    a_xx = np.full(len(omega_arr), np.nan, dtype=float)
    a_xz = np.full(len(omega_arr), np.nan, dtype=float)
    a_zx = np.full(len(omega_arr), np.nan, dtype=float)
    a_zz = np.full(len(omega_arr), np.nan, dtype=float)
    delta_opts = np.full(len(omega_arr), np.inf, dtype=float)
    sqls = np.zeros(len(omega_arr), dtype=float)
    ratios = np.full(len(omega_arr), np.inf, dtype=float)
    exps = np.zeros(len(omega_arr), dtype=float)
    vars_ = np.zeros(len(omega_arr), dtype=float)
    d_exps = np.zeros(len(omega_arr), dtype=float)
    n_starts_arr = np.full(len(omega_arr), n_starts, dtype=int)
    n_conv_arr = np.zeros(len(omega_arr), dtype=int)
    grad_norm_arr = np.full(len(omega_arr), np.nan, dtype=float)

    total_omega = len(omega_arr)
    for i, omega_val in enumerate(omega_arr):
        t0 = time.time()
        opt = optimise_four_params(
            N=N_val,
            omega=omega_val,
            ops=ops,
            psi0=psi0,
            protocol="dual",
            n_starts=n_starts,
            T_hold=DEFAULT_T_hold,
        )
        elapsed = time.time() - t0
        sql = 1.0 / (np.sqrt(N_val) * DEFAULT_T_hold)
        ratio = (
            opt.delta_omega_opt / sql
            if np.isfinite(opt.delta_omega_opt) and sql > 0
            else float("inf")
        )

        omegas[i] = omega_val
        a_xx[i] = opt.alpha_opt[0]
        a_xz[i] = opt.alpha_opt[1]
        a_zx[i] = opt.alpha_opt[2]
        a_zz[i] = opt.alpha_opt[3]
        delta_opts[i] = opt.delta_omega_opt
        sqls[i] = sql
        ratios[i] = ratio
        exps[i] = opt.expectation_Jz
        vars_[i] = opt.variance_Jz
        d_exps[i] = opt.d_expectation
        n_conv_arr[i] = opt.n_converged
        grad_norm_arr[i] = opt.gradient_norm

        print(
            f"  [N={N_val:2d}, ω={omega_val:.1f}] Δω={opt.delta_omega_opt:.6f}, "
            f"ratio={ratio:.4f}, conv={opt.n_converged}/{n_starts}, "
            f"t={elapsed:.0f}s ({i + 1}/{total_omega})"
        )

    result = FourParamSweepResult(
        omega_values=omegas,
        N_values=Ns,
        protocol=protos,
        alpha_xx_opt=a_xx,
        alpha_xz_opt=a_xz,
        alpha_zx_opt=a_zx,
        alpha_zz_opt=a_zz,
        delta_omega_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        n_starts=n_starts_arr,
        n_converged=n_conv_arr,
        gradient_norm=grad_norm_arr,
        T_hold=DEFAULT_T_hold,
    )
    result.save_parquet(indv_path)
    print(f"  [save] N={N_val}: {indv_path}")
    return indv_path


def run_sonly_N(N_val: int, omega_arr: np.ndarray, force: bool = False) -> Path | None:
    """Run all omega for a single N in S-only sweep."""
    indv_path = parquet_path(f"sonly-mzi-N{N_val:02d}")
    if indv_path.exists() and not force:
        print(f"  [skip] N={N_val}: {indv_path.name} exists")
        return indv_path

    n_starts = _n_starts_for_N(N_val)
    ops = embed_combined_operators(N_val)
    psi0 = initial_state(N_val)

    omegas = np.zeros(len(omega_arr), dtype=float)
    Ns = np.full(len(omega_arr), N_val, dtype=int)
    protos = ["S-only"] * len(omega_arr)
    a_xx = np.full(len(omega_arr), np.nan, dtype=float)
    a_xz = np.full(len(omega_arr), np.nan, dtype=float)
    a_zx = np.full(len(omega_arr), np.nan, dtype=float)
    a_zz = np.full(len(omega_arr), np.nan, dtype=float)
    delta_opts = np.full(len(omega_arr), np.inf, dtype=float)
    sqls = np.zeros(len(omega_arr), dtype=float)
    ratios = np.full(len(omega_arr), np.inf, dtype=float)
    exps = np.zeros(len(omega_arr), dtype=float)
    vars_ = np.zeros(len(omega_arr), dtype=float)
    d_exps = np.zeros(len(omega_arr), dtype=float)
    n_starts_arr = np.full(len(omega_arr), n_starts, dtype=int)
    n_conv_arr = np.zeros(len(omega_arr), dtype=int)
    grad_norm_arr = np.full(len(omega_arr), np.nan, dtype=float)

    total_omega = len(omega_arr)
    for i, omega_val in enumerate(omega_arr):
        t0 = time.time()
        opt = optimise_four_params(
            N=N_val,
            omega=omega_val,
            ops=ops,
            psi0=psi0,
            protocol="S-only",
            n_starts=n_starts,
            T_hold=DEFAULT_T_hold,
        )
        elapsed = time.time() - t0
        sql = 1.0 / (np.sqrt(N_val) * DEFAULT_T_hold)
        ratio = (
            opt.delta_omega_opt / sql
            if np.isfinite(opt.delta_omega_opt) and sql > 0
            else float("inf")
        )

        omegas[i] = omega_val
        a_xx[i] = opt.alpha_opt[0]
        a_xz[i] = opt.alpha_opt[1]
        a_zx[i] = opt.alpha_opt[2]
        a_zz[i] = opt.alpha_opt[3]
        delta_opts[i] = opt.delta_omega_opt
        sqls[i] = sql
        ratios[i] = ratio
        exps[i] = opt.expectation_Jz
        vars_[i] = opt.variance_Jz
        d_exps[i] = opt.d_expectation
        n_conv_arr[i] = opt.n_converged
        grad_norm_arr[i] = opt.gradient_norm

        print(
            f"  [S-only N={N_val:2d}, ω={omega_val:.1f}] Δω={opt.delta_omega_opt:.6f}, "
            f"ratio={ratio:.4f}, conv={opt.n_converged}/{n_starts}, "
            f"t={elapsed:.0f}s ({i + 1}/{total_omega})"
        )

    result = FourParamSweepResult(
        omega_values=omegas,
        N_values=Ns,
        protocol=protos,
        alpha_xx_opt=a_xx,
        alpha_xz_opt=a_xz,
        alpha_zx_opt=a_zx,
        alpha_zz_opt=a_zz,
        delta_omega_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        n_starts=n_starts_arr,
        n_converged=n_conv_arr,
        gradient_norm=grad_norm_arr,
        T_hold=DEFAULT_T_hold,
    )
    result.save_parquet(indv_path)
    print(f"  [save] N={N_val}: {indv_path}")
    return indv_path


def combine_dual_sweep() -> None:
    """Combine individual N parquets into full sweep result."""
    csv_p = parquet_path("dual-mzi-sweep")
    all_paths = sorted(Path(csv_p).parent.glob("20260523-dual-mzi-N*.parquet"))
    if not all_paths:
        print("  [skip] No individual N files found for dual MZI")
        return None

    all_results = []
    for p in all_paths:
        r = FourParamSweepResult.from_parquet(p)
        all_results.append(r)

    import pandas as pd

    df = pd.concat([r.to_dataframe() for r in all_results], ignore_index=True)
    df = df.sort_values(["N", "omega"]).reset_index(drop=True)

    combined = FourParamSweepResult(
        omega_values=df["omega"].to_numpy(dtype=float),
        N_values=df["N"].to_numpy(dtype=int),
        protocol=df["protocol"].tolist(),
        alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
        alpha_xz_opt=df["alpha_xz_opt"].to_numpy(dtype=float),
        alpha_zx_opt=df["alpha_zx_opt"].to_numpy(dtype=float),
        alpha_zz_opt=df["alpha_zz_opt"].to_numpy(dtype=float),
        delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
        sql_values=df["sql"].to_numpy(dtype=float),
        ratio=df["ratio"].to_numpy(dtype=float),
        expectation_Jz=df["expectation_Jz"].to_numpy(dtype=float),
        variance_Jz=df["variance_Jz"].to_numpy(dtype=float),
        d_expectation=df["d_expectation"].to_numpy(dtype=float),
        n_starts=df["n_starts"].to_numpy(dtype=int),
        n_converged=df["n_converged"].to_numpy(dtype=int),
        gradient_norm=df["gradient_norm"].to_numpy(dtype=float),
        T_hold=DEFAULT_T_hold,
    )
    combined.save_parquet(csv_p)
    print(f"  [save] Combined dual sweep: {csv_p}")

    # Generate figures
    fig_ratio = csv_p.parent.parent / "figures" / "20260523-dual-mzi-ratio-heatmap.svg"
    plot_ratio_heatmap(combined, fig_ratio, title_suffix="Dual MZI")
    print(f"  [fig]  {fig_ratio}")

    fig_scan = csv_p.parent.parent / "figures" / "20260523-dual-mzi-omega-scan-N5.svg"
    plot_omega_scan(combined, fig_scan, N_fixed=5)
    print(f"  [fig]  {fig_scan}")

    return combined


def combine_sonly_sweep() -> None:
    """Combine individual N parquets into full S-only sweep result."""
    csv_p = parquet_path("sonly-mzi-sweep")
    all_paths = sorted(Path(csv_p).parent.glob("20260523-sonly-mzi-N*.parquet"))
    if not all_paths:
        print("  [skip] No individual N files found for S-only MZI")
        return

    all_results = []
    for p in all_paths:
        r = FourParamSweepResult.from_parquet(p)
        all_results.append(r)

    import pandas as pd

    df = pd.concat([r.to_dataframe() for r in all_results], ignore_index=True)
    df = df.sort_values(["N", "omega"]).reset_index(drop=True)

    combined = FourParamSweepResult(
        omega_values=df["omega"].to_numpy(dtype=float),
        N_values=df["N"].to_numpy(dtype=int),
        protocol=df["protocol"].tolist(),
        alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
        alpha_xz_opt=df["alpha_xz_opt"].to_numpy(dtype=float),
        alpha_zx_opt=df["alpha_zx_opt"].to_numpy(dtype=float),
        alpha_zz_opt=df["alpha_zz_opt"].to_numpy(dtype=float),
        delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
        sql_values=df["sql"].to_numpy(dtype=float),
        ratio=df["ratio"].to_numpy(dtype=float),
        expectation_Jz=df["expectation_Jz"].to_numpy(dtype=float),
        variance_Jz=df["variance_Jz"].to_numpy(dtype=float),
        d_expectation=df["d_expectation"].to_numpy(dtype=float),
        n_starts=df["n_starts"].to_numpy(dtype=int),
        n_converged=df["n_converged"].to_numpy(dtype=int),
        gradient_norm=df["gradient_norm"].to_numpy(dtype=float),
        T_hold=DEFAULT_T_hold,
    )
    combined.save_parquet(csv_p)
    print(f"  [save] Combined S-only sweep: {csv_p}")

    fig_ratio = csv_p.parent.parent / "figures" / "20260523-sonly-mzi-ratio-heatmap.svg"
    plot_ratio_heatmap(combined, fig_ratio, title_suffix="S-only MZI")
    print(f"  [fig]  {fig_ratio}")


def run_dual_sweep(force: bool = False) -> None:
    """Run all N values for dual MZI sweep."""
    omega_arr = np.array(OMEGA_VALS, dtype=float)
    N_vals = DUAL_MZI_N_VALS
    print(
        f"Dual MZI sweep: {len(N_vals)} N × {len(omega_arr)} ω = {len(N_vals) * len(omega_arr)} points (N=1..10)"
    )

    for N_val in N_vals:
        print(f"\n--- N={N_val} ---")
        n_starts = _n_starts_for_N(N_val)
        print(f"  Starts: {n_starts}")
        total_per_N = n_starts * len(omega_arr)
        est_sec = 0
        # Rough estimate: from N=1 (0.5s/start) scaling roughly as d^2.5
        d = (N_val + 1) ** 2
        d1 = 4  # N=1 dimension
        scaling = (d / d1) ** 2.5
        est_per_point = 0.5 * scaling
        est_sec = est_per_point * total_per_N
        if est_sec > 60:
            print(f"  Est. time: {est_sec / 60:.0f} min")
        else:
            print(f"  Est. time: {est_sec:.0f}s")

        run_dual_N(N_val, omega_arr, force=force)


def run_sonly_sweep(force: bool = False) -> None:
    """Run all N values for S-only MZI sweep."""
    omega_arr = np.array(OMEGA_VALS, dtype=float)
    N_vals = SONLY_MZI_N_VALS
    print(
        f"S-only MZI sweep: {len(N_vals)} N × {len(omega_arr)} ω = {len(N_vals) * len(omega_arr)} points"
    )

    for N_val in N_vals:
        print(f"\n--- S-only N={N_val} ---")
        run_sonly_N(N_val, omega_arr, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental sweep runner")
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        help="Task: dual-sweep, sonly-sweep, combine-dual, combine-sonly, all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Ensure directories exist
    (Path(__file__).parent / "raw_data").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "figures").mkdir(parents=True, exist_ok=True)

    task = args.task or "all"

    if task in ("dual-sweep", "all"):
        print("\n=== Dual MZI Sweep ===")
        run_dual_sweep(force=args.force)
        print("\n--- Combining dual sweep ---")
        combine_dual_sweep()

    if task in ("sonly-sweep", "all"):
        print("\n=== S-only MZI Sweep ===")
        run_sonly_sweep(force=args.force)
        print("\n--- Combining S-only sweep ---")
        combine_sonly_sweep()

    if task == "combine-dual":
        combine_dual_sweep()

    if task == "combine-sonly":
        combine_sonly_sweep()

    if task == "all":
        print("\n=== Quick tasks ===")
        generate_decoupled_baseline(force=args.force)
        generate_sonly_reproduction(force=args.force)

        print("\n=== Post-processing ===")
        from local import (
            generate_n_scaling,
            generate_omega_dependence,
            generate_scaling_analysis,
        )

        generate_n_scaling(force=args.force)
        generate_omega_dependence(force=args.force)
        generate_scaling_analysis(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
