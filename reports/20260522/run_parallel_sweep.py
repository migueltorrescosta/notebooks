"""
Parallel sweep for 2026-05-22 Multi-Particle XX-Coupling Dual-MZI.

Uses multiprocessing to parallelise the full ω×N sweep.
With n_coarse=11, estimated ~5 minutes on 16 cores.

Usage:
    uv run python reports/20260522/run_parallel_sweep.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add report dir to path for module-level import in workers
_report_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, _report_dir)
del _report_dir


def _worker(args):
    """Optimise a single (N, omega) pair."""
    N, omega, n_coarse, t_hold = args
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Import inside worker (separate process)
    from local import embed_combined_operators, initial_state, optimise_alpha_xx

    ops = embed_combined_operators(N)
    psi0 = initial_state(N)
    r = optimise_alpha_xx(
        N=N,
        omega=omega,
        ops=ops,
        psi0=psi0,
        n_coarse=n_coarse,
        t_hold=t_hold,
    )
    r["N"] = N
    r["omega"] = omega
    return r


def run_sweep(
    omega_vals: np.ndarray,
    N_vals: np.ndarray,
    t_hold: float = 10.0,
    n_coarse: int = 11,
    max_workers: int = 16,
) -> list[dict]:
    """Parallel sweep over (ω, N)."""
    tasks = [
        (int(N), float(omega), int(n_coarse), float(t_hold))
        for N in N_vals
        for omega in omega_vals
    ]
    total = len(tasks)
    print(f"[parallel] {total} tasks, {max_workers} workers, {n_coarse} coarse pts")
    sys.stdout.flush()

    results: list[dict] = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for done, fut in enumerate(as_completed(futs), start=1):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                task = futs[fut]
                print(f"[error] N={task[0]} ω={task[1]:.1f}: {e}")
                sys.stdout.flush()
                results.append(
                    {
                        "N": task[0],
                        "omega": task[1],
                        "alpha_xx_opt": float("nan"),
                        "delta_omega_opt": float("inf"),
                        "expectation_Jz": 0.0,
                        "variance_Jz": 0.0,
                        "d_expectation": 0.0,
                        "sql": 1.0 / (np.sqrt(task[0]) * t_hold),
                    }
                )

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.0f}s ({rate:.1f}/s)")
                sys.stdout.flush()

    print(f"[done] {total} tasks in {time.time() - t_start:.0f}s")
    sys.stdout.flush()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-coarse", type=int, default=11)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    report_dir = Path(__file__).resolve().parent
    raw_dir = report_dir / "raw_data"
    fig_dir = report_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = raw_dir / "20260522-dual-mzi-sweep.parquet"
    if sweep_path.exists() and not args.force:
        print(f"[skip] {sweep_path.name} exists (use --force)")
        return

    omega_vals = np.linspace(0.1, 5.0, 50)
    N_vals = np.arange(1, 21)
    print(f"ω: {len(omega_vals)} pts ({omega_vals[0]:.1f}–{omega_vals[-1]:.1f})")
    print(f"N: {len(N_vals)} pts ({N_vals[0]}–{N_vals[-1]})")
    sys.stdout.flush()

    results = run_sweep(
        omega_vals, N_vals, n_coarse=args.n_coarse, max_workers=args.workers
    )

    # Sort by N then ω
    results.sort(key=lambda r: (r["N"], r["omega"]))

    from local import (
        DualMZISweepResult,
        fit_scaling_exponents,
        plot_alpha_opt_heatmap,
        plot_n_scaling,
        plot_omega_dependence,
        plot_ratio_heatmap,
        plot_scaling_exponents,
    )

    # Build result object
    ratios = np.array(
        [
            r["delta_omega_opt"] / r["sql"]
            if np.isfinite(r["delta_omega_opt"]) and r["sql"] > 0
            else float("inf")
            for r in results
        ]
    )
    sweep = DualMZISweepResult(
        omega_values=np.array([r["omega"] for r in results]),
        N_values=np.array([r["N"] for r in results], dtype=int),
        alpha_xx_opt=np.array([r["alpha_xx_opt"] for r in results]),
        delta_omega_opt=np.array([r["delta_omega_opt"] for r in results]),
        sql_values=np.array([r["sql"] for r in results]),
        ratio=ratios,
        expectation_Jz=np.array([r["expectation_Jz"] for r in results]),
        variance_Jz=np.array([r["variance_Jz"] for r in results]),
        d_expectation=np.array([r["d_expectation"] for r in results]),
        t_hold=10.0,
    )
    sweep.save_parquet(sweep_path)
    print(f"[save] {sweep_path}")

    # Stats
    alphas = sweep.alpha_xx_opt[np.isfinite(sweep.alpha_xx_opt)]
    ratios_finite = sweep.ratio[np.isfinite(sweep.ratio)]
    print(
        f"  α* min={np.min(alphas):.6f} max={np.max(alphas):.6f} mean={np.mean(alphas):.6f}"
    )
    print(f"  ratio min={np.min(ratios_finite):.6f} max={np.max(ratios_finite):.6f}")
    print(f"  All α*==0: {np.allclose(alphas, 0.0, atol=1e-10)}")
    print(f"  All ratio==1: {np.allclose(ratios_finite, 1.0, atol=1e-5)}")

    # Generate figures
    print("\nGenerating figures...")
    plot_ratio_heatmap(sweep, fig_dir / "20260522-dual-mzi-ratio-heatmap.svg")
    plot_alpha_opt_heatmap(sweep, fig_dir / "20260522-dual-mzi-alpha-opt-heatmap.svg")
    for th, nm in [(0.3, "omega0.3"), (1.0, "omega1.0"), (3.0, "omega3.0")]:
        plot_n_scaling(
            sweep, fig_dir / f"20260522-dual-mzi-n-scaling-{nm}.svg", omega_fixed=th
        )
    for N, nm in [(1, "N1"), (5, "N5"), (20, "N20")]:
        plot_omega_dependence(
            sweep, fig_dir / f"20260522-dual-mzi-omega-{nm}.svg", N_fixed=N
        )

    scaling = fit_scaling_exponents(sweep)
    scaling.save_parquet(raw_dir / "20260522-dual-mzi-scaling.parquet")
    plot_scaling_exponents(scaling, fig_dir / "20260522-dual-mzi-scaling-exponents.svg")
    print("All figures generated.")


if __name__ == "__main__":
    main()
