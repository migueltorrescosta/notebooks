"""
Adaptive Step 3 N-scaling (J_A = N/2) continuation.

Uses progressively fewer random samples as N grows:
  N=6-7:  2000 samples + Nelder-Mead (100 iter)
  N=8-9:  1000 samples + Nelder-Mead (50 iter)
  N=10-11: 500 samples + Nelder-Mead (50 iter)
  N=12-13: 200 samples + Nelder-Mead (50 iter)

Runs ω values in parallel for each N (ThreadPoolExecutor).
"""

import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

local_path = Path(__file__).resolve().parent / "general_4param_omega_drive.py"
spec = importlib.util.spec_from_file_location(
    "general_4param_omega_drive", str(local_path)
)
module = importlib.util.module_from_spec(spec)
sys.modules["general_4param_omega_drive"] = module
spec.loader.exec_module(module)

NM_MAXITER = {
    6: 100,
    7: 100,
    8: 50,
    9: 50,
    10: 50,
    11: 50,
    12: 50,
    13: 50,
}
N_RANDOM_MAP = {
    6: 2000,
    7: 2000,
    8: 1000,
    9: 1000,
    10: 500,
    11: 500,
    12: 200,
    13: 200,
}

print("Adaptive Step 3 N-scaling continuation")
print(f"N range: {module.N_VALS_FULL_ANCILLA}")
sys.stdout.flush()


def _random_search_and_refine(
    rng: np.random.Generator,
    n_random: int,
    N: int,
    psi0: np.ndarray,
    omega: float,
    ops: dict,
    ancilla_dim: int,
    nm_maxiter: int,
) -> tuple[float, np.ndarray]:
    """Random parameter search + Nelder-Mead refinement for a single (N, ω).

    Returns (best_delta, best_sample).
    """
    best_delta = float("inf")
    best_sample = np.zeros(7, dtype=float)

    for _i in range(n_random):
        params = np.array(
            [
                rng.uniform(-5, 5),
                rng.uniform(-5, 5),
                rng.uniform(-5, 5),
                rng.uniform(-20, 20),
                rng.uniform(-20, 20),
                rng.uniform(-20, 20),
                rng.uniform(-20, 20),
            ]
        )
        d = module.compute_combined_sensitivity(
            N,
            psi0,
            module.T_BS,
            module.T_HOLD,
            omega,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            float(params[4]),
            float(params[5]),
            float(params[6]),
            ops,
            ancilla_dim,
        )
        if np.isfinite(d) and d < best_delta:
            best_delta = d
            best_sample = params.copy()

    # Nelder-Mead refinement (gradient-free)
    if np.isfinite(best_delta) and best_delta < 1e6:

        def _nm_obj(p: np.ndarray) -> float:
            return module.compute_combined_sensitivity(
                N,
                psi0,
                module.T_BS,
                module.T_HOLD,
                omega,
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(p[5]),
                float(p[6]),
                ops,
                ancilla_dim,
            )

        try:
            nm_res = minimize(
                _nm_obj,
                best_sample,
                method="Nelder-Mead",
                bounds=[(-5, 5)] * 3 + [(-20, 20)] * 4,
                options={"maxiter": nm_maxiter, "xatol": 1e-3, "fatol": 1e-4},
            )
            if np.isfinite(nm_res.fun) and nm_res.fun < best_delta:
                best_delta = float(nm_res.fun)
                best_sample = nm_res.x.copy()
        except Exception:
            pass

    return best_delta, best_sample


def _run_single_n_omega(
    N: int,
    omega: float,
    ancilla_dim: int,
) -> dict:
    """Run random search + Nelder-Mead for a single (N, ω)."""
    n_random = N_RANDOM_MAP.get(N, 500)
    nm_maxiter = NM_MAXITER.get(N, 50)
    t0 = time.time()

    # Build operators
    if ancilla_dim == N + 1:
        ops = module.build_full_ancilla_combined_operators(N)
    else:
        ops = module.build_fixed_ancilla_combined_operators(N)
    d_tot = (N + 1) * ancilla_dim
    psi0 = module.combined_initial_state(d_tot)

    rng = np.random.default_rng(42 + N * 100 + int(omega * 10))
    best_delta, best_sample = _random_search_and_refine(
        rng, n_random, N, psi0, omega, ops, ancilla_dim, nm_maxiter
    )

    # Diagnostics
    psi_final = module.evolve_combined_circuit(
        N,
        psi0,
        module.T_BS,
        module.T_HOLD,
        omega,
        float(best_sample[0]),
        float(best_sample[1]),
        float(best_sample[2]),
        float(best_sample[3]),
        float(best_sample[4]),
        float(best_sample[5]),
        float(best_sample[6]),
        ops,
        ancilla_dim,
    )
    from src.analysis.ancilla_optimization import compute_expectation_and_variance

    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])
    sql_val = module.sql_reference(N)
    ratio_val = sql_val / best_delta if best_delta > 0 else float("nan")

    t1 = time.time()
    print(
        f"  [N={N}, ω={omega:.1f}] Δω={best_delta:.6f}, R={ratio_val:.3f}, "
        f"t={t1 - t0:.1f}s, n_rand={n_random}"
    )
    sys.stdout.flush()

    return {
        "N": N,
        "omega": omega,
        "delta_omega_opt": best_delta,
        "sql": sql_val,
        "ratio": ratio_val,
        "a_x_opt": float(best_sample[0]),
        "a_y_opt": float(best_sample[1]),
        "a_z_opt": float(best_sample[2]),
        "alpha_xx_opt": float(best_sample[3]),
        "alpha_xz_opt": float(best_sample[4]),
        "alpha_zx_opt": float(best_sample[5]),
        "alpha_zz_opt": float(best_sample[6]),
        "expectation_Jz": float(exp_val),
        "variance_Jz": float(var_val),
        "success": 1,
        "nfev": n_random,
        "n_starts": n_random,
        "n_converged": 1,
    }


checkpoint_dir = (
    module._REPORTS_DIR / module._REPORT_DATE / "raw_data" / "checkpoints_step3"
)
parquet_path = module._parquet_path("step3-n-scaling")

# Find already-completed N values
completed: set[int] = set()
if checkpoint_dir.exists():
    for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
        try:
            df_ckpt = pd.read_parquet(ckpt_file)
            for n_val in df_ckpt["N"].unique():
                completed.add(int(n_val))
        except Exception:
            pass

print(f"\nAlready completed N values: {sorted(completed)}")
remaining = [N for N in module.N_VALS_FULL_ANCILLA if N not in completed]
print(f"Remaining N values: {remaining}")

if not remaining:
    print("All N values already checkpointed. Consolidating...")
else:
    print(
        f"\nRunning {len(remaining)} N values × 5 ω values = {len(remaining) * 5} pairs"
    )
    sys.stdout.flush()

    for N in remaining:
        ancilla_dim = N + 1
        omega_items = module.OMEGA_VALS_N_SCALING
        print(
            f"\n[N={N}, dim={ancilla_dim}] "
            f"{len(omega_items)} ω values, "
            f"n_rand={N_RANDOM_MAP.get(N, 500)}"
        )
        sys.stdout.flush()

        # Run 5 ω values in parallel with ThreadPoolExecutor
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_run_single_n_omega, N, omega, ancilla_dim): omega
                for omega in omega_items
            }
            for future in as_completed(futures):
                try:
                    rdict = future.result()
                    results.append(rdict)
                except Exception as exc:
                    print(f"    [error] Worker failed: {exc}")

        if results:
            n_ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
            n_ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt_df = pd.DataFrame(results)
            ckpt_df.to_parquet(n_ckpt, index=False)
            print(f"  [ckpt] saved {n_ckpt.name} with {len(results)} results")
        else:
            print(f"  [warn] No valid results for N={N}")
        sys.stdout.flush()

# Consolidate
print(f"\n{'=' * 60}")
print("  Consolidating all Step 3 results...")
print(f"{'=' * 60}")
sys.stdout.flush()

all_dfs: list = []
for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
    try:
        df = pd.read_parquet(ckpt_file)
        n_val = int(df["N"].iloc[0])
        this_n = df[df["N"] == n_val]
        all_dfs.append(this_n)
        print(f"  {ckpt_file.name}: {len(this_n)} rows for N={n_val}")
    except Exception as exc:
        print(f"  [warn] Skipping {ckpt_file}: {exc}")

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["N", "omega"], keep="last")
    print(f"Total: {len(combined_df)} rows (after dedup)")

    combined_df.to_parquet(parquet_path, index=False)
    print(f"[save] {parquet_path}")

    print("\nSummary:")
    summary_rows: list[str] = []
    for N in sorted(combined_df["N"].unique()):
        sub = combined_df[combined_df["N"] == N].sort_values("omega")
        for _, row in sub.iterrows():
            line = (
                f"N={int(row['N'])}, ω={row['omega']:.1f}: "
                f"Δω={row['delta_omega_opt']:.6f}, R={row['ratio']:.3f}"
            )
            summary_rows.append(line)
            print(f"  {line}")
else:
    print("[warn] No results to consolidate!")

print("\nDone!")
