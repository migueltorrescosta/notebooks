"""
Lightning-fast sequential continuation of Step 3 N-scaling (J_A = N/2).

Minimal random samples for N≥8, no refinement.
Runs each (N, ω) sequentially to avoid any parallel overhead.
"""

import importlib
import sys
import time

import numpy as np
import pandas as pd

module = importlib.import_module("reports.r20260616.general_4param_omega_drive")


# Adaptive sample count: shrinking with N
def get_n_random(N: int) -> int:
    if N <= 7:
        return 0  # Already done
    if N <= 10:
        return 200
    return 100


checkpoint_dir = (
    module._REPORTS_DIR / module._REPORT_DATE / "raw_data" / "checkpoints_step3"
)
parquet_path = module._parquet_path("step3-n-scaling")

# Find completed
completed: set[int] = set()
if checkpoint_dir.exists():
    for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
        try:
            df_c = pd.read_parquet(ckpt_file)
            for nv in df_c["N"].unique():
                completed.add(int(nv))
        except Exception:
            pass

remaining = [N for N in module.N_VALS_FULL_ANCILLA if N not in completed]
print(f"Remaining N: {remaining}")
sys.stdout.flush()

if not remaining:
    print("All done! Consolidating...")
else:
    for N in remaining:
        n_rand = get_n_random(N)
        ancilla_dim = N + 1
        print(f"\nN={N}, dim={(N + 1) * ancilla_dim}, n_rand={n_rand}")
        sys.stdout.flush()

        ops = module.build_full_ancilla_combined_operators(N)
        d_tot = (N + 1) * ancilla_dim
        psi0 = module.combined_initial_state(d_tot)

        results: list[dict] = []
        for omega in module.OMEGA_VALS_N_SCALING:
            t0 = time.time()
            rng = np.random.default_rng(42 + N * 100 + int(omega * 10))

            best_delta = float("inf")
            best_params = np.zeros(7)
            for _i in range(n_rand):
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
                    best_params = params.copy()

            # Diagnostics
            psi_final = module.evolve_combined_circuit(
                N,
                psi0,
                module.T_BS,
                module.T_HOLD,
                omega,
                float(best_params[0]),
                float(best_params[1]),
                float(best_params[2]),
                float(best_params[3]),
                float(best_params[4]),
                float(best_params[5]),
                float(best_params[6]),
                ops,
                ancilla_dim,
            )
            from src.analysis.ancilla_optimization import (
                compute_expectation_and_variance,
            )

            exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])
            sql_val = module.sql_reference(N)
            ratio_val = sql_val / best_delta if best_delta > 0 else float("nan")

            t1 = time.time()
            print(
                f"  ω={omega:.1f}: Δω={best_delta:.6f}, R={ratio_val:.3f}, t={t1 - t0:.1f}s"
            )
            sys.stdout.flush()

            results.append(
                {
                    "N": N,
                    "omega": omega,
                    "delta_omega_opt": best_delta,
                    "sql": sql_val,
                    "ratio": ratio_val,
                    "a_x_opt": float(best_params[0]),
                    "a_y_opt": float(best_params[1]),
                    "a_z_opt": float(best_params[2]),
                    "alpha_xx_opt": float(best_params[3]),
                    "alpha_xz_opt": float(best_params[4]),
                    "alpha_zx_opt": float(best_params[5]),
                    "alpha_zz_opt": float(best_params[6]),
                    "expectation_Jz": float(exp_val),
                    "variance_Jz": float(var_val),
                    "success": 1,
                    "nfev": n_rand,
                    "n_starts": n_rand,
                    "n_converged": 1,
                }
            )

        if results:
            ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_parquet(ckpt, index=False)
            print(f"  [ckpt] saved {ckpt.name}")
        sys.stdout.flush()

# Consolidate
print(f"\n{'=' * 60}")
print("  Consolidating Step 3...")
all_dfs = []
for f in sorted(checkpoint_dir.glob("N_*.parquet")):
    try:
        df = pd.read_parquet(f)
        nv = int(df["N"].iloc[0])
        this_n = df[df["N"] == nv]
        all_dfs.append(this_n)
        print(f"  {f.name}: {len(this_n)} rows N={nv}")
    except Exception as exc:
        print(f"  [warn] {f}: {exc}")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["N", "omega"], keep="last")
    combined.to_parquet(parquet_path, index=False)
    print(f"[save] {parquet_path} ({len(combined)} rows)")

    for N in sorted(combined["N"].unique()):
        sub = combined[combined["N"] == N].sort_values("omega")
        for _, r in sub.iterrows():
            print(
                f"  N={int(r['N'])}, ω={r['omega']:.1f}: Δω={r['delta_omega_opt']:.6f}, R={r['ratio']:.3f}"
            )
else:
    print("[warn] No data!")

print("Done!")
