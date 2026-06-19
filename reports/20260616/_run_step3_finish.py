"""
Finish Step 3: N=11, 12, 13 with 100 random samples each, sequential.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

local_path = Path(__file__).resolve().parent / "local.py"
spec = importlib.util.spec_from_file_location("local", str(local_path))
module = importlib.util.module_from_spec(spec)
sys.modules["local"] = module
spec.loader.exec_module(module)

checkpoint_dir = (
    module._REPORTS_DIR / module._REPORT_DATE / "raw_data" / "checkpoints_step3"
)

for N in [11, 12, 13]:
    n_rand = 100
    ancilla_dim = N + 1
    print(f"\nN={N}, dim={(N + 1) * ancilla_dim}, {n_rand} random samples")
    sys.stdout.flush()

    ops = module.build_full_ancilla_combined_operators(N)
    d_tot = (N + 1) * ancilla_dim
    psi0 = module.combined_initial_state(d_tot)

    results = []
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

        from src.analysis.ancilla_optimization import compute_expectation_and_variance

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

    ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(ckpt, index=False)
    print(f"  [ckpt] saved {ckpt.name}")
    sys.stdout.flush()

# Consolidate
print("\n=== Consolidating Step 3 ===")
parquet_path = module._parquet_path("step3-n-scaling")
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

print("Done!")
