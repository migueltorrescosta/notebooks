"""
Run Step 3 N-scaling (J_A = N/2) with reduced BFGS refinements.
Step 2 (J_A = 1/2) already complete for N=1..13.
Now running the same N range with ancilla_dim = N+1.
"""

import importlib.util
import sys
import time
from pathlib import Path

local_path = Path(__file__).resolve().parent / "general_4param_omega_drive.py"
spec = importlib.util.spec_from_file_location(
    "general_4param_omega_drive", str(local_path)
)
module = importlib.util.module_from_spec(spec)
sys.modules["general_4param_omega_drive"] = module
spec.loader.exec_module(module)

# Override constants for faster execution
REDUCED_RANDOM = 1000
REDUCED_BFGS = 20

print(f"Using N_RANDOM={REDUCED_RANDOM}, N_BFGS_REFINE={REDUCED_BFGS}")
print(f"N range: {module.N_VALS_FULL_ANCILLA}")
sys.stdout.flush()


def _run_worker(args: tuple[int, float, int]) -> dict:
    """Modified worker with reduced samples/refinements."""
    N, omega, ancilla_dim = args
    t0 = time.time()
    print(f"  [N={N}, omega={omega}, dim={ancilla_dim}] starting...")
    sys.stdout.flush()
    result = module.run_combined_single_n_omega(
        N,
        omega,
        ancilla_dim,
        n_random=REDUCED_RANDOM,
        n_bfgs_refine=REDUCED_BFGS,
    )
    t1 = time.time()
    print(
        f"  [N={N}, omega={omega}] delta_omega={result.delta_omega_opt:.6f}, "
        f"R={result.ratio:.3f}, n_conv={result.n_converged}, "
        f"t={(t1 - t0) / 60:.1f}min"
    )
    sys.stdout.flush()
    return {
        "N": result.N,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "alpha_xx_opt": result.alpha_xx_opt,
        "alpha_xz_opt": result.alpha_xz_opt,
        "alpha_zx_opt": result.alpha_zx_opt,
        "alpha_zz_opt": result.alpha_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
        "n_starts": result.n_starts,
        "n_converged": result.n_converged,
    }


def run_step3(
    N_values: list[int],
    omega_values: list[float],
    label: str,
    parquet_path: Path,
    checkpoint_dir: Path,
):
    """Run Step 3 N-scaling sequentially with checkpointing, no Step 2."""
    import shutil

    import pandas as pd

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for N in sorted(N_values):
        ancilla_dim = N + 1
        print(f"\n[{label}] N={N}, dim={ancilla_dim}: {len(omega_values)} omega values")
        sys.stdout.flush()
        n_ckpt = checkpoint_dir / f"N_{N:03d}.parquet"

        for omega in omega_values:
            rdict = _run_worker((N, omega, ancilla_dim))
            results.append(rdict)

        if results:
            ckpt_df = pd.DataFrame(results)
            ckpt_df.to_parquet(n_ckpt, index=False)
            print(f"  [ckpt] saved {n_ckpt.name}")
            sys.stdout.flush()

    combined = module.CombinedNScalingScanResult(
        results=[module._make_result_from_dict(r) for r in results]
    )
    combined.save_parquet(parquet_path)
    print(f"[save] {parquet_path}")
    sys.stdout.flush()


# Step 3 only: J_A = N/2, N=1..13
print("\n" + "=" * 60)
print("  Step 3: N-Scaling (J_A = N/2), N=1..13")
print("  WARNING: large Hilbert spaces at N≥10 (~dim=1331..2744)")
print("=" * 60)
sys.stdout.flush()
t0 = time.time()
run_step3(
    N_values=module.N_VALS_FULL_ANCILLA,
    omega_values=module.OMEGA_VALS_N_SCALING,
    label="Step3",
    parquet_path=module._parquet_path("step3-n-scaling"),
    checkpoint_dir=module._REPORTS_DIR
    / module._REPORT_DATE
    / "raw_data"
    / "checkpoints_step3",
)
t1 = time.time()
print(f"Step 3 total: {(t1 - t0) / 60:.1f} min")
sys.stdout.flush()

print("\nDone!")
