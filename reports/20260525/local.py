"""
Local module for the 2026-05-25 Multi-Particle XX-Coupling Dual-MZI
with Optimised System–Ancilla Joint Measurement report.

Contains all code exclusive to this report:
- Multi-particle operator construction (Dicke basis, N up to 20)
- Dual MZI circuit: BS(S)⊗BS(A) → Hold → BS(S)⊗BS(A) → full-state measurement M(φ)
- Full-state expectation and variance computation (no partial trace)
- Sensitivity via error propagation with central finite differences
- Joint (α_xx, φ) optimisation via L-BFGS-B with 20 random starts per (ω, N)
- Full 2D sweep over ω ∈ [0.1, 5.0] and N ∈ [1, 20]
- Decoupled baseline verification (α_xx = 0, φ-optimised)
- Scaling analysis (log-log fit Δω ∝ N^α)
- Exclusive plot functions for heatmaps, scaling curves, and landscape slices
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260525/local.py --force

This module is **not** importable as ``reports.20260525.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Use spawn start method to avoid fork + BLAS threading deadlock.
mp.set_start_method("spawn", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.multi_mzi_scaling import (  # noqa: E402
    ScalingAnalysisResult,
    fit_scaling_exponents,
)
from src.physics.multi_mzi import (  # noqa: E402
    build_hold_hamiltonian,  # noqa: F401 — re-exported for tests
    dual_bs_unitary,  # noqa: F401 — re-exported for tests
    embed_combined_operators,
    evolve_circuit,
    hold_unitary_dicke,  # noqa: F401 — re-exported for tests
    single_bs_unitary,  # noqa: F401 — re-exported for tests
)

if TYPE_CHECKING:
    from collections.abc import Callable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_hold: float = 10.0  # Holding time (SQL reference)
AXX_BOUNDS: tuple[float, float] = (0.0, 20.0)  # α_xx optimisation range
PSI_BOUNDS: tuple[float, float] = (-np.pi, np.pi)  # ψ optimisation range
N_RANDOM_STARTS: int = 20  # L-BFGS-B random starts per (ω, N) for small N
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
# Full-State Measurement (No Partial Trace)
# ============================================================================


def build_measurement_operator(
    N: int, psi: float, ops: dict[str, np.ndarray]
) -> np.ndarray:
    """Build the joint measurement operator M(ψ) in the full S⊗A space.

    M(ψ) = cosψ · J_z^S + sinψ · J_z^A

    The coefficients automatically satisfy m_s² + m_a² = 1 with
    m_s = cosψ, m_a = sinψ.

    Args:
        N: Particle number per subsystem.
        psi: Measurement weight angle.
        ops: Embedded operators (must contain 'Jz_S', 'Jz_A').

    Returns:
        (N+1)² × (N+1)² Hermitian measurement matrix.
    """
    M = np.cos(psi) * ops["Jz_S"] + np.sin(psi) * ops["Jz_A"]
    return 0.5 * (M + M.conj().T)


def full_state_expectation_and_variance(
    psi: np.ndarray, meas_op: np.ndarray
) -> tuple[float, float]:
    """Compute ⟨M⟩ and Var(M) directly from a pure state (no partial trace).

    ⟨M⟩ = ⟨ψ|M|ψ⟩
    Var(M) = ⟨ψ|M²|ψ⟩ - ⟨ψ|M|ψ⟩²

    Args:
        psi: Pure state vector (length (N+1)²).
        meas_op: Measurement operator (d × d Hermitian matrix).

    Returns:
        Tuple (expectation, variance). Variance is clamped to zero
        when below 1e-12 due to numerical round-off.
    """
    M_psi = meas_op @ psi
    exp_val = float(np.real(np.vdot(psi, M_psi)))
    M2_psi = meas_op @ M_psi
    exp_sq = float(np.real(np.vdot(psi, M2_psi)))
    raw_var = exp_sq - exp_val**2

    # Clamp negative variance near zero (numerical round-off)
    if raw_var < 0 and raw_var > -1e-12:
        raw_var = 0.0
    assert raw_var >= -1e-12, f"Unphysical negative variance: {raw_var:.2e}"

    return float(exp_val), float(max(0.0, raw_var))


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_sensitivity_full(
    N: int,
    psi0: np.ndarray,
    omega_true: float,
    alpha_xx: float,
    psi: float,
    ops: dict[str, np.ndarray],
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
    T_BS: float = DEFAULT_T_BS,
    T_hold: float = DEFAULT_T_hold,
) -> tuple[float, float, float, float]:
    """Compute the error-propagation sensitivity Δω with full-state measurement.

    Δω = √Var(M) / |∂⟨M⟩/∂ω|

    where M = cosψ·J_z^S + sinψ·J_z^A is the joint measurement operator.

    Also returns ⟨M⟩, Var(M), and ∂⟨M⟩/∂ω at omega_true.

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector.
        omega_true: True phase rate.
        alpha_xx: XX coupling strength.
        psi: Measurement weight angle.
        ops: Embedded operators.
        meas_op: Pre-computed measurement operator (default: built from psi).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        T_hold: Holding time.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        meas_op = build_measurement_operator(N, psi, ops)

    # Evaluate at omega_true
    state = evolve_circuit(N, psi0, omega_true, alpha_xx, ops, T_BS, T_hold)
    exp_val, var_val = full_state_expectation_and_variance(state, meas_op)

    # Central finite difference for ∂⟨M⟩/∂ω
    psi_plus = evolve_circuit(
        N, psi0, omega_true + fd_step, alpha_xx, ops, T_BS, T_hold
    )
    psi_minus = evolve_circuit(
        N, psi0, omega_true - fd_step, alpha_xx, ops, T_BS, T_hold
    )

    exp_plus = full_state_expectation_and_variance(psi_plus, meas_op)[0]
    exp_minus = full_state_expectation_and_variance(psi_minus, meas_op)[0]
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_omega = float(np.sqrt(var_val) / abs(d_exp))
    return delta_omega, exp_val, var_val, d_exp


# ============================================================================
# 2D Optimisation over (α_xx, φ) via L-BFGS-B
# ============================================================================


def _objective_joint(
    params: np.ndarray,
    N: int,
    psi0: np.ndarray,
    omega: float,
    ops: dict[str, np.ndarray],
    T_hold: float,
    fd_step: float,
) -> float:
    """Objective function for joint (α_xx, ψ) optimisation.

    Returns Δω at the given (α_xx, ψ). Returns a large finite value
    if the sensitivity diverges (fringe extremum).

    Args:
        params: Array [alpha_xx, psi].
        N: Particle number per subsystem.
        psi0: Initial state.
        omega: Phase rate.
        ops: Embedded operators.
        T_hold: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Δω (finite, large if fringe extremum).
    """
    alpha_xx, psi = float(params[0]), float(params[1])
    dt, _, _, _ = compute_sensitivity_full(
        N, psi0, omega, alpha_xx, psi, ops, T_hold=T_hold, fd_step=fd_step
    )
    if np.isfinite(dt) and dt > 0:
        return dt
    return 1e10  # Large penalty for divergent points


def _maxiter_for_n(N: int) -> int:
    """Adaptive L-BFGS-B maxiter based on Hilbert space dimension.

    Large N is exponentially more expensive per objective evaluation,
    so we cap maxiter to keep runtime manageable.

    Args:
        N: Particle number per subsystem.

    Returns:
        Max iterations for L-BFGS-B.
    """
    if N <= 5:
        return 200
    if N <= 10:
        return 100
    return 50


def optimise_joint(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    axx_bounds: tuple[float, float] = AXX_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    n_starts: int = N_RANDOM_STARTS,
    maxiter: int | None = None,
    T_BS: float = DEFAULT_T_BS,
    T_hold: float = DEFAULT_T_hold,
    fd_step: float = FD_STEP,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """Optimise Δω over (α_xx, φ) for a given (ω, N) pair.

    Uses L-BFGS-B with n_starts random starting points to avoid local minima.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        axx_bounds: (min, max) for α_xx.
        psi_bounds: (min, max) for ψ.
        n_starts: Number of random starting points.
        T_BS: Beam-splitter angle.
        T_hold: Holding time.
        fd_step: Finite-difference step.
        rng_seed: Optional seed for reproducibility.

    Returns:
        Dict with keys:
            'alpha_xx_opt': optimal α_xx value.
            'psi_opt': optimal φ value.
            'ms_opt': m_s = cos(φ*) at optimum.
            'ma_opt': m_a = sin(φ*) at optimum.
            'delta_omega_opt': minimal Δω.
            'expectation_M': ⟨M⟩ at optimum.
            'variance_M': Var(M) at optimum.
            'd_expectation': ∂⟨M⟩/∂ω at optimum.
            'sql_2n': SQL = 1/(√(2N) * T_hold) reference.
            'n_starts_converged': number of starts that converged.
            'n_starts_at_best': number of starts that reached the best optimum
                (within 1% relative tolerance of delta_omega_opt).
    """
    if psi0 is None:
        psi0 = initial_state(N)

    sql_2n = 1.0 / (np.sqrt(2 * N) * T_hold)
    rng = np.random.default_rng(rng_seed)

    if maxiter is None:
        maxiter = _maxiter_for_n(N)
    bounds = [axx_bounds, psi_bounds]
    best_result: dict[str, float] = {
        "alpha_xx_opt": 0.0,
        "psi_opt": 0.0,
        "ms_opt": 1.0,
        "ma_opt": 0.0,
        "delta_omega_opt": float("inf"),
        "expectation_M": 0.0,
        "variance_M": 0.0,
        "d_expectation": 0.0,
        "sql_2n": sql_2n,
        "n_starts_converged": 0,
        "n_starts_at_best": 0,
    }

    n_converged = 0
    # Collect all converged delta_omega values for clustering diagnostic
    _all_dt: list[float] = []

    for _ in range(n_starts):
        # Random initial point within bounds
        alpha0 = rng.uniform(axx_bounds[0], axx_bounds[1])
        psi_start = rng.uniform(psi_bounds[0], psi_bounds[1])
        x0 = np.array([alpha0, psi_start])

        try:
            result = minimize(
                _objective_joint,
                x0,
                args=(N, psi0, omega, ops, T_hold, fd_step),
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": maxiter},
            )

            if not result.success:
                continue

            n_converged += 1
            alpha_opt = float(result.x[0])
            psi_opt = float(result.x[1])

            # Ensure α_xx is within bounds (L-BFGS-B may slightly exceed)
            alpha_opt = np.clip(alpha_opt, axx_bounds[0], axx_bounds[1])

            dt_opt, exp_opt, var_opt, d_exp_opt = compute_sensitivity_full(
                N, psi0, omega, alpha_opt, psi_opt, ops, T_hold=T_hold, fd_step=fd_step
            )

            if np.isfinite(dt_opt):
                _all_dt.append(dt_opt)

            if np.isfinite(dt_opt) and dt_opt < best_result["delta_omega_opt"]:
                best_result["alpha_xx_opt"] = alpha_opt
                best_result["psi_opt"] = psi_opt
                best_result["ms_opt"] = np.cos(psi_opt)
                best_result["ma_opt"] = np.sin(psi_opt)
                best_result["delta_omega_opt"] = dt_opt
                best_result["expectation_M"] = exp_opt
                best_result["variance_M"] = var_opt
                best_result["d_expectation"] = d_exp_opt

        except (ValueError, np.linalg.LinAlgError):
            continue

    best_result["n_starts_converged"] = n_converged
    # Count how many starts landed within 1% of the best delta_omega
    best_dt = best_result["delta_omega_opt"]
    if np.isfinite(best_dt) and best_dt > 0 and _all_dt:
        best_result["n_starts_at_best"] = sum(
            1
            for dt in _all_dt
            if np.isfinite(dt) and abs(dt - best_dt) / best_dt < 0.01
        )
    return best_result


# ============================================================================
# Dataclass for Optimised Sweep Results
# ============================================================================


@dataclass
class DualMZIOptimisedResult:
    """Full 2D sweep over ω and N with joint (α_xx, φ) optimisation per point.

    All array fields have the same length (n_omega × n_N), stored in
    row-major order (N varies slowest, ω varies fastest).

    Attributes:
        omega_values: ω values for each point.
        N_values: N values for each point.
        alpha_xx_opt: Optimal α_xx at each point.
        psi_opt: Optimal ψ at each point.
        ms_opt: m_s = cos(ψ*) at each point.
        ma_opt: m_a = sin(ψ*) at each point.
        delta_omega_opt: Minimal Δω at each point.
        sql_2n: 2N-SQL = 1/(√(2N) T_hold) at each point.
        ratio: Δω_opt / SQL_2N at each point.
        expectation_M: ⟨M⟩ at optimum.
        variance_M: Var(M) at optimum.
        d_expectation: ∂⟨M⟩/∂ω at optimum.
        n_starts_converged: Number of L-BFGS-B starts that converged per point.
        n_starts_at_best: Number of starts that reached the best optimum
            (within 1% relative tolerance of delta_omega_opt).
        T_hold: Holding time (scalar).
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    alpha_xx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    psi_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    ms_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    ma_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_2n: np.ndarray = field(default_factory=lambda: np.array([]))
    ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_M: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_M: np.ndarray = field(default_factory=lambda: np.array([]))
    d_expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    n_starts_converged: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    n_starts_at_best: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    T_hold: float = DEFAULT_T_hold

    def __post_init__(self) -> None:
        # Ensure int dtype for N_values
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "N": self.N_values,
                "T_hold": np.full(len(self.omega_values), self.T_hold),
                "alpha_xx_opt": self.alpha_xx_opt,
                "psi_opt": self.psi_opt,
                "ms_opt": self.ms_opt,
                "ma_opt": self.ma_opt,
                "delta_omega_opt": self.delta_omega_opt,
                "sql_2n": self.sql_2n,
                "ratio": self.ratio,
                "expectation_M": self.expectation_M,
                "variance_M": self.variance_M,
                "d_expectation": self.d_expectation,
                "n_starts_converged": self.n_starts_converged,
                "n_starts_at_best": self.n_starts_at_best,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DualMZIOptimisedResult:
        df = pd.read_parquet(path)
        required = {
            "omega",
            "N",
            "T_hold",
            "alpha_xx_opt",
            "psi_opt",
            "ms_opt",
            "ma_opt",
            "delta_omega_opt",
            "sql_2n",
            "ratio",
            "expectation_M",
            "variance_M",
            "d_expectation",
            "n_starts_converged",
            "n_starts_at_best",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )

        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            N_values=df["N"].to_numpy(dtype=int),
            alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
            psi_opt=df["psi_opt"].to_numpy(dtype=float),
            ms_opt=df["ms_opt"].to_numpy(dtype=float),
            ma_opt=df["ma_opt"].to_numpy(dtype=float),
            delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
            sql_2n=df["sql_2n"].to_numpy(dtype=float),
            ratio=df["ratio"].to_numpy(dtype=float),
            expectation_M=df["expectation_M"].to_numpy(dtype=float),
            variance_M=df["variance_M"].to_numpy(dtype=float),
            d_expectation=df["d_expectation"].to_numpy(dtype=float),
            n_starts_converged=df["n_starts_converged"].to_numpy(dtype=int),
            n_starts_at_best=df["n_starts_at_best"].to_numpy(dtype=int),
            T_hold=float(df["T_hold"].iloc[0]),
        )

    @property
    def n_points(self) -> int:
        return len(self.omega_values)

    @property
    def n_omega_unique(self) -> int:
        return len(np.unique(self.omega_values))

    @property
    def n_N_unique(self) -> int:
        return len(np.unique(self.N_values))


# ============================================================================
# Sweep Execution
# ============================================================================


def _unpack_worker_result(r: dict[str, object]) -> dict[str, float | int]:
    """Extract typed values from a worker result dict (process-boundary safe)."""
    return {
        "omega": float(r["omega"]),  # type: ignore[arg-type]
        "N": int(r["N"]),  # type: ignore[arg-type]
        "alpha_xx_opt": float(r["alpha_xx_opt"]),  # type: ignore[arg-type]
        "psi_opt": float(r["psi_opt"]),  # type: ignore[arg-type]
        "ms_opt": float(r["ms_opt"]),  # type: ignore[arg-type]
        "ma_opt": float(r["ma_opt"]),  # type: ignore[arg-type]
        "delta_omega_opt": float(r["delta_omega_opt"]),  # type: ignore[arg-type]
        "sql_2n": float(r["sql_2n"]),  # type: ignore[arg-type]
        "expectation_M": float(r["expectation_M"]),  # type: ignore[arg-type]
        "variance_M": float(r["variance_M"]),  # type: ignore[arg-type]
        "d_expectation": float(r["d_expectation"]),  # type: ignore[arg-type]
        "n_starts_at_best": int(r["n_starts_at_best"]),  # type: ignore[arg-type]
        "n_starts_converged": int(r["n_starts_converged"]),  # type: ignore[arg-type]
    }


def _n_starts_for_n(N: int) -> int:
    """Adaptive number of random starts based on Hilbert space dimension.

    Large N is exponentially more expensive, so we use fewer starts
    to keep total runtime manageable.

    Args:
        N: Particle number per subsystem.

    Returns:
        Number of random starts for optimisation at this N.
    """
    if N <= 5:
        return 20
    if N <= 10:
        return 10
    if N <= 15:
        return 5
    return 3


def _optimise_one_point(
    args: tuple[int, float, float, int, float, int | None],
) -> dict[str, Any]:
    """Worker function (picklable) for per-point parallel sweep.

    Args:
        args: Tuple (N, omega, T_hold, n_starts, fd_step, seed).

    Returns:
        Dict with keys: 'N', 'omega', plus all optimise_joint result keys.
    """
    N, omega, T_hold, n_starts, fd_step, seed = args
    ops = embed_combined_operators(N)
    psi0 = initial_state(N)
    result = optimise_joint(
        N=N,
        omega=omega,
        ops=ops,
        psi0=psi0,
        n_starts=n_starts,
        T_hold=T_hold,
        fd_step=fd_step,
        rng_seed=seed,
    )
    n_best = result["n_starts_at_best"]
    if n_best <= 1 and np.isfinite(result["delta_omega_opt"]):
        print(
            f"  [diag] N={N}, ω={omega:.1f}: best Δω={result['delta_omega_opt']:.6e} "
            f"found by only {n_best}/{result['n_starts_converged']} starts "
            f"(α_xx*={result['alpha_xx_opt']:.4f}, φ*={result['psi_opt']:.4f})"
        )
    return {**result, "N": N, "omega": omega}


def run_sweep(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    T_hold: float = DEFAULT_T_hold,
    n_starts: int = N_RANDOM_STARTS,
    progress_callback: Callable[[int, int], None] | None = None,
    seed: int | None = None,
    parallel: int = 0,
) -> DualMZIOptimisedResult:
    """Run the full 2D sweep over ω and N with joint (α_xx, φ) optimisation.

    Args:
        omega_values: ω values to sweep (default: 0.1 to 5.0 step 0.1).
        N_values: N values to sweep (default: 1 to 20 inclusive).
        T_hold: Holding time.
        n_starts: Random starts per (ω, N) pair.
        progress_callback: Optional callback (current, total).
        seed: Optional seed for reproducibility.
        parallel: Number of worker processes (0 = serial).

    Returns:
        DualMZIOptimisedResult with all optimised points.
    """
    if omega_values is None:
        omega_values = np.arange(OMEGA_MIN, OMEGA_MAX + 1e-9, OMEGA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N

    omegas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    alpha_opts = np.full(total, np.nan, dtype=float)
    psi_opts = np.full(total, np.nan, dtype=float)
    ms_opts = np.full(total, np.nan, dtype=float)
    ma_opts = np.full(total, np.nan, dtype=float)
    delta_opts = np.full(total, np.inf, dtype=float)
    sqls = np.zeros(total, dtype=float)
    ratios = np.full(total, np.inf, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)
    n_best_arr = np.zeros(total, dtype=int)
    n_converged_arr = np.zeros(total, dtype=int)

    if parallel > 0:
        # ── Parallel execution via ProcessPoolExecutor (per-point) ──
        global_start_seed = 0 if seed is None else seed
        n_omega = len(omega_values)
        n_N = len(N_values)

        # Build tasks: one per (N, ω) pair with adaptive n_starts
        tasks: list[tuple[int, float, float, int, float, int | None]] = [
            (
                int(N_i),
                float(omega_i),
                T_hold,
                _n_starts_for_n(N_i),
                FD_STEP,
                global_start_seed,
            )
            for N_i in N_values
            for omega_i in omega_values
        ]

        print(f"[info] Parallel sweep: {len(tasks)} tasks, {parallel} workers")
        n_done = 0
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(_optimise_one_point, t): i for i, t in enumerate(tasks)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"  [err] Worker failed for task {i}: {exc}")
                    continue
                n_done += 1
                omegas[i] = float(r["omega"])
                Ns[i] = int(r["N"])
                alpha_opts[i] = float(r["alpha_xx_opt"])
                psi_opts[i] = float(r["psi_opt"])
                ms_opts[i] = float(r["ms_opt"])
                ma_opts[i] = float(r["ma_opt"])
                delta_opts[i] = float(r["delta_omega_opt"])
                sqls[i] = float(r["sql_2n"])
                ratios[i] = (
                    float(r["delta_omega_opt"]) / float(r["sql_2n"])
                    if np.isfinite(r["delta_omega_opt"]) and float(r["sql_2n"]) > 0
                    else float("inf")
                )
                exps[i] = float(r["expectation_M"])
                vars_[i] = float(r["variance_M"])
                d_exps[i] = float(r["d_expectation"])
                n_best_arr[i] = int(r["n_starts_at_best"])
                n_converged_arr[i] = int(r["n_starts_converged"])
                if progress_callback is not None:
                    progress_callback(n_done, total)
                if n_done % 50 == 0:
                    print(f"  [progress] {n_done}/{total} tasks completed")
    else:
        # ── Serial execution (original path) ──
        idx = 0
        global_start_seed = 0 if seed is None else seed

        for N in N_values:
            ops = embed_combined_operators(N)
            psi0 = initial_state(N)
            for omega in omega_values:
                omegas[idx] = omega
                Ns[idx] = N

                opt_result = optimise_joint(
                    N=N,
                    omega=omega,
                    ops=ops,
                    psi0=psi0,
                    n_starts=_n_starts_for_n(N),
                    T_hold=T_hold,
                    rng_seed=global_start_seed + idx if seed is not None else None,
                )

                alpha_opts[idx] = opt_result["alpha_xx_opt"]
                psi_opts[idx] = opt_result["psi_opt"]
                ms_opts[idx] = opt_result["ms_opt"]
                ma_opts[idx] = opt_result["ma_opt"]
                delta_opts[idx] = opt_result["delta_omega_opt"]
                sqls[idx] = opt_result["sql_2n"]
                ratios[idx] = (
                    opt_result["delta_omega_opt"] / opt_result["sql_2n"]
                    if np.isfinite(opt_result["delta_omega_opt"])
                    and opt_result["sql_2n"] > 0
                    else float("inf")
                )
                exps[idx] = opt_result["expectation_M"]
                vars_[idx] = opt_result["variance_M"]
                d_exps[idx] = opt_result["d_expectation"]
                n_best_arr[idx] = opt_result["n_starts_at_best"]
                n_converged_arr[idx] = opt_result["n_starts_converged"]

                # Convergence diagnostic
                n_best = opt_result["n_starts_at_best"]
                if n_best <= 1 and np.isfinite(opt_result["delta_omega_opt"]):
                    print(
                        f"  [diag] N={N}, ω={omega:.1f}: best Δω={opt_result['delta_omega_opt']:.6e} "
                        f"found by only {n_best}/{opt_result['n_starts_converged']} starts "
                        f"(α_xx*={opt_result['alpha_xx_opt']:.4f}, φ*={opt_result['psi_opt']:.4f})"
                    )

                idx += 1
                if progress_callback is not None:
                    progress_callback(idx, total)

    return DualMZIOptimisedResult(
        omega_values=omegas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        psi_opt=psi_opts,
        ms_opt=ms_opts,
        ma_opt=ma_opts,
        delta_omega_opt=delta_opts,
        sql_2n=sqls,
        ratio=ratios,
        expectation_M=exps,
        variance_M=vars_,
        d_expectation=d_exps,
        n_starts_converged=n_converged_arr,
        n_starts_at_best=n_best_arr,
        T_hold=T_hold,
    )


# ============================================================================
# Decoupled Baseline Verification (α_xx = 0, φ-sweep)
# ============================================================================


def compute_sensitivity_at_psi(
    N: int,
    omega: float,
    psi: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    T_hold: float = DEFAULT_T_hold,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float]:
    """Compute sensitivity at α_xx=0 with a given ψ.

    At α_xx = 0, this should give:
    - ψ = 0:     Δω = 1/(√N T_hold) (the N-SQL, worse than 2N-SQL by √2)
    - ψ = π/4:   Δω = 1/(√(2N) T_hold) = Δω_SQL (the 2N-SQL, optimal separable)

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        psi: Measurement weight angle.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        T_hold: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
    """
    if psi0 is None:
        psi0 = initial_state(N)
    meas_op = build_measurement_operator(N, psi, ops)
    return compute_sensitivity_full(
        N,
        psi0,
        omega,
        alpha_xx=0.0,
        psi=psi,
        ops=ops,
        meas_op=meas_op,
        T_hold=T_hold,
        fd_step=fd_step,
    )


def compute_decoupled_baseline(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    T_hold: float = DEFAULT_T_hold,
    fd_step: float = FD_STEP,
) -> DualMZIOptimisedResult:
    """Verify the decoupled baseline (α_xx = 0) at the analytically optimal φ = π/4.

    At α_xx = 0, the optimal measurement is φ = π/4, giving
    Δω = 1/(√(2N) T_hold) = Δω_SQL (the 2N-SQL).

    Args:
        omega_values: ω values (default: sweep range).
        N_values: N values (default: 1 to 20).
        T_hold: Holding time.

    Returns:
        DualMZIOptimisedResult with α_xx=0, φ=π/4 results.
    """
    if omega_values is None:
        omega_values = np.arange(OMEGA_MIN, OMEGA_MAX + 1e-9, OMEGA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N

    omegas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    sqls = np.zeros(total, dtype=float)
    psi_vals = np.full(total, np.pi / 4.0, dtype=float)  # Optimal ψ at α_xx=0
    ms_vals = np.full(total, np.cos(np.pi / 4.0), dtype=float)
    ma_vals = np.full(total, np.sin(np.pi / 4.0), dtype=float)
    delta_opts = np.zeros(total, dtype=float)
    alpha_opts = np.zeros(total, dtype=float)  # all zero
    ratios = np.zeros(total, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)
    n_best_arr = np.zeros(total, dtype=int)  # No random starts in baseline
    n_converged_arr = np.zeros(total, dtype=int)  # No random starts in baseline

    idx = 0
    for N in N_values:
        ops = embed_combined_operators(N)
        psi0 = initial_state(N)
        for omega in omega_values:
            omegas[idx] = omega
            Ns[idx] = N
            sql = 1.0 / (np.sqrt(2 * N) * T_hold)
            sqls[idx] = sql

            dt, exp_val, var_val, d_exp_val = compute_sensitivity_at_psi(
                N,
                omega,
                psi=np.pi / 4.0,
                ops=ops,
                psi0=psi0,
                T_hold=T_hold,
                fd_step=fd_step,
            )
            delta_opts[idx] = dt
            ratios[idx] = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
            exps[idx] = exp_val
            vars_[idx] = var_val
            d_exps[idx] = d_exp_val

            idx += 1

    return DualMZIOptimisedResult(
        omega_values=omegas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        psi_opt=psi_vals,
        ms_opt=ms_vals,
        ma_opt=ma_vals,
        delta_omega_opt=delta_opts,
        sql_2n=sqls,
        ratio=ratios,
        expectation_M=exps,
        variance_M=vars_,
        d_expectation=d_exps,
        n_starts_converged=n_converged_arr,
        n_starts_at_best=n_best_arr,
        T_hold=T_hold,
    )


# ============================================================================
# Plot Functions
# ============================================================================


def plot_ratio_heatmap(
    sweep: DualMZIOptimisedResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of Δω_opt / SQL_2N ratio across (ω, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(sweep.omega_values)
    N_vals = np.unique(sweep.N_values)
    ratio_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                ratio_map[j, i] = float(sweep.ratio[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(0.0, float(np.nanmin(ratio_map)))
    vmax = max(2.0, float(np.nanmax(ratio_map[ratio_map < 10])))

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        ratio_map,
        shading="nearest",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(
        im,
        ax=ax,
        label=r"$\Delta\omega_{\mathrm{opt}} / \Delta\omega_{\mathrm{SQL}}^{2N}$",
    )
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(
        "Sensitivity Ratio: Optimised Joint Measurement\n(lower = better; 1.0 = 2N-SQL)"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alpha_opt_heatmap(
    sweep: DualMZIOptimisedResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal α_xx across (ω, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(sweep.omega_values)
    N_vals = np.unique(sweep.N_values)
    alpha_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                alpha_map[j, i] = float(sweep.alpha_xx_opt[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmax_val = float(np.nanmax(alpha_map)) if np.any(np.isfinite(alpha_map)) else 20.0

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        alpha_map,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_val,
    )
    fig.colorbar(im, ax=ax, label=r"$\alpha_{xx}^*$")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(r"Optimal $\alpha_{xx}$: Joint Optimised Measurement")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_psi_opt_heatmap(
    sweep: DualMZIOptimisedResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal φ across (ω, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(sweep.omega_values)
    N_vals = np.unique(sweep.N_values)
    psi_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                psi_map[j, i] = float(sweep.psi_opt[mask][0])

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        psi_map,
        shading="nearest",
        cmap="RdBu",
        vmin=-np.pi,
        vmax=np.pi,
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$\psi^*$ (rad)")
    # Mark π/4 reference line
    cbar.ax.axhline(y=np.pi / 4.0, color="gray", linewidth=1.0, linestyle=":")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(r"Optimal $\psi$: Joint Optimised Measurement")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling(
    sweep: DualMZIOptimisedResult,
    save_path: str | Path,
    omega_fixed: float | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δω_opt vs N at a fixed ω, with 2N-SQL and N-SQL reference lines.

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        omega_fixed: ω value to plot. If None, uses the first ω.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if omega_fixed is None:
        omega_fixed = float(np.unique(sweep.omega_values)[0])

    mask = np.isclose(sweep.omega_values, omega_fixed)
    N_vals = sweep.N_values[mask].astype(float)
    delta_vals = sweep.delta_omega_opt[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # Reference lines
    N_dense = np.logspace(np.log10(1), np.log10(20), 100)
    sql_2n_dense = 1.0 / (np.sqrt(2 * N_dense) * sweep.T_hold)
    sql_n_dense = 1.0 / (np.sqrt(N_dense) * sweep.T_hold)
    hl_dense = 1.0 / (N_dense * sweep.T_hold)

    ax.loglog(N_dense, sql_2n_dense, "--", color="gray", alpha=0.7, label="2N-SQL")
    ax.loglog(N_dense, sql_n_dense, "-.", color="gray", alpha=0.5, label="N-SQL")
    ax.loglog(N_dense, hl_dense, ":", color="gray", alpha=0.5, label="HL")

    # Data
    finite_mask = np.isfinite(delta_vals) & (delta_vals > 0)
    if np.any(finite_mask):
        ax.loglog(
            N_vals[finite_mask],
            delta_vals[finite_mask],
            "o-",
            color="C0",
            markersize=8,
            linewidth=1.8,
            label=rf"$\Delta\omega_{{\mathrm{{opt}}}}(\omega={omega_fixed:.2f})$",
        )

    ax.set_xlabel(r"$N$ (particles per subsystem)")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(
        f"N-Scaling at $\\omega={omega_fixed:.2f}$:\nOptimised Joint Measurement"
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling_exponents(
    scaling: ScalingAnalysisResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot the scaling exponent α vs ω from log-log fits.

    Args:
        scaling: Scaling analysis result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: exponent vs ω
    valid_exp = np.isfinite(scaling.exponents)
    if np.any(valid_exp):
        ax1.plot(
            scaling.omega_values[valid_exp],
            scaling.exponents[valid_exp],
            "o-",
            color="C1",
            markersize=6,
            linewidth=1.5,
        )
    ax1.axhline(
        y=-0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="SQL (α = −0.5)",
    )
    ax1.axhline(
        y=-1.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="HL (α = −1.0)",
    )
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"Scaling exponent $\alpha$")
    ax1.set_title(r"Exponent $\alpha$ from $\Delta\omega = C N^{\alpha}$")
    ax1.legend(fontsize=9)

    # Right: R² vs ω
    valid_r2 = np.isfinite(scaling.r_squared)
    if np.any(valid_r2):
        ax2.plot(
            scaling.omega_values[valid_r2],
            scaling.r_squared[valid_r2],
            "s-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_title("Goodness of Fit")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_landscape(
    N: int,
    omega: float,
    axx_vals: np.ndarray,
    psi_vals: np.ndarray,
    delta_map: np.ndarray,
    sql_2n: float,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot a 2D contour of Δω(α_xx, ψ) at a given (ω, N).

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        axx_vals: α_xx grid values (1D).
        psi_vals: ψ grid values (1D).
        delta_map: 2D array of Δω values (len(axx_vals) × len(psi_vals)).
        sql_2n: 2N-SQL reference value for contour line.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Mask infinite values for contouring
    delta_plot = np.where(np.isfinite(delta_map), delta_map, np.nan)

    levels = np.linspace(
        np.nanmin(delta_plot) if np.any(np.isfinite(delta_plot)) else 0.01,
        min(
            np.nanpercentile(delta_plot, 95)
            if np.any(np.isfinite(delta_plot))
            else 0.1,
            0.5,
        ),
        30,
    )

    cf = ax.contourf(
        axx_vals,
        psi_vals,
        delta_plot.T,
        levels=levels,
        cmap="viridis",
        extend="both",
    )
    fig.colorbar(cf, ax=ax, label=r"$\Delta\omega$")

    # SQL contour line (where Δω = SQL)
    ax.contour(
        axx_vals,
        psi_vals,
        delta_plot.T,
        levels=[sql_2n],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    # Add a legend entry for the SQL contour using a proxy Line2D (compatible
    # with matplotlib 3.8+ where ContourSet.collections was removed).
    from matplotlib.lines import Line2D

    proxy_line = Line2D(
        [0],
        [0],
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=rf"SQL = {sql_2n:.5f}",
    )
    ax.legend(handles=[proxy_line], fontsize=9, loc="upper right")

    ax.set_xlabel(r"$\alpha_{xx}$")
    ax.set_ylabel(r"$\psi$ (rad)")
    ax.set_title(
        f"$\\Delta\\omega(\\alpha_{{xx}}, \\psi)$ for $N={N},\\omega={omega:.1f}$"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


class _TracedDataProxy:
    """Simple namespace for traced-out data loaded from Parquet."""

    def __init__(
        self,
        N_values: np.ndarray,
        ratio: np.ndarray,
        omega_values: np.ndarray,
    ) -> None:
        self.N_values = N_values
        self.ratio = ratio
        self.omega_values = omega_values


def plot_comparison_traced_out(
    sweep_joint: DualMZIOptimisedResult,
    sweep_traced: DualMZIOptimisedResult | _TracedDataProxy | None = None,
    traced_data_path: str | Path | None = None,
    save_path: str | Path = "",
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot comparison between optimised joint measurement and traced-out protocol.

    Shows r_joint = Δω_opt / (1/√(2N) T_hold) vs r_trace = Δω_trace / (1/√N T_hold).

    Args:
        sweep_joint: This report's sweep result.
        sweep_traced: 2026-05-22 sweep result (optional).
        traced_data_path: Path to 2026-05-22 data Parquet (alternative).
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load traced-out data from 2026-05-22 Parquet directly
    traced = None
    if sweep_traced is not None:
        traced = sweep_traced
    elif traced_data_path is not None:
        try:
            df_traced = pd.read_parquet(traced_data_path)
            # Build a simple namespace with the fields we need
            traced = _TracedDataProxy(
                N_values=df_traced["N"].to_numpy(dtype=int),
                ratio=df_traced["ratio"].to_numpy(dtype=float),
                omega_values=df_traced["omega"].to_numpy(dtype=float),
            )
        except (FileNotFoundError, ValueError):
            pass

    fig, ax = plt.subplots(figsize=figsize)

    # Joint measurement data
    joint_ratio = sweep_joint.ratio
    joint_finite = np.isfinite(joint_ratio)

    ax.scatter(
        sweep_joint.N_values[joint_finite],
        joint_ratio[joint_finite],
        c=sweep_joint.omega_values[joint_finite],
        cmap="viridis",
        alpha=0.6,
        s=20,
        label="Joint (this report, 2N-SQL ref)",
    )

    # Traced-out data
    if traced is not None:
        N_vals_t = traced.N_values
        ratio_t = traced.ratio
        omega_t = traced.omega_values
        finite_t = np.isfinite(ratio_t)
        ax.scatter(
            N_vals_t[finite_t],
            ratio_t[finite_t],
            c=omega_t[finite_t],
            cmap="plasma",
            alpha=0.4,
            s=10,
            marker="x",
            label="Traced-out (2026-05-22, N-SQL ref)",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="SQL baseline")
    ax.set_xlabel(r"$N$ (particles per subsystem)")
    ax.set_ylabel(r"$\Delta\omega / \Delta\omega_{\mathrm{SQL}}$")
    ax.set_title("Joint Optimised vs Traced-Out Measurement")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260525"
# Date prefix for filenames uses dashes (YYYY-MM-DD) for human readability.
_REPORT_DATE_PREFIX = "2026-05-25"
OMEGA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
N_VALS: list[int] = list(range(1, 21))


def parquet_path(name: str) -> Path:
    """Return path to a raw_data Parquet file for this report."""
    return (
        REPORTS_DIR / REPORT_DATE / "raw_data" / f"{_REPORT_DATE_PREFIX}-{name}.parquet"
    )


def fig_path(name: str) -> Path:
    """Return path to a figures SVG file for this report."""
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{_REPORT_DATE_PREFIX}-{name}.svg"


# Backward-compatible aliases
_parquet_path = parquet_path
_fig_path = fig_path


# ── Generator Functions ────────────────────────────────────────────────


def generate_sweep(force: bool = False, parallel: int = 0) -> None:
    """Run the full ω × N sweep with joint (α_xx, ψ) optimisation."""
    csv_p = _parquet_path("optimised-measurement-sweep")
    fig_ratio = _fig_path("ratio-heatmap")
    fig_alpha = _fig_path("alpha-opt-heatmap")
    fig_psi = _fig_path("psi-opt-heatmap")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZIOptimisedResult.from_parquet(csv_p)
    else:
        print(
            "[run]  Computing joint (α_xx, φ) optimisation ω×N sweep "
            f"({len(OMEGA_VALS)}×{len(N_VALS)} = {len(OMEGA_VALS) * len(N_VALS)} points)..."
        )
        omega_arr = np.array(OMEGA_VALS, dtype=float)
        N_arr = np.array(N_VALS, dtype=int)
        result = run_sweep(omega_values=omega_arr, N_values=N_arr, parallel=parallel)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio)
    print(f"[fig]  {fig_ratio}")
    plot_alpha_opt_heatmap(result, fig_alpha)
    print(f"[fig]  {fig_alpha}")
    plot_psi_opt_heatmap(result, fig_psi)

    print(f"[fig]  {fig_psi}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline (α_xx = 0, φ-optimised) verification."""
    csv_p = _parquet_path("optimised-measurement-decoupled-baseline")
    fig_p = _fig_path("decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZIOptimisedResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline (α_xx = 0, φ=π/4)...")
        N_arr = np.array(N_VALS, dtype=int)
        omega_subset = np.array([v for i, v in enumerate(OMEGA_VALS) if i % 5 == 0])
        result = compute_decoupled_baseline(
            omega_values=omega_subset,
            N_values=N_arr,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Create a verification figure: heatmap of |ratio - 1| on log scale
    from matplotlib.colors import LogNorm

    omega_vals = np.unique(result.omega_values)
    N_vals = np.unique(result.N_values)
    dev_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(result.omega_values, omega) & (result.N_values == N_val)
            if np.any(mask):
                r = float(result.ratio[mask][0])
                dev_map[j, i] = abs(r - 1.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = dev_map[np.isfinite(dev_map)]
    if len(finite) > 0:
        vmin_val = float(np.min(finite))
        vmax_val = float(np.max(finite))
    else:
        vmin_val, vmax_val = 1e-15, 1.0

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        dev_map,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=max(vmin_val, 1e-16), vmax=vmax_val),
    )
    fig.colorbar(
        im, ax=ax, label=r"$|\Delta\omega/\Delta\omega_{\mathrm{SQL}}^{2N} - 1|$"
    )

    max_dev = float(np.max(finite)) if len(finite) > 0 else 0.0
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(
        f"Decoupled Baseline Verification ($\\alpha_{{xx}} = 0$, $\\psi = \\pi/4$, "
        f"$T_hold = {result.T_hold}$)\n"
        f"Max $|\\Delta\\omega/\\mathrm{{SQL}}^{{2N}} - 1| = {max_dev:.2e}$, "
        f"points checked: {len(finite)}"
    )

    fig.tight_layout()
    fig.savefig(fig_p, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {fig_p}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling analysis from the sweep data.

    Args:
        force: If True, overwrite existing figure files.
    """
    csv_p = _parquet_path("optimised-measurement-sweep")
    fig_n3 = _fig_path("n-scaling-omega0.3")
    fig_n1 = _fig_path("n-scaling-omega1.0")
    fig_n3p = _fig_path("n-scaling-omega3.0")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZIOptimisedResult.from_parquet(csv_p)

    # Plot at three representative ω values
    for omega_val, fig_p in [(0.3, fig_n3), (1.0, fig_n1), (3.0, fig_n3p)]:
        if fig_p.exists() and not force:
            print(f"[skip] {fig_p.name} exists (use --force to overwrite)")
            continue
        plot_n_scaling(result, fig_p, omega_fixed=omega_val)
        print(f"[fig]  {fig_p}")


def generate_scaling_analysis(force: bool = False) -> None:
    """Scaling analysis (exponents) from the sweep data."""
    csv_p = _parquet_path("optimised-measurement-sweep")
    scaling_csv = _parquet_path("optimised-measurement-scaling")
    fig_p = _fig_path("scaling-exponents")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZIOptimisedResult.from_parquet(csv_p)

    if scaling_csv.exists() and not force:
        scaling = ScalingAnalysisResult.from_parquet(scaling_csv)
        print(f"[skip] {scaling_csv.name} exists")
    else:
        print("[run]  Fitting scaling exponents...")
        scaling = fit_scaling_exponents(
            result.omega_values,
            result.N_values,
            result.delta_omega_opt,
        )
        scaling.save_parquet(scaling_csv)
        print(f"[save] {scaling_csv}")

    plot_scaling_exponents(scaling, fig_p)
    print(f"[fig]  {fig_p}")


def evaluate_coarse_grid(
    N: int,
    omega: float,
    n_axx: int = 21,
    n_psi: int = 21,
    fd_step: float = FD_STEP,
    T_hold: float = DEFAULT_T_hold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate Δω on a coarse 2D grid of (α_xx, ψ) for landscape validation.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        n_axx: Number of α_xx grid points.
        n_psi: Number of ψ grid points.
        fd_step: Finite-difference step.
        T_hold: Holding time.

    Returns:
        Tuple (axx_vals, psi_vals, delta_map, grid_best_dt) where
        axx_vals: 1D array of α_xx values (n_axx).
        psi_vals: 1D array of ψ values (n_psi).
        delta_map: 2D (n_axx × n_psi) array of Δω values.
        grid_best_dt: Minimum Δω on the grid.
    """
    axx_vals = np.linspace(AXX_BOUNDS[0], AXX_BOUNDS[1], n_axx)
    psi_vals = np.linspace(PSI_BOUNDS[0], PSI_BOUNDS[1], n_psi)
    ops = embed_combined_operators(N)
    psi0 = initial_state(N)

    delta_map = np.full((n_axx, n_psi), np.nan, dtype=float)
    grid_best = float("inf")

    for i, axx in enumerate(axx_vals):
        for j, psi in enumerate(psi_vals):
            dt, _, _, _ = compute_sensitivity_full(
                N,
                psi0,
                omega,
                axx,
                psi,
                ops,
                T_hold=T_hold,
                fd_step=fd_step,
            )
            delta_map[i, j] = dt
            if np.isfinite(dt) and dt < grid_best:
                grid_best = dt

    return axx_vals, psi_vals, delta_map, grid_best


def generate_landscapes(force: bool = False) -> None:
    """Generate 2D landscape contour figures for representative (ω,N) points.

    Also validates BFGS optimisation against a coarse grid for these points.
    """
    # Representative points: (N, ω, filename-suffix)
    rep_points = [
        (1, 0.5, "N1-omega0.5"),
        (5, 2.0, "N5-omega2.0"),
        (20, 4.0, "N20-omega4.0"),
    ]

    # Load sweep data for BFGS comparison if available
    csv_p = _parquet_path("optimised-measurement-sweep")
    sweep = None
    if csv_p.exists():
        try:
            sweep = DualMZIOptimisedResult.from_parquet(csv_p)
        except (FileNotFoundError, ValueError):
            pass

    for N, omega, suffix in rep_points:
        fig_p = _fig_path(f"landscape-{suffix}")
        if fig_p.exists() and not force:
            print(f"[skip] {fig_p.name} exists (use --force to overwrite)")
            continue

        print(f"[run]  Computing landscape for N={N}, ω={omega:.1f} (21×21 grid)...")
        axx_vals, psi_vals, delta_map, grid_best = evaluate_coarse_grid(N, omega)

        sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_T_hold)

        plot_landscape(N, omega, axx_vals, psi_vals, delta_map, sql_2n, fig_p)
        print(f"[fig]  {fig_p}")

        # Validate BFGS result against coarse grid best
        if sweep is not None:
            mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
            if np.any(mask):
                bfgs_dt = float(sweep.delta_omega_opt[mask][0])
                bfgs_axx = float(sweep.alpha_xx_opt[mask][0])
                bfgs_psi = float(sweep.psi_opt[mask][0])
                rel_diff = (
                    abs(bfgs_dt - grid_best) / grid_best if grid_best > 0 else 0.0
                )
                if rel_diff > 0.05 and grid_best < float("inf"):
                    print(
                        f"  [warn] N={N}, ω={omega:.1f}: BFGS Δω={bfgs_dt:.6e} "
                        f"(α_xx={bfgs_axx:.4f}, ψ={bfgs_psi:.4f}) is "
                        f"{rel_diff * 100:.1f}% above grid best Δω={grid_best:.6e} — "
                        "possible local minimum"
                    )
                else:
                    print(
                        f"  [ok]   BFGS Δω={bfgs_dt:.6e} vs grid best={grid_best:.6e} "
                        f"(rel diff {rel_diff * 100:.2f}%)"
                    )


def generate_comparison_traced_out(force: bool = False) -> None:
    """Generate the joint-vs-traced-out comparison figure.

    Loads the 2026-05-22 sweep Parquet and the current sweep Parquet.
    """
    joint_csv = _parquet_path("optimised-measurement-sweep")
    fig_p = _fig_path("comparison-traced-out")

    if not joint_csv.exists():
        print("[skip] Current sweep data not found; run 'sweep' first")
        return

    if fig_p.exists() and not force:
        print(f"[skip] {fig_p.name} exists (use --force to overwrite)")
        return

    joint_result = DualMZIOptimisedResult.from_parquet(joint_csv)

    # Locate the 2026-05-22 sweep Parquet
    traced_path = (
        REPORTS_DIR / "20260522" / "raw_data" / "20260522-dual-mzi-sweep.parquet"
    )

    if not traced_path.exists():
        print(f"[skip] Traced-out data not found at {traced_path}")
        print("[info] Generating joint-only comparison figure...")
        plot_comparison_traced_out(joint_result, save_path=fig_p, traced_data_path=None)
    else:
        print("[run]  Generating joint vs traced-out comparison...")
        plot_comparison_traced_out(
            joint_result, save_path=fig_p, traced_data_path=traced_path
        )

    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-25 report figures and Parquet data",
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of worker processes for parallel sweep (0 = serial)",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Callable[..., None]] = {
        "decoupled-baseline": generate_decoupled_baseline,
        "sweep": lambda **_: generate_sweep(force=args.force, parallel=args.parallel),  # type: ignore[dict-item]
        "n-scaling": generate_n_scaling,
        "scaling-analysis": generate_scaling_analysis,
        "landscapes": generate_landscapes,
        "comparison-traced-out": generate_comparison_traced_out,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            if name == "sweep":
                func()
            else:
                func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
