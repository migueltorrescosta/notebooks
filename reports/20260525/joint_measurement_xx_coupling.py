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
    uv run python reports/20260525/joint_measurement_xx_coupling.py --force

This module is importable via ``importlib.import_module("reports.20260525.joint_measurement_xx_coupling")``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable
from src.visualization.coupling_heatmaps import (
    plot_alpha_opt_heatmap,
    plot_ratio_heatmap,
)

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.coupling_sweep import resolve_sweep_defaults, run_sweep_base
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
    plot_decoupled_baseline_heatmap,
)
from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,  # noqa: F401 — re-exported for tests
    fit_scaling_exponents,  # noqa: F401 — re-exported for tests
    generate_scaling_analysis,
)
from src.analysis.n_scaling_sweep import (
    generate_n_scaling_plots,
)
from src.physics.multi_mzi import (
    build_hold_hamiltonian,  # noqa: F401 — re-exported for tests
    dual_bs_unitary,  # noqa: F401 — re-exported for tests
    embed_combined_operators,
    evolve_circuit,
    hold_unitary_dicke,  # noqa: F401 — re-exported for tests
    single_bs_unitary,  # noqa: F401 — re-exported for tests
)

# Use spawn start method to avoid fork + BLAS threading deadlock. Must run
# before any ProcessPoolExecutor is created, but after all imports.
mp.set_start_method("spawn", force=True)

if TYPE_CHECKING:
    from collections.abc import Callable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL reference)
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
    t_hold: float = DEFAULT_t_hold,
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
        t_hold: Holding time.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        meas_op = build_measurement_operator(N, psi, ops)

    # Evaluate at omega_true
    state = evolve_circuit(N, psi0, omega_true, alpha_xx, ops, T_BS, t_hold)
    exp_val, var_val = full_state_expectation_and_variance(state, meas_op)

    # Central finite difference for ∂⟨M⟩/∂ω
    psi_plus = evolve_circuit(
        N, psi0, omega_true + fd_step, alpha_xx, ops, T_BS, t_hold
    )
    psi_minus = evolve_circuit(
        N, psi0, omega_true - fd_step, alpha_xx, ops, T_BS, t_hold
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
    t_hold: float,
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
        t_hold: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Δω (finite, large if fringe extremum).
    """
    alpha_xx, psi = float(params[0]), float(params[1])
    dt, _, _, _ = compute_sensitivity_full(
        N, psi0, omega, alpha_xx, psi, ops, t_hold=t_hold, fd_step=fd_step
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


def _run_single_joint_start(
    x0: np.ndarray,
    N: int,
    psi0: np.ndarray,
    omega: float,
    ops: dict[str, np.ndarray],
    bounds: list[tuple[float, float]],
    maxiter: int,
    t_hold: float,
    fd_step: float,
    axx_bounds: tuple[float, float],
) -> dict[str, float] | None:
    """Run a single L-BFGS-B optimisation from one starting point.

    Args:
        x0: 2-element starting vector [alpha_xx, psi].
        N: Particle number per subsystem.
        psi0: Initial state.
        omega: Phase rate.
        ops: Embedded operators.
        bounds: L-BFGS-B bounds [(axx_lo, axx_hi), (psi_lo, psi_hi)].
        maxiter: Maximum L-BFGS-B iterations.
        t_hold: Holding time.
        fd_step: Finite-difference step.
        axx_bounds: (min, max) for α_xx clipping.

    Returns:
        Dict with 'alpha_xx_opt', 'psi_opt', 'delta_omega_opt',
        'expectation_M', 'variance_M', 'd_expectation', or None if
        optimisation failed.
    """
    try:
        result = minimize(
            _objective_joint,
            x0,
            args=(N, psi0, omega, ops, t_hold, fd_step),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": maxiter},
        )

        if not result.success:
            return None

        alpha_opt = float(result.x[0])
        psi_opt = float(result.x[1])
        alpha_opt = np.clip(alpha_opt, axx_bounds[0], axx_bounds[1])

        dt_opt, exp_opt, var_opt, d_exp_opt = compute_sensitivity_full(
            N, psi0, omega, alpha_opt, psi_opt, ops, t_hold=t_hold, fd_step=fd_step
        )

        if not np.isfinite(dt_opt):
            return None

        return {
            "alpha_xx_opt": alpha_opt,
            "psi_opt": psi_opt,
            "delta_omega_opt": dt_opt,
            "expectation_M": exp_opt,
            "variance_M": var_opt,
            "d_expectation": d_exp_opt,
        }

    except (ValueError, np.linalg.LinAlgError):
        return None


def _count_starts_near_best(
    all_dt: list[float], best_dt: float, rtol: float = 0.01
) -> int:
    """Count how many deltas are within a relative tolerance of the best.

    Args:
        all_dt: List of converged Δω values.
        best_dt: Best (minimum) Δω.
        rtol: Relative tolerance (default 0.01 = 1%).

    Returns:
        Number of values within rtol of best_dt.
    """
    if best_dt <= 0:
        return 0
    count = 0
    for dt in all_dt:
        if np.isfinite(dt) and abs(dt - best_dt) / best_dt < rtol:
            count += 1
    return count


def _resolve_defaults(
    N: int, psi0: np.ndarray | None, maxiter: int | None
) -> tuple[np.ndarray, int]:
    """Resolve optional psi0 and maxiter to concrete values."""
    _psi0 = initial_state(N) if psi0 is None else psi0
    _maxiter = _maxiter_for_n(N) if maxiter is None else maxiter
    return _psi0, _maxiter


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
    t_hold: float = DEFAULT_t_hold,
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
        t_hold: Holding time.
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
            'sql_2n': SQL = 1/(√(2N) * t_hold) reference.
            'n_starts_converged': number of starts that converged.
            'n_starts_at_best': number of starts that reached the best optimum
                (within 1% relative tolerance of delta_omega_opt).
    """
    _psi0, _maxiter = _resolve_defaults(N, psi0, maxiter)
    sql_2n = 1.0 / (np.sqrt(2 * N) * t_hold)
    rng = np.random.default_rng(rng_seed)
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
    all_dt: list[float] = []

    for _ in range(n_starts):
        alpha0 = rng.uniform(axx_bounds[0], axx_bounds[1])
        psi_start = rng.uniform(psi_bounds[0], psi_bounds[1])
        x0 = np.array([alpha0, psi_start])

        single_result = _run_single_joint_start(
            x0, N, _psi0, omega, ops, bounds, _maxiter, t_hold, fd_step, axx_bounds
        )
        if single_result is None:
            continue

        n_converged += 1
        dt = single_result["delta_omega_opt"]
        all_dt.append(dt)

        if dt < best_result["delta_omega_opt"]:
            best_result["alpha_xx_opt"] = single_result["alpha_xx_opt"]
            best_result["psi_opt"] = single_result["psi_opt"]
            best_result["ms_opt"] = np.cos(single_result["psi_opt"])
            best_result["ma_opt"] = np.sin(single_result["psi_opt"])
            best_result["delta_omega_opt"] = dt
            best_result["expectation_M"] = single_result["expectation_M"]
            best_result["variance_M"] = single_result["variance_M"]
            best_result["d_expectation"] = single_result["d_expectation"]

    best_result["n_starts_converged"] = n_converged
    best_result["n_starts_at_best"] = _count_starts_near_best(
        all_dt, best_result["delta_omega_opt"]
    )
    return best_result


# ============================================================================
# Dataclass for Optimised Sweep Results
# ============================================================================


@dataclass
class DualMZIOptimisedResult(ParquetSerializable):
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
        sql_2n: 2N-SQL = 1/(√(2N) t_hold) at each point.
        ratio: Δω_opt / SQL_2N at each point.
        expectation_M: ⟨M⟩ at optimum.
        variance_M: Var(M) at optimum.
        d_expectation: ∂⟨M⟩/∂ω at optimum.
        n_starts_converged: Number of L-BFGS-B starts that converged per point.
        n_starts_at_best: Number of starts that reached the best optimum
            (within 1% relative tolerance of delta_omega_opt).
        t_hold: Holding time (scalar).
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
    t_hold: float = DEFAULT_t_hold

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "N",
        "t_hold",
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
    ]

    def __post_init__(self) -> None:
        # Ensure int dtype for N_values
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "N": self.N_values,
                "t_hold": np.full(len(self.omega_values), self.t_hold),
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

    @classmethod
    def from_parquet(cls, path: str | Path) -> DualMZIOptimisedResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)

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
            t_hold=float(df["t_hold"].iloc[0]),
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


def _unpack_worker_result(r: dict[str, Any]) -> dict[str, float | int]:
    """Extract typed values from a worker result dict (process-boundary safe)."""
    return {
        "omega": float(r["omega"]),
        "N": int(r["N"]),
        "alpha_xx_opt": float(r["alpha_xx_opt"]),
        "psi_opt": float(r["psi_opt"]),
        "ms_opt": float(r["ms_opt"]),
        "ma_opt": float(r["ma_opt"]),
        "delta_omega_opt": float(r["delta_omega_opt"]),
        "sql_2n": float(r["sql_2n"]),
        "expectation_M": float(r["expectation_M"]),
        "variance_M": float(r["variance_M"]),
        "d_expectation": float(r["d_expectation"]),
        "n_starts_at_best": int(r["n_starts_at_best"]),
        "n_starts_converged": int(r["n_starts_converged"]),
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
        args: Tuple (N, omega, t_hold, n_starts, fd_step, seed).

    Returns:
        Dict with keys: 'N', 'omega', plus all optimise_joint result keys.
    """
    N, omega, t_hold, n_starts, fd_step, seed = args
    ops = embed_combined_operators(N)
    psi0 = initial_state(N)
    result = optimise_joint(
        N=N,
        omega=omega,
        ops=ops,
        psi0=psi0,
        n_starts=n_starts,
        t_hold=t_hold,
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


def _empty_sweep_arrays(total: int) -> dict[str, np.ndarray]:
    """Allocate empty (pre-filled) result arrays for a sweep.

    Args:
        total: Total number of sweep points.

    Returns:
        Dict with all 1D array keys except 'ratios'.
    """
    return {
        "omegas": np.zeros(total, dtype=float),
        "Ns": np.zeros(total, dtype=int),
        "alpha_opts": np.full(total, np.nan, dtype=float),
        "psi_opts": np.full(total, np.nan, dtype=float),
        "ms_opts": np.full(total, np.nan, dtype=float),
        "ma_opts": np.full(total, np.nan, dtype=float),
        "delta_opts": np.full(total, np.inf, dtype=float),
        "sqls": np.zeros(total, dtype=float),
        "exps": np.zeros(total, dtype=float),
        "vars_": np.zeros(total, dtype=float),
        "d_exps": np.zeros(total, dtype=float),
        "n_best_arr": np.zeros(total, dtype=int),
        "n_converged_arr": np.zeros(total, dtype=int),
    }


def _unpack_result_into_arrays(
    r: dict[str, Any],
    idx: int,
    data: dict[str, np.ndarray],
) -> None:
    """Unpack one optimiser result dict into sweep arrays at position idx.

    The result dict must contain keys: omega, N, alpha_xx_opt, psi_opt,
    ms_opt, ma_opt, delta_omega_opt, sql_2n, expectation_M, variance_M,
    d_expectation, n_starts_at_best, n_starts_converged.
    """
    data["omegas"][idx] = float(r["omega"])
    data["Ns"][idx] = int(r["N"])  # type: ignore[call-overload]
    data["alpha_opts"][idx] = float(r["alpha_xx_opt"])
    data["psi_opts"][idx] = float(r["psi_opt"])
    data["ms_opts"][idx] = float(r["ms_opt"])
    data["ma_opts"][idx] = float(r["ma_opt"])
    data["delta_opts"][idx] = float(r["delta_omega_opt"])
    data["sqls"][idx] = float(r["sql_2n"])
    data["exps"][idx] = float(r["expectation_M"])
    data["vars_"][idx] = float(r["variance_M"])
    data["d_exps"][idx] = float(r["d_expectation"])
    data["n_best_arr"][idx] = int(r["n_starts_at_best"])  # type: ignore[call-overload]
    data["n_converged_arr"][idx] = int(r["n_starts_converged"])  # type: ignore[call-overload]


def _compute_sweep_ratios(
    delta_opts: np.ndarray,
    sqls: np.ndarray,
) -> np.ndarray:
    """Compute Δω / SQL ratio, vectorized. Inf where not finite or SQL ≤ 0."""
    return np.where(
        np.isfinite(delta_opts) & (sqls > 0),
        delta_opts / sqls,
        np.inf,
    )


def _compute_sweep_serial(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    t_hold: float,
    seed: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> dict[str, np.ndarray]:
    """Serial sweep over all (N, ω) pairs using per-point optimisations.

    Args:
        omega_values: ω values to sweep.
        N_values: N values to sweep.
        t_hold: Holding time.
        seed: Optional seed for reproducibility.
        progress_callback: Optional callback (current, total).

    Returns:
        Dict of 1D arrays (keys: omegas, Ns, alpha_opts, psi_opts, ms_opts,
        ma_opts, delta_opts, sqls, ratios, exps, vars_, d_exps,
        n_best_arr, n_converged_arr).
    """
    global_start_seed = 0 if seed is None else seed
    n_starts_fn = _n_starts_for_n

    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, Any]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        r = _optimise_one_point(
            (N, omega, t_hold, n_starts_fn(N), FD_STEP, global_start_seed),
        )
        return {
            "delta_omega": r["delta_omega_opt"],
            "sql": r["sql_2n"],
            "expectation_M": r["expectation_M"],
            "variance_M": r["variance_M"],
            "d_expectation": r["d_expectation"],
            "alpha_xx_opt": r["alpha_xx_opt"],
            "psi_opt": r["psi_opt"],
            "ms_opt": r["ms_opt"],
            "ma_opt": r["ma_opt"],
            "n_best_arr": r["n_starts_at_best"],
            "n_converged_arr": r["n_starts_converged"],
        }

    data = run_sweep_base(
        omega_values, N_values, per_point, progress_callback=progress_callback
    )
    # Map keys to match what _assemble_sweep_result expects
    data["ratios"] = _compute_sweep_ratios(data["delta_opts"], data["sqls"])
    data["alpha_opts"] = data["alpha_xx_opt"]
    data["psi_opts"] = data["psi_opt"]
    data["ms_opts"] = data["ms_opt"]
    data["ma_opts"] = data["ma_opt"]
    data["exps"] = data["expectation_M"]
    data["vars_"] = data["variance_M"]
    data["d_exps"] = data["d_expectation"]
    return data


def _build_sweep_tasks(
    N_values: np.ndarray,
    omega_values: np.ndarray,
    t_hold: float,
    fd_step: float,
    global_start_seed: int | None,
) -> list[tuple[int, float, float, int, float, int | None]]:
    """Build optimisation task tuples for all (N, ω) pairs.

    Each task is a picklable tuple consumable by _optimise_one_point.
    """
    return [
        (
            int(N_i),
            float(omega_i),
            t_hold,
            _n_starts_for_n(N_i),
            fd_step,
            global_start_seed,
        )
        for N_i in N_values
        for omega_i in omega_values
    ]


def _get_future_result(
    fut: Any,
    futures: dict[Any, int],
) -> tuple[int, dict[str, Any] | None]:
    """Try to get the result from a completed future, returning (i, r) or (i, None).

    Args:
        fut: Completed future from ProcessPoolExecutor.
        futures: Dict mapping future → task index.

    Returns:
        Tuple (task_index, result_dict_or_None).
    """
    i = futures[fut]
    try:
        return i, fut.result()
    except Exception as exc:
        print(f"  [err] Worker failed for task {i}: {exc}")
        return i, None


def _on_result_received(
    r: dict[str, Any],
    i: int,
    n_done: int,
    total: int,
    data: dict[str, np.ndarray],
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    """Process one successful optimisation result: unpack + report progress."""
    _unpack_result_into_arrays(r, i, data)
    if progress_callback is not None:
        progress_callback(n_done, total)
    if n_done % 50 == 0:
        print(f"  [progress] {n_done}/{total} tasks completed")


def _compute_sweep_parallel(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    t_hold: float,
    seed: int | None,
    progress_callback: Callable[[int, int], None] | None,
    parallel: int,
) -> dict[str, np.ndarray]:
    """Parallel sweep over all (N, ω) pairs via ProcessPoolExecutor.

    Args:
        omega_values: ω values to sweep.
        N_values: N values to sweep.
        t_hold: Holding time.
        seed: Optional seed for reproducibility.
        progress_callback: Optional callback (current, total).
        parallel: Number of worker processes.

    Returns:
        Dict of 1D arrays (same keys as _compute_sweep_serial).
    """
    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N
    global_start_seed = 0 if seed is None else seed

    # Build tasks: one per (N, ω) pair with adaptive n_starts
    tasks = _build_sweep_tasks(
        N_values, omega_values, t_hold, FD_STEP, global_start_seed
    )

    print(f"[info] Parallel sweep: {len(tasks)} tasks, {parallel} workers")
    n_done = 0
    data = _empty_sweep_arrays(total)
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_optimise_one_point, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, r = _get_future_result(fut, futures)
            if r is not None:
                n_done += 1
                _on_result_received(r, i, n_done, total, data, progress_callback)

    data["ratios"] = _compute_sweep_ratios(data["delta_opts"], data["sqls"])
    return data


def _assemble_sweep_result(
    data: dict[str, np.ndarray],
    t_hold: float,
) -> DualMZIOptimisedResult:
    """Assemble DualMZIOptimisedResult from computed array dict.

    Args:
        data: Dict with keys matching _compute_sweep_* output.
        t_hold: Holding time (scalar attribute).

    Returns:
        Fully populated DualMZIOptimisedResult.
    """
    return DualMZIOptimisedResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        alpha_xx_opt=data["alpha_opts"],
        psi_opt=data["psi_opts"],
        ms_opt=data["ms_opts"],
        ma_opt=data["ma_opts"],
        delta_omega_opt=data["delta_opts"],
        sql_2n=data["sqls"],
        ratio=data["ratios"],
        expectation_M=data["exps"],
        variance_M=data["vars_"],
        d_expectation=data["d_exps"],
        n_starts_converged=data["n_converged_arr"],
        n_starts_at_best=data["n_best_arr"],
        t_hold=t_hold,
    )


def run_sweep(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    t_hold: float = DEFAULT_t_hold,
    n_starts: int = N_RANDOM_STARTS,
    progress_callback: Callable[[int, int], None] | None = None,
    seed: int | None = None,
    parallel: int = 0,
) -> DualMZIOptimisedResult:
    """Run the full 2D sweep over ω and N with joint (α_xx, φ) optimisation.

    Args:
        omega_values: ω values to sweep (default: 0.1 to 5.0 step 0.1).
        N_values: N values to sweep (default: 1 to 20 inclusive).
        t_hold: Holding time.
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

    if parallel > 0:
        data = _compute_sweep_parallel(
            omega_values,
            N_values,
            t_hold,
            seed,
            progress_callback,
            parallel,
        )
    else:
        data = _compute_sweep_serial(
            omega_values,
            N_values,
            t_hold,
            seed,
            progress_callback,
        )

    return _assemble_sweep_result(data, t_hold)


# ============================================================================
# Decoupled Baseline Verification (α_xx = 0, φ-sweep)
# ============================================================================


def compute_sensitivity_at_psi(
    N: int,
    omega: float,
    psi: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float]:
    """Compute sensitivity at α_xx=0 with a given ψ.

    At α_xx = 0, this should give:
    - ψ = 0:     Δω = 1/(√N t_hold) (the N-SQL, worse than 2N-SQL by √2)
    - ψ = π/4:   Δω = 1/(√(2N) t_hold) = Δω_SQL (the 2N-SQL, optimal separable)

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        psi: Measurement weight angle.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        t_hold: Holding time.
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
        t_hold=t_hold,
        fd_step=fd_step,
    )


def compute_decoupled_baseline(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
) -> DualMZIOptimisedResult:
    """Verify the decoupled baseline (α_xx = 0) at the analytically optimal φ = π/4.

    At α_xx = 0, the optimal measurement is φ = π/4, giving
    Δω = 1/(√(2N) t_hold) = Δω_SQL (the 2N-SQL).

    Args:
        omega_values: ω values (default: sweep range).
        N_values: N values (default: 1 to 20).
        t_hold: Holding time.

    Returns:
        DualMZIOptimisedResult with α_xx=0, φ=π/4 results.
    """
    omega_values, N_values = resolve_sweep_defaults(
        omega_values,
        N_values,
        default_omega_range=(OMEGA_MIN, OMEGA_MAX, OMEGA_STEP),
        default_N_range=(N_MIN, N_MAX),
    )

    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, Any]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        sql = 1.0 / (np.sqrt(2 * N) * t_hold)
        dt, exp_val, var_val, d_exp_val = compute_sensitivity_at_psi(
            N,
            omega,
            psi=np.pi / 4.0,
            ops=_cache["ops"],
            psi0=_cache["psi0"],
            t_hold=t_hold,
            fd_step=fd_step,
        )
        ratio = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
        return {
            "delta_omega": dt,
            "sql": sql,
            "expectation_M": exp_val,
            "variance_M": var_val,
            "d_expectation": d_exp_val,
            "alpha_xx_opt": 0.0,
            "psi_opt": np.pi / 4.0,
            "ms_opt": np.cos(np.pi / 4.0),
            "ma_opt": np.sin(np.pi / 4.0),
            "n_best_arr": 0,
            "n_converged_arr": 0,
            "ratio": ratio,
        }

    data = run_sweep_base(omega_values, N_values, per_point)

    return DualMZIOptimisedResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        alpha_xx_opt=data["alpha_xx_opt"],
        psi_opt=data["psi_opt"],
        ms_opt=data["ms_opt"],
        ma_opt=data["ma_opt"],
        delta_omega_opt=data["delta_opts"],
        sql_2n=data["sqls"],
        ratio=data["ratio"],
        expectation_M=data["expectation_M"],
        variance_M=data["variance_M"],
        d_expectation=data["d_expectation"],
        n_starts_converged=data["n_converged_arr"],
        n_starts_at_best=data["n_best_arr"],
        t_hold=t_hold,
    )


# ============================================================================
# Plot Functions
# ============================================================================


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

    Shows r_joint = Δω_opt / (1/√(2N) t_hold) vs r_trace = Δω_trace / (1/√N t_hold).

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


parquet_path, fig_path = report_path_fn(REPORTS_DIR, _REPORT_DATE_PREFIX)

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

    plot_ratio_heatmap(
        result.omega_values,
        result.N_values,
        result.ratio,
        fig_ratio,
        title="Sensitivity Ratio: Optimised Joint Measurement\n(lower = better; 1.0 = 2N-SQL)",
        cbar_label=r"$\Delta\omega_{\mathrm{opt}} / \Delta\omega_{\mathrm{SQL}}^{2N}$",
    )
    print(f"[fig]  {fig_ratio}")
    plot_alpha_opt_heatmap(
        result.omega_values,
        result.N_values,
        result.alpha_xx_opt,
        fig_alpha,
        vmin=0.0,
        title=r"Optimal $\alpha_{xx}$: Joint Optimised Measurement",
    )
    print(f"[fig]  {fig_alpha}")
    plot_psi_opt_heatmap(result, fig_psi)

    print(f"[fig]  {fig_psi}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling analysis from the sweep data.

    Args:
        force: If True, overwrite existing figure files.
    """
    omega_fig_pairs = [
        (0.3, _fig_path("n-scaling-omega0.3")),
        (1.0, _fig_path("n-scaling-omega1.0")),
        (3.0, _fig_path("n-scaling-omega3.0")),
    ]

    def _plot_with_title(
        df: Any,
        omega_fixed: float,
        save_path: Path,
        t_hold: float | None = None,
        include_2n_sql: bool = False,
    ) -> None:
        from src.visualization.scaling_plots import plot_n_scaling_single_omega

        title = (
            f"N-Scaling at $\\omega={omega_fixed:.2f}$:\nOptimised Joint Measurement"
        )
        plot_n_scaling_single_omega(
            df,
            omega_fixed=omega_fixed,
            save_path=save_path,
            t_hold=t_hold,
            include_2n_sql=include_2n_sql,
            title=title,
        )

    generate_n_scaling_plots(
        force=force,
        parquet_path=_parquet_path("optimised-measurement-sweep"),
        result_cls=DualMZIOptimisedResult,
        omega_fig_pairs=omega_fig_pairs,
        include_2n_sql=True,
        plot_fn=_plot_with_title,
        label="optimised-measurement n-scaling",
    )


def evaluate_coarse_grid(
    N: int,
    omega: float,
    n_axx: int = 21,
    n_psi: int = 21,
    fd_step: float = FD_STEP,
    t_hold: float = DEFAULT_t_hold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate Δω on a coarse 2D grid of (α_xx, ψ) for landscape validation.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        n_axx: Number of α_xx grid points.
        n_psi: Number of ψ grid points.
        fd_step: Finite-difference step.
        t_hold: Holding time.

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
                t_hold=t_hold,
                fd_step=fd_step,
            )
            delta_map[i, j] = dt
            if np.isfinite(dt) and dt < grid_best:
                grid_best = dt

    return axx_vals, psi_vals, delta_map, grid_best


def _load_sweep_for_validation() -> DualMZIOptimisedResult | None:
    """Load sweep Parquet for BFGS vs grid validation, or None if unavailable."""
    csv_p = _parquet_path("optimised-measurement-sweep")
    if csv_p.exists():
        try:
            return DualMZIOptimisedResult.from_parquet(csv_p)
        except (FileNotFoundError, ValueError):
            pass
    return None


def _validate_bfgs_against_grid(
    sweep: DualMZIOptimisedResult,
    N: int,
    omega: float,
    grid_best: float,
) -> None:
    """Compare BFGS-optimised result to coarse-grid minimum at (N, ω).

    Prints a warning if the relative difference exceeds 5%, otherwise OK.
    """
    mask = np.isclose(sweep.omega_values, omega) & (sweep.N_values == N)
    if not np.any(mask):
        return

    bfgs_dt = float(sweep.delta_omega_opt[mask][0])
    bfgs_axx = float(sweep.alpha_xx_opt[mask][0])
    bfgs_psi = float(sweep.psi_opt[mask][0])
    rel_diff = abs(bfgs_dt - grid_best) / grid_best if grid_best > 0 else 0.0

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


def _process_landscape_point(
    N: int,
    omega: float,
    suffix: str,
    sweep: DualMZIOptimisedResult | None,
    force: bool,
) -> bool:
    """Compute, plot, and BFGS-validate one landscape point.

    Returns True if the figure was generated, False if skipped (cached).
    """
    fig_p = _fig_path(f"landscape-{suffix}")
    if fig_p.exists() and not force:
        print(f"[skip] {fig_p.name} exists (use --force to overwrite)")
        return False

    print(f"[run]  Computing landscape for N={N}, ω={omega:.1f} (21×21 grid)...")
    axx_vals, psi_vals, delta_map, grid_best = evaluate_coarse_grid(N, omega)
    sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)

    plot_landscape(N, omega, axx_vals, psi_vals, delta_map, sql_2n, fig_p)
    print(f"[fig]  {fig_p}")

    if sweep is not None:
        _validate_bfgs_against_grid(sweep, N, omega, grid_best)

    return True


def generate_landscapes(force: bool = False) -> None:
    """Generate 2D landscape contour figures for representative (ω,N) points.

    Also validates BFGS optimisation against a coarse grid for these points.
    """
    rep_points = [
        (1, 0.5, "N1-omega0.5"),
        (5, 2.0, "N5-omega2.0"),
        (20, 4.0, "N20-omega4.0"),
    ]
    sweep = _load_sweep_for_validation()

    for N, omega, suffix in rep_points:
        _process_landscape_point(N, omega, suffix, sweep, force)


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

    tasks: dict[str, Callable[..., Any]] = {
        "decoupled-baseline": lambda force=False: generate_decoupled_baseline(
            force=force,
            parquet_path=_parquet_path("optimised-measurement-decoupled-baseline"),
            fig_path=_fig_path("decoupled-baseline"),
            compute_fn=compute_decoupled_baseline,
            compute_kwargs={
                "omega_values": np.array(
                    [v for i, v in enumerate(OMEGA_VALS) if i % 5 == 0]
                ),
                "N_values": np.array(N_VALS, dtype=int),
            },
            result_cls=DualMZIOptimisedResult,
            plot_fn=lambda r, p: plot_decoupled_baseline_heatmap(
                r,
                p,
                title_prefix=(
                    r"Decoupled Baseline Verification ($\alpha_{xx} = 0$, $\psi = \pi/4$"
                ),
                sql_label=r"$|\Delta\omega/\Delta\omega_{\mathrm{SQL}}^{2N} - 1|$",
                sql_ref_label="SQL^{2N}",
            ),
            label="decoupled baseline (α_xx = 0, φ=π/4)",
        ),
        "sweep": lambda **_: generate_sweep(force=args.force, parallel=args.parallel),  # type: ignore[dict-item]
        "n-scaling": generate_n_scaling,
        "scaling-analysis": partial(
            generate_scaling_analysis,
            parquet_path=_parquet_path("optimised-measurement-sweep"),
            scaling_path=_parquet_path("optimised-measurement-scaling"),
            fig_path=_fig_path("scaling-exponents"),
            result_cls=DualMZIOptimisedResult,
            label="optimised-measurement scaling analysis",
        ),
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
