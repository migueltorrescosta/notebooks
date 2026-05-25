"""
Local module for the 2026-05-22 Multi-Particle XX-Coupling Dual-MZI report.

Contains all code exclusive to this report:
- Multi-particle operator construction (Dicke basis, N up to 20)
- Dual MZI circuit: BS(S)⊗BS(A) → Hold → BS(S)⊗BS(A) → Tr_A → measure J_z^S
- Sensitivity via error propagation with central finite differences
- α_xx optimisation (coarse grid + bounded 1D refinement) per (θ, N) pair
- Full 2D sweep over θ ∈ [0.1, 5.0] and N ∈ [1, 20]
- Scaling analysis (log-log fit Δθ ∝ N^α)
- Exclusive plot functions for heatmaps and scaling curves
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/2026-05-22/local.py --force

This module is **not** importable as ``reports.2026-05-22.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

# Ensure project root is on sys.path for shared-module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_H: float = 10.0  # Holding time (SQL reference)
AXX_BOUNDS: tuple[float, float] = (0.0, 20.0)  # α_xx optimisation range
N_COARSE_GRID: int = 101  # Coarse grid points for α_xx scan
FD_STEP: float = 1e-6  # Central finite-difference step

# θ and N sweep ranges (from report)
THETA_MIN: float = 0.1
THETA_MAX: float = 5.0
THETA_STEP: float = 0.1
N_MIN: int = 1
N_MAX: int = 20


# ============================================================================
# Operator / Hamiltonian Construction (Multi-Particle Dicke Basis)
# ============================================================================


def dicke_single_operators(N: int) -> dict[str, np.ndarray]:
    """Build single-subsystem Dicke-basis operators for N particles.

    Returns (N+1) × (N+1) matrices for J_z, J_x, J_y in the Dicke basis
    with m descending from +N/2 to -N/2.

    Args:
        N: Number of particles (dim = N+1).

    Returns:
        Dict with keys 'Jz', 'Jx', 'Jy'.
    """
    return {
        "Jz": jz_operator(N),
        "Jx": jx_operator(N),
        "Jy": jy_operator(N),
    }


def embed_combined_operators(N: int) -> dict[str, np.ndarray]:
    """Build (N+1)² × (N+1)² operators in the combined S⊗A space.

    J_k^S = J_k ⊗ I_{N+1}
    J_k^A = I_{N+1} ⊗ J_k

    The basis ordering is |m_S, m_A⟩ with both m descending from +J to -J.
    System index rows, ancilla index columns.

    Args:
        N: Particle number per subsystem (dim = N+1 per subsystem).

    Returns:
        Dict with keys 'Jz_S', 'Jz_A', 'Jx_S', 'Jx_A', 'Jy_S', 'Jy_A'.
    """
    single = dicke_single_operators(N)
    eye = np.eye(N + 1, dtype=float)
    return {
        "Jz_S": np.kron(single["Jz"], eye),
        "Jz_A": np.kron(eye, single["Jz"]),
        "Jx_S": np.kron(single["Jx"], eye),
        "Jx_A": np.kron(eye, single["Jx"]),
        "Jy_S": np.kron(single["Jy"], eye),
        "Jy_A": np.kron(eye, single["Jy"]),
    }


def build_hold_hamiltonian(
    N: int,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the total holding Hamiltonian in the combined S⊗A space.

    H = θ (J_z^S + J_z^A) + α_xx J_x^S J_x^A

    Args:
        N: Particle number per subsystem.
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed embedded operators. If None, built fresh.

    Returns:
        (N+1)² × (N+1)² Hermitian Hamiltonian matrix.
    """
    if ops is None:
        ops = embed_combined_operators(N)
    dim = (N + 1) ** 2
    H = np.zeros((dim, dim), dtype=complex)
    H += theta * (ops["Jz_S"] + ops["Jz_A"])
    if alpha_xx != 0.0:
        # J_x^S J_x^A = (J_x ⊗ I)(I ⊗ J_x) = J_x ⊗ J_x
        H += alpha_xx * (ops["Jx_S"] @ ops["Jx_A"])
    return 0.5 * (H + H.conj().T)


def single_bs_unitary(N: int, T: float = DEFAULT_T_BS) -> np.ndarray:
    """Single-subsystem 50/50 beam-splitter unitary.

    U_BS = exp(-i T J_x)

    Args:
        N: Particle number (dim = N+1).
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1) × (N+1) unitary matrix.
    """
    Jx = jx_operator(N)
    U = expm(-1j * T * Jx)
    assert np.allclose(U @ U.conj().T, np.eye(N + 1), atol=1e-12)
    return U


def dual_bs_unitary(N: int, T: float = DEFAULT_T_BS) -> np.ndarray:
    """Dual beam-splitter unitary: BS on both S and A.

    U_BS = exp(-i T J_x) ⊗ exp(-i T J_x)

    Args:
        N: Particle number per subsystem.
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    U_single = single_bs_unitary(N, T)
    return np.kron(U_single, U_single)


def hold_unitary(
    N: int,
    T_H: float,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Holding-time unitary in the combined S⊗A space.

    U_hold(T_H) = exp(-i T_H H)

    Args:
        N: Particle number per subsystem.
        T_H: Holding time.
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed operators.

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    H = build_hold_hamiltonian(N, theta, alpha_xx, ops)
    U = expm(-1j * T_H * H)
    dim = (N + 1) ** 2
    assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12)
    return U


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


def evolve_circuit(
    N: int,
    psi0: np.ndarray,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    T_BS: float = DEFAULT_T_BS,
    T_H: float = DEFAULT_T_H,
) -> np.ndarray:
    """Run the full dual-MZI circuit.

    |ψ_final⟩ = U_BS · U_hold(T_H) · U_BS · |ψ₀⟩

    where U_BS acts on both S and A (dual MZI).

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector (length (N+1)²).
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        T_BS: Beam-splitter angle (default π/2).
        T_H: Holding time (default 10).

    Returns:
        Final state vector (length (N+1)²).
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"
    U_bs = dual_bs_unitary(N, T_BS)
    psi = U_bs @ psi0
    psi = hold_unitary(N, T_H, theta, alpha_xx, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


# ============================================================================
# Ancilla Trace-Out and Reduced Variance
# ============================================================================


def _reduced_expectation(psi_state: np.ndarray, N: int, Jz: np.ndarray) -> float:
    """Compute ⟨J_z⟩ from the reduced density matrix of the system.

    Args:
        psi_state: Pure state vector in S⊗A space (length (N+1)²).
        N: Particle number per subsystem.
        Jz: (N+1)×(N+1) single-subsystem J_z operator.

    Returns:
        Expectation value ⟨J_z^S⟩.
    """
    psi_m = psi_state.reshape(N + 1, N + 1)
    rho = psi_m @ psi_m.conj().T
    return float(np.real(np.trace(rho @ Jz)))


def compute_reduced_expectation_and_variance(
    psi: np.ndarray,
    N: int,
    meas_op: np.ndarray,
) -> tuple[float, float]:
    """Compute ⟨J_z^S⟩ and Var(J_z^S) after tracing out the ancilla.

    The final pure state |ψ⟩ of length (N+1)² is reshaped into an
    (N+1) × (N+1) matrix (rows = system index, columns = ancilla index).
    The reduced density matrix is ρ_S = Ψ Ψ^†.

    Args:
        psi: Final pure state vector (length (N+1)²).
        N: Particle number per subsystem.
        meas_op: (N+1) × (N+1) measurement operator (e.g., J_z).

    Returns:
        Tuple (expectation, variance).
    """
    # Reshape: rows = system, columns = ancilla
    psi_mat = psi.reshape(N + 1, N + 1)  # (N+1, N+1)
    rho_S = psi_mat @ psi_mat.conj().T  # (N+1, N+1)

    # Verify trace preservation
    trace = float(np.real(np.trace(rho_S)))
    assert np.isclose(trace, 1.0, atol=1e-12), f"Reduced trace = {trace} != 1"

    # Expectation and variance
    exp_val = float(np.real(np.trace(rho_S @ meas_op)))
    op_sq = meas_op @ meas_op
    exp_sq = float(np.real(np.trace(rho_S @ op_sq)))
    raw_var = exp_sq - exp_val**2

    # Clamp negative variance near zero (numerical round-off)
    if raw_var < 0 and raw_var > -1e-12:
        raw_var = 0.0
    assert raw_var >= -1e-12, f"Unphysical negative variance: {raw_var:.2e}"

    return float(exp_val), float(max(0.0, raw_var))


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_sensitivity(
    N: int,
    psi0: np.ndarray,
    theta_true: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
    T_BS: float = DEFAULT_T_BS,
    T_H: float = DEFAULT_T_H,
) -> tuple[float, float, float, float]:
    """Compute the error-propagation sensitivity Δθ.

    Δθ = √Var(J_z^S) / |∂⟨J_z^S⟩/∂θ|

    Also returns ⟨J_z^S⟩, Var(J_z^S), and ∂⟨J_z^S⟩/∂θ at theta_true.

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector.
        theta_true: True phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        meas_op: (N+1)×(N+1) measurement operator (default = J_z^S single).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        T_H: Holding time.

    Returns:
        Tuple (delta_theta, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    psi = evolve_circuit(N, psi0, theta_true, alpha_xx, ops, T_BS, T_H)
    # Use single-subsystem J_z for the reduced expectation/variance
    Jz_single = jz_operator(N)
    exp_val, var_val = compute_reduced_expectation_and_variance(psi, N, Jz_single)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    psi_plus = evolve_circuit(N, psi0, theta_true + fd_step, alpha_xx, ops, T_BS, T_H)
    psi_minus = evolve_circuit(N, psi0, theta_true - fd_step, alpha_xx, ops, T_BS, T_H)

    exp_plus = _reduced_expectation(psi_plus, N, Jz_single)
    exp_minus = _reduced_expectation(psi_minus, N, Jz_single)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_theta = float(np.sqrt(var_val) / abs(d_exp))
    return delta_theta, exp_val, var_val, d_exp


# ============================================================================
# α_xx Optimisation (Coarse Grid + Bounded 1D Refinement)
# ============================================================================


def optimise_alpha_xx(
    N: int,
    theta: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    axx_bounds: tuple[float, float] = AXX_BOUNDS,
    n_coarse: int = N_COARSE_GRID,
    T_BS: float = DEFAULT_T_BS,
    T_H: float = DEFAULT_T_H,
    fd_step: float = FD_STEP,
) -> dict[str, float]:
    """Optimise Δθ over α_xx for a given (θ, N) pair.

    Two-stage approach:
    Stage 1: Evaluate Δθ on a coarse grid of n_coarse points.
    Stage 2: Bounded 1D refinement via scipy.optimize.minimize_scalar,
             seeded at the best grid point.

    Args:
        N: Particle number per subsystem.
        theta: Phase rate.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        axx_bounds: (min, max) for α_xx.
        n_coarse: Number of coarse grid points.
        T_BS: Beam-splitter angle.
        T_H: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Dict with keys:
            'alpha_xx_opt': optimal α_xx value.
            'delta_theta_opt': minimal Δθ.
            'expectation_Jz': ⟨J_z^S⟩ at optimum.
            'variance_Jz': Var(J_z^S) at optimum.
            'd_expectation': ∂⟨J_z^S⟩/∂θ at optimum.
            'sql': SQL = 1/(√N * T_H) reference.
    """
    if psi0 is None:
        psi0 = initial_state(N)

    sql = 1.0 / (np.sqrt(N) * T_H)

    # Stage 1: coarse grid
    alpha_vals = np.linspace(axx_bounds[0], axx_bounds[1], n_coarse)
    delta_vals = np.full(n_coarse, np.inf, dtype=float)

    for i, a_val in enumerate(alpha_vals):
        dt, _, _, _ = compute_sensitivity(N, psi0, theta, a_val, ops, fd_step=fd_step)
        delta_vals[i] = dt

    # Find best grid point
    finite_mask = np.isfinite(delta_vals)
    if not np.any(finite_mask):
        return {
            "alpha_xx_opt": float("nan"),
            "delta_theta_opt": float("inf"),
            "expectation_Jz": 0.0,
            "variance_Jz": 0.0,
            "d_expectation": 0.0,
            "sql": sql,
        }

    finite_indices = np.where(finite_mask)[0]
    best_idx = finite_indices[int(np.argmin(delta_vals[finite_mask]))]
    seed_alpha = float(alpha_vals[best_idx])

    # Stage 2: bounded 1D refinement
    def _objective(a: float) -> float:
        dt, _, _, _ = compute_sensitivity(N, psi0, theta, a, ops, fd_step=fd_step)
        return dt if np.isfinite(dt) else 1e10

    result = minimize_scalar(
        _objective,
        bounds=axx_bounds,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 100},
    )

    alpha_opt = float(result.x)
    delta_opt = float(result.fun)

    # Re-evaluate at optimum for expectation and variance
    dt_opt, exp_opt, var_opt, d_exp = compute_sensitivity(
        N, psi0, theta, alpha_opt, ops, fd_step=fd_step
    )

    # If the refine result is worse than the grid, prefer the grid result
    if delta_opt > delta_vals[best_idx]:
        alpha_opt = seed_alpha
        dt_opt, exp_opt, var_opt, d_exp = compute_sensitivity(
            N, psi0, theta, alpha_opt, ops, fd_step=fd_step
        )
        delta_opt = dt_opt

    return {
        "alpha_xx_opt": alpha_opt,
        "delta_theta_opt": delta_opt,
        "expectation_Jz": exp_opt,
        "variance_Jz": var_opt,
        "d_expectation": d_exp,
        "sql": sql,
    }


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class DualMZISweepResult:
    """Full 2D sweep over θ and N with α_xx optimisation per point.

    All array fields have the same length (n_theta × n_N), stored in
    row-major order (θ varies fastest, then N).

    Attributes:
        theta_values: θ values for each point.
        N_values: N values for each point.
        alpha_xx_opt: Optimal α_xx at each point.
        delta_theta_opt: Minimal Δθ at each point.
        sql_values: SQL = 1/(√N T_H) at each point.
        ratio: Δθ_opt / SQL at each point.
        expectation_Jz: ⟨J_z^S⟩ at optimum.
        variance_Jz: Var(J_z^S) at optimum.
        d_expectation: ∂⟨J_z^S⟩/∂θ at optimum.
        T_H: Holding time (scalar).
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    alpha_xx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_theta_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    d_expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    T_H: float = DEFAULT_T_H

    def __post_init__(self) -> None:
        # Ensure int dtype for N_values
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "theta": self.theta_values,
                "N": self.N_values,
                "T_H": np.full(len(self.theta_values), self.T_H),
                "alpha_xx_opt": self.alpha_xx_opt,
                "delta_theta_opt": self.delta_theta_opt,
                "sql": self.sql_values,
                "ratio": self.ratio,
                "expectation_Jz": self.expectation_Jz,
                "variance_Jz": self.variance_Jz,
                "d_expectation": self.d_expectation,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DualMZISweepResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "N",
            "T_H",
            "alpha_xx_opt",
            "delta_theta_opt",
            "sql",
            "ratio",
            "expectation_Jz",
            "variance_Jz",
            "d_expectation",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )

        return cls(
            theta_values=df["theta"].to_numpy(dtype=float),
            N_values=df["N"].to_numpy(dtype=int),
            alpha_xx_opt=df["alpha_xx_opt"].to_numpy(dtype=float),
            delta_theta_opt=df["delta_theta_opt"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
            ratio=df["ratio"].to_numpy(dtype=float),
            expectation_Jz=df["expectation_Jz"].to_numpy(dtype=float),
            variance_Jz=df["variance_Jz"].to_numpy(dtype=float),
            d_expectation=df["d_expectation"].to_numpy(dtype=float),
            T_H=float(df["T_H"].iloc[0]),
        )

    @property
    def n_points(self) -> int:
        return len(self.theta_values)

    @property
    def n_theta_unique(self) -> int:
        return len(np.unique(self.theta_values))

    @property
    def n_N_unique(self) -> int:
        return len(np.unique(self.N_values))


# ============================================================================
# Sweep Execution
# ============================================================================


def run_sweep(
    theta_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    T_H: float = DEFAULT_T_H,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DualMZISweepResult:
    """Run the full 2D sweep over θ and N with α_xx optimisation.

    Args:
        theta_values: θ values to sweep (default: 0.1 to 5.0 step 0.1).
        N_values: N values to sweep (default: 1 to 20 inclusive).
        T_H: Holding time.
        progress_callback: Optional callback (current, total).

    Returns:
        DualMZISweepResult with all optimised points.
    """
    if theta_values is None:
        theta_values = np.arange(THETA_MIN, THETA_MAX + 1e-9, THETA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_theta = len(theta_values)
    n_N = len(N_values)
    total = n_theta * n_N

    thetas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    alpha_opts = np.full(total, np.nan, dtype=float)
    delta_opts = np.full(total, np.inf, dtype=float)
    sqls = np.zeros(total, dtype=float)
    ratios = np.full(total, np.inf, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)

    idx = 0
    for N in N_values:
        ops = embed_combined_operators(N)
        psi0 = initial_state(N)
        for theta in theta_values:
            thetas[idx] = theta
            Ns[idx] = N

            opt_result = optimise_alpha_xx(
                N=N,
                theta=theta,
                ops=ops,
                psi0=psi0,
                T_H=T_H,
            )
            alpha_opts[idx] = opt_result["alpha_xx_opt"]
            delta_opts[idx] = opt_result["delta_theta_opt"]
            sqls[idx] = opt_result["sql"]
            ratios[idx] = (
                opt_result["delta_theta_opt"] / opt_result["sql"]
                if np.isfinite(opt_result["delta_theta_opt"]) and opt_result["sql"] > 0
                else float("inf")
            )
            exps[idx] = opt_result["expectation_Jz"]
            vars_[idx] = opt_result["variance_Jz"]
            d_exps[idx] = opt_result["d_expectation"]

            idx += 1
            if progress_callback is not None:
                progress_callback(idx, total)

    return DualMZISweepResult(
        theta_values=thetas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        delta_theta_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        T_H=T_H,
    )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def compute_decoupled_baseline(
    theta_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    T_H: float = DEFAULT_T_H,
    fd_step: float = FD_STEP,
) -> DualMZISweepResult:
    """Verify the decoupled baseline (α_xx = 0) for all (θ, N) pairs.

    At α_xx = 0, the sensitivity should equal SQL = 1/(√N T_H).

    Args:
        theta_values: θ values (default: sweep range).
        N_values: N values (default: 1 to 20).
        T_H: Holding time.

    Returns:
        DualMZISweepResult with α_xx=0 results.
    """
    if theta_values is None:
        theta_values = np.arange(THETA_MIN, THETA_MAX + 1e-9, THETA_STEP)
    if N_values is None:
        N_values = np.arange(N_MIN, N_MAX + 1, dtype=int)

    n_theta = len(theta_values)
    n_N = len(N_values)
    total = n_theta * n_N

    thetas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    sqls = np.zeros(total, dtype=float)
    delta_opts = np.zeros(total, dtype=float)
    alpha_opts = np.zeros(total, dtype=float)  # all zero
    ratios = np.zeros(total, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)

    idx = 0
    for N in N_values:
        ops = embed_combined_operators(N)
        psi0 = initial_state(N)
        for theta in theta_values:
            thetas[idx] = theta
            Ns[idx] = N
            sql = 1.0 / (np.sqrt(N) * T_H)
            sqls[idx] = sql

            dt, exp_val, var_val, d_exp_val = compute_sensitivity(
                N, psi0, theta, 0.0, ops, T_H=T_H, fd_step=fd_step
            )
            delta_opts[idx] = dt
            ratios[idx] = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
            exps[idx] = exp_val
            vars_[idx] = var_val
            d_exps[idx] = d_exp_val

            idx += 1

    return DualMZISweepResult(
        theta_values=thetas,
        N_values=Ns,
        alpha_xx_opt=alpha_opts,
        delta_theta_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        T_H=T_H,
    )


# ============================================================================
# Scaling Analysis
# ============================================================================


@dataclass
class ScalingAnalysisResult:
    """Log-log fit results for Δθ ∝ N^α at each θ value.

    Attributes:
        theta_values: θ values analysed.
        exponents: Exponent α from Δθ ∝ N^α for each θ.
        prefactors: Prefactor C in Δθ = C N^α for each θ.
        r_squared: R² goodness-of-fit for each θ.
        sql_exponent: SQL exponent = -0.5 (reference).
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    exponents: np.ndarray = field(default_factory=lambda: np.array([]))
    prefactors: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_exponent: float = -0.5

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "theta": self.theta_values,
                "exponent": self.exponents,
                "prefactor": self.prefactors,
                "r_squared": self.r_squared,
                "sql_exponent": [self.sql_exponent] * len(self.theta_values),
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> ScalingAnalysisResult:
        df = pd.read_parquet(path)
        required = {"theta", "exponent", "prefactor", "r_squared", "sql_exponent"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        return cls(
            theta_values=df["theta"].to_numpy(dtype=float),
            exponents=df["exponent"].to_numpy(dtype=float),
            prefactors=df["prefactor"].to_numpy(dtype=float),
            r_squared=df["r_squared"].to_numpy(dtype=float),
            sql_exponent=float(df["sql_exponent"].iloc[0]),
        )


def fit_scaling_exponents(sweep: DualMZISweepResult) -> ScalingAnalysisResult:
    """Fit Δθ = C N^α at each θ from the sweep data.

    Performs a log-log linear fit: log(Δθ) = α log(N) + log(C).

    Args:
        sweep: Sweep result with (θ, N) points.

    Returns:
        ScalingAnalysisResult with exponent α and prefactor C at each θ.
    """
    theta_vals = np.unique(sweep.theta_values)
    exponents = np.full(len(theta_vals), np.nan, dtype=float)
    prefactors = np.full(len(theta_vals), np.nan, dtype=float)
    r_squared_vals = np.full(len(theta_vals), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        mask = np.isclose(sweep.theta_values, theta)
        N_vals = sweep.N_values[mask].astype(float)
        delta_vals = sweep.delta_theta_opt[mask]

        # Filter finite values
        finite_mask = np.isfinite(delta_vals) & (delta_vals > 0) & (N_vals > 0)
        N_finite = N_vals[finite_mask]
        delta_finite = delta_vals[finite_mask]

        if len(N_finite) < 3:
            continue

        log_N = np.log(N_finite)
        log_delta = np.log(delta_finite)

        # Linear fit
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        coeffs, *_ = np.linalg.lstsq(A, log_delta, rcond=None)
        alpha = coeffs[0]
        log_C = coeffs[1]

        exponents[i] = alpha
        prefactors[i] = np.exp(log_C)

        # R² calculation
        if len(log_delta) > 2:
            ss_res = np.sum((log_delta - A @ coeffs) ** 2)
            ss_tot = np.sum((log_delta - np.mean(log_delta)) ** 2)
            r_squared_vals[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ScalingAnalysisResult(
        theta_values=theta_vals,
        exponents=exponents,
        prefactors=prefactors,
        r_squared=r_squared_vals,
    )


# ============================================================================
# Plot Functions
# ============================================================================


def plot_ratio_heatmap(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of Δθ_opt / SQL ratio across (θ, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    theta_vals = np.unique(sweep.theta_values)
    N_vals = np.unique(sweep.N_values)
    ratio_map = np.full((len(N_vals), len(theta_vals)), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.theta_values, theta) & (sweep.N_values == N)
            if np.any(mask):
                ratio_map[j, i] = float(sweep.ratio[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(0.0, float(np.nanmin(ratio_map)))
    vmax = max(2.0, float(np.nanmax(ratio_map[ratio_map < 10])))

    im = ax.pcolormesh(
        theta_vals,
        N_vals,
        ratio_map,
        shading="nearest",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(
        im, ax=ax, label=r"$\Delta\theta_{\mathrm{opt}} / \Delta\theta_{\mathrm{SQL}}$"
    )
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title("Sensitivity Ratio: Dual-MZI XX Coupling\n(lower = better; 1.0 = SQL)")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alpha_opt_heatmap(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal α_xx across (θ, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    theta_vals = np.unique(sweep.theta_values)
    N_vals = np.unique(sweep.N_values)
    alpha_map = np.full((len(N_vals), len(theta_vals)), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        for j, N in enumerate(N_vals):
            mask = np.isclose(sweep.theta_values, theta) & (sweep.N_values == N)
            if np.any(mask):
                alpha_map[j, i] = float(sweep.alpha_xx_opt[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(alpha_map)) if np.any(np.isfinite(alpha_map)) else 20.0

    im = ax.pcolormesh(
        theta_vals,
        N_vals,
        alpha_map,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=r"$\alpha_{xx}^*$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(r"Optimal $\alpha_{xx}$: Dual-MZI XX Coupling")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    theta_fixed: float | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δθ_opt vs N at a fixed θ, with SQL and HL reference lines.

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        theta_fixed: θ value to plot. If None, uses the first θ.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if theta_fixed is None:
        theta_fixed = float(np.unique(sweep.theta_values)[0])

    mask = np.isclose(sweep.theta_values, theta_fixed)
    N_vals = sweep.N_values[mask].astype(float)
    delta_vals = sweep.delta_theta_opt[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference: Δθ = 1/(√N T_H)
    N_dense = np.logspace(np.log10(1), np.log10(20), 100)
    sql_dense = 1.0 / (np.sqrt(N_dense) * sweep.T_H)
    hl_dense = 1.0 / (N_dense * sweep.T_H)

    ax.loglog(N_dense, sql_dense, "--", color="gray", alpha=0.7, label="SQL")
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
            label=rf"$\Delta\theta_{{\mathrm{{opt}}}}(\theta={theta_fixed:.2f})$",
        )

    ax.set_xlabel(r"$N$ (particles per subsystem)")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"N-Scaling at $\\theta={theta_fixed:.2f}$:\nDual-MZI XX Coupling")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_theta_dependence(
    sweep: DualMZISweepResult,
    save_path: str | Path,
    N_fixed: int | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δθ_opt vs θ at fixed N, with SQL reference line.

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        N_fixed: N value to plot. If None, uses the first N.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if N_fixed is None:
        N_fixed = int(np.unique(sweep.N_values)[0])

    mask = sweep.N_values == N_fixed
    theta_vals = sweep.theta_values[mask]
    delta_vals = sweep.delta_theta_opt[mask]
    sql_vals = sweep.sql_values[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference (flat line for fixed N)
    sql_val = 1.0 / (np.sqrt(N_fixed) * sweep.T_H) if len(sql_vals) > 0 else 0.1
    ax.axhline(
        y=sql_val,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_val:.5f}",
    )

    finite_mask = np.isfinite(delta_vals)
    if np.any(finite_mask):
        ax.plot(
            theta_vals[finite_mask],
            delta_vals[finite_mask],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.5,
            label=rf"$\Delta\theta_{{\mathrm{{opt}}}}(N={N_fixed})$",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"$\\theta$-Dependence at $N={N_fixed}$:\nDual-MZI XX Coupling")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling_exponents(
    scaling: ScalingAnalysisResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot the scaling exponent α vs θ from log-log fits.

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

    # Left: exponent vs θ
    valid_exp = np.isfinite(scaling.exponents)
    if np.any(valid_exp):
        ax1.plot(
            scaling.theta_values[valid_exp],
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
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"Scaling exponent $\alpha$")
    ax1.set_title("Exponent $\\alpha$ from\n$\\Delta\\theta = C N^{\\alpha}$")
    ax1.legend(fontsize=9)

    # Right: R² vs θ
    valid_r2 = np.isfinite(scaling.r_squared)
    if np.any(valid_r2):
        ax2.plot(
            scaling.theta_values[valid_r2],
            scaling.r_squared[valid_r2],
            "s-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_title("Goodness of Fit")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_DATE = "2026-05-22"
THETA_VALS: list[float] = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
N_VALS: list[int] = list(range(1, 21))


def parquet_path(name: str) -> Path:
    """Return path to a raw_data Parquet file for this report."""
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def fig_path(name: str) -> Path:
    """Return path to a figures SVG file for this report."""
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# Backward-compatible aliases
_parquet_path = parquet_path
_fig_path = fig_path


# ── Generator Functions ────────────────────────────────────────────────


def generate_sweep(force: bool = False) -> None:
    """Run the full θ × N sweep with α_xx optimisation."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_ratio = _fig_path("dual-mzi-ratio-heatmap")
    fig_alpha = _fig_path("dual-mzi-alpha-opt-heatmap")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZISweepResult.from_parquet(csv_p)
    else:
        print(
            "[run]  Computing dual-MZI θ×N sweep "
            f"({len(THETA_VALS)}×{len(N_VALS)} = {len(THETA_VALS) * len(N_VALS)} points)..."
        )
        theta_arr = np.array(THETA_VALS, dtype=float)
        N_arr = np.array(N_VALS, dtype=int)
        result = run_sweep(theta_values=theta_arr, N_values=N_arr)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio)
    print(f"[fig]  {fig_ratio}")
    plot_alpha_opt_heatmap(result, fig_alpha)
    print(f"[fig]  {fig_alpha}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline (α_xx = 0) verification."""
    csv_p = _parquet_path("dual-mzi-decoupled-baseline")
    fig_p = _fig_path("dual-mzi-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DualMZISweepResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline (α_xx = 0)...")
        N_arr = np.array(N_VALS, dtype=int)
        # Use a subset for speed (every 5th θ, all N)
        theta_subset = np.array([v for i, v in enumerate(THETA_VALS) if i % 5 == 0])
        result = compute_decoupled_baseline(
            theta_values=theta_subset,
            N_values=N_arr,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Create a verification figure: heatmap of |ratio - 1| on log scale
    from matplotlib.colors import LogNorm

    theta_vals = np.unique(result.theta_values)
    N_vals = np.unique(result.N_values)
    dev_map = np.full((len(N_vals), len(theta_vals)), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(result.theta_values, theta) & (result.N_values == N_val)
            if np.any(mask):
                r = float(result.ratio[mask][0])
                dev_map[j, i] = abs(r - 1.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    finite = dev_map[np.isfinite(dev_map)]
    if len(finite) > 0:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin, vmax = 1e-15, 1.0

    im = ax.pcolormesh(
        theta_vals,
        N_vals,
        dev_map,
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=max(vmin, 1e-16), vmax=vmax),
    )
    cbar = fig.colorbar(
        im, ax=ax, label=r"$|\Delta\theta/\Delta\theta_{\mathrm{SQL}} - 1|$"
    )

    max_dev = float(np.max(finite)) if len(finite) > 0 else 0.0
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(
        f"Decoupled Baseline Verification ($\\alpha_{{xx}} = 0$, $T_H = {result.T_H}$)\n"
        f"Max $|\\Delta\\theta/\\mathrm{{SQL}} - 1| = {max_dev:.2e}$, "
        f"points checked: {len(finite)}"
    )

    fig.tight_layout()
    fig.savefig(fig_p, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {fig_p}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling analysis from the sweep data."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_n3 = _fig_path("dual-mzi-n-scaling-theta0.3")
    fig_n1 = _fig_path("dual-mzi-n-scaling-theta1.0")
    fig_n3p = _fig_path("dual-mzi-n-scaling-theta3.0")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    # Plot at three representative θ values
    for theta_val, fig_p in [(0.3, fig_n3), (1.0, fig_n1), (3.0, fig_n3p)]:
        plot_n_scaling(result, fig_p, theta_fixed=theta_val)
        print(f"[fig]  {fig_p}")


def generate_theta_dependence(force: bool = False) -> None:
    """θ-dependence plots at fixed N values."""
    csv_p = _parquet_path("dual-mzi-sweep")
    fig_n1 = _fig_path("dual-mzi-theta-N1")
    fig_n5 = _fig_path("dual-mzi-theta-N5")
    fig_n20 = _fig_path("dual-mzi-theta-N20")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    for N_fixed, fig_p in [(1, fig_n1), (5, fig_n5), (20, fig_n20)]:
        plot_theta_dependence(result, fig_p, N_fixed=N_fixed)
        print(f"[fig]  {fig_p}")


def generate_scaling_analysis(force: bool = False) -> None:
    """Scaling analysis (exponents) from the sweep data."""
    csv_p = _parquet_path("dual-mzi-sweep")
    scaling_csv = _parquet_path("dual-mzi-scaling")
    fig_p = _fig_path("dual-mzi-scaling-exponents")

    if not csv_p.exists():
        print("[skip] Sweep data not found; run 'sweep' first")
        return

    result = DualMZISweepResult.from_parquet(csv_p)

    if scaling_csv.exists() and not force:
        scaling = ScalingAnalysisResult.from_parquet(scaling_csv)
        print(f"[skip] {scaling_csv.name} exists")
    else:
        print("[run]  Fitting scaling exponents...")
        scaling = fit_scaling_exponents(result)
        scaling.save_parquet(scaling_csv)
        print(f"[save] {scaling_csv}")

    plot_scaling_exponents(scaling, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-22 report figures and Parquet data",
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
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Callable[..., None]] = {
        "decoupled-baseline": generate_decoupled_baseline,
        "sweep": generate_sweep,
        "n-scaling": generate_n_scaling,
        "theta-dependence": generate_theta_dependence,
        "scaling-analysis": generate_scaling_analysis,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
