"""
Local module for the 2026-05-23 Four-Parameter Coupling Multi-Particle Dual MZI report.

Contains all code exclusive to this report:
- Multi-particle Dicke-basis operator construction (N up to 10)
- Two protocols: dual MZI (BS on S and A) and S-only MZI (BS on S only)
- Four-parameter interaction: H_int = α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
                                   + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A
- Sensitivity via error propagation with central finite differences
- Multi-start L-BFGS-B optimisation over (α_xx, α_xz, α_zx, α_zz)
- Sweeps over θ ∈ [0.5, 5.0], N ∈ [1, 10] (dual MZI) and N ∈ {1, 5, 10} (S-only)
- Scaling analysis (log-log fit Δθ ∝ N^α)
- JSON/manual reproduction of the 2026-05-21 result (S-only MZI, N=1)
- Exclusive plot functions for heatmaps, scaling curves, and θ-dependence
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/2026-05-23/local.py --force

This module is **not** importable as ``reports.2026-05-23.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import functools
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
from scipy.optimize import minimize

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
ALPHA_BOUND: float = 20.0  # |α_ij| ≤ 20 for optimisation bounds
N_LBFGS_STARTS: int = 25  # Number of L-BFGS-B random starts (20-30 per report)
FD_STEP: float = 1e-6  # Central finite-difference step

# θ and N sweep ranges (from report)
THETA_VALS: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
DUAL_MZI_N_VALS: list[int] = list(range(1, 11))
SONLY_MZI_N_VALS: list[int] = [1, 5, 10]

# Adaptive start counts: more starts for small N, fewer for large N
def _n_starts_for_N(N: int) -> int:
    """Return the number of L-BFGS-B starts appropriate for N.
    
    The expm time scales super-linearly with (N+1)^2, and the number of
    L-BFGS-B iterations also grows with N (more rugged landscape).
    This function balances optimisation quality against computational cost.
    """
    if N <= 3:
        return 25
    if N <= 5:
        return 15
    if N <= 7:
        return 8
    if N <= 10:
        return 5
    if N <= 15:
        return 3
    return 2


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
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the total holding Hamiltonian in the combined S⊗A space.

    H = θ (J_z^S + J_z^A)
        + α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
        + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A

    Args:
        N: Particle number per subsystem.
        theta: Unknown phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Pre-computed embedded operators. If None, built fresh.

    Returns:
        (N+1)² × (N+1)² Hermitian Hamiltonian matrix.
    """
    if ops is None:
        ops = embed_combined_operators(N)

    a_xx, a_xz, a_zx, a_zz = alpha
    dim = (N + 1) ** 2
    H = np.zeros((dim, dim), dtype=complex)

    # Phase-encoding terms
    H += theta * (ops["Jz_S"] + ops["Jz_A"])

    # Interaction terms
    if a_xx != 0.0:
        H += a_xx * (ops["Jx_S"] @ ops["Jx_A"])
    if a_xz != 0.0:
        H += a_xz * (ops["Jx_S"] @ ops["Jz_A"])
    if a_zx != 0.0:
        H += a_zx * (ops["Jz_S"] @ ops["Jx_A"])
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])

    return 0.5 * (H + H.conj().T)


@functools.lru_cache(maxsize=128)
def _single_bs_unitary_cached(N: int, T: float) -> np.ndarray:
    """Cached helper: compute exp(-i T J_x) for given N."""
    Jx = jx_operator(N)
    U = expm(-1j * T * Jx)
    return U


def single_bs_unitary(N: int, T: float = DEFAULT_T_BS) -> np.ndarray:
    """Single-subsystem 50/50 beam-splitter unitary (cached).

    U_BS = exp(-i T J_x)

    Results are cached by (N, T) to avoid repeated matrix exponentiation.

    Args:
        N: Particle number (dim = N+1).
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1) × (N+1) unitary matrix.
    """
    return _single_bs_unitary_cached(N, T)


def protocol_bs_unitary(
    N: int,
    protocol: str = "dual",
    T: float = DEFAULT_T_BS,
) -> np.ndarray:
    """Beam-splitter unitary based on protocol.

    Dual MZI: U_BS = exp(-i T J_x) ⊗ exp(-i T J_x)
    S-only MZI: U_BS = exp(-i T J_x) ⊗ I_{N+1}

    Args:
        N: Particle number per subsystem.
        protocol: 'dual' or 'S-only'.
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    U_single = single_bs_unitary(N, T)
    if protocol == "dual":
        U = np.kron(U_single, U_single)
    elif protocol == "S-only":
        U = np.kron(U_single, np.eye(N + 1, dtype=complex))
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}. Use 'dual' or 'S-only'.")
    return U


def hold_unitary(
    N: int,
    T_H: float,
    theta: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Holding-time unitary in the combined S⊗A space.

    U_hold(T_H) = exp(-i T_H H)

    Args:
        N: Particle number per subsystem.
        T_H: Holding time.
        theta: Unknown phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Pre-computed operators.

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    H = build_hold_hamiltonian(N, theta, alpha, ops)
    U = expm(-1j * T_H * H)
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
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    protocol: str = "dual",
    T_BS: float = DEFAULT_T_BS,
    T_H: float = DEFAULT_T_H,
) -> np.ndarray:
    """Run the full MZI circuit for the given protocol.

    |ψ_final⟩ = U_BS · U_hold(T_H) · U_BS · |ψ₀⟩

    where U_BS depends on the protocol (dual or S-only).

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector (length (N+1)²).
        theta: Unknown phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Embedded operators.
        protocol: 'dual' (BS on both) or 'S-only' (BS on system only).
        T_BS: Beam-splitter angle (default π/2).
        T_H: Holding time (default 10).

    Returns:
        Final state vector (length (N+1)²).
    """
    U_bs = protocol_bs_unitary(N, protocol, T_BS)
    psi = U_bs @ psi0
    psi = hold_unitary(N, T_H, theta, alpha, ops) @ psi
    psi = U_bs @ psi
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

    # Expectation and variance
    exp_val = float(np.real(np.trace(rho_S @ meas_op)))
    op_sq = meas_op @ meas_op
    exp_sq = float(np.real(np.trace(rho_S @ op_sq)))
    raw_var = exp_sq - exp_val**2

    # Clamp negative variance near zero (numerical round-off)
    if raw_var < 0 and raw_var > -1e-12:
        raw_var = 0.0

    return float(exp_val), float(max(0.0, raw_var))


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_sensitivity(
    N: int,
    psi0: np.ndarray,
    theta_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    protocol: str = "dual",
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
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Embedded operators.
        protocol: 'dual' or 'S-only'.
        meas_op: (N+1)×(N+1) measurement operator (default = J_z).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        T_H: Holding time.

    Returns:
        Tuple (delta_theta, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        Jz_single = jz_operator(N)
    else:
        Jz_single = meas_op

    # Evaluate at theta_true
    psi = evolve_circuit(N, psi0, theta_true, alpha, ops, protocol, T_BS, T_H)
    exp_val, var_val = compute_reduced_expectation_and_variance(psi, N, Jz_single)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    psi_plus = evolve_circuit(
        N,
        psi0,
        theta_true + fd_step,
        alpha,
        ops,
        protocol,
        T_BS,
        T_H,
    )
    psi_minus = evolve_circuit(
        N,
        psi0,
        theta_true - fd_step,
        alpha,
        ops,
        protocol,
        T_BS,
        T_H,
    )

    exp_plus = _reduced_expectation(psi_plus, N, Jz_single)
    exp_minus = _reduced_expectation(psi_minus, N, Jz_single)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_theta = float(np.sqrt(var_val) / abs(d_exp))
    return delta_theta, exp_val, var_val, d_exp


# ============================================================================
# L-BFGS-B Objective and Multi-Start Optimisation
# ============================================================================


def _sensitivity_objective(
    alpha_params: np.ndarray,
    N: int,
    theta_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    protocol: str,
    T_BS: float,
    T_H: float,
    fd_step: float,
) -> float:
    """Objective function for L-BFGS-B optimisation.

    f(α) = Δθ(α; θ_true, N, protocol)

    Args:
        alpha_params: 4-element array [α_xx, α_xz, α_zx, α_zz].
        N: Particle number per subsystem.
        theta_true: True phase rate.
        ops: Embedded operators.
        psi0: Initial state.
        protocol: 'dual' or 'S-only'.
        T_BS: Beam-splitter angle.
        T_H: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Δθ (positive float). Returns large number if fringe extremum.
    """
    alpha = (
        float(alpha_params[0]),
        float(alpha_params[1]),
        float(alpha_params[2]),
        float(alpha_params[3]),
    )
    dt, _, _, _ = compute_sensitivity(
        N,
        psi0,
        theta_true,
        alpha,
        ops,
        protocol,
        fd_step=fd_step,
        T_BS=T_BS,
        T_H=T_H,
    )
    return dt if np.isfinite(dt) else 1e10


def optimise_four_params(
    N: int,
    theta: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    protocol: str = "dual",
    alpha_bounds: tuple[float, float] = (-ALPHA_BOUND, ALPHA_BOUND),
    n_starts: int = N_LBFGS_STARTS,
    T_BS: float = DEFAULT_T_BS,
    T_H: float = DEFAULT_T_H,
    fd_step: float = FD_STEP,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> FourParamOptResult:
    """Run multi-start L-BFGS-B optimisation for a given (θ, N, protocol).

    For each start:
    1. Generate random initial α in [alpha_bounds]^4.
    2. Run L-BFGS-B with bounded optimisation.
    3. Select the run with lowest Δθ.

    Args:
        N: Particle number per subsystem.
        theta: Phase rate.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        protocol: 'dual' or 'S-only'.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts.
        T_BS: Beam-splitter angle.
        T_H: Holding time.
        fd_step: Finite-difference step.
        seed: Base random seed (incremented per start).
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        FourParamOptResult with optimal parameters found.
    """
    if psi0 is None:
        psi0 = initial_state(N)

    lo, hi = alpha_bounds
    base_seed = seed if seed is not None else 42
    bounds_ls = [(lo, hi)] * 4
    sql = 1.0 / (np.sqrt(N) * T_H)

    best_delta = float("inf")
    best_alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    best_exp = 0.0
    best_var = 0.0
    best_d_exp = 0.0
    best_grad_norm = float("inf")
    n_converged = 0

    for start in range(n_starts):
        rng = np.random.default_rng(base_seed + int(theta * 1000) + start * 7)
        x0 = rng.uniform(lo, hi, size=4)

        result = minimize(
            _sensitivity_objective,
            x0,
            args=(N, theta, ops, psi0, protocol, T_BS, T_H, fd_step),
            method="L-BFGS-B",
            bounds=bounds_ls,
            options={
                "maxiter": maxiter,
                "gtol": gtol,
                "ftol": 1e-12,
            },
        )

        if result.success:
            n_converged += 1

        delta_val = float(result.fun)
        if np.isfinite(delta_val) and delta_val < best_delta:
            best_delta = delta_val
            best_alpha = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
                float(result.x[3]),
            )
            # Capture projected gradient norm at termination
            if result.jac is not None:
                best_grad_norm = float(np.linalg.norm(result.jac))
            else:
                best_grad_norm = float("inf")
            # Re-evaluate at optimum for diagnostics
            _, exp_val, var_val, d_exp = compute_sensitivity(
                N,
                psi0,
                theta,
                best_alpha,
                ops,
                protocol,
                fd_step=fd_step,
                T_BS=T_BS,
                T_H=T_H,
            )
            best_exp, best_var, best_d_exp = exp_val, var_val, d_exp

    return FourParamOptResult(
        theta_value=theta,
        N=N,
        protocol=protocol,
        alpha_opt=best_alpha,
        delta_theta_opt=best_delta,
        sql=sql,
        expectation_Jz=best_exp,
        variance_Jz=best_var,
        d_expectation=best_d_exp,
        n_starts=n_starts,
        n_converged=n_converged,
        gradient_norm=best_grad_norm,
    )


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class FourParamOptResult:
    """Result from a multi-start L-BFGS-B optimisation at a single (θ, N, protocol).

    Attributes:
        theta_value: θ at which the optimisation was performed.
        N: Particle number per subsystem.
        protocol: 'dual' or 'S-only'.
        alpha_opt: Optimal (α_xx, α_xz, α_zx, α_zz) found.
        delta_theta_opt: Minimal Δθ found.
        sql: SQL = 1/(√N T_H) reference value.
        expectation_Jz: ⟨J_z^S⟩ at the optimal point.
        variance_Jz: Var(J_z^S) at the optimal point.
        d_expectation: ∂⟨J_z^S⟩/∂θ at the optimal point.
        n_starts: Number of random starts used.
        n_converged: Number of starts that converged successfully.
        gradient_norm: L-BFGS-B projected gradient norm at optimum.
    """

    theta_value: float
    N: int
    protocol: str = "dual"
    alpha_opt: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    delta_theta_opt: float = float("inf")
    sql: float = 0.1
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_expectation: float = 0.0
    n_starts: int = N_LBFGS_STARTS
    n_converged: int = 0
    gradient_norm: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with all metadata."""
        return pd.DataFrame(
            {
                "theta": [self.theta_value],
                "N": [self.N],
                "protocol": [self.protocol],
                "T_H": [DEFAULT_T_H],
                "alpha_xx_opt": [self.alpha_opt[0]],
                "alpha_xz_opt": [self.alpha_opt[1]],
                "alpha_zx_opt": [self.alpha_opt[2]],
                "alpha_zz_opt": [self.alpha_opt[3]],
                "delta_theta_opt": [self.delta_theta_opt],
                "sql": [self.sql],
                "ratio": [
                    self.delta_theta_opt / self.sql
                    if np.isfinite(self.delta_theta_opt) and self.sql > 0
                    else float("inf")
                ],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "d_expectation": [self.d_expectation],
                "n_starts": [self.n_starts],
                "n_converged": [self.n_converged],
                "gradient_norm": [self.gradient_norm],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path


@dataclass
class FourParamSweepResult:
    """Full sweep over (θ, N, protocol) with optimisation per point.

    All array fields have the same length (n_points), stored in row-major
    order (θ varies fastest, then N).

    Attributes:
        theta_values: θ values for each point.
        N_values: N values for each point.
        protocol: Protocol string for each point (list of str).
        alpha_xx_opt: Optimal α_xx at each point.
        alpha_xz_opt: Optimal α_xz at each point.
        alpha_zx_opt: Optimal α_zx at each point.
        alpha_zz_opt: Optimal α_zz at each point.
        delta_theta_opt: Minimal Δθ at each point.
        sql_values: SQL = 1/(√N T_H) at each point.
        ratio: Δθ_opt / SQL at each point.
        expectation_Jz: ⟨J_z^S⟩ at optimum.
        variance_Jz: Var(J_z^S) at optimum.
        d_expectation: ∂⟨J_z^S⟩/∂θ at optimum.
        n_starts: Number of random starts per point.
        n_converged: Number of converged starts per point.
        gradient_norm: L-BFGS-B projected gradient norm at optimum per point.
        T_H: Holding time (scalar).
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    protocol: list[str] = field(default_factory=list)
    alpha_xx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xz_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zz_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_theta_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    d_expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    n_starts: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_converged: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    gradient_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    T_H: float = DEFAULT_T_H

    def __post_init__(self) -> None:
        # Ensure int dtype for integer arrays
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)
        if self.n_starts.dtype.kind != "i":
            self.n_starts = self.n_starts.astype(int)
        if self.n_converged.dtype.kind != "i":
            self.n_converged = self.n_converged.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.theta_values)
        # Pad protocol list if needed
        protocol_list = self.protocol
        if len(protocol_list) < n:
            protocol_list = protocol_list + ["unknown"] * (n - len(protocol_list))
        return pd.DataFrame(
            {
                "theta": self.theta_values,
                "N": self.N_values,
                "protocol": protocol_list[:n],
                "T_H": np.full(n, self.T_H),
                "alpha_xx_opt": self.alpha_xx_opt,
                "alpha_xz_opt": self.alpha_xz_opt,
                "alpha_zx_opt": self.alpha_zx_opt,
                "alpha_zz_opt": self.alpha_zz_opt,
                "delta_theta_opt": self.delta_theta_opt,
                "sql": self.sql_values,
                "ratio": self.ratio,
                "expectation_Jz": self.expectation_Jz,
                "variance_Jz": self.variance_Jz,
                "d_expectation": self.d_expectation,
                "n_starts": self.n_starts,
                "n_converged": self.n_converged,
                "gradient_norm": self.gradient_norm,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> FourParamSweepResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "N",
            "protocol",
            "T_H",
            "alpha_xx_opt",
            "alpha_xz_opt",
            "alpha_zx_opt",
            "alpha_zz_opt",
            "delta_theta_opt",
            "sql",
            "ratio",
            "expectation_Jz",
            "variance_Jz",
            "d_expectation",
            "n_starts",
            "n_converged",
            "gradient_norm",
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

    def filter_protocol(self, protocol: str) -> FourParamSweepResult:
        """Return a new result filtered to a single protocol."""
        mask = [p == protocol for p in self.protocol]
        return FourParamSweepResult(
            theta_values=self.theta_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_theta_opt=self.delta_theta_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            T_H=self.T_H,
        )

    def filter_theta(self, theta: float) -> FourParamSweepResult:
        """Return a new result filtered to a single θ value."""
        mask = np.isclose(self.theta_values, theta)
        return FourParamSweepResult(
            theta_values=self.theta_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_theta_opt=self.delta_theta_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            T_H=self.T_H,
        )

    def filter_N(self, N: int) -> FourParamSweepResult:
        """Return a new result filtered to a single N value."""
        mask = self.N_values == N
        return FourParamSweepResult(
            theta_values=self.theta_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_theta_opt=self.delta_theta_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            T_H=self.T_H,
        )


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


# ============================================================================
# Sweep Execution
# ============================================================================


def run_sweep(
    theta_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    protocol: str = "dual",
    T_H: float = DEFAULT_T_H,
    n_starts: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FourParamSweepResult:
    """Run the full sweep over θ and N with four-parameter optimisation.

    Args:
        theta_values: θ values to sweep (default: [0.5, 1.0, ..., 5.0]).
        N_values: N values to sweep (default: 1 to 20 for dual, 1/5/10 for S-only).
        protocol: 'dual' or 'S-only'.
        T_H: Holding time.
        n_starts: Number of L-BFGS-B random starts per point.
        progress_callback: Optional callback (current, total).

    Returns:
        FourParamSweepResult with all optimised points.
    """
    if theta_values is None:
        theta_values = np.array(THETA_VALS, dtype=float)

    n_theta = len(theta_values)
    n_N = len(N_values) if N_values is not None else 0
    total = n_theta * n_N

    # Choose default N values based on protocol
    if N_values is None:
        if protocol == "dual":
            N_values = np.array(DUAL_MZI_N_VALS, dtype=int)
        else:
            N_values = np.array(SONLY_MZI_N_VALS, dtype=int)

    thetas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    protos: list[str] = []
    a_xx_opts = np.full(total, np.nan, dtype=float)
    a_xz_opts = np.full(total, np.nan, dtype=float)
    a_zx_opts = np.full(total, np.nan, dtype=float)
    a_zz_opts = np.full(total, np.nan, dtype=float)
    delta_opts = np.full(total, np.inf, dtype=float)
    sqls = np.zeros(total, dtype=float)
    ratios = np.full(total, np.inf, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)
    n_starts_arr = np.zeros(total, dtype=int)
    n_conv_arr = np.zeros(total, dtype=int)
    grad_norm_arr = np.full(total, np.nan, dtype=float)

    idx = 0
    for N_val in N_values:
        ops = embed_combined_operators(N_val)
        psi0 = initial_state(N_val)
        actual_starts = _n_starts_for_N(N_val) if n_starts is None else n_starts
        for theta_val in theta_values:
            thetas[idx] = theta_val
            Ns[idx] = N_val
            protos.append(protocol)

            opt_result = optimise_four_params(
                N=N_val,
                theta=theta_val,
                ops=ops,
                psi0=psi0,
                protocol=protocol,
                n_starts=actual_starts,
                T_H=T_H,
            )
            a_xx_opts[idx] = opt_result.alpha_opt[0]
            a_xz_opts[idx] = opt_result.alpha_opt[1]
            a_zx_opts[idx] = opt_result.alpha_opt[2]
            a_zz_opts[idx] = opt_result.alpha_opt[3]
            delta_opts[idx] = opt_result.delta_theta_opt
            sqls[idx] = opt_result.sql
            ratios[idx] = (
                opt_result.delta_theta_opt / opt_result.sql
                if np.isfinite(opt_result.delta_theta_opt) and opt_result.sql > 0
                else float("inf")
            )
            exps[idx] = opt_result.expectation_Jz
            vars_[idx] = opt_result.variance_Jz
            d_exps[idx] = opt_result.d_expectation
            n_starts_arr[idx] = actual_starts
            n_conv_arr[idx] = opt_result.n_converged
            grad_norm_arr[idx] = opt_result.gradient_norm

            idx += 1
            if progress_callback is not None:
                progress_callback(idx, total)

    return FourParamSweepResult(
        theta_values=thetas,
        N_values=Ns,
        protocol=protos,
        alpha_xx_opt=a_xx_opts,
        alpha_xz_opt=a_xz_opts,
        alpha_zx_opt=a_zx_opts,
        alpha_zz_opt=a_zz_opts,
        delta_theta_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        n_starts=n_starts_arr,
        n_converged=n_conv_arr,
        gradient_norm=grad_norm_arr,
        T_H=T_H,
    )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def compute_decoupled_baseline(
    theta_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    protocol: str = "dual",
    T_H: float = DEFAULT_T_H,
    fd_step: float = FD_STEP,
) -> FourParamSweepResult:
    """Verify the decoupled baseline (α = 0) for all (θ, N, protocol) pairs.

    At α = (0,0,0,0), the sensitivity should equal SQL = 1/(√N T_H).

    Args:
        theta_values: θ values (default: sweep range).
        N_values: N values (default: 1 to 20 for dual).
        protocol: 'dual' or 'S-only'.
        T_H: Holding time.

    Returns:
        FourParamSweepResult with α = 0 results.
    """
    if theta_values is None:
        theta_values = np.array(THETA_VALS, dtype=float)
    if N_values is None:
        if protocol == "dual":
            N_values = np.array(DUAL_MZI_N_VALS, dtype=int)
        else:
            N_values = np.array(SONLY_MZI_N_VALS, dtype=int)

    n_theta = len(theta_values)
    n_N = len(N_values)
    total = n_theta * n_N

    thetas = np.zeros(total, dtype=float)
    Ns = np.zeros(total, dtype=int)
    protos: list[str] = []
    sqls = np.zeros(total, dtype=float)
    delta_opts = np.zeros(total, dtype=float)
    a_xx_opts = np.zeros(total, dtype=float)
    a_xz_opts = np.zeros(total, dtype=float)
    a_zx_opts = np.zeros(total, dtype=float)
    a_zz_opts = np.zeros(total, dtype=float)
    ratios = np.zeros(total, dtype=float)
    exps = np.zeros(total, dtype=float)
    vars_ = np.zeros(total, dtype=float)
    d_exps = np.zeros(total, dtype=float)
    n_starts_arr = np.zeros(total, dtype=int)
    n_conv_arr = np.zeros(total, dtype=int)
    grad_norm_arr = np.full(total, np.nan, dtype=float)

    zero_alpha = (0.0, 0.0, 0.0, 0.0)

    idx = 0
    for N_val in N_values:
        ops = embed_combined_operators(N_val)
        psi0 = initial_state(N_val)
        for theta_val in theta_values:
            thetas[idx] = theta_val
            Ns[idx] = N_val
            protos.append(protocol)
            sql = 1.0 / (np.sqrt(N_val) * T_H)
            sqls[idx] = sql

            dt, exp_val, var_val, d_exp_val = compute_sensitivity(
                N_val,
                psi0,
                theta_val,
                zero_alpha,
                ops,
                protocol=protocol,
                T_H=T_H,
                fd_step=fd_step,
            )
            delta_opts[idx] = dt
            ratios[idx] = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
            exps[idx] = exp_val
            vars_[idx] = var_val
            d_exps[idx] = d_exp_val

            idx += 1

    return FourParamSweepResult(
        theta_values=thetas,
        N_values=Ns,
        protocol=protos,
        alpha_xx_opt=a_xx_opts,
        alpha_xz_opt=a_xz_opts,
        alpha_zx_opt=a_zx_opts,
        alpha_zz_opt=a_zz_opts,
        delta_theta_opt=delta_opts,
        sql_values=sqls,
        ratio=ratios,
        expectation_Jz=exps,
        variance_Jz=vars_,
        d_expectation=d_exps,
        n_starts=n_starts_arr,
        n_converged=n_conv_arr,
        gradient_norm=grad_norm_arr,
        T_H=T_H,
    )


# ============================================================================
# Scaling Analysis
# ============================================================================


def fit_scaling_exponents(sweep: FourParamSweepResult) -> ScalingAnalysisResult:
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
    sweep: FourParamSweepResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
    title_suffix: str = "",
) -> Path:
    """Plot a heatmap of Δθ_opt / SQL ratio across (θ, N).

    Args:
        sweep: Sweep result.
        save_path: Output SVG path.
        figsize: Figure size.
        title_suffix: Optional suffix for the plot title.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    theta_vals = np.unique(sweep.theta_values)
    N_vals = np.unique(sweep.N_values)
    ratio_map = np.full((len(N_vals), len(theta_vals)), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(sweep.theta_values, theta) & (sweep.N_values == N_val)
            if np.any(mask):
                ratio_map[j, i] = float(sweep.ratio[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(0.0, float(np.nanmin(ratio_map)))
    finite_mask_plot = ratio_map < 10
    if np.any(finite_mask_plot):
        vmax = max(2.0, float(np.nanmax(ratio_map[finite_mask_plot])))
    else:
        vmax = 2.0

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
    title = "Sensitivity Ratio: 4-Parameter Interaction"
    if title_suffix:
        title += f"\n{title_suffix}"
    else:
        title += "\n(lower = better; 1.0 = SQL)"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alpha_opt_heatmap(
    sweep: FourParamSweepResult,
    alpha_label: str,
    alpha_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal α values across (θ, N).

    Args:
        sweep: Sweep result.
        alpha_label: Label for colorbar (e.g., r'$\alpha_{xx}^*$').
        alpha_values: Array of optimal α values (same length as sweep).
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
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(sweep.theta_values, theta) & (sweep.N_values == N_val)
            if np.any(mask):
                alpha_map[j, i] = float(alpha_values[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    finite_alpha = alpha_map[np.isfinite(alpha_map)]
    if len(finite_alpha) > 0:
        vmax = (
            float(np.nanmax(np.abs(finite_alpha))) if np.any(finite_alpha != 0) else 1.0
        )
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    im = ax.pcolormesh(
        theta_vals,
        N_vals,
        alpha_map,
        shading="nearest",
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=alpha_label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(f"Optimal {alpha_label}")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling(
    sweep: FourParamSweepResult,
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
    N_dense = np.logspace(
        np.log10(1), np.log10(max(N_vals) if len(N_vals) > 0 else 20), 100
    )
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

    protocol_label = sweep.protocol[0] if sweep.protocol else "dual"
    ax.set_xlabel(r"$N$ (particles per subsystem)")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"N-Scaling at $\\theta={theta_fixed:.2f}$:\n{protocol_label} MZI")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_theta_dependence(
    sweep: FourParamSweepResult,
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

    protocol_label = sweep.protocol[0] if sweep.protocol else "dual"
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"$\\theta$-Dependence at $N={N_fixed}$:\n{protocol_label} MZI")
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


def plot_theta_scan(
    sweep: FourParamSweepResult,
    save_path: str | Path,
    N_fixed: int | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δθ_opt vs θ with SQL reference and optimal α parameters as secondary axis.

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

    filtered = sweep.filter_N(N_fixed)
    theta = filtered.theta_values
    sql_val = 1.0 / (np.sqrt(N_fixed) * sweep.T_H)

    fig, ax1 = plt.subplots(figsize=figsize)

    # SQL reference line
    ax1.axhline(
        y=sql_val,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql_val:.4f}",
    )

    # Δθ vs θ
    valid = np.isfinite(filtered.delta_theta_opt)
    if np.any(valid):
        ax1.plot(
            theta[valid],
            filtered.delta_theta_opt[valid],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.8,
            label=r"$\Delta\theta_{\mathrm{opt}}$",
        )
        # Annotate best point
        best_idx = int(np.argmin(filtered.delta_theta_opt[valid]))
        best_theta = float(theta[valid][best_idx])
        best_val = float(filtered.delta_theta_opt[valid][best_idx])
        best_ratio = best_val / sql_val if sql_val > 0 else float("inf")
        ax1.annotate(
            rf"Best: $\Delta\theta$={best_val:.5f} ({best_ratio:.3f}$\times$SQL)"
            rf" at $\theta$={best_theta:.2f}",
            xy=(best_theta, best_val),
            xytext=(best_theta + 0.8, best_val + 0.02),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    protocol_label = filtered.protocol[0] if filtered.protocol else "dual"
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title(
        f"4-Parameter Sensitivity vs $\\theta$ at $N={N_fixed}$:\n{protocol_label} MZI"
    )

    # Secondary axis: optimal α parameters
    ax2 = ax1.twinx()
    for label, arr, color, marker in [
        (r"$\alpha_{xx}^*$", filtered.alpha_xx_opt, "C1", "s"),
        (r"$\alpha_{xz}^*$", filtered.alpha_xz_opt, "C2", "d"),
        (r"$\alpha_{zx}^*$", filtered.alpha_zx_opt, "C3", "^"),
        (r"$\alpha_{zz}^*$", filtered.alpha_zz_opt, "C4", "v"),
    ]:
        valid_a = np.isfinite(arr)
        if np.any(valid_a):
            ax2.plot(
                theta[valid_a],
                arr[valid_a],
                marker + "-",
                color=color,
                markersize=5,
                linewidth=1.0,
                alpha=0.6,
                label=label,
            )
    ax2.set_ylabel(r"$\alpha_{ij}^*$")
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_DATE = "2026-05-23"


def parquet_path(name: str) -> Path:
    """Return path to a raw_data Parquet file for this report."""
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def fig_path(name: str) -> Path:
    """Return path to a figures SVG file for this report."""
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ── Generator Functions ────────────────────────────────────────────────


def generate_dual_sweep(force: bool = False) -> None:
    """Run the dual MZI full sweep (N=1-20, θ from 0.5 to 5.0)."""
    csv_p = parquet_path("dual-mzi-sweep")
    fig_ratio = fig_path("dual-mzi-ratio-heatmap")
    fig_theta_scan = fig_path("dual-mzi-theta-scan-N5")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = FourParamSweepResult.from_parquet(csv_p)
    else:
        N_arr = np.array(DUAL_MZI_N_VALS, dtype=int)
        theta_arr = np.array(THETA_VALS, dtype=float)
        print(
            f"[run]  Computing dual MZI sweep "
            f"({len(theta_arr)}×{len(N_arr)} = {len(theta_arr) * len(N_arr)} points)..."
        )
        result = run_sweep(theta_values=theta_arr, N_values=N_arr, protocol="dual")
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio, title_suffix="Dual MZI")
    print(f"[fig]  {fig_ratio}")
    plot_theta_scan(result, fig_theta_scan, N_fixed=5)
    print(f"[fig]  {fig_theta_scan}")


def generate_sonly_sweep(force: bool = False) -> None:
    """Run the S-only MZI comparison sweep (N=1,5,10)."""
    csv_p = parquet_path("sonly-mzi-sweep")
    fig_ratio = fig_path("sonly-mzi-ratio-heatmap")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = FourParamSweepResult.from_parquet(csv_p)
    else:
        N_arr = np.array(SONLY_MZI_N_VALS, dtype=int)
        theta_arr = np.array(THETA_VALS, dtype=float)
        print(
            f"[run]  Computing S-only MZI sweep "
            f"({len(theta_arr)}×{len(N_arr)} = {len(theta_arr) * len(N_arr)} points)..."
        )
        result = run_sweep(theta_values=theta_arr, N_values=N_arr, protocol="S-only")
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(result, fig_ratio, title_suffix="S-only MZI")
    print(f"[fig]  {fig_ratio}")


def generate_sonly_reproduction(force: bool = False) -> None:
    """Reproduce the 2026-05-21 result: S-only MZI, N=1, θ=3.8.

    Expected: Δθ/Δθ_SQL ≤ 0.690 (i.e., Δθ ≤ 0.0690 at T_H=10).
    """
    csv_p = parquet_path("sonly-reproduction-n1")
    N_val = 1
    theta_val = 3.8
    sql = 1.0 / (np.sqrt(N_val) * DEFAULT_T_H)

    print("[run]  2026-05-21 reproduction: S-only MZI, N=1, θ=3.8")

    ops = embed_combined_operators(N_val)
    psi0 = initial_state(N_val)
    result = optimise_four_params(
        N=N_val,
        theta=theta_val,
        ops=ops,
        psi0=psi0,
        protocol="S-only",
        n_starts=N_LBFGS_STARTS,  # Per report: 20-30 starts
    )

    ratio = (
        result.delta_theta_opt / sql
        if np.isfinite(result.delta_theta_opt)
        else float("inf")
    )
    print(
        f"  Δθ_opt = {result.delta_theta_opt:.6f}, SQL = {sql:.6f}, ratio = {ratio:.4f}"
    )
    print(
        f"  α* = ({result.alpha_opt[0]:.4f}, {result.alpha_opt[1]:.4f}, "
        f"{result.alpha_opt[2]:.4f}, {result.alpha_opt[3]:.4f})"
    )
    print(f"  Converged: {result.n_converged}/{result.n_starts}")

    # Save result
    result.save_parquet(csv_p)
    print(f"[save] {csv_p}")

    # Create verification figure
    fig_p = fig_path("sonly-reproduction-n1")
    text = (
        f"2026-05-21 Reproduction (S-only MZI, N=1, θ=3.8)\n"
        f"Δθ_opt = {result.delta_theta_opt:.6f}\n"
        f"SQL = {sql:.6f}\n"
        f"Ratio = {ratio:.4f}\n"
        f"Expected ratio ≤ 0.690\n"
        f"α* = ({result.alpha_opt[0]:.3f}, {result.alpha_opt[1]:.3f}, "
        f"{result.alpha_opt[2]:.3f}, {result.alpha_opt[3]:.3f})\n"
        f"Converged: {result.n_converged}/{result.n_starts}"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_p, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {fig_p}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline (α = 0) verification for both protocols."""
    csv_dual = parquet_path("decoupled-baseline-dual")
    csv_sonly = parquet_path("decoupled-baseline-sonly")

    # Dual MZI baseline
    if csv_dual.exists() and not force:
        print(f"[skip] {csv_dual.name} exists")
        result_dual = FourParamSweepResult.from_parquet(csv_dual)
    else:
        print("[run]  Computing decoupled baseline (dual MZI)...")
        N_arr = np.array(DUAL_MZI_N_VALS, dtype=int)
        theta_subset = np.array(THETA_VALS, dtype=float)
        result_dual = compute_decoupled_baseline(
            theta_values=theta_subset,
            N_values=N_arr,
            protocol="dual",
        )
        result_dual.save_parquet(csv_dual)
        print(f"[save] {csv_dual}")

    # S-only MZI baseline
    if csv_sonly.exists() and not force:
        print(f"[skip] {csv_sonly.name} exists")
        result_sonly = FourParamSweepResult.from_parquet(csv_sonly)
    else:
        print("[run]  Computing decoupled baseline (S-only MZI)...")
        N_arr = np.array(SONLY_MZI_N_VALS, dtype=int)
        theta_subset = np.array(THETA_VALS, dtype=float)
        result_sonly = compute_decoupled_baseline(
            theta_values=theta_subset,
            N_values=N_arr,
            protocol="S-only",
        )
        result_sonly.save_parquet(csv_sonly)
        print(f"[save] {csv_sonly}")

    # Create verification figures: heatmaps of |ratio - 1| on log scale
    from matplotlib.colors import LogNorm

    for protocol, result in [("dual", result_dual), ("S-only", result_sonly)]:
        theta_vals = np.unique(result.theta_values)
        N_vals = np.unique(result.N_values)
        dev_map = np.full((len(N_vals), len(theta_vals)), np.nan, dtype=float)

        for i, theta in enumerate(theta_vals):
            for j, N_val in enumerate(N_vals):
                mask = np.isclose(result.theta_values, theta) & (
                    result.N_values == N_val
                )
                if np.any(mask):
                    r = float(result.ratio[mask][0])
                    dev_map[j, i] = abs(r - 1.0)

        fig_p = fig_path(f"decoupled-baseline-{protocol}")
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
        protocol_label = "dual" if protocol == "dual" else "S-only"
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$N$ (particles per subsystem)")
        ax.set_title(
            f"Decoupled Baseline Verification ($\\alpha = 0$, {protocol_label} MZI, "
            f"$T_H = {result.T_H}$)\n"
            f"Max $|\\Delta\\theta/\\mathrm{{SQL}} - 1| = {max_dev:.2e}$, "
            f"points checked: {len(finite)}"
        )

        fig.tight_layout()
        fig.savefig(fig_p, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[fig]  {fig_p}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling plots from the dual MZI sweep data."""
    csv_p = parquet_path("dual-mzi-sweep")

    if not csv_p.exists():
        print("[skip] Dual MZI sweep data not found; run 'dual-sweep' first")
        return

    result = FourParamSweepResult.from_parquet(csv_p)

    # Plot at three representative θ values
    for theta_val in [0.5, 2.5, 5.0]:
        fig_p = fig_path(f"dual-mzi-n-scaling-theta{theta_val:.1f}")
        plot_n_scaling(result, fig_p, theta_fixed=theta_val)
        print(f"[fig]  {fig_p}")


def generate_theta_dependence(force: bool = False) -> None:
    """θ-dependence plots at fixed N values from sweep data."""
    csv_dual = parquet_path("dual-mzi-sweep")

    if csv_dual.exists():
        result_dual = FourParamSweepResult.from_parquet(csv_dual)
        for N_fixed in [1, 5, 10, 20]:
            fig_p = fig_path(f"dual-mzi-theta-N{N_fixed}")
            plot_theta_dependence(result_dual, fig_p, N_fixed=N_fixed)
            print(f"[fig]  {fig_p}")


def generate_scaling_analysis(force: bool = False) -> None:
    """Scaling analysis from the dual MZI sweep data."""
    csv_p = parquet_path("dual-mzi-sweep")
    scaling_csv = parquet_path("dual-mzi-scaling")
    fig_p = fig_path("dual-mzi-scaling-exponents")

    if not csv_p.exists():
        print("[skip] Dual MZI sweep data not found; run 'dual-sweep' first")
        return

    result = FourParamSweepResult.from_parquet(csv_p)

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
        description="Generate 2026-05-23 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'dual-sweep'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Callable[..., None]] = {
        "decoupled-baseline": generate_decoupled_baseline,
        "sonly-reproduction": generate_sonly_reproduction,
        "dual-sweep": generate_dual_sweep,
        "sonly-sweep": generate_sonly_sweep,
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
