"""
Local module for the 2026-05-23 Four-Parameter Coupling Multi-Particle Dual MZI report.

Contains all code exclusive to this report:
- Multi-particle Dicke-basis operator construction (N up to 10)
- Two protocols: dual MZI (BS on S and A) and S-only MZI (BS on S only)
- Four-parameter interaction: H_int = α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
                                   + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A
- Sensitivity via error propagation with central finite differences
- Multi-start L-BFGS-B optimisation over (α_xx, α_xz, α_zx, α_zz)
- Sweeps over ω ∈ [0.5, 5.0], N ∈ [1, 10] (dual MZI) and N ∈ {1, 5, 10} (S-only)
- Scaling analysis (log-log fit Δω ∝ N^α)
- JSON/manual reproduction of the 2026-05-21 result (S-only MZI, N=1)
- Exclusive plot functions for heatmaps, scaling curves, and ω-dependence
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260523/four_param_coupling_multi_particle.py --force

This module is importable via ``importlib.import_module("reports.20260523.four_param_coupling_multi_particle")``.
"""

from __future__ import annotations

__all__ = [
    "ScalingAnalysisResult",
    "fit_scaling_exponents",
]

import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.coupling_sweep import run_sweep_base
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
    plot_decoupled_baseline_heatmap,
)
from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,
    fit_scaling_exponents,
    generate_scaling_analysis,
)
from src.analysis.n_scaling_sweep import (
    generate_n_scaling_plots,
)
from src.physics.dicke_basis import jz_operator
from src.physics.multi_mzi import (
    compute_reduced_expectation_and_variance,
    embed_combined_operators,
    single_bs_unitary,
)
from src.utils.enums import OperatorBasis
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable
from src.visualization.coupling_heatmaps import plot_ratio_heatmap

if TYPE_CHECKING:
    from collections.abc import Callable

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL reference)
ALPHA_BOUND: float = 20.0  # |α_ij| ≤ 20 for optimisation bounds
N_LBFGS_STARTS: int = 25  # Number of L-BFGS-B random starts (20-30 per report)
FD_STEP: float = 1e-6  # Central finite-difference step

# ω and N sweep ranges (from report)
OMEGA_VALS: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
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


def build_hold_hamiltonian(
    N: int,
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the total holding Hamiltonian in the combined S⊗A space.

    H = ω (J_z^S + J_z^A)
        + α_xx J_x^S J_x^A + α_xz J_x^S J_z^A
        + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A

    Args:
        N: Particle number per subsystem.
        omega: Unknown phase rate.
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
    H += omega * (ops["Jz_S"] + ops["Jz_A"])

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


def protocol_bs_unitary(
    N: int,
    protocol: str = "dual",
    T_BS: float = DEFAULT_T_BS,
) -> np.ndarray:
    """Beam-splitter unitary based on protocol.

    Dual MZI: U_BS = exp(-i T_BS J_x) ⊗ exp(-i T_BS J_x)
    S-only MZI: U_BS = exp(-i T_BS J_x) ⊗ I_{N+1}

    Args:
        N: Particle number per subsystem.
        protocol: 'dual' or 'S-only'.
        T_BS: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    U_single = single_bs_unitary(N, T_BS)
    if protocol == "dual":
        U = np.kron(U_single, U_single)
    elif protocol == "S-only":
        U = np.kron(U_single, np.eye(N + 1, dtype=complex))
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}. Use 'dual' or 'S-only'.")
    return U


def hold_unitary(
    N: int,
    t_hold: float,
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Holding-time unitary in the combined S⊗A space.

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        N: Particle number per subsystem.
        t_hold: Holding time.
        omega: Unknown phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Pre-computed operators.

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    H = build_hold_hamiltonian(N, omega, alpha, ops)
    return expm(-1j * t_hold * H)


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
    omega: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    protocol: str = "dual",
    T_BS: float = DEFAULT_T_BS,
    t_hold: float = DEFAULT_t_hold,
) -> np.ndarray:
    """Run the full MZI circuit for the given protocol.

    |ψ_final⟩ = U_BS · U_hold(t_hold) · U_BS · |ψ₀⟩

    where U_BS depends on the protocol (dual or S-only).

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector (length (N+1)²).
        omega: Unknown phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Embedded operators.
        protocol: 'dual' (BS on both) or 'S-only' (BS on system only).
        T_BS: Beam-splitter angle (default π/2).
        t_hold: Holding time (default 10).

    Returns:
        Final state vector (length (N+1)²).
    """
    U_bs = protocol_bs_unitary(N, protocol, T_BS)
    psi = U_bs @ psi0
    psi = hold_unitary(N, t_hold, omega, alpha, ops) @ psi
    return U_bs @ psi


# ============================================================================
# Ancilla Trace-Out and Reduced Variance
# ============================================================================


# _reduced_expectation replaced by compute_reduced_expectation_and_variance from src


# ============================================================================
# Sensitivity Computation (Error Propagation)
# ============================================================================


def compute_sensitivity(
    N: int,
    psi0: np.ndarray,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    protocol: str = "dual",
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
    T_BS: float = DEFAULT_T_BS,
    t_hold: float = DEFAULT_t_hold,
) -> tuple[float, float, float, float]:
    """Compute the error-propagation sensitivity Δω.

    Δω = √Var(J_z^S) / |∂⟨J_z^S⟩/∂ω|

    Also returns ⟨J_z^S⟩, Var(J_z^S), and ∂⟨J_z^S⟩/∂ω at omega_true.

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector.
        omega_true: True phase rate.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Embedded operators.
        protocol: 'dual' or 'S-only'.
        meas_op: (N+1)×(N+1) measurement operator (default = J_z).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        Jz_single = jz_operator(N, basis=OperatorBasis.DICKE)
    else:
        Jz_single = meas_op

    # Evaluate at omega_true
    psi = evolve_circuit(N, psi0, omega_true, alpha, ops, protocol, T_BS, t_hold)
    exp_val, var_val = compute_reduced_expectation_and_variance(psi, N, Jz_single)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_circuit(
        N,
        psi0,
        omega_true + fd_step,
        alpha,
        ops,
        protocol,
        T_BS,
        t_hold,
    )
    psi_minus = evolve_circuit(
        N,
        psi0,
        omega_true - fd_step,
        alpha,
        ops,
        protocol,
        T_BS,
        t_hold,
    )

    exp_plus, _ = compute_reduced_expectation_and_variance(psi_plus, N, Jz_single)
    exp_minus, _ = compute_reduced_expectation_and_variance(psi_minus, N, Jz_single)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_omega = float(np.sqrt(var_val) / abs(d_exp))
    return delta_omega, exp_val, var_val, d_exp


# ============================================================================
# L-BFGS-B Objective and Multi-Start Optimisation
# ============================================================================


def _sensitivity_objective(
    alpha_params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    protocol: str,
    T_BS: float,
    t_hold: float,
    fd_step: float,
) -> float:
    """Objective function for L-BFGS-B optimisation.

    f(α) = Δω(α; ω_true, N, protocol)

    Args:
        alpha_params: 4-element array [α_xx, α_xz, α_zx, α_zz].
        N: Particle number per subsystem.
        omega_true: True phase rate.
        ops: Embedded operators.
        psi0: Initial state.
        protocol: 'dual' or 'S-only'.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.
        fd_step: Finite-difference step.

    Returns:
        Δω (positive float). Returns large number if fringe extremum.
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
        omega_true,
        alpha,
        ops,
        protocol,
        fd_step=fd_step,
        T_BS=T_BS,
        t_hold=t_hold,
    )
    return dt if np.isfinite(dt) else 1e10


def optimise_four_params(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray | None = None,
    protocol: str = "dual",
    alpha_bounds: tuple[float, float] = (-ALPHA_BOUND, ALPHA_BOUND),
    n_starts: int = N_LBFGS_STARTS,
    T_BS: float = DEFAULT_T_BS,
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
    seed: int | None = 42,
    maxiter: int = 1000,
    gtol: float = 1e-6,
) -> FourParamOptResult:
    """Run multi-start L-BFGS-B optimisation for a given (ω, N, protocol).

    For each start:
    1. Generate random initial α in [alpha_bounds]^4.
    2. Run L-BFGS-B with bounded optimisation.
    3. Select the run with lowest Δω.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        ops: Embedded operators.
        psi0: Initial state (default: built fresh).
        protocol: 'dual' or 'S-only'.
        alpha_bounds: (min, max) for all α coefficients.
        n_starts: Number of random starts.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.
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
    sql = 1.0 / (np.sqrt(N) * t_hold)

    best_delta = float("inf")
    best_alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    best_exp = 0.0
    best_var = 0.0
    best_d_exp = 0.0
    best_grad_norm = float("inf")
    n_converged = 0

    for start in range(n_starts):
        rng = np.random.default_rng(base_seed + int(omega * 1000) + start * 7)
        x0 = rng.uniform(lo, hi, size=4)

        result = minimize(
            _sensitivity_objective,
            x0,
            args=(N, omega, ops, psi0, protocol, T_BS, t_hold, fd_step),
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
                omega,
                best_alpha,
                ops,
                protocol,
                fd_step=fd_step,
                T_BS=T_BS,
                t_hold=t_hold,
            )
            best_exp, best_var, best_d_exp = exp_val, var_val, d_exp

    return FourParamOptResult(
        omega_value=omega,
        N=N,
        protocol=protocol,
        alpha_opt=best_alpha,
        delta_omega_opt=best_delta,
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
class FourParamOptResult(ParquetSerializable):
    """Result from a multi-start L-BFGS-B optimisation at a single (ω, N, protocol).

    Attributes:
        omega_value: ω at which the optimisation was performed.
        N: Particle number per subsystem.
        protocol: 'dual' or 'S-only'.
        alpha_opt: Optimal (α_xx, α_xz, α_zx, α_zz) found.
        delta_omega_opt: Minimal Δω found.
        sql: SQL = 1/(√N t_hold) reference value.
        expectation_Jz: ⟨J_z^S⟩ at the optimal point.
        variance_Jz: Var(J_z^S) at the optimal point.
        d_expectation: ∂⟨J_z^S⟩/∂ω at the optimal point.
        n_starts: Number of random starts used.
        n_converged: Number of starts that converged successfully.
        gradient_norm: L-BFGS-B projected gradient norm at optimum.
    """

    omega_value: float
    N: int
    protocol: str = "dual"
    alpha_opt: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    delta_omega_opt: float = float("inf")
    sql: float = 0.1
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_expectation: float = 0.0
    n_starts: int = N_LBFGS_STARTS
    n_converged: int = 0
    gradient_norm: float = 0.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "N",
        "protocol",
        "t_hold",
        "alpha_xx_opt",
        "alpha_xz_opt",
        "alpha_zx_opt",
        "alpha_zz_opt",
        "delta_omega_opt",
        "sql",
        "ratio",
        "expectation_Jz",
        "variance_Jz",
        "d_expectation",
        "n_starts",
        "n_converged",
        "gradient_norm",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with all metadata."""
        return pd.DataFrame(
            {
                "omega": [self.omega_value],
                "N": [self.N],
                "protocol": [self.protocol],
                "t_hold": [DEFAULT_t_hold],
                "alpha_xx_opt": [self.alpha_opt[0]],
                "alpha_xz_opt": [self.alpha_opt[1]],
                "alpha_zx_opt": [self.alpha_opt[2]],
                "alpha_zz_opt": [self.alpha_opt[3]],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [
                    self.delta_omega_opt / self.sql
                    if np.isfinite(self.delta_omega_opt) and self.sql > 0
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

    @classmethod
    def from_parquet(cls, path: str | Path) -> FourParamOptResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            omega_value=float(row["omega"]),
            N=int(row["N"]),
            protocol=str(row["protocol"]),
            alpha_opt=(
                float(row["alpha_xx_opt"]),
                float(row["alpha_xz_opt"]),
                float(row["alpha_zx_opt"]),
                float(row["alpha_zz_opt"]),
            ),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            d_expectation=float(row["d_expectation"]),
            n_starts=int(row["n_starts"]),
            n_converged=int(row["n_converged"]),
            gradient_norm=float(row["gradient_norm"]),
        )


@dataclass
class FourParamSweepResult(ParquetSerializable):
    """Full sweep over (ω, N, protocol) with optimisation per point.

    All array fields have the same length (n_points), stored in row-major
    order (ω varies fastest, then N).

    Attributes:
        omega_values: ω values for each point.
        N_values: N values for each point.
        protocol: Protocol string for each point (list of str).
        alpha_xx_opt: Optimal α_xx at each point.
        alpha_xz_opt: Optimal α_xz at each point.
        alpha_zx_opt: Optimal α_zx at each point.
        alpha_zz_opt: Optimal α_zz at each point.
        delta_omega_opt: Minimal Δω at each point.
        sql_values: SQL = 1/(√N t_hold) at each point.
        ratio: Δω_opt / SQL at each point.
        expectation_Jz: ⟨J_z^S⟩ at optimum.
        variance_Jz: Var(J_z^S) at optimum.
        d_expectation: ∂⟨J_z^S⟩/∂ω at optimum.
        n_starts: Number of random starts per point.
        n_converged: Number of converged starts per point.
        gradient_norm: L-BFGS-B projected gradient norm at optimum per point.
        t_hold: Holding time (scalar).
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    protocol: list[str] = field(default_factory=list)
    alpha_xx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_xz_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zx_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_zz_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz: np.ndarray = field(default_factory=lambda: np.array([]))
    d_expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    n_starts: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_converged: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    gradient_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    t_hold: float = DEFAULT_t_hold

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "N",
        "protocol",
        "t_hold",
        "alpha_xx_opt",
        "alpha_xz_opt",
        "alpha_zx_opt",
        "alpha_zz_opt",
        "delta_omega_opt",
        "sql",
        "ratio",
        "expectation_Jz",
        "variance_Jz",
        "d_expectation",
        "n_starts",
        "n_converged",
        "gradient_norm",
    ]

    def __post_init__(self) -> None:
        # Ensure int dtype for integer arrays
        if self.N_values.dtype.kind != "i":
            self.N_values = self.N_values.astype(int)
        if self.n_starts.dtype.kind != "i":
            self.n_starts = self.n_starts.astype(int)
        if self.n_converged.dtype.kind != "i":
            self.n_converged = self.n_converged.astype(int)

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.omega_values)
        # Pad protocol list if needed
        protocol_list = self.protocol
        if len(protocol_list) < n:
            protocol_list = protocol_list + ["unknown"] * (n - len(protocol_list))
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "N": self.N_values,
                "protocol": protocol_list[:n],
                "t_hold": np.full(n, self.t_hold),
                "alpha_xx_opt": self.alpha_xx_opt,
                "alpha_xz_opt": self.alpha_xz_opt,
                "alpha_zx_opt": self.alpha_zx_opt,
                "alpha_zz_opt": self.alpha_zz_opt,
                "delta_omega_opt": self.delta_omega_opt,
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

    @classmethod
    def from_parquet(cls, path: str | Path) -> FourParamSweepResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
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

    def filter_protocol(self, protocol: str) -> FourParamSweepResult:
        """Return a new result filtered to a single protocol."""
        mask = [p == protocol for p in self.protocol]
        return FourParamSweepResult(
            omega_values=self.omega_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_omega_opt=self.delta_omega_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            t_hold=self.t_hold,
        )

    def filter_omega(self, omega: float) -> FourParamSweepResult:
        """Return a new result filtered to a single ω value."""
        mask = np.isclose(self.omega_values, omega)
        return FourParamSweepResult(
            omega_values=self.omega_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_omega_opt=self.delta_omega_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            t_hold=self.t_hold,
        )

    def filter_N(self, N: int) -> FourParamSweepResult:
        """Return a new result filtered to a single N value."""
        mask = self.N_values == N
        return FourParamSweepResult(
            omega_values=self.omega_values[mask],
            N_values=self.N_values[mask],
            protocol=[self.protocol[i] for i in range(len(mask)) if mask[i]],
            alpha_xx_opt=self.alpha_xx_opt[mask],
            alpha_xz_opt=self.alpha_xz_opt[mask],
            alpha_zx_opt=self.alpha_zx_opt[mask],
            alpha_zz_opt=self.alpha_zz_opt[mask],
            delta_omega_opt=self.delta_omega_opt[mask],
            sql_values=self.sql_values[mask],
            ratio=self.ratio[mask],
            expectation_Jz=self.expectation_Jz[mask],
            variance_Jz=self.variance_Jz[mask],
            d_expectation=self.d_expectation[mask],
            n_starts=self.n_starts[mask],
            n_converged=self.n_converged[mask],
            gradient_norm=self.gradient_norm[mask],
            t_hold=self.t_hold,
        )


# ============================================================================
# Sweep Execution
# ============================================================================


def run_sweep(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    protocol: str = "dual",
    t_hold: float = DEFAULT_t_hold,
    n_starts: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FourParamSweepResult:
    """Run the full sweep over ω and N with four-parameter optimisation.

    Args:
        omega_values: ω values to sweep (default: [0.5, 1.0, ..., 5.0]).
        N_values: N values to sweep (default: 1 to 20 for dual, 1/5/10 for S-only).
        protocol: 'dual' or 'S-only'.
        t_hold: Holding time.
        n_starts: Number of L-BFGS-B random starts per point.
        progress_callback: Optional callback (current, total).

    Returns:
        FourParamSweepResult with all optimised points.
    """
    # Resolve defaults with protocol-specific N ranges
    if omega_values is None:
        omega_values = np.array(OMEGA_VALS, dtype=float)

    if N_values is None:
        n_defaults = DUAL_MZI_N_VALS if protocol == "dual" else SONLY_MZI_N_VALS
        N_values = np.array(n_defaults, dtype=int)

    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, Any]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        actual_starts = _n_starts_for_N(N) if n_starts is None else n_starts
        opt = optimise_four_params(
            N=N,
            omega=omega,
            ops=_cache["ops"],
            psi0=_cache["psi0"],
            protocol=protocol,
            n_starts=actual_starts,
            t_hold=t_hold,
        )
        ratio = (
            opt.delta_omega_opt / opt.sql
            if np.isfinite(opt.delta_omega_opt) and opt.sql > 0
            else float("inf")
        )
        return {
            "delta_omega": opt.delta_omega_opt,
            "sql": opt.sql,
            "expectation_Jz": opt.expectation_Jz,
            "variance_Jz": opt.variance_Jz,
            "d_expectation": opt.d_expectation,
            "alpha_xx_opt": opt.alpha_opt[0],
            "alpha_xz_opt": opt.alpha_opt[1],
            "alpha_zx_opt": opt.alpha_opt[2],
            "alpha_zz_opt": opt.alpha_opt[3],
            "n_starts_arr": actual_starts,
            "n_conv_arr": opt.n_converged,
            "grad_norm_arr": opt.gradient_norm,
            "protocol": protocol,
            "ratio": ratio,
        }

    data = run_sweep_base(
        omega_values, N_values, per_point, progress_callback=progress_callback
    )

    # Convert protocol str array back to list (FourParamSweepResult expects list)
    proto_list: list[str] = list(
        data.get("protocol", np.full(len(data["omegas"]), protocol, dtype=object))
    )  # type: ignore[arg-type]

    return FourParamSweepResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        protocol=proto_list,
        alpha_xx_opt=data.get("alpha_xx_opt", np.full(len(data["omegas"]), np.nan)),
        alpha_xz_opt=data.get("alpha_xz_opt", np.full(len(data["omegas"]), np.nan)),
        alpha_zx_opt=data.get("alpha_zx_opt", np.full(len(data["omegas"]), np.nan)),
        alpha_zz_opt=data.get("alpha_zz_opt", np.full(len(data["omegas"]), np.nan)),
        delta_omega_opt=data["delta_opts"],
        sql_values=data["sqls"],
        ratio=data["ratio"],
        expectation_Jz=data.get("expectation_Jz", np.zeros(len(data["omegas"]))),
        variance_Jz=data.get("variance_Jz", np.zeros(len(data["omegas"]))),
        d_expectation=data.get("d_expectation", np.zeros(len(data["omegas"]))),
        n_starts=data.get("n_starts_arr", np.zeros(len(data["omegas"]), dtype=int)),
        n_converged=data.get("n_conv_arr", np.zeros(len(data["omegas"]), dtype=int)),
        gradient_norm=data.get("grad_norm_arr", np.full(len(data["omegas"]), np.nan)),
        t_hold=t_hold,
    )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def compute_decoupled_baseline(
    omega_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    protocol: str = "dual",
    t_hold: float = DEFAULT_t_hold,
    fd_step: float = FD_STEP,
) -> FourParamSweepResult:
    """Verify the decoupled baseline (α = 0) for all (ω, N, protocol) pairs.

    At α = (0,0,0,0), the sensitivity should equal SQL = 1/(√N t_hold).

    Args:
        omega_values: ω values (default: sweep range).
        N_values: N values (default: 1 to 20 for dual).
        protocol: 'dual' or 'S-only'.
        t_hold: Holding time.

    Returns:
        FourParamSweepResult with α = 0 results.
    """
    if omega_values is None:
        omega_values = np.array(OMEGA_VALS, dtype=float)
    if N_values is None:
        N_defaults = DUAL_MZI_N_VALS if protocol == "dual" else SONLY_MZI_N_VALS
        N_values = np.array(N_defaults, dtype=int)

    zero_alpha = (0.0, 0.0, 0.0, 0.0)
    _cache: dict[str, Any] = {"last_N": None, "ops": None, "psi0": None}

    def per_point(N: int, omega: float) -> dict[str, Any]:
        if _cache["last_N"] != N:
            _cache["ops"] = embed_combined_operators(N)
            _cache["psi0"] = initial_state(N)
            _cache["last_N"] = N
        sql = 1.0 / (np.sqrt(N) * t_hold)
        dt, exp_val, var_val, d_exp_val = compute_sensitivity(
            N,
            _cache["psi0"],
            omega,
            zero_alpha,
            _cache["ops"],
            protocol=protocol,
            t_hold=t_hold,
            fd_step=fd_step,
        )
        ratio = dt / sql if np.isfinite(dt) and sql > 0 else float("inf")
        return {
            "delta_omega": dt,
            "sql": sql,
            "expectation_Jz": exp_val,
            "variance_Jz": var_val,
            "d_expectation": d_exp_val,
            "alpha_xx_opt": 0.0,
            "alpha_xz_opt": 0.0,
            "alpha_zx_opt": 0.0,
            "alpha_zz_opt": 0.0,
            "n_starts_arr": 0,
            "n_conv_arr": 0,
            "grad_norm_arr": float("nan"),
            "protocol": protocol,
            "ratio": ratio,
        }

    data = run_sweep_base(
        omega_values,
        N_values,
        per_point,
    )

    proto_list: list[str] = list(
        data.get("protocol", np.full(len(data["omegas"]), protocol, dtype=object))
    )  # type: ignore[arg-type]

    return FourParamSweepResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        protocol=proto_list,
        alpha_xx_opt=data.get("alpha_xx_opt", np.zeros(len(data["omegas"]))),
        alpha_xz_opt=data.get("alpha_xz_opt", np.zeros(len(data["omegas"]))),
        alpha_zx_opt=data.get("alpha_zx_opt", np.zeros(len(data["omegas"]))),
        alpha_zz_opt=data.get("alpha_zz_opt", np.zeros(len(data["omegas"]))),
        delta_omega_opt=data["delta_opts"],
        sql_values=data["sqls"],
        ratio=data["ratio"],
        expectation_Jz=data.get("expectation_Jz", np.zeros(len(data["omegas"]))),
        variance_Jz=data.get("variance_Jz", np.zeros(len(data["omegas"]))),
        d_expectation=data.get("d_expectation", np.zeros(len(data["omegas"]))),
        n_starts=data.get("n_starts_arr", np.zeros(len(data["omegas"]), dtype=int)),
        n_converged=data.get("n_conv_arr", np.zeros(len(data["omegas"]), dtype=int)),
        gradient_norm=data.get("grad_norm_arr", np.full(len(data["omegas"]), np.nan)),
        t_hold=t_hold,
    )


# ============================================================================
# Scaling Analysis
# ============================================================================


# ============================================================================
# Plot Functions
# ============================================================================


def plot_omega_dependence(
    sweep: FourParamSweepResult,
    save_path: str | Path,
    N_fixed: int | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot Δω_opt vs ω at fixed N, with SQL reference line.

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
    omega_vals = sweep.omega_values[mask]
    delta_vals = sweep.delta_omega_opt[mask]
    sql_vals = sweep.sql_values[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference (flat line for fixed N)
    sql_val = 1.0 / (np.sqrt(N_fixed) * sweep.t_hold) if len(sql_vals) > 0 else 0.1
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
            omega_vals[finite_mask],
            delta_vals[finite_mask],
            "o-",
            color="C0",
            markersize=6,
            linewidth=1.5,
            label=rf"$\Delta\omega_{{\mathrm{{opt}}}}(N={N_fixed})$",
        )

    protocol_label = sweep.protocol[0] if sweep.protocol else "dual"
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(f"$\\omega$-Dependence at $N={N_fixed}$:\n{protocol_label} MZI")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_omega_scan(
    sweep: FourParamSweepResult,
    save_path: str | Path,
    N_fixed: int | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δω_opt vs ω with SQL reference and optimal α parameters as secondary axis.

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
    omega = filtered.omega_values
    sql_val = 1.0 / (np.sqrt(N_fixed) * sweep.t_hold)

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

    # Δω vs ω
    valid = np.isfinite(filtered.delta_omega_opt)
    if np.any(valid):
        ax1.plot(
            omega[valid],
            filtered.delta_omega_opt[valid],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.8,
            label=r"$\Delta\omega_{\mathrm{opt}}$",
        )
        # Annotate best point
        best_idx = int(np.argmin(filtered.delta_omega_opt[valid]))
        best_omega = float(omega[valid][best_idx])
        best_val = float(filtered.delta_omega_opt[valid][best_idx])
        best_ratio = best_val / sql_val if sql_val > 0 else float("inf")
        ax1.annotate(
            rf"Best: $\Delta\omega$={best_val:.5f} ({best_ratio:.3f}$\times$SQL)"
            rf" at $\omega$={best_omega:.2f}",
            xy=(best_omega, best_val),
            xytext=(best_omega + 0.8, best_val + 0.02),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    protocol_label = filtered.protocol[0] if filtered.protocol else "dual"
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(
        f"4-Parameter Sensitivity vs $\\omega$ at $N={N_fixed}$:\n{protocol_label} MZI"
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
                omega[valid_a],
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

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260523"


parquet_path, fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ── Generator Functions ────────────────────────────────────────────────


def generate_dual_sweep(force: bool = False) -> None:
    """Run the dual MZI full sweep (N=1-20, ω from 0.5 to 5.0)."""
    csv_p = parquet_path("dual-mzi-sweep")
    fig_ratio = fig_path("dual-mzi-ratio-heatmap")
    fig_omega_scan = fig_path("dual-mzi-omega-scan-N5")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = FourParamSweepResult.from_parquet(csv_p)
    else:
        N_arr = np.array(DUAL_MZI_N_VALS, dtype=int)
        omega_arr = np.array(OMEGA_VALS, dtype=float)
        print(
            f"[run]  Computing dual MZI sweep "
            f"({len(omega_arr)}×{len(N_arr)} = {len(omega_arr) * len(N_arr)} points)..."
        )
        result = run_sweep(omega_values=omega_arr, N_values=N_arr, protocol="dual")
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(
        result.omega_values,
        result.N_values,
        result.ratio,
        fig_ratio,
        title_suffix="Dual MZI",
    )
    print(f"[fig]  {fig_ratio}")
    plot_omega_scan(result, fig_omega_scan, N_fixed=5)
    print(f"[fig]  {fig_omega_scan}")


def generate_sonly_sweep(force: bool = False) -> None:
    """Run the S-only MZI comparison sweep (N=1,5,10)."""
    csv_p = parquet_path("sonly-mzi-sweep")
    fig_ratio = fig_path("sonly-mzi-ratio-heatmap")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = FourParamSweepResult.from_parquet(csv_p)
    else:
        N_arr = np.array(SONLY_MZI_N_VALS, dtype=int)
        omega_arr = np.array(OMEGA_VALS, dtype=float)
        print(
            f"[run]  Computing S-only MZI sweep "
            f"({len(omega_arr)}×{len(N_arr)} = {len(omega_arr) * len(N_arr)} points)..."
        )
        result = run_sweep(omega_values=omega_arr, N_values=N_arr, protocol="S-only")
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_ratio_heatmap(
        result.omega_values,
        result.N_values,
        result.ratio,
        fig_ratio,
        title_suffix="S-only MZI",
    )
    print(f"[fig]  {fig_ratio}")


def generate_sonly_reproduction(force: bool = False) -> None:
    """Reproduce the 2026-05-21 result: S-only MZI, N=1, ω=3.8.

    Expected: Δω/Δω_SQL ≤ 0.690 (i.e., Δω ≤ 0.0690 at t_hold=10).
    """
    csv_p = parquet_path("sonly-reproduction-n1")
    N_val = 1
    omega_val = 3.8
    sql = 1.0 / (np.sqrt(N_val) * DEFAULT_t_hold)

    print("[run]  2026-05-21 reproduction: S-only MZI, N=1, ω=3.8")

    ops = embed_combined_operators(N_val)
    psi0 = initial_state(N_val)
    result = optimise_four_params(
        N=N_val,
        omega=omega_val,
        ops=ops,
        psi0=psi0,
        protocol="S-only",
        n_starts=N_LBFGS_STARTS,  # Per report: 20-30 starts
    )

    ratio = (
        result.delta_omega_opt / sql
        if np.isfinite(result.delta_omega_opt)
        else float("inf")
    )
    print(
        f"  Δω_opt = {result.delta_omega_opt:.6f}, SQL = {sql:.6f}, ratio = {ratio:.4f}"
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
        f"2026-05-21 Reproduction (S-only MZI, N=1, ω=3.8)\n"
        f"Δω_opt = {result.delta_omega_opt:.6f}\n"
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


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling plots from the dual MZI sweep data."""
    # Load the sweep to determine the protocol label for the title
    csv_p = parquet_path("dual-mzi-sweep")
    if not csv_p.exists():
        print("[skip] Dual MZI sweep data not found; run 'dual-sweep' first")
        return

    result = FourParamSweepResult.from_parquet(csv_p)
    protocol_label = result.protocol[0] if result.protocol else "dual"

    def _plot_with_title(
        df: Any,
        omega_fixed: float,
        save_path: Path,
        t_hold: float | None = None,
        include_2n_sql: bool = False,
    ) -> None:
        from src.visualization.scaling_plots import plot_n_scaling_single_omega

        title = f"N-Scaling at $\\omega={omega_fixed:.2f}$:\\n{protocol_label} MZI"
        plot_n_scaling_single_omega(
            df,
            omega_fixed=omega_fixed,
            save_path=save_path,
            t_hold=t_hold,
            include_2n_sql=include_2n_sql,
            title=title,
        )

    omega_fig_pairs = [
        (0.5, fig_path(f"dual-mzi-n-scaling-omega{omega_val:.1f}"))
        for omega_val in [0.5, 2.5, 5.0]
    ]
    generate_n_scaling_plots(
        force=force,
        parquet_path=csv_p,
        result_cls=FourParamSweepResult,
        omega_fig_pairs=omega_fig_pairs,
        plot_fn=_plot_with_title,
        label="four-param n-scaling",
    )


def generate_omega_dependence(force: bool = False) -> None:
    """ω-dependence plots at fixed N values from sweep data."""
    csv_dual = parquet_path("dual-mzi-sweep")

    if csv_dual.exists():
        result_dual = FourParamSweepResult.from_parquet(csv_dual)
        for N_fixed in [1, 5, 10, 20]:
            fig_p = fig_path(f"dual-mzi-omega-N{N_fixed}")
            plot_omega_dependence(result_dual, fig_p, N_fixed=N_fixed)
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

    tasks: dict[str, Callable[..., Any]] = {
        "decoupled-baseline": lambda force=False: (
            generate_decoupled_baseline(
                force=force,
                parquet_path=parquet_path("decoupled-baseline-dual"),
                fig_path=fig_path("decoupled-baseline-dual"),
                compute_fn=compute_decoupled_baseline,
                compute_kwargs={
                    "omega_values": np.array(OMEGA_VALS, dtype=float),
                    "N_values": np.array(DUAL_MZI_N_VALS, dtype=int),
                    "protocol": "dual",
                },
                result_cls=FourParamSweepResult,
                plot_fn=lambda r, p: plot_decoupled_baseline_heatmap(
                    r,
                    p,
                    title_prefix=r"Decoupled Baseline Verification ($\alpha = 0$, dual MZI",
                ),
                label="decoupled baseline (dual MZI, α = 0)",
            )
            or generate_decoupled_baseline(
                force=force,
                parquet_path=parquet_path("decoupled-baseline-sonly"),
                fig_path=fig_path("decoupled-baseline-sonly"),
                compute_fn=compute_decoupled_baseline,
                compute_kwargs={
                    "omega_values": np.array(OMEGA_VALS, dtype=float),
                    "N_values": np.array(SONLY_MZI_N_VALS, dtype=int),
                    "protocol": "S-only",
                },
                result_cls=FourParamSweepResult,
                plot_fn=lambda r, p: plot_decoupled_baseline_heatmap(
                    r,
                    p,
                    title_prefix=r"Decoupled Baseline Verification ($\alpha = 0$, S-only MZI",
                ),
                label="decoupled baseline (S-only MZI, α = 0)",
            )
        ),
        "sonly-reproduction": generate_sonly_reproduction,
        "dual-sweep": generate_dual_sweep,
        "sonly-sweep": generate_sonly_sweep,
        "n-scaling": generate_n_scaling,
        "omega-dependence": generate_omega_dependence,
        "scaling-analysis": partial(
            generate_scaling_analysis,
            parquet_path=parquet_path("dual-mzi-sweep"),
            scaling_path=parquet_path("dual-mzi-scaling"),
            fig_path=fig_path("dual-mzi-scaling-exponents"),
            result_cls=FourParamSweepResult,
            label="dual-mzi scaling analysis",
        ),
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
