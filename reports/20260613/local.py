"""
Local module for the 2026-06-13 ω-Modulated Drive with Weighted Joint Measurement report.

Combines the ω-modulated ancilla drive (20260519) with the weighted joint
measurement M(ψ) = cosψ·J_z^S + sinψ·J_z^A (20260525).

Circuit: BS_S → Hold → BS_S → measure M(ψ).

The optimisation space is 5D: (a_x, a_y, a_z, a_zz, ψ).

Usage:
    uv run python reports/20260613/local.py --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_decoupled_baseline,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)

# ============================================================================
# Constants
# ============================================================================

PSI_BOUNDS: tuple[float, float] = (-np.pi, np.pi)  # Measurement angle bounds

# Local copies of shared constants (originally in src/physics/n_particle_drive.py)
T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Parameter bounds

# ω values for the scan
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values for the scaling scan (1 to 20)
N_VALS: list[int] = list(range(1, 21))

# Random search parameters
N_RANDOM: int = 1000  # Increased for 5D
N_NM_REFINE: int = 50
NM_MAXITER: int = 5000

# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260613"


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, float], bool]:
    """Verify the decoupled baseline for all (N, ω) pairs.

    At zero drive and zero interaction, the sensitivity must equal
    Δω = 1/(√N × T_HOLD) to machine precision.

    Args:
        N_values: List of N values (default: 1 to 20).
        omega_values: List of ω values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping (N, ω) → PASS/FAIL (True/False).
    """
    if N_values is None:
        N_values = N_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N)
        for omega in omega_values:
            delta = compute_n_particle_decoupled_baseline(N, omega)
            results[(N, omega)] = bool(np.isclose(delta, sql_ref, rtol=rtol))
    return results


def build_joint_measurement_operator(
    N: int,
    psi: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the weighted joint measurement operator M(ψ).

    M(ψ) = cosψ · J_z^S + sinψ · J_z^A

    The coefficients automatically satisfy m_s² + m_a² = 1 with
    m_s = cosψ, m_a = sinψ.

    Args:
        N: Number of system particles (for dimension check).
        psi: Measurement weight angle (radians).
        ops: Operators from build_n_particle_operators(N).

    Returns:
        2(N+1) × 2(N+1) Hermitian matrix.
    """
    d_tot = 2 * (N + 1)
    M = np.cos(psi) * ops["Jz_S"] + np.sin(psi) * ops["Jz_A"]
    M = 0.5 * (M + M.conj().T)
    assert M.shape == (d_tot, d_tot), (
        f"M has shape {M.shape}, expected ({d_tot}, {d_tot})"
    )
    assert np.allclose(M, M.conj().T, atol=1e-12), (
        "Joint measurement operator not Hermitian"
    )
    return M


# ============================================================================
# 2D Slice Scan (ψ × a_zz)
# ============================================================================


@dataclass
class Joint2DSliceResult:
    """Result of a 2D slice scan over (ψ, a_zz) for the joint measurement.

    Attributes:
        psi_values: Array of ψ values (radians).
        azz_values: Array of a_zz values.
        delta_omega_grid: 2D grid of Δω values (psi × azz).
        N: Number of system particles.
        omega_value: Phase rate value.
        sql: SQL reference value.
    """

    psi_values: np.ndarray
    azz_values: np.ndarray
    delta_omega_grid: np.ndarray
    N: int
    omega_value: float
    sql: float


def joint_2d_psi_azz_slice(
    N: int,
    omega: float,
    psi_range: tuple[float, float] = PSI_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_psi: int = 101,
    n_azz: int = 101,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
) -> Joint2DSliceResult:
    """Run a 2D slice scan over (ψ, a_zz) with fixed a_x=a_y=a_z=0.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        psi_range: (min, max) for ψ.
        azz_range: (min, max) for a_zz.
        n_psi: Number of ψ points.
        n_azz: Number of a_zz points.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.

    Returns:
        Joint2DSliceResult with the sensitivity grid.
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    psi_vals = np.linspace(psi_range[0], psi_range[1], n_psi)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)
    grid = np.full((n_psi, n_azz), np.inf, dtype=float)

    for i, pv in enumerate(psi_vals):
        M = build_joint_measurement_operator(N, pv, ops)
        for j, av in enumerate(azz_vals):
            domega = compute_n_particle_sensitivity(
                N,
                psi0,
                T_bs,
                t_hold,
                omega,
                0.0,
                0.0,
                0.0,
                av,
                ops,
                meas_op=M,
            )
            grid[i, j] = domega

    return Joint2DSliceResult(
        psi_values=psi_vals,
        azz_values=azz_vals,
        delta_omega_grid=grid,
        N=N,
        omega_value=omega,
        sql=sql_reference(N),
    )


# ============================================================================
# Joint Measurement Sensitivity with ψ in parameter vector
# ============================================================================


def compute_joint_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    psi: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> float:
    """Compute sensitivity using the weighted joint measurement M(ψ).

    Builds M(ψ) = cosψ·J_z^S + sinψ·J_z^A and computes Δω.

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        psi: Measurement weight angle.
        ops: Operators from build_n_particle_operators(N).
        fd_step: Finite-difference step size.

    Returns:
        Sensitivity Δω (positive float).
    """
    M = build_joint_measurement_operator(N, psi, ops)
    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        fd_step=fd_step,
        meas_op=M,
    )


# ============================================================================
# 5D Random Search
# ============================================================================


@dataclass
class JointRandomSearchResult:
    """Result of a 5D random search for the joint measurement protocol.

    Attributes:
        samples: Array of shape (n_samples, 5) with parameter values.
        delta_omega_values: Array of Δω values for each sample.
        best_params: Best 5-element parameter vector [ax, ay, az, azz, psi].
        best_delta_omega: Best Δω found.
        omega_value: Phase rate value.
        sql: SQL reference value.
        t_hold: Holding time.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, float, float, float, float]
    best_delta_omega: float
    omega_value: float
    sql: float
    t_hold: float


def joint_random_search(
    N: int,
    omega: float,
    n_samples: int = N_RANDOM,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    seed: int | None = 42,
) -> JointRandomSearchResult:
    """Random search over the 5D parameter space (a_x, a_y, a_z, a_zz, ψ).

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four drive coefficients.
        psi_bounds: (min, max) for ψ.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        JointRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    lo, hi = bounds
    psi_lo, psi_hi = psi_bounds

    # 5D samples: [a_x, a_y, a_z, a_zz, ψ]
    samples_4d = rng.uniform(lo, hi, size=(n_samples, 4))
    psi_samples = rng.uniform(psi_lo, psi_hi, size=(n_samples, 1))
    samples = np.concatenate([samples_4d, psi_samples], axis=1)

    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])
        psi_val = float(samples[i, 4])

        domega = compute_joint_sensitivity(
            N,
            psi0,
            T_bs,
            t_hold,
            omega,
            ax,
            ay,
            az,
            azz,
            psi_val,
            ops,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
        float(samples[best_idx, 4]),
    )

    return JointRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N),
        t_hold=t_hold,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def joint_sensitivity_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω with joint measurement.

    params = [a_x, a_y, a_z, a_zz, ψ] (5 elements).

    Args:
        params: 5-element parameter vector.
        N: Number of system particles.
        omega_true: True phase rate.
        ops: N-particle operators.
        psi0: Initial state vector.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for drive parameters.
        psi_bounds: (min, max) for ψ.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])
    psi_val = float(params[4])

    lo, hi = bounds
    psi_lo, psi_hi = psi_bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2
    if psi_val < psi_lo:
        penalty += penalty_scale * (psi_lo - psi_val) ** 2
    if psi_val > psi_hi:
        penalty += penalty_scale * (psi_val - psi_hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_joint_sensitivity(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        ax,
        ay,
        az,
        azz,
        psi_val,
        ops,
        fd_step,
    )


# ============================================================================
# Nelder-Mead Result Dataclass
# ============================================================================


@dataclass
class JointNMSensitivityResult:
    """Result of a single Nelder-Mead optimisation run for joint measurement.

    Attributes:
        delta_omega_opt: Best sensitivity Δω found.
        params_opt: Optimal 5-element parameter vector [ax, ay, az, azz, psi].
        omega_true: True ω used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        expectation_M: ⟨M⟩ at the optimal operating point.
        variance_M: Var(M) at the optimal operating point.
        d_expectation: ∂⟨M⟩/∂ω at the optimal operating point.
        history: Objective values per iteration (if tracked).
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    history: list[float] = field(default_factory=list)


def run_joint_nelder_mead(
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    track_history: bool = False,
) -> JointNMSensitivityResult:
    """Run Nelder-Mead optimisation for the N-particle joint measurement protocol.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        ops: N-particle operators (built if None).
        psi0: Initial state (built if None).
        x0: Initial 5-parameter vector [ax, ay, az, azz, psi]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for drive parameters.
        psi_bounds: (min, max) for ψ.
        track_history: If True, record objective values per iteration.

    Returns:
        JointNMSensitivityResult.
    """
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        psi_lo, psi_hi = psi_bounds
        x0 = np.concatenate(
            [
                rng.uniform(lo, hi, size=4),
                rng.uniform(psi_lo, psi_hi, size=1),
            ]
        )
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (5,), f"x0 must have 5 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return joint_sensitivity_objective(
            p,
            N,
            omega_true,
            ops,
            psi0,
            bounds=bounds,
            psi_bounds=psi_bounds,
        )

    history: list[float] = []

    def callback(xk: np.ndarray) -> None:
        if track_history:
            val = objective(xk)
            history.append(val)

    result = minimize(  # type: ignore[call-overload]
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val_s, var_val_s = compute_expectation_and_variance(psi_final, ops["Jz_S"])
    opt_psi = float(opt_params[4])
    M_opt = build_joint_measurement_operator(N, opt_psi, ops)
    exp_val_m, var_val_m = compute_expectation_and_variance(psi_final, M_opt)

    # Compute derivative of M at the optimal point
    psi_plus = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true + FD_STEP,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    psi_minus = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true - FD_STEP,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    d_exp = (
        np.real(psi_plus.conj() @ M_opt @ psi_plus)
        - np.real(psi_minus.conj() @ M_opt @ psi_minus)
    ) / (2.0 * FD_STEP)

    return JointNMSensitivityResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=float(exp_val_s),
        variance_Jz=float(var_val_s),
        expectation_M=float(exp_val_m),
        variance_M=float(var_val_m),
        d_expectation=float(d_exp),
        history=history.copy(),
    )


# ============================================================================
# Single (N, ω) Sensitivity Computation
# ============================================================================


def run_single_joint_n_omega(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
) -> JointNScalingResult:
    """Nelder-Mead optimisation for a single (N, ω) pair with joint meas.

    Returns a JointNScalingResult wrapping the NM result with metadata.

    Args:
        N: Number of system particles.
        omega: True phase rate.
        ops: Pre-built operators (built if None).
        psi0: Initial state (built if None).
        seed: Random seed for initial guess.
        maxiter: Max Nelder-Mead iterations.

    Returns:
        JointNScalingResult with sensitivity and optimal parameters.
    """
    nm_result = run_joint_nelder_mead(
        N=N,
        omega_true=omega,
        ops=ops,
        psi0=psi0,
        seed=seed,
        maxiter=maxiter,
    )
    sql = 1.0 / (np.sqrt(N) * T_HOLD)
    ratio = nm_result.delta_omega_opt / sql if sql > 0 else 0.0
    params = nm_result.params_opt
    return JointNScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=nm_result.delta_omega_opt,
        sql=sql,
        ratio=ratio,
        a_x_opt=float(params[0]),
        a_y_opt=float(params[1]),
        a_z_opt=float(params[2]),
        a_zz_opt=float(params[3]),
        psi_opt=float(params[4]),
        expectation_Jz=nm_result.expectation_Jz,
        variance_Jz=nm_result.variance_Jz,
        expectation_M=nm_result.expectation_M,
        variance_M=nm_result.variance_M,
        d_expectation=nm_result.d_expectation,
        t_hold=T_HOLD,
        success=nm_result.success,
        nfev=nm_result.nfev,
    )


# ============================================================================
# S-Only Nelder-Mead (comparison baseline)
# ============================================================================


def sonly_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective for S-only optimisation (measures J_z^S only).

    Enforces bounds via quadratic penalty for fair comparison with joint protocol.
    Delegates to compute_n_particle_sensitivity with J_z^S measurement.
    """
    ax, ay, az, azz = (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
    )

    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2
    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        ax,
        ay,
        az,
        azz,
        ops,
        meas_op=ops["Jz_S"],
    )


def run_single_sonly_n_omega(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
) -> JointNScalingResult:
    """S-only Nelder-Mead for a single (N, ω) pair, returns JointNScalingResult."""
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)

    rng = np.random.default_rng(seed)
    lo, hi = DRIVE_BOUNDS
    x0 = rng.uniform(lo, hi, size=4)

    def objective(p: np.ndarray) -> float:
        return sonly_objective(p, N, omega, ops, psi0)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )

    opt_params = result.x.copy()
    psi_final = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])
    sql = 1.0 / (np.sqrt(N) * T_HOLD)

    return JointNScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=float(result.fun),
        sql=sql,
        ratio=float(result.fun) / sql if sql > 0 else 0.0,
        a_x_opt=float(opt_params[0]),
        a_y_opt=float(opt_params[1]),
        a_z_opt=float(opt_params[2]),
        a_zz_opt=float(opt_params[3]),
        psi_opt=0.0,
        expectation_Jz=float(exp_val),
        variance_Jz=float(var_val),
        t_hold=T_HOLD,
        success=bool(result.success),
        nfev=int(result.nfev),
    )


# ============================================================================
# Scaling Dataclasses
# ============================================================================


@dataclass
class JointNScalingResult:
    """Result of joint measurement optimisation for a single (N, ω) pair.

    Attributes:
        N: Number of particles.
        omega: True ω used.
        delta_omega_opt: Best Δω found.
        sql: SQL reference value (= 1/(√N × T_HOLD)).
        ratio: Δω_opt / Δω_SQL.
        a_x_opt: Optimal a_x drive parameter.
        a_y_opt: Optimal a_y drive parameter.
        a_z_opt: Optimal a_z drive parameter.
        a_zz_opt: Optimal a_zz drive parameter.
        psi_opt: Optimal ψ.
        expectation_Jz: ⟨J_z^S⟩ at optimum.
        variance_Jz: Var(J_z^S) at optimum.
        expectation_M: ⟨M⟩ at optimum.
        variance_M: Var(M) at optimum.
        d_expectation: ∂⟨M⟩/∂ω at optimum.
        t_hold: Holding time used.
        success: NM convergence flag.
        nfev: Number of function evaluations.
    """

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    psi_opt: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    t_hold: float = T_HOLD
    success: bool = False
    nfev: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Serialize to a single-row DataFrame (self-describing)."""
        return pd.DataFrame(
            [
                {
                    "N": self.N,
                    "omega": self.omega,
                    "delta_omega_opt": self.delta_omega_opt,
                    "sql": self.sql,
                    "ratio": self.ratio,
                    "a_x_opt": self.a_x_opt,
                    "a_y_opt": self.a_y_opt,
                    "a_z_opt": self.a_z_opt,
                    "a_zz_opt": self.a_zz_opt,
                    "psi_opt": self.psi_opt,
                    "expectation_Jz": self.expectation_Jz,
                    "variance_Jz": self.variance_Jz,
                    "expectation_M": self.expectation_M,
                    "variance_M": self.variance_M,
                    "d_expectation": self.d_expectation,
                    "t_hold": self.t_hold,
                    "success": int(self.success),
                    "nfev": self.nfev,
                }
            ]
        )

    def save_parquet(self, path: str) -> None:
        """Write to Parquet."""
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str) -> JointNScalingResult:
        """Load from Parquet with fail-fast column validation.

        Raises:
            ValueError: If any required column is missing.
        """
        required = {
            "N",
            "omega",
            "delta_omega_opt",
            "sql",
            "ratio",
            "a_x_opt",
            "a_y_opt",
            "a_z_opt",
            "a_zz_opt",
            "psi_opt",
            "expectation_Jz",
            "variance_Jz",
            "expectation_M",
            "variance_M",
            "d_expectation",
            "t_hold",
            "success",
            "nfev",
        }
        df = pd.read_parquet(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet file {path} missing columns: {sorted(missing)}. "
                "Re-run the simulation that generated this file."
            )
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            psi_opt=float(row["psi_opt"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            expectation_M=float(row["expectation_M"]),
            variance_M=float(row["variance_M"]),
            d_expectation=float(row["d_expectation"]),
            t_hold=float(row["t_hold"]),
            success=bool(row["success"]),
            nfev=int(row["nfev"]),
        )


@dataclass
class JointNScalingScanResult:
    """Collection of JointNScalingResult for multiple (N, ω) combinations.

    Attributes:
        results: List of individual scaling results.
    """

    results: list[JointNScalingResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten all results into a self-describing DataFrame."""
        return pd.DataFrame(
            [
                {
                    "N": r.N,
                    "omega": r.omega,
                    "delta_omega_opt": r.delta_omega_opt,
                    "sql": r.sql,
                    "ratio": r.ratio,
                    "a_x_opt": r.a_x_opt,
                    "a_y_opt": r.a_y_opt,
                    "a_z_opt": r.a_z_opt,
                    "a_zz_opt": r.a_zz_opt,
                    "psi_opt": r.psi_opt,
                    "expectation_Jz": r.expectation_Jz,
                    "variance_Jz": r.variance_Jz,
                    "expectation_M": r.expectation_M,
                    "variance_M": r.variance_M,
                    "d_expectation": r.d_expectation,
                    "t_hold": r.t_hold,
                    "success": int(r.success),
                    "nfev": r.nfev,
                }
                for r in self.results
            ]
        )

    def save_parquet(self, path: str) -> None:
        """Write to Parquet."""
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str) -> JointNScalingScanResult:
        """Load from Parquet with fail-fast column validation.

        Raises:
            ValueError: If any required column is missing.
        """
        required = {
            "N",
            "omega",
            "delta_omega_opt",
            "sql",
            "ratio",
            "a_x_opt",
            "a_y_opt",
            "a_z_opt",
            "a_zz_opt",
            "psi_opt",
            "expectation_Jz",
            "variance_Jz",
            "expectation_M",
            "variance_M",
            "d_expectation",
            "t_hold",
            "success",
            "nfev",
        }
        df = pd.read_parquet(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet file {path} missing columns: {sorted(missing)}. "
                "Re-run the simulation that generated this file."
            )
        results = []
        for _, row in df.iterrows():
            results.append(
                JointNScalingResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    psi_opt=float(row["psi_opt"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    expectation_M=float(row["expectation_M"]),
                    variance_M=float(row["variance_M"]),
                    d_expectation=float(row["d_expectation"]),
                    t_hold=float(row["t_hold"]),
                    success=bool(row["success"]),
                    nfev=int(row["nfev"]),
                )
            )
        return cls(results=results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> JointNScalingResult:
        return self.results[idx]

    def to_numpy_arrays(self) -> dict[str, np.ndarray]:
        """Extract all fields as numpy arrays for plotting."""
        d: dict[str, list] = {
            "N": [],
            "omega": [],
            "delta_omega_opt": [],
            "sql": [],
            "ratio": [],
            "a_x_opt": [],
            "a_y_opt": [],
            "a_z_opt": [],
            "a_zz_opt": [],
            "psi_opt": [],
            "expectation_Jz": [],
            "variance_Jz": [],
            "expectation_M": [],
            "variance_M": [],
            "d_expectation": [],
            "t_hold": [],
            "success": [],
            "nfev": [],
        }
        for r in self.results:
            for key, val_list in d.items():
                val_list.append(getattr(r, key))
        return {k: np.array(v) for k, v in d.items()}


# ============================================================================
# Data Generation
# ============================================================================


def generate_joint_scaling_data(
    omega_value: float = 0.2,
    N_range: list[int] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> JointNScalingScanResult:
    """Scan N values, running Nelder-Mead at each N for joint protocol only.

    Args:
        omega_value: True ω for all runs.
        N_range: List of N values (default [1, 2, 3, 4, 5, 8, 10, 15, 20]).
        seed: Base RNG seed (incremented per N).
        maxiter: Max Nelder-Mead iterations per run.
        progress_callback: Optional fn(current, total) for progress.

    Returns:
        JointNScalingScanResult with individual results.
    """
    if N_range is None:
        N_range = [1, 2, 3, 4, 5, 8, 10, 15, 20]

    results: list[JointNScalingResult] = []

    for idx, N in enumerate(N_range):
        if progress_callback:
            progress_callback(idx + 1, len(N_range))

        ops = build_n_particle_operators(N)
        psi0 = n_particle_initial_state(N)
        n_seed = seed + N

        joint_res = run_single_joint_n_omega(
            N=N,
            omega=omega_value,
            ops=ops,
            psi0=psi0,
            seed=n_seed,
            maxiter=maxiter,
        )

        results.append(joint_res)

    return JointNScalingScanResult(results=results)


def generate_joint_and_sonly_scaling_data(
    omega_value: float = 0.2,
    N_range: list[int] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[JointNScalingScanResult, JointNScalingScanResult]:
    """Scan N values, running NM for both joint and S-only protocols.

    Returns:
        (joint_result, sonly_result) — scaling results for both protocols.
    """
    if N_range is None:
        N_range = [1, 2, 3, 4, 5, 8, 10, 15, 20]

    joint_list: list[JointNScalingResult] = []
    sonly_list: list[JointNScalingResult] = []

    for idx, N in enumerate(N_range):
        if progress_callback:
            progress_callback(idx + 1, len(N_range))

        ops = build_n_particle_operators(N)
        psi0 = n_particle_initial_state(N)
        n_seed = seed + N

        joint_res = run_single_joint_n_omega(
            N=N,
            omega=omega_value,
            ops=ops,
            psi0=psi0,
            seed=n_seed,
            maxiter=maxiter,
        )
        sonly_res = run_single_sonly_n_omega(
            N=N,
            omega=omega_value,
            ops=ops,
            psi0=psi0,
            seed=n_seed + 1000,
            maxiter=maxiter,
        )

        joint_list.append(joint_res)
        sonly_list.append(sonly_res)

    return (
        JointNScalingScanResult(results=joint_list),
        JointNScalingScanResult(results=sonly_list),
    )


# ============================================================================
# Plotting
# ============================================================================


def _arrays_from_scan(scan: JointNScalingScanResult) -> dict[str, np.ndarray]:
    """Extract numpy arrays from a JointNScalingScanResult for plotting."""
    return scan.to_numpy_arrays()


def plot_sensitivity_vs_n(
    joint_result: JointNScalingScanResult,
    sonly_result: JointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Ratio Δω/Δω_SQL vs N for joint and S-only protocols.

    Args:
        joint_result: Joint protocol scaling results.
        sonly_result: S-only protocol scaling results.
        save_path: Optional path to save SVG.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    j = _arrays_from_scan(joint_result)
    s = _arrays_from_scan(sonly_result)

    ratio_joint = j["delta_omega_opt"] / j["sql"]
    ratio_sonly = s["delta_omega_opt"] / s["sql"]

    ax.plot(
        j["N"],
        ratio_joint,
        "o-",
        color="C0",
        label="Joint M(ψ) optimised",
        markersize=6,
    )
    ax.plot(
        s["N"], ratio_sonly, "s--", color="C1", label="S-only J$_z^S$", markersize=6
    )
    ax.axhline(1.0, color="gray", linestyle=":", label="SQL (Δω/Δω_SQL = 1)")

    ax.set_xlabel("N (number of system particles)", fontsize=13)
    ax.set_ylabel("Δω / Δω_SQL", fontsize=13)
    ax.set_title(f"Phase Sensitivity vs N (ω = {j['omega'][0]})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_optimal_params(
    result: JointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot optimal drive parameters and ψ vs N.

    Args:
        result: Joint protocol scaling results.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    arr = _arrays_from_scan(result)
    Ns = arr["N"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    params = [
        (arr["a_x_opt"], "$a_x$"),
        (arr["a_y_opt"], "$a_y$"),
        (arr["a_z_opt"], "$a_z$"),
        (arr["a_zz_opt"], "$a_{zz}$"),
        (arr["psi_opt"], "$\\psi$"),
    ]
    for ax, (vals, label) in zip(axes, params, strict=False):
        ax.plot(Ns, vals, "o-", markersize=5)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("N", fontsize=13)
    fig.suptitle(f"Optimal Parameters vs N (ω = {arr['omega'][0]})", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_psi_vs_azz_heatmap(
    result: JointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of optimal ψ vs a_{zz} across N (coloured by N).

    Args:
        result: Joint protocol scaling results.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    arr = _arrays_from_scan(result)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        arr["a_zz_opt"],
        arr["psi_opt"],
        c=arr["N"],
        cmap="viridis",
        s=60,
        edgecolors="k",
    )
    fig.colorbar(sc, ax=ax, label="N")
    ax.set_xlabel("$a_{zz}$ (optimal)", fontsize=13)
    ax.set_ylabel("$\\psi$ (optimal)", fontsize=13)
    ax.set_title("Optimal (ψ, a$_{zz}$) across N", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_joint_vs_sonly_comparison(
    joint_result: JointNScalingScanResult,
    sonly_result: JointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Absolute Δω comparison: joint vs S-only with SQL band.

    Args:
        joint_result: Joint protocol scaling results.
        sonly_result: S-only protocol scaling results.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    j = _arrays_from_scan(joint_result)
    s = _arrays_from_scan(sonly_result)
    Ns = j["N"]
    sql_vals = j["sql"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        Ns,
        sql_vals / np.sqrt(Ns),  # Heisenberg bound ~ SQL/√N
        sql_vals,
        alpha=0.1,
        color="gray",
        label="SQL to HL band",
    )
    ax.plot(
        Ns, j["delta_omega_opt"], "o-", color="C0", label="Joint M(ψ)", markersize=6
    )
    ax.plot(
        Ns,
        s["delta_omega_opt"],
        "s--",
        color="C1",
        label="S-only J$_z^S$",
        markersize=6,
    )
    ax.plot(Ns, sql_vals, ":", color="gray", label="SQL ($1/\\sqrt{N} T_H$)")

    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Δω", fontsize=13)
    ax.set_title(f"Absolute Phase Sensitivity (ω = {j['omega'][0]})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """CLI entry point for generating joint measurement scaling data."""

    parser = argparse.ArgumentParser(
        description="Generate joint measurement scaling data for ω-modulated drive."
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.2,
        help="True phase rate ω (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=NM_MAXITER,
        help=f"Nelder-Mead max iterations (default: {NM_MAXITER}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="20260613-joint-scaling.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 8, 10, 15, 20],
        help="N values to scan.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=".",
        help="Directory for output SVG plots.",
    )

    args = parser.parse_args()

    print(f"Generating joint scaling data: ω={args.omega}, N={args.N}")
    print(f"Seed={args.seed}, maxiter={args.maxiter}")

    joint_result, sonly_result = generate_joint_and_sonly_scaling_data(
        omega_value=args.omega,
        N_range=args.N,
        seed=args.seed,
        maxiter=args.maxiter,
    )

    joint_result.save_parquet(args.output)
    print(f"Saved joint scaling data to {args.output}")

    sonly_output = args.output.replace(".parquet", "-sonly.parquet")
    sonly_result.save_parquet(sonly_output)
    print(f"Saved S-only scaling data to {sonly_output}")

    # Generate plots
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_sensitivity_vs_n(
        joint_result,
        sonly_result,
        save_path=str(plot_dir / "sensitivity_vs_n.svg"),
    )
    print("Saved sensitivity_vs_n.svg")

    plot_optimal_params(
        joint_result,
        save_path=str(plot_dir / "optimal_params.svg"),
    )
    print("Saved optimal_params.svg")

    plot_psi_vs_azz_heatmap(
        joint_result,
        save_path=str(plot_dir / "psi_vs_azz.svg"),
    )
    print("Saved psi_vs_azz.svg")

    plot_joint_vs_sonly_comparison(
        joint_result,
        sonly_result,
        save_path=str(plot_dir / "joint_vs_sonly.svg"),
    )
    print("Saved joint_vs_sonly.svg")


if __name__ == "__main__":
    main()
