"""
Local module for the 2026-05-19 Ancilla-Drive Phase-Modulated Metrology report.

Contains all code exclusive to this report:
- Core physics simulation (θ-modulated Hamiltonian operators, circuit evolution,
  sensitivity computation)
- 4D random search, Nelder–Mead refinement, and θ-scan orchestration
- Exclusive plot functions (combined sensitivity, NM expectation/variance,
  cross-experiment comparison, fraction below SQL)
- Data and figure generation pipeline (``generate_phase_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260519/local.py --force
    uv run python reports/20260519/local.py --only phase-decoupled-baseline

This module is **not** importable as ``reports.20260519.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.  See ``tests/test_ancilla_drive_phase_modulated.py``
for an example.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as _mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# Shared primitives (used by all reports)
from src.analysis.ancilla_drive_metrology import (  # noqa: E402
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
    build_iszz_interaction,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (  # noqa: E402
    build_two_qubit_operators,
    compute_expectation_and_variance,
)

sns.set_theme(style="whitegrid")

I_4 = np.eye(4, dtype=complex)

# ============================================================================
# Physical constants  (moved from ancilla_drive_phase_modulated.py)
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_H: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_T_H  # Δθ_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients

# ============================================================================
# Operator Construction  (from ancilla_drive_phase_modulated.py)
# ============================================================================


def build_phase_modulated_drive_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the θ-modulated ancilla drive Hamiltonian.

    H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    The critical difference from the fixed-drive protocol is the leading θ
    factor: the ancilla drive scales with the unknown phase, creating a
    parametric amplification effect in ∂⟨J_z^S⟩/∂θ.

    Args:
        theta: Unknown phase rate parameter (scales the whole drive).
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the θ-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = theta * H  # θ-modulation: entire drive scales with the unknown phase
    return 0.5 * (H + H.conj().T)


def build_phase_modulated_hold_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with θ-modulated ancilla drive.

    H = θ J_z^S + H_A + H_int
      = θ J_z^S + θ (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A
      = θ [J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A] + a_zz J_z^S ⊗ J_z^A

    The θ factor on the drive terms means ∂H/∂θ = J_z^S + a_x J_x^A + a_y J_y^A
    + a_z J_z^A, which includes ancilla operators. This extra contribution to
    the derivative is the key mechanism for potential SQL violation.

    Args:
        theta: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = theta * ops["Jz_S"]
    H += build_phase_modulated_drive_hamiltonian(theta, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def phase_modulated_hold_unitary(
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the θ-modulated ancilla protocol.

    U_hold(T_H) = exp(-i T_H H)
    where H = θ J_z^S + θ(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        T_H: Holding-time strength.
        theta: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_phase_modulated_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * T_H * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Phase-modulated hold unitary not unitary for T_H={T_H}, θ={theta}"
    )
    return U


def evolve_phase_modulated_circuit(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full θ-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(T_H) · U_BS_S · |ψ₀⟩

    The hold unitary uses the θ-modulated H_A = θ (a_x J_x^A + ...).

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        T_H: Holding-time strength.
        theta: Phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = phase_modulated_hold_unitary(T_H, theta, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_phase_modulated_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δθ.

    Δθ = sqrt(Var(O)) / |∂⟨O⟩/∂θ|

    where O is the measurement operator (default: J_z^S).

    IMPORTANT: Because θ now appears in both H_S (= θ J_z^S) and H_A
    (= θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)), the central finite-difference
    step captures the FULL θ-dependence (both channels) automatically —
    the circuit is re-evaluated at θ ± δ, and both H_S and H_A change.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        T_H: Holding-time strength.
        theta_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δθ (positive float). Returns inf if derivative is zero
        (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    psi = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂θ
    psi_plus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ meas_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ meas_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_phase_modulated_decoupled_baseline(
    T_H: float = DEFAULT_T_H,
    theta_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δθ.

    At (a_x = a_y = a_z = a_zz = 0), the θ-modulated ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δθ = 1/T_H. The θ factor in H_A is irrelevant when all a_k = 0.

    Args:
        T_H: Holding-time strength.
        theta_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    dtheta = compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        T_H,
        theta_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        T_H_value=T_H,
        delta_theta=dtheta,
        sql=1.0 / T_H,
    )


# ============================================================================
# 2D Slice Scan
# ============================================================================


def phase_modulated_2d_slice(
    theta: float,
    drive_range: tuple[float, float] = DRIVE_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz) with θ-modulated ancilla drive.

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).

    Args:
        theta: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax' or 'ay'.
        T_H: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).

    Returns:
        Drive2DSliceResult with the sensitivity grid.
    """
    if slice_type not in ("ax", "ay"):
        raise ValueError(f"slice_type must be 'ax' or 'ay', got {slice_type}")

    ops = build_two_qubit_operators()
    drive_vals = np.linspace(drive_range[0], drive_range[1], n_drive)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)
    grid = np.full((n_drive, n_azz), np.inf, dtype=float)

    for i, d_val in enumerate(drive_vals):
        for j, a_val in enumerate(azz_vals):
            if slice_type == "ax":
                ax, ay, az = d_val, 0.0, 0.0
            else:
                ax, ay, az = 0.0, d_val, 0.0

            dtheta = compute_phase_modulated_sensitivity(
                DEFAULT_PSI0,
                T_BS,
                T_H,
                theta,
                ax,
                ay,
                az,
                a_val,
                ops,
            )
            grid[i, j] = dtheta

    return Drive2DSliceResult(
        drive_values=drive_vals,
        azz_values=azz_vals,
        delta_theta_grid=grid,
        theta_value=theta,
        slice_type=slice_type,
        sql=1.0 / T_H,
    )


# ============================================================================
# 4D Random Search
# ============================================================================


def phase_modulated_random_search(
    theta: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Uses the θ-modulated ancilla drive H_A = θ (a_x J_x^A + ...).

    Args:
        theta: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])

        dtheta = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            T_BS,
            T_H,
            theta,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = dtheta

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    return DriveRandomSearchResult(
        samples=samples,
        delta_theta_values=deltas,
        best_params=best_params,
        best_delta_theta=float(deltas[best_idx]),
        theta_value=theta,
        sql=1.0 / T_H,
        T_H=T_H,
    )


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


def phase_modulated_sensitivity_objective(
    params: np.ndarray,
    theta_true: float,
    ops: dict[str, np.ndarray],
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δθ in the θ-modulated protocol.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed T_H.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        theta_true: True phase rate.
        ops: Two-qubit operators.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δθ (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])

    # Bound enforcement
    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        T_BS,
        T_H,
        theta_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_phase_modulated_nelder_mead(
    theta_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder--Mead optimisation for the θ-modulated ancilla protocol.

    Args:
        theta_true: True phase rate parameter.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        bounds: (min, max) for all four parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
    """
    ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return phase_modulated_sensitivity_objective(
            p,
            theta_true,
            ops,
            T_H=T_H,
            T_BS=T_BS,
            bounds=bounds,
        )

    history: list[float] = []

    def callback(intermediate_result: Any) -> None:
        if track_history:
            val = objective(intermediate_result.x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_phase_modulated_circuit(
        DEFAULT_PSI0,
        T_BS,
        T_H,
        theta_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_theta_opt=float(result.fun),
        params_opt=opt_params,
        theta_true=theta_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# θ Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_phase_modulated_theta_scan(
    theta_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> DriveThetaScanResult:
    """Scan over θ values with 4D random search and Nelder--Mead refinement.

    For each θ:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        theta_values: θ values to scan.
        n_random: Number of random search points per θ.
        n_nm_refine: Number of Nelder--Mead refinements per θ.
        seed: Base random seed (incremented per θ).
        maxiter: Maximum Nelder--Mead iterations.
        bounds: (min, max) for all parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveThetaScanResult with optimal parameters and sensitivities.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    base_seed = seed if seed is not None else 42

    best_params_list: list[tuple[float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []
    all_results_dict: dict[float, list[DriveNelderMeadResult]] = {}

    for theta in theta_arr:
        # Stage 1: Random search
        rs_result = phase_modulated_random_search(
            theta,
            n_samples=n_random,
            bounds=bounds,
            T_H=T_H,
            T_BS=T_BS,
            seed=base_seed + int(theta * 1000),
        )

        # Sort random-search results by Δθ, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_theta_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[DriveNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            x0 = rs_result.samples[idx].copy()
            nm = run_phase_modulated_nelder_mead(
                theta_true=theta,
                x0=x0,
                seed=base_seed + int(theta * 1000) + 10000 + rank,
                maxiter=maxiter,
                bounds=bounds,
                T_H=T_H,
                T_BS=T_BS,
                track_history=False,
            )
            nm_results.append(nm)

        # Sort Nelder--Mead results by Δθ
        nm_results.sort(key=lambda r: r.delta_theta_opt)
        best_nm = nm_results[0]

        best_params_list.append(
            (
                float(best_nm.params_opt[0]),
                float(best_nm.params_opt[1]),
                float(best_nm.params_opt[2]),
                float(best_nm.params_opt[3]),
            )
        )
        best_deltas.append(best_nm.delta_theta_opt)
        sql_vals.append(1.0 / T_H)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)
        all_results_dict[float(theta)] = nm_results

    return DriveThetaScanResult(
        theta_values=theta_arr,
        best_params_per_theta=best_params_list,
        best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
        variance_Jz_per_theta=np.array(var_vals, dtype=float),
        all_results=all_results_dict,
    )


# ============================================================================
# Exclusive Plot Functions
# (moved from src/visualization/ancilla_drive_plots.py)
# ============================================================================


def plot_drive_combined_sensitivity(
    theta_values: np.ndarray,
    best_ax_slice: np.ndarray,
    best_ay_slice: np.ndarray,
    best_random: np.ndarray,
    best_nm: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot comparing Δθ from 2D slices, 4D random search, NM refinement, and SQL.

    Args:
        theta_values: Array of θ values.
        best_ax_slice: Best Δθ from (a_x, a_zz) slice at each θ.
        best_ay_slice: Best Δθ from (a_y, a_zz) slice at each θ.
        best_random: Best Δθ from 4D random search at each θ.
        best_nm: Best Δθ from Nelder–Mead refinement at each θ.
        sql_values: SQL reference at each θ (constant).
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference line
    sql = float(sql_values[0]) if len(sql_values) > 0 else 0.1
    ax.axhline(
        y=sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql:.4f}",
    )

    methods: list[tuple[np.ndarray, str, str, str]] = [
        (best_ax_slice, "o-", "C0", r"2D slice $(a_x, a_{zz})$"),
        (best_ay_slice, "s-", "C1", r"2D slice $(a_y, a_{zz})$"),
        (best_random, "^-", "C2", "4D random search"),
        (best_nm, "D-", "C3", "4D Nelder–Mead"),
    ]

    for data, fmt, colour, label in methods:
        valid = np.isfinite(data)
        if np.any(valid):
            ax.plot(
                theta_values[valid],
                data[valid],
                fmt,
                color=colour,
                label=label,
                markersize=6,
                linewidth=1.5,
                markerfacecolor=colour,
            )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(
        "Sensitivity vs $\\theta$: 2D slices, 4D random search, Nelder–Mead refinement"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_nm_expectation_variance(
    theta_values: np.ndarray,
    expectation_Jz: np.ndarray,
    variance_Jz: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Side-by-side plot of ⟨J_z^S⟩ and Var(J_z^S) at the NM optimum vs θ.

    Args:
        theta_values: Array of θ values.
        expectation_Jz: ⟨J_z^S⟩ at NM optimum for each θ.
        variance_Jz: Var(J_z^S) at NM optimum for each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: expectation
    valid_exp = np.isfinite(expectation_Jz)
    if np.any(valid_exp):
        ax1.plot(
            theta_values[valid_exp],
            expectation_Jz[valid_exp],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.5,
        )
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\langle J_z^S \rangle$")
    ax1.set_title(r"Expectation $\langle J_z^S\rangle$ at NM optimum")

    # Right panel: variance
    valid_var = np.isfinite(variance_Jz)
    if np.any(valid_var):
        ax2.plot(
            theta_values[valid_var],
            variance_Jz[valid_var],
            "s-",
            color="C1",
            markersize=7,
            linewidth=1.5,
        )
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax2.set_title(r"Variance $\mathrm{Var}(J_z^S)$ at NM optimum")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_cross_experiment_comparison(
    theta_values: np.ndarray,
    best_delta_19: np.ndarray,
    best_delta_18: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Compare Δθ from the fixed-drive (2026-05-18) and modulated-drive
    (2026-05-19) experiments in a 2×1 vertically stacked figure.

    Upper panel: Overlaid line plots of Δθ vs θ for both experiments,
    with the SQL shown as a dashed reference line.

    Lower panel: Ratio Δθ_19 / Δθ_18 vs θ. A horizontal line at y=1
    separates regimes where the fixed drive (above 1) or modulated drive
    (below 1) performs better.

    Args:
        theta_values: Common θ grid (50 points from the modulated-drive scan).
        best_delta_19: Δθ from the modulated-drive scan (2026-05-19).
        best_delta_18: Δθ from the fixed-drive scan (2026-05-18),
            interpolated to the same θ grid.
        sql_values: SQL reference values (constant, 0.1) at each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δθ vs θ ──────────────────────────────────────────
    sql_ref = float(sql_values[0]) if len(sql_values) > 0 else 0.1

    ax1.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=rf"SQL = {sql_ref:.4f}",
    )

    ax1.plot(
        theta_values,
        best_delta_18,
        marker="s",
        linestyle="-",
        color="C0",
        markersize=5,
        linewidth=1.8,
        label=r"Fixed drive (2026-05-18)",
    )
    ax1.plot(
        theta_values,
        best_delta_19,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=5,
        linewidth=1.8,
        label=r"Modulated drive (2026-05-19)",
    )

    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title("Cross-experiment comparison: fixed vs modulated drive")
    ax1.legend(fontsize=9)

    # ── Lower panel: ratio Δθ_19 / Δθ_18 ──────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(best_delta_18) & (best_delta_18 > 0),
            best_delta_19 / best_delta_18,
            np.nan,
        )

    ax2.plot(
        theta_values,
        ratio,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=4,
        linewidth=1.5,
    )
    ax2.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="y = 1"
    )

    # Annotate the minimum ratio
    valid = np.isfinite(ratio)
    if np.any(valid):
        min_idx = np.argmin(ratio[valid])
        min_ratio = float(ratio[valid][min_idx])
        min_theta = float(theta_values[valid][min_idx])
        ax2.annotate(
            f"Best = {min_ratio:.3f}$\\times$ at $\\theta$={min_theta:.1f}",
            xy=(min_theta, min_ratio),
            xytext=(min_theta + 0.6, min_ratio + 0.15),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\Delta\theta_{19} \;/\; \Delta\theta_{18}$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_fraction_below_sql(
    theta_values: np.ndarray,
    fractions_2d_ax: np.ndarray,
    fractions_2d_ay: np.ndarray,
    fractions_random: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of the fraction of parameter space below SQL as a function of θ.

    Args:
        theta_values: Array of θ values.
        fractions_2d_ax: Fraction below SQL from (a_x, a_zz) slices at each θ.
        fractions_2d_ay: Fraction below SQL from (a_y, a_zz) slices at each θ.
        fractions_random: Fraction below SQL from 4D random search at each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        theta_values,
        fractions_2d_ax,
        "o-",
        color="C0",
        label=r"2D slice $(a_x, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        theta_values,
        fractions_2d_ay,
        "s-",
        color="C1",
        label=r"2D slice $(a_y, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        theta_values,
        fractions_random,
        "^-",
        color="C2",
        label="4D random search",
        markersize=6,
        linewidth=1.5,
    )

    # Reference lines at y=0 and y=1
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Fraction below SQL")
    ax.set_title("Robustness of SQL violation: fraction of parameter space below SQL")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# (moved from src/visualization/report_figures.py)
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
PHASE_DATE = "20260519"
PHASE_THETA_VALS = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
PHASE_N_GRID = 201


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / PHASE_DATE / "raw_data" / f"{PHASE_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / PHASE_DATE / "figures" / f"{PHASE_DATE}-{name}.svg"


# ── Parallel dispatch helper ──────────────────────────────────────────────


def _parallel_map(
    worker_fn,
    items,
    desc: str = "Processing",
    max_workers: int | None = None,
) -> None:
    """Run *worker_fn(item)* for each *item* in parallel via process pool.

    Each worker is a top-level function (or ``functools.partial`` wrapping
    one) that performs its own file I/O.  Results are implicitly persisted to
    disk by the worker — this function only waits for completion and re-raises
    the first exception encountered.

    Args:
        worker_fn: Callable taking a single item argument.
        items: Iterable of items (typically θ values).
        desc: Short description for progress logging.
        max_workers: Number of subprocess workers (default: CPU count).
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)
    item_list = list(items)
    print(f"  [parallel] {desc}: {len(item_list)} items, {max_workers} workers")

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_item = {executor.submit(worker_fn, item): item for item in item_list}
        for future in concurrent.futures.as_completed(fut_to_item):
            item = fut_to_item[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item}: {exc}")
                raise


# ── Generator functions ───────────────────────────────────────────────────


def generate_phase_decoupled_baseline(force: bool = False) -> None:
    """Phase-modulated decoupled baseline verification."""
    csv_p = _parquet_path("phase-decoupled-baseline")
    fig_p = _fig_path("phase-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing phase-modulated decoupled baseline...")
        result = compute_phase_modulated_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Use the shared plot function from the main codebase
    from src.visualization.ancilla_drive_plots import plot_drive_decoupled_baseline

    plot_drive_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def _run_phase_2d_slice(
    theta: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a phase-modulated 2D slice scan for a single θ value."""
    tag = f"phase-2d-slice-{slice_type}-azz-theta{theta}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing phase ({slice_type}, a_zz) slice at θ={theta}...")
        result = phase_modulated_2d_slice(
            theta=theta,
            slice_type=slice_type,
            n_drive=PHASE_N_GRID,
            n_azz=PHASE_N_GRID,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_2d_slice_heatmap

    plot_drive_2d_slice_heatmap(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_phase_2d_slice_ax_azz(force: bool = False) -> None:
    """Phase-modulated 2D slice scans over (a_x, a_zz) at all θ values."""
    n = len(PHASE_THETA_VALS)
    print(f"[run]  (a_x, a_zz) phase slice at {n} θ values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ax", force=force)
    _parallel_map(worker, PHASE_THETA_VALS, desc="(a_x, a_zz) slices")


def generate_phase_2d_slice_ay_azz(force: bool = False) -> None:
    """Phase-modulated 2D slice scans over (a_y, a_zz) at all θ values."""
    n = len(PHASE_THETA_VALS)
    print(f"[run]  (a_y, a_zz) phase slice at {n} θ values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ay", force=force)
    _parallel_map(worker, PHASE_THETA_VALS, desc="(a_y, a_zz) slices")


def _run_phase_random_search(theta: float, force: bool) -> None:
    """Run a phase-modulated 4D random search for a single θ value."""
    tag = f"phase-random-search-theta{theta}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveRandomSearchResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Running phase 4D random search at θ={theta} (500 samples)...")
        result = phase_modulated_random_search(
            theta=theta,
            n_samples=500,
            seed=42,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_random_search_histogram

    plot_drive_random_search_histogram(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_phase_random_search(force: bool = False) -> None:
    """Phase-modulated 4D random search at all θ values (parallel)."""
    n = len(PHASE_THETA_VALS)
    print(f"[run]  4D phase random search at {n} θ values (parallel)")
    worker = partial(_run_phase_random_search, force=force)
    _parallel_map(worker, PHASE_THETA_VALS, desc="random search")


def _run_phase_theta_scan_single(theta: float) -> dict[str, float | np.ndarray]:
    """Run random search + NM refinement for a single θ value.

    Returns a dict with per-θ results that can be aggregated into a
    ``DriveThetaScanResult``.
    """
    base_seed: int = 42
    n_random: int = 500
    n_nm_refine: int = 50
    maxiter_val: int = 5000
    bounds: tuple[float, float] = (-5.0, 5.0)

    # Stage 1: Random search
    rs_result = phase_modulated_random_search(
        theta,
        n_samples=n_random,
        bounds=bounds,
        seed=base_seed + int(theta * 1000),
    )

    # Sort by Δθ, take top n_nm_refine
    sorted_indices = np.argsort(rs_result.delta_theta_values)
    top_indices = sorted_indices[:n_nm_refine]

    # Stage 2: Nelder--Mead refinement from each top point
    nm_results: list[DriveNelderMeadResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        nm = run_phase_modulated_nelder_mead(
            theta_true=theta,
            x0=x0,
            seed=base_seed + int(theta * 1000) + 10000 + rank,
            maxiter=maxiter_val,
            bounds=bounds,
            track_history=False,
        )
        nm_results.append(nm)

    nm_results.sort(key=lambda r: r.delta_theta_opt)
    best_nm = nm_results[0]

    return {
        "theta": theta,
        "best_delta_theta": best_nm.delta_theta_opt,
        "a_x": float(best_nm.params_opt[0]),
        "a_y": float(best_nm.params_opt[1]),
        "a_z": float(best_nm.params_opt[2]),
        "a_zz": float(best_nm.params_opt[3]),
        "expectation_Jz": best_nm.expectation_Jz,
        "variance_Jz": best_nm.variance_Jz,
    }


def generate_phase_theta_scan(force: bool = False) -> None:
    """Phase-modulated θ-scan with Nelder-Mead refinement (parallel)."""
    csv_p = _parquet_path("phase-theta-scan")
    fig_p = _fig_path("phase-theta-scan")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveThetaScanResult.from_parquet(csv_p)
    else:
        n = len(PHASE_THETA_VALS)
        print(f"[run]  Computing phase θ-scan for {n} θ values (parallel)...")

        max_workers = min(32, os.cpu_count() or 1)
        print(f"  [parallel] Using {max_workers} workers for θ-scan")

        per_theta_results: list[dict] = []
        mp_ctx = _mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
        ) as executor:
            fut_to_theta = {
                executor.submit(_run_phase_theta_scan_single, theta): theta
                for theta in PHASE_THETA_VALS
            }
            for future in concurrent.futures.as_completed(fut_to_theta):
                theta = fut_to_theta[future]
                try:
                    per_theta_results.append(future.result())
                    print(f"  [done] θ={theta}")
                except Exception as exc:
                    print(f"  [ERROR] θ={theta}: {exc}")
                    raise

        # Sort by θ and construct the full result
        per_theta_results.sort(key=lambda r: float(r["theta"]))

        theta_arr = np.array([r["theta"] for r in per_theta_results], dtype=float)
        best_deltas = [float(r["best_delta_theta"]) for r in per_theta_results]
        best_params = [
            (
                float(r["a_x"]),
                float(r["a_y"]),
                float(r["a_z"]),
                float(r["a_zz"]),
            )
            for r in per_theta_results
        ]
        exp_vals = [float(r["expectation_Jz"]) for r in per_theta_results]
        var_vals = [float(r["variance_Jz"]) for r in per_theta_results]
        sql_vals = [1.0 / 10.0] * len(theta_arr)

        result = DriveThetaScanResult(
            theta_values=theta_arr,
            best_params_per_theta=best_params,
            best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
            sql_values=np.array(sql_vals, dtype=float),
            expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
            variance_Jz_per_theta=np.array(var_vals, dtype=float),
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_theta_scan

    plot_drive_theta_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_optimal_params(force: bool = False) -> None:
    """Phase-modulated optimal parameter evolution vs θ."""
    csv_p = _parquet_path("phase-theta-scan")
    fig_p = _fig_path("phase-optimal-params")

    result = DriveThetaScanResult.from_parquet(csv_p)
    from src.visualization.ancilla_drive_plots import plot_drive_optimal_params

    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_combined_sensitivity(force: bool = False) -> None:
    """Combined sensitivity plot + NM expectation/variance plot.

    Reads existing per-θ Parquets (2D slices, random search) and the theta-scan
    result to produce two figures:
        - phase-combined-sensitivity.svg : Δθ vs θ for all methods + SQL
        - phase-nm-expectation-variance.svg : ⟨J_z^S⟩ and Var(J_z^S) at NM optimum

    This must be run AFTER the data-generation steps
    (phase-2d-slice-*, phase-random-search, phase-theta-scan).
    """
    fig_p1 = _fig_path("phase-combined-sensitivity")
    fig_p2 = _fig_path("phase-nm-expectation-variance")

    # Load NM result
    theta_scan_pq = _parquet_path("phase-theta-scan")
    if not theta_scan_pq.exists():
        print(
            "[skip] phase-theta-scan.parquet does not exist; run 'phase-theta-scan' first"
        )
        return
    nm_result = DriveThetaScanResult.from_parquet(theta_scan_pq)

    theta_vals = np.array(PHASE_THETA_VALS, dtype=float)
    n_theta = len(theta_vals)

    best_nm = np.full(n_theta, np.nan)
    exp_vals = np.full(n_theta, np.nan)
    var_vals = np.full(n_theta, np.nan)

    nm_theta = nm_result.theta_values
    if len(nm_theta) >= n_theta:
        for i in range(n_theta):
            best_nm[i] = float(nm_result.best_delta_theta_per_theta[i])
            exp_vals[i] = float(nm_result.expectation_Jz_per_theta[i])
            var_vals[i] = float(nm_result.variance_Jz_per_theta[i])

    # Collect per-θ minima from 2D slice and random search Parquets
    best_ax = np.full(n_theta, np.nan)
    best_ay = np.full(n_theta, np.nan)
    best_rs = np.full(n_theta, np.nan)

    def _safe_grid_min(grid: np.ndarray) -> float:
        finite_vals = grid[np.isfinite(grid)]
        if finite_vals.size == 0:
            return np.nan
        return float(np.min(finite_vals))

    for i, theta in enumerate(theta_vals):
        for slice_type, best_arr in [("ax", best_ax), ("ay", best_ay)]:
            tag = f"phase-2d-slice-{slice_type}-azz-theta{theta}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                result_slice = Drive2DSliceResult.from_parquet(csv_p)
                best_arr[i] = _safe_grid_min(result_slice.delta_theta_grid)

        tag_rs = f"phase-random-search-theta{theta}"
        csv_p_rs = _parquet_path(tag_rs)
        if csv_p_rs.exists():
            result_rs = DriveRandomSearchResult.from_parquet(csv_p_rs)
            best_rs[i] = result_rs.best_delta_theta

    print(f"  [debug] best_ax finite: {np.sum(np.isfinite(best_ax))} / {n_theta}")
    print(f"  [debug] best_ay finite: {np.sum(np.isfinite(best_ay))} / {n_theta}")
    print(f"  [debug] best_rs finite: {np.sum(np.isfinite(best_rs))} / {n_theta}")
    print(f"  [debug] best_nm finite: {np.sum(np.isfinite(best_nm))} / {n_theta}")

    sql_vals = np.full(n_theta, 0.1)

    # Generate combined sensitivity plot (local function)
    plot_drive_combined_sensitivity(
        theta_vals,
        best_ax,
        best_ay,
        best_rs,
        best_nm,
        sql_vals,
        fig_p1,
    )
    print(f"[fig]  {fig_p1}")

    # Generate NM expectation/variance plot (local function)
    plot_drive_nm_expectation_variance(
        theta_vals,
        exp_vals,
        var_vals,
        fig_p2,
    )
    print(f"[fig]  {fig_p2}")


def generate_phase_fraction_below_sql(force: bool = False) -> None:
    """Fraction of parameter space below SQL vs θ for all methods.

    Reads existing per-θ Parquets (2D slices, random search) and computes
    the fraction of points whose Δθ falls below the SQL for each θ.
    """
    fig_p = _fig_path("phase-fraction-below-sql")

    theta_vals = np.array(PHASE_THETA_VALS, dtype=float)
    n_theta = len(theta_vals)

    fractions_ax = np.full(n_theta, np.nan)
    fractions_ay = np.full(n_theta, np.nan)
    fractions_rs = np.full(n_theta, np.nan)

    for i, theta in enumerate(theta_vals):
        tag_ax = f"phase-2d-slice-ax-azz-theta{theta}"
        csv_ax = _parquet_path(tag_ax)
        if csv_ax.exists():
            result = Drive2DSliceResult.from_parquet(csv_ax)
            fractions_ax[i] = (
                np.sum(result.delta_theta_grid < result.sql)
                / result.delta_theta_grid.size
            )

        tag_ay = f"phase-2d-slice-ay-azz-theta{theta}"
        csv_ay = _parquet_path(tag_ay)
        if csv_ay.exists():
            result = Drive2DSliceResult.from_parquet(csv_ay)
            fractions_ay[i] = (
                np.sum(result.delta_theta_grid < result.sql)
                / result.delta_theta_grid.size
            )

        tag_rs = f"phase-random-search-theta{theta}"
        csv_rs = _parquet_path(tag_rs)
        if csv_rs.exists():
            result = DriveRandomSearchResult.from_parquet(csv_rs)
            fractions_rs[i] = np.sum(result.delta_theta_values < result.sql) / len(
                result.delta_theta_values
            )

    plot_drive_fraction_below_sql(
        theta_vals,
        fractions_ax,
        fractions_ay,
        fractions_rs,
        fig_p,
    )
    print(f"[fig]  {fig_p}")


def generate_phase_cross_experiment_comparison(force: bool = False) -> None:
    """Comparison of fixed-drive (2026-05-18) vs modulated-drive (2026-05-19)
    θ-scan results.

    Loads both Parquets, interpolates the sparse 2026-05-18 data to the fine
    50-point θ grid of the 2026-05-19 scan, and produces a 2×1 figure
    showing Δθ vs θ (upper) and the ratio Δθ_19/Δθ_18 (lower).
    """
    fig_p = _fig_path("phase-cross-experiment-comparison")

    # Load modulated-drive result (2026-05-19, 50 points)
    pq_19 = _parquet_path("phase-theta-scan")
    if not pq_19.exists():
        print(
            "[skip] 2026-05-19-phase-theta-scan.parquet does not exist; "
            "run 'phase-theta-scan' first"
        )
        return
    result_19 = DriveThetaScanResult.from_parquet(pq_19)

    # Load fixed-drive result (2026-05-18, 5 points)
    csv_18 = REPORTS_DIR / "20260518" / "raw_data" / "2026-05-18-drive-theta-scan.csv"
    if not csv_18.exists():
        print(
            "[skip] 2026-05-18-drive-theta-scan.csv does not exist; "
            "run 'drive-theta-scan' first"
        )
        return
    result_18 = DriveThetaScanResult.from_parquet(csv_18)

    theta_fine = result_19.theta_values
    theta_coarse = result_18.theta_values
    delta_18_coarse = result_18.best_delta_theta_per_theta
    delta_18_fine = np.interp(theta_fine, theta_coarse, delta_18_coarse)

    sql_fine = result_19.sql_values

    plot_drive_cross_experiment_comparison(
        theta_values=theta_fine,
        best_delta_19=result_19.best_delta_theta_per_theta,
        best_delta_18=delta_18_fine,
        sql_values=sql_fine,
        save_path=fig_p,
    )
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-19 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'phase-decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / PHASE_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / PHASE_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "phase-decoupled-baseline": generate_phase_decoupled_baseline,
        "phase-2d-slice-ax-azz": generate_phase_2d_slice_ax_azz,
        "phase-2d-slice-ay-azz": generate_phase_2d_slice_ay_azz,
        "phase-random-search": generate_phase_random_search,
        "phase-theta-scan": generate_phase_theta_scan,
        "phase-optimal-params": generate_phase_optimal_params,
        "phase-combined-sensitivity": generate_phase_combined_sensitivity,
        "phase-fraction-below-sql": generate_phase_fraction_below_sql,
        "phase-cross-experiment-comparison": generate_phase_cross_experiment_comparison,
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
