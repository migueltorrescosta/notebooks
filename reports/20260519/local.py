"""
Local module for the 2026-05-19 Ancilla-Drive Phase-Modulated Metrology report.

Contains all code exclusive to this report:
- Core physics simulation (ω-modulated Hamiltonian operators, circuit evolution,
  sensitivity computation)
- 4D random search, Nelder–Mead refinement, and ω-scan orchestration
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
from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
    build_iszz_interaction,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
)
from src.utils.constants import I_4
from src.utils.parallel import parallel_map
from src.utils.paths import fig_path, parquet_path

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants  (moved from ancilla_drive_phase_modulated.py)
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_t_hold  # Δω_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients

# ============================================================================
# Operator Construction  (from ancilla_drive_phase_modulated.py)
# ============================================================================


def build_phase_modulated_drive_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ω-modulated ancilla drive Hamiltonian.

    H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    The critical difference from the fixed-drive protocol is the leading ω
    factor: the ancilla drive scales with the unknown phase, creating a
    parametric amplification effect in ∂⟨J_z^S⟩/∂ω.

    Args:
        omega: Unknown phase rate parameter (scales the whole drive).
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the ω-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H  # ω-modulation: entire drive scales with the unknown phase
    return 0.5 * (H + H.conj().T)


def build_phase_modulated_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with ω-modulated ancilla drive.

    H = ω J_z^S + H_A + H_int
      = ω J_z^S + ω (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A
      = ω [J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A] + a_zz J_z^S ⊗ J_z^A

    The ω factor on the drive terms means ∂H/∂ω = J_z^S + a_x J_x^A + a_y J_y^A
    + a_z J_z^A, which includes ancilla operators. This extra contribution to
    the derivative is the key mechanism for potential SQL violation.

    Args:
        omega: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_phase_modulated_drive_hamiltonian(omega, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def phase_modulated_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the ω-modulated ancilla protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω J_z^S + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_phase_modulated_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Phase-modulated hold unitary not unitary for t_hold={t_hold}, ω={omega}"
    )
    return U


def evolve_phase_modulated_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full ω-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    The hold unitary uses the ω-modulated H_A = ω (a_x J_x^A + ...).

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
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
    psi = phase_modulated_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_phase_modulated_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δω.

    Δω = sqrt(Var(O)) / |∂⟨O⟩/∂ω|

    where O is the measurement operator (default: J_z^S).

    IMPORTANT: Because ω now appears in both H_S (= ω J_z^S) and H_A
    (= ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)), the central finite-difference
    step captures the FULL ω-dependence (both channels) automatically —
    the circuit is re-evaluated at ω ± δ, and both H_S and H_A change.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δω (positive float). Returns inf if derivative is zero
        (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂ω
    psi_plus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
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
    t_hold: float = DEFAULT_t_hold,
    omega_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω.

    At (a_x = a_y = a_z = a_zz = 0), the ω-modulated ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δω = 1/t_hold. The ω factor in H_A is irrelevant when all a_k = 0.

    Args:
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    domega = compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        t_hold,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        t_hold_value=t_hold,
        delta_omega=domega,
        sql=1.0 / t_hold,
    )


# ============================================================================
# 2D Slice Scan
# ============================================================================


def phase_modulated_2d_slice(
    omega: float,
    drive_range: tuple[float, float] = DRIVE_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz) with ω-modulated ancilla drive.

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).

    Args:
        omega: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax' or 'ay'.
        t_hold: Holding time (default 10).
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

            domega = compute_phase_modulated_sensitivity(
                DEFAULT_PSI0,
                T_BS,
                t_hold,
                omega,
                ax,
                ay,
                az,
                a_val,
                ops,
            )
            grid[i, j] = domega

    return Drive2DSliceResult(
        drive_values=drive_vals,
        azz_values=azz_vals,
        delta_omega_grid=grid,
        omega_value=omega,
        slice_type=slice_type,
        sql=1.0 / t_hold,
    )


# ============================================================================
# 4D Random Search
# ============================================================================


def phase_modulated_random_search(
    omega: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Uses the ω-modulated ancilla drive H_A = ω (a_x J_x^A + ...).

    Args:
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        t_hold: Holding time.
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

        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            T_BS,
            t_hold,
            omega,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    return DriveRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=1.0 / t_hold,
        t_hold=t_hold,
    )


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


def phase_modulated_sensitivity_objective(
    params: np.ndarray,
    omega_true: float,
    ops: dict[str, np.ndarray],
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω in the ω-modulated protocol.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed t_hold.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        omega_true: True phase rate.
        ops: Two-qubit operators.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
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
        t_hold,
        omega_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_phase_modulated_nelder_mead(
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder--Mead optimisation for the ω-modulated ancilla protocol.

    Args:
        omega_true: True phase rate parameter.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        bounds: (min, max) for all four parameters.
        t_hold: Holding time.
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
            omega_true,
            ops,
            t_hold=t_hold,
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
        t_hold,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# ω Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_phase_modulated_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> DriveOmegaScanResult:
    """Scan over ω values with 4D random search and Nelder--Mead refinement.

    For each ω:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search points per ω.
        n_nm_refine: Number of Nelder--Mead refinements per ω.
        seed: Base random seed (incremented per ω).
        maxiter: Maximum Nelder--Mead iterations.
        bounds: (min, max) for all parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveOmegaScanResult with optimal parameters and sensitivities.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    base_seed = seed if seed is not None else 42

    best_params_list: list[tuple[float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []
    all_results_dict: dict[float, list[DriveNelderMeadResult]] = {}

    for omega in omega_arr:
        # Stage 1: Random search
        rs_result = phase_modulated_random_search(
            omega,
            n_samples=n_random,
            bounds=bounds,
            t_hold=t_hold,
            T_BS=T_BS,
            seed=base_seed + int(omega * 1000),
        )

        # Sort random-search results by Δω, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_omega_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[DriveNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            x0 = rs_result.samples[idx].copy()
            nm = run_phase_modulated_nelder_mead(
                omega_true=omega,
                x0=x0,
                seed=base_seed + int(omega * 1000) + 10000 + rank,
                maxiter=maxiter,
                bounds=bounds,
                t_hold=t_hold,
                T_BS=T_BS,
                track_history=False,
            )
            nm_results.append(nm)

        # Sort Nelder--Mead results by Δω
        nm_results.sort(key=lambda r: r.delta_omega_opt)
        best_nm = nm_results[0]

        best_params_list.append(
            (
                float(best_nm.params_opt[0]),
                float(best_nm.params_opt[1]),
                float(best_nm.params_opt[2]),
                float(best_nm.params_opt[3]),
            )
        )
        best_deltas.append(best_nm.delta_omega_opt)
        sql_vals.append(1.0 / t_hold)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)
        all_results_dict[float(omega)] = nm_results

    return DriveOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params_list,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
        all_results=all_results_dict,
    )


# ============================================================================
# Exclusive Plot Functions
# (moved from src/visualization/ancilla_drive_plots.py)
# ============================================================================


def plot_drive_combined_sensitivity(
    omega_values: np.ndarray,
    best_ax_slice: np.ndarray,
    best_ay_slice: np.ndarray,
    best_random: np.ndarray,
    best_nm: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot comparing Δω from 2D slices, 4D random search, NM refinement, and SQL.

    Args:
        omega_values: Array of ω values.
        best_ax_slice: Best Δω from (a_x, a_zz) slice at each ω.
        best_ay_slice: Best Δω from (a_y, a_zz) slice at each ω.
        best_random: Best Δω from 4D random search at each ω.
        best_nm: Best Δω from Nelder–Mead refinement at each ω.
        sql_values: SQL reference at each ω (constant).
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
                omega_values[valid],
                data[valid],
                fmt,
                color=colour,
                label=label,
                markersize=6,
                linewidth=1.5,
                markerfacecolor=colour,
            )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(
        "Sensitivity vs $\\omega$: 2D slices, 4D random search, Nelder–Mead refinement"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_nm_expectation_variance(
    omega_values: np.ndarray,
    expectation_Jz: np.ndarray,
    variance_Jz: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Side-by-side plot of ⟨J_z^S⟩ and Var(J_z^S) at the NM optimum vs ω.

    Args:
        omega_values: Array of ω values.
        expectation_Jz: ⟨J_z^S⟩ at NM optimum for each ω.
        variance_Jz: Var(J_z^S) at NM optimum for each ω.
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
            omega_values[valid_exp],
            expectation_Jz[valid_exp],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.5,
        )
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\langle J_z^S \rangle$")
    ax1.set_title(r"Expectation $\langle J_z^S\rangle$ at NM optimum")

    # Right panel: variance
    valid_var = np.isfinite(variance_Jz)
    if np.any(valid_var):
        ax2.plot(
            omega_values[valid_var],
            variance_Jz[valid_var],
            "s-",
            color="C1",
            markersize=7,
            linewidth=1.5,
        )
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax2.set_title(r"Variance $\mathrm{Var}(J_z^S)$ at NM optimum")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_cross_experiment_comparison(
    omega_values: np.ndarray,
    best_delta_19: np.ndarray,
    best_delta_18: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Compare Δω from the fixed-drive (2026-05-18) and modulated-drive
    (2026-05-19) experiments in a 2×1 vertically stacked figure.

    Upper panel: Overlaid line plots of Δω vs ω for both experiments,
    with the SQL shown as a dashed reference line.

    Lower panel: Ratio Δω_19 / Δω_18 vs ω. A horizontal line at y=1
    separates regimes where the fixed drive (above 1) or modulated drive
    (below 1) performs better.

    Args:
        omega_values: Common ω grid (50 points from the modulated-drive scan).
        best_delta_19: Δω from the modulated-drive scan (2026-05-19).
        best_delta_18: Δω from the fixed-drive scan (2026-05-18),
            interpolated to the same ω grid.
        sql_values: SQL reference values (constant, 0.1) at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δω vs ω ──────────────────────────────────────────
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
        omega_values,
        best_delta_18,
        marker="s",
        linestyle="-",
        color="C0",
        markersize=5,
        linewidth=1.8,
        label=r"Fixed drive (2026-05-18)",
    )
    ax1.plot(
        omega_values,
        best_delta_19,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=5,
        linewidth=1.8,
        label=r"Modulated drive (2026-05-19)",
    )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title("Cross-experiment comparison: fixed vs modulated drive")
    ax1.legend(fontsize=9)

    # ── Lower panel: ratio Δω_19 / Δω_18 ──────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(best_delta_18) & (best_delta_18 > 0),
            best_delta_19 / best_delta_18,
            np.nan,
        )

    ax2.plot(
        omega_values,
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
        min_omega = float(omega_values[valid][min_idx])
        ax2.annotate(
            f"Best = {min_ratio:.3f}$\\times$ at $\\omega$={min_omega:.1f}",
            xy=(min_omega, min_ratio),
            xytext=(min_omega + 0.6, min_ratio + 0.15),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega_{19} \;/\; \Delta\omega_{18}$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_fraction_below_sql(
    omega_values: np.ndarray,
    fractions_2d_ax: np.ndarray,
    fractions_2d_ay: np.ndarray,
    fractions_random: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of the fraction of parameter space below SQL as a function of ω.

    Args:
        omega_values: Array of ω values.
        fractions_2d_ax: Fraction below SQL from (a_x, a_zz) slices at each ω.
        fractions_2d_ay: Fraction below SQL from (a_y, a_zz) slices at each ω.
        fractions_random: Fraction below SQL from 4D random search at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        omega_values,
        fractions_2d_ax,
        "o-",
        color="C0",
        label=r"2D slice $(a_x, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_2d_ay,
        "s-",
        color="C1",
        label=r"2D slice $(a_y, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
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

    ax.set_xlabel(r"$\omega$")
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
PHASE_OMEGA_VALS = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
PHASE_N_GRID = 201


def _parquet_path(name: str) -> Path:
    return parquet_path(REPORTS_DIR, PHASE_DATE, name)


def _fig_path(name: str) -> Path:
    return fig_path(REPORTS_DIR, PHASE_DATE, name)


# ── Parallel dispatch helper ──────────────────────────────────────────────


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
    omega: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a phase-modulated 2D slice scan for a single ω value."""
    tag = f"phase-2d-slice-{slice_type}-azz-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing phase ({slice_type}, a_zz) slice at ω={omega}...")
        result = phase_modulated_2d_slice(
            omega=omega,
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
    """Phase-modulated 2D slice scans over (a_x, a_zz) at all ω values."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  (a_x, a_zz) phase slice at {n} ω values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ax", force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="(a_x, a_zz) slices")


def generate_phase_2d_slice_ay_azz(force: bool = False) -> None:
    """Phase-modulated 2D slice scans over (a_y, a_zz) at all ω values."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  (a_y, a_zz) phase slice at {n} ω values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ay", force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="(a_y, a_zz) slices")


def _run_phase_random_search(omega: float, force: bool) -> None:
    """Run a phase-modulated 4D random search for a single ω value."""
    tag = f"phase-random-search-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveRandomSearchResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Running phase 4D random search at ω={omega} (500 samples)...")
        result = phase_modulated_random_search(
            omega=omega,
            n_samples=500,
            seed=42,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_random_search_histogram

    plot_drive_random_search_histogram(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_phase_random_search(force: bool = False) -> None:
    """Phase-modulated 4D random search at all ω values (parallel)."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  4D phase random search at {n} ω values (parallel)")
    worker = partial(_run_phase_random_search, force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="random search")


def _run_phase_omega_scan_single(omega: float) -> dict[str, float | np.ndarray]:
    """Run random search + NM refinement for a single ω value.

    Returns a dict with per-ω results that can be aggregated into a
    ``DriveOmegaScanResult``.
    """
    base_seed: int = 42
    n_random: int = 500
    n_nm_refine: int = 50
    maxiter_val: int = 5000
    bounds: tuple[float, float] = (-5.0, 5.0)

    # Stage 1: Random search
    rs_result = phase_modulated_random_search(
        omega,
        n_samples=n_random,
        bounds=bounds,
        seed=base_seed + int(omega * 1000),
    )

    # Sort by Δω, take top n_nm_refine
    sorted_indices = np.argsort(rs_result.delta_omega_values)
    top_indices = sorted_indices[:n_nm_refine]

    # Stage 2: Nelder--Mead refinement from each top point
    nm_results: list[DriveNelderMeadResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        nm = run_phase_modulated_nelder_mead(
            omega_true=omega,
            x0=x0,
            seed=base_seed + int(omega * 1000) + 10000 + rank,
            maxiter=maxiter_val,
            bounds=bounds,
            track_history=False,
        )
        nm_results.append(nm)

    nm_results.sort(key=lambda r: r.delta_omega_opt)
    best_nm = nm_results[0]

    return {
        "omega": omega,
        "best_delta_omega": best_nm.delta_omega_opt,
        "a_x": float(best_nm.params_opt[0]),
        "a_y": float(best_nm.params_opt[1]),
        "a_z": float(best_nm.params_opt[2]),
        "a_zz": float(best_nm.params_opt[3]),
        "expectation_Jz": best_nm.expectation_Jz,
        "variance_Jz": best_nm.variance_Jz,
    }


def _check_omega_scan_cache(
    parquet_path: Path, force: bool
) -> DriveOmegaScanResult | None:
    """Check if cached parquet exists and load it.

    Args:
        parquet_path: Path to cached Parquet file.
        force: If True, ignore cache and return None.

    Returns:
        Loaded result if cached and not forced, else None.
    """
    if not parquet_path.exists() or force:
        return None
    print(f"[skip] {parquet_path.name} exists (use --force to overwrite)")
    return DriveOmegaScanResult.from_parquet(parquet_path)


def _compute_omega_scan_core() -> DriveOmegaScanResult:
    """Run the parallel ω-scan computation (random search + NM refinement)."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  Computing phase ω-scan for {n} ω values (parallel)...")

    max_workers = min(32, os.cpu_count() or 1)
    print(f"  [parallel] Using {max_workers} workers for ω-scan")

    per_omega_results: list[dict[str, Any]] = []
    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_omega = {
            executor.submit(_run_phase_omega_scan_single, omega): omega
            for omega in PHASE_OMEGA_VALS
        }
        for future in concurrent.futures.as_completed(fut_to_omega):
            omega = fut_to_omega[future]
            try:
                per_omega_results.append(future.result())
                print(f"  [done] ω={omega}")
            except Exception as exc:
                print(f"  [ERROR] ω={omega}: {exc}")
                raise

    per_omega_results.sort(key=lambda r: float(r["omega"]))

    omega_arr = np.array([r["omega"] for r in per_omega_results], dtype=float)
    best_deltas = [float(r["best_delta_omega"]) for r in per_omega_results]
    best_params = [
        (
            float(r["a_x"]),
            float(r["a_y"]),
            float(r["a_z"]),
            float(r["a_zz"]),
        )
        for r in per_omega_results
    ]
    exp_vals = [float(r["expectation_Jz"]) for r in per_omega_results]
    var_vals = [float(r["variance_Jz"]) for r in per_omega_results]

    return DriveOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.full(len(omega_arr), 1.0 / DEFAULT_t_hold, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
    )


def _save_and_plot_omega_scan(
    result: DriveOmegaScanResult,
    parquet_path: Path,
    fig_path: Path,
) -> None:
    """Save result to Parquet and generate the ω-scan figure."""
    result.save_parquet(parquet_path)
    print(f"[save] {parquet_path}")

    from src.visualization.ancilla_drive_plots import plot_drive_omega_scan

    plot_drive_omega_scan(result, fig_path)
    print(f"[fig]  {fig_path}")


def generate_phase_omega_scan(force: bool = False) -> None:
    """Phase-modulated ω-scan with Nelder-Mead refinement (parallel).

    Check cache → compute (if needed) → save and plot.
    """
    from src.visualization.ancilla_drive_plots import plot_drive_omega_scan

    csv_p = _parquet_path("phase-omega-scan")
    fig_p = _fig_path("phase-omega-scan")

    result = _check_omega_scan_cache(csv_p, force)
    if result is None:
        result = _compute_omega_scan_core()
        _save_and_plot_omega_scan(result, csv_p, fig_p)
    else:
        plot_drive_omega_scan(result, fig_p)
        print(f"[fig]  {fig_p}")


def generate_phase_optimal_params(force: bool = False) -> None:
    """Phase-modulated optimal parameter evolution vs ω."""
    csv_p = _parquet_path("phase-omega-scan")
    fig_p = _fig_path("phase-optimal-params")

    result = DriveOmegaScanResult.from_parquet(csv_p)
    from src.visualization.ancilla_drive_plots import plot_drive_optimal_params

    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_combined_sensitivity(force: bool = False) -> None:
    """Combined sensitivity plot + NM expectation/variance plot.

    Reads existing per-ω Parquets (2D slices, random search) and the omega-scan
    result to produce two figures:
        - phase-combined-sensitivity.svg : Δω vs ω for all methods + SQL
        - phase-nm-expectation-variance.svg : ⟨J_z^S⟩ and Var(J_z^S) at NM optimum

    This must be run AFTER the data-generation steps
    (phase-2d-slice-*, phase-random-search, phase-omega-scan).
    """
    fig_p1 = _fig_path("phase-combined-sensitivity")
    fig_p2 = _fig_path("phase-nm-expectation-variance")

    # Load NM result
    omega_scan_pq = _parquet_path("phase-omega-scan")
    if not omega_scan_pq.exists():
        print(
            "[skip] phase-omega-scan.parquet does not exist; run 'phase-omega-scan' first"
        )
        return
    nm_result = DriveOmegaScanResult.from_parquet(omega_scan_pq)

    omega_vals = np.array(PHASE_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    best_nm = np.full(n_omega, np.nan)
    exp_vals = np.full(n_omega, np.nan)
    var_vals = np.full(n_omega, np.nan)

    nm_omega = nm_result.omega_values
    if len(nm_omega) >= n_omega:
        for i in range(n_omega):
            best_nm[i] = float(nm_result.best_delta_omega_per_omega[i])
            exp_vals[i] = float(nm_result.expectation_Jz_per_omega[i])
            var_vals[i] = float(nm_result.variance_Jz_per_omega[i])

    # Collect per-ω minima from 2D slice and random search Parquets
    best_ax = np.full(n_omega, np.nan)
    best_ay = np.full(n_omega, np.nan)
    best_rs = np.full(n_omega, np.nan)

    def _safe_grid_min(grid: np.ndarray) -> float:
        finite_vals = grid[np.isfinite(grid)]
        if finite_vals.size == 0:
            return np.nan
        return float(np.min(finite_vals))

    for i, omega in enumerate(omega_vals):
        for slice_type, best_arr in [("ax", best_ax), ("ay", best_ay)]:
            tag = f"phase-2d-slice-{slice_type}-azz-omega{omega}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                result_slice = Drive2DSliceResult.from_parquet(csv_p)
                best_arr[i] = _safe_grid_min(result_slice.delta_omega_grid)

        tag_rs = f"phase-random-search-omega{omega}"
        csv_p_rs = _parquet_path(tag_rs)
        if csv_p_rs.exists():
            result_rs = DriveRandomSearchResult.from_parquet(csv_p_rs)
            best_rs[i] = result_rs.best_delta_omega

    print(f"  [debug] best_ax finite: {np.sum(np.isfinite(best_ax))} / {n_omega}")
    print(f"  [debug] best_ay finite: {np.sum(np.isfinite(best_ay))} / {n_omega}")
    print(f"  [debug] best_rs finite: {np.sum(np.isfinite(best_rs))} / {n_omega}")
    print(f"  [debug] best_nm finite: {np.sum(np.isfinite(best_nm))} / {n_omega}")

    sql_vals = np.full(n_omega, 0.1)

    # Generate combined sensitivity plot (local function)
    plot_drive_combined_sensitivity(
        omega_vals,
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
        omega_vals,
        exp_vals,
        var_vals,
        fig_p2,
    )
    print(f"[fig]  {fig_p2}")


def generate_phase_fraction_below_sql(force: bool = False) -> None:
    """Fraction of parameter space below SQL vs ω for all methods.

    Reads existing per-ω Parquets (2D slices, random search) and computes
    the fraction of points whose Δω falls below the SQL for each ω.
    """
    fig_p = _fig_path("phase-fraction-below-sql")

    omega_vals = np.array(PHASE_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    fractions_ax = np.full(n_omega, np.nan)
    fractions_ay = np.full(n_omega, np.nan)
    fractions_rs = np.full(n_omega, np.nan)

    for i, omega in enumerate(omega_vals):
        tag_ax = f"phase-2d-slice-ax-azz-omega{omega}"
        csv_ax = _parquet_path(tag_ax)
        if csv_ax.exists():
            result = Drive2DSliceResult.from_parquet(csv_ax)
            fractions_ax[i] = (
                np.sum(result.delta_omega_grid < result.sql)
                / result.delta_omega_grid.size
            )

        tag_ay = f"phase-2d-slice-ay-azz-omega{omega}"
        csv_ay = _parquet_path(tag_ay)
        if csv_ay.exists():
            result = Drive2DSliceResult.from_parquet(csv_ay)
            fractions_ay[i] = (
                np.sum(result.delta_omega_grid < result.sql)
                / result.delta_omega_grid.size
            )

        tag_rs = f"phase-random-search-omega{omega}"
        csv_rs = _parquet_path(tag_rs)
        if csv_rs.exists():
            result = DriveRandomSearchResult.from_parquet(csv_rs)
            fractions_rs[i] = np.sum(result.delta_omega_values < result.sql) / len(
                result.delta_omega_values
            )

    plot_drive_fraction_below_sql(
        omega_vals,
        fractions_ax,
        fractions_ay,
        fractions_rs,
        fig_p,
    )
    print(f"[fig]  {fig_p}")


def generate_phase_cross_experiment_comparison(force: bool = False) -> None:
    """Comparison of fixed-drive (2026-05-18) vs modulated-drive (2026-05-19)
    ω-scan results.

    Loads both Parquets, interpolates the sparse 2026-05-18 data to the fine
    50-point ω grid of the 2026-05-19 scan, and produces a 2×1 figure
    showing Δω vs ω (upper) and the ratio Δω_19/Δω_18 (lower).
    """
    fig_p = _fig_path("phase-cross-experiment-comparison")

    # Load modulated-drive result (2026-05-19, 50 points)
    pq_19 = _parquet_path("phase-omega-scan")
    if not pq_19.exists():
        print(
            "[skip] 20260519-phase-omega-scan.parquet does not exist; "
            "run 'phase-omega-scan' first"
        )
        return
    result_19 = DriveOmegaScanResult.from_parquet(pq_19)

    # Load fixed-drive result (2026-05-18, 5 points)
    csv_18 = REPORTS_DIR / "20260518" / "raw_data" / "20260518-drive-omega-scan.csv"
    if not csv_18.exists():
        print(
            "[skip] 20260518-drive-omega-scan.csv does not exist; "
            "run 'drive-omega-scan' first"
        )
        return
    result_18 = DriveOmegaScanResult.from_parquet(csv_18)

    omega_fine = result_19.omega_values
    omega_coarse = result_18.omega_values
    delta_18_coarse = result_18.best_delta_omega_per_omega
    delta_18_fine = np.interp(omega_fine, omega_coarse, delta_18_coarse)

    sql_fine = result_19.sql_values

    plot_drive_cross_experiment_comparison(
        omega_values=omega_fine,
        best_delta_19=result_19.best_delta_omega_per_omega,
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
        "phase-omega-scan": generate_phase_omega_scan,
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
