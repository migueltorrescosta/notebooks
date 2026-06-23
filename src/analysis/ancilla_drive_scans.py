"""
Scan and optimisation routines for the driven-ancilla metrology protocol.

Provides 2D slice scans, random search, Nelder--Mead optimisation,
and ω-scan pipelines built on top of the core circuit operators
in ``ancilla_drive_metrology`` and result dataclasses in
``ancilla_drive_results``.

References:
- Report ``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``
"""

from __future__ import annotations

import concurrent.futures
import os

import numpy as np
from scipy.optimize import minimize

from src.analysis.ancilla_drive_metrology import (
    compute_drive_sensitivity,
    evolve_drive_circuit,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
)

# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_drive_decoupled_baseline(
    t_hold: float = 10.0,
    omega_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω.

    At (a_x = a_y = a_z = a_zz = 0), the driving-ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δω = 1/t_hold.

    Args:
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    domega = compute_drive_sensitivity(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.pi / 2.0,
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
        omega_value=omega_true,
    )


# ============================================================================
# 2D Slice Scan (ax, ay, or az)
# ============================================================================


def _evaluate_grid_point(
    d_val: float,
    a_val: float,
    slice_type: str,
    omega: float,
    t_hold: float,
    T_BS: float,
    ops: dict[str, np.ndarray],
) -> float:
    """Evaluate Δω at a single (drive, a_zz) grid point.

    Args:
        d_val: Drive coefficient value (a_x, a_y, or a_z depending on slice_type).
        a_val: Interaction coefficient a_zz value.
        slice_type: 'ax', 'ay', or 'az'.
        omega: Phase rate value.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        ops: Two-qubit operators.

    Returns:
        Δω sensitivity at this grid point.
    """
    if slice_type == "ax":
        ax, ay, az = d_val, 0.0, 0.0
    elif slice_type == "ay":
        ax, ay, az = 0.0, d_val, 0.0
    else:
        ax, ay, az = 0.0, 0.0, d_val
    return compute_drive_sensitivity(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        T_BS,
        t_hold,
        omega,
        ax,
        ay,
        az,
        a_val,
        ops,
    )


def _drive_slice_chunk_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker for parallel 2D slice evaluation (module-level for pickling).

    Args:
        args: Tuple (omega, drive_chunk, azz_vals, slice_type, t_hold, T_BS, start_idx).

    Returns:
        Tuple (start_idx, chunk_grid) where chunk_grid has shape
        (len(drive_chunk), len(azz_vals)).
    """
    omega, drive_chunk, azz_vals, slice_type, t_hold, T_BS, start_idx = args
    local_ops = build_two_qubit_operators()
    n_d = len(drive_chunk)
    n_a = len(azz_vals)
    chunk_grid = np.full((n_d, n_a), np.inf, dtype=float)
    for i, d_val in enumerate(drive_chunk):
        for j, a_val in enumerate(azz_vals):
            chunk_grid[i, j] = _evaluate_grid_point(
                d_val,
                a_val,
                slice_type,
                omega,
                t_hold,
                T_BS,
                local_ops,
            )
    return start_idx, chunk_grid


def _evaluate_sequential_slice(
    drive_vals: np.ndarray,
    azz_vals: np.ndarray,
    slice_type: str,
    omega: float,
    t_hold: float,
    T_BS: float,
) -> np.ndarray:
    """Evaluate the 2D slice sequentially.

    Args:
        drive_vals: Drive coefficient values.
        azz_vals: Interaction coefficient values.
        slice_type: 'ax', 'ay', or 'az'.
        omega: Phase rate value.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        2D grid of Δω values, shape (len(drive_vals), len(azz_vals)).
    """
    ops = build_two_qubit_operators()
    n_d = len(drive_vals)
    n_a = len(azz_vals)
    grid = np.full((n_d, n_a), np.inf, dtype=float)
    for i, d_val in enumerate(drive_vals):
        for j, a_val in enumerate(azz_vals):
            grid[i, j] = _evaluate_grid_point(
                d_val,
                a_val,
                slice_type,
                omega,
                t_hold,
                T_BS,
                ops,
            )
    return grid


def _evaluate_parallel_slice(
    drive_vals: np.ndarray,
    azz_vals: np.ndarray,
    slice_type: str,
    omega: float,
    t_hold: float,
    T_BS: float,
    n_jobs: int,
) -> np.ndarray:
    """Evaluate the 2D slice in parallel using process workers.

    The drive dimension is split into ``n_jobs`` chunks, each evaluated
    by a separate worker process.

    Args:
        drive_vals: Drive coefficient values.
        azz_vals: Interaction coefficient values.
        slice_type: 'ax', 'ay', or 'az'.
        omega: Phase rate value.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        n_jobs: Number of parallel workers (-1 uses all CPUs).

    Returns:
        2D grid of Δω values, shape (len(drive_vals), len(azz_vals)).
    """
    n_workers = max(1, os.cpu_count() or 4) if n_jobs == -1 else n_jobs
    n_drive = len(drive_vals)
    drive_indices = np.arange(n_drive)
    chunks = np.array_split(drive_indices, n_workers)
    worker_args = [
        (
            omega,
            drive_vals[chunk],
            azz_vals,
            slice_type,
            t_hold,
            T_BS,
            int(chunk[0]),
        )
        for chunk in chunks
    ]

    grid = np.full((n_drive, len(azz_vals)), np.inf, dtype=float)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
    ) as executor:
        futures = {
            executor.submit(_drive_slice_chunk_worker, args): args
            for args in worker_args
        }
        for future in concurrent.futures.as_completed(futures):
            start_idx, chunk_grid = future.result()
            n_chunk = chunk_grid.shape[0]
            grid[start_idx : start_idx + n_chunk, :] = chunk_grid
    return grid


def drive_2d_slice(
    omega: float,
    drive_range: tuple[float, float] = (-5.0, 5.0),
    azz_range: tuple[float, float] = (-5.0, 5.0),
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
    n_jobs: int | None = None,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz).

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).
    For slice_type='az': varies a_z (with a_x = a_y = 0).

    When n_jobs > 1, the grid is split across ``n_jobs`` worker processes
    for parallel evaluation.

    Args:
        omega: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax', 'ay', or 'az'.
        t_hold: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).
        n_jobs: Number of parallel workers. ``None`` (default) = sequential.
            Pass ``-1`` to use all available CPUs.

    Returns:
        Drive2DSliceResult with the sensitivity grid.
    """
    if slice_type not in ("ax", "ay", "az"):
        raise ValueError(f"slice_type must be 'ax', 'ay' or 'az', got {slice_type}")

    drive_vals = np.linspace(drive_range[0], drive_range[1], n_drive)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)

    if n_jobs is None or n_jobs == 1:
        grid = _evaluate_sequential_slice(
            drive_vals,
            azz_vals,
            slice_type,
            omega,
            t_hold,
            T_BS,
        )
    else:
        grid = _evaluate_parallel_slice(
            drive_vals,
            azz_vals,
            slice_type,
            omega,
            t_hold,
            T_BS,
            n_jobs,
        )

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


def drive_random_search(
    omega: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = (-5.0, 5.0),
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

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

        domega = compute_drive_sensitivity(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
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


def drive_sensitivity_objective(
    params: np.ndarray,
    omega_true: float,
    ops: dict[str, np.ndarray],
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
    fd_step: float = 1e-6,
    bounds: tuple[float, float] = (-5.0, 5.0),
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω in the driven-ancilla protocol.

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

    return compute_drive_sensitivity(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
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


def run_drive_nelder_mead(
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = (-5.0, 5.0),
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder--Mead optimisation for the driven-ancilla protocol.

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
        return drive_sensitivity_objective(
            p,
            omega_true,
            ops,
            t_hold=t_hold,
            T_BS=T_BS,
            bounds=bounds,
        )

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,  # type: ignore[arg-type]
        options={  # type: ignore[call-overload]
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_drive_circuit(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
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


def run_drive_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = (-5.0, 5.0),
    t_hold: float = 10.0,
    T_BS: float = np.pi / 2.0,
) -> DriveOmegaScanResult:
    """Scan over ω values with 4D random search and Nelder--Mead refinement.

    For each ω:
    1. Run ``n_random`` random evaluations in the 4D parameter space.
    2. Select the best ``n_nm_refine`` points.
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

    for omega_val in omega_arr:
        # Stage 1: Random search
        rs_result = drive_random_search(
            omega_val,
            n_samples=n_random,
            bounds=bounds,
            t_hold=t_hold,
            T_BS=T_BS,
            seed=base_seed + int(omega_val * 1000),
        )

        # Sort random-search results by Δω, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_omega_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[DriveNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            x0 = rs_result.samples[idx].copy()
            nm = run_drive_nelder_mead(
                omega_true=omega_val,
                x0=x0,
                seed=base_seed + int(omega_val * 1000) + 10000 + rank,
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
        all_results_dict[float(omega_val)] = nm_results

    return DriveOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params_list,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
        all_results=all_results_dict,
    )
