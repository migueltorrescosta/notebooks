"""
Generate CSV data files and SVG figures for a report.

Usage:
    uv run python -m src.visualization.report_figures
    uv run python -m src.visualization.report_figures --force
    uv run python -m src.visualization.report_figures --only decoupled-baseline

Conventions:
    - Raw data goes to ``reports/raw_data/{date}-{name}.csv``
    - Figures go to ``reports/figures/{date}-{name}.svg``
    - Each ``generate_*`` function: (1) checks if CSV exists (skip unless
      ``--force``), (2) runs the simulation and saves CSV, (3) renders the
      SVG figure from the CSV.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
# Limit OpenMP threads to avoid oversubscription with forking.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.ancilla_drive_metrology import (  # noqa: E402
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
    compute_drive_decoupled_baseline,
    drive_2d_slice,
    drive_random_search,
    run_drive_theta_scan,
)
from src.analysis.ancilla_drive_phase_modulated import (  # noqa: E402
    compute_phase_modulated_decoupled_baseline,
    phase_modulated_2d_slice,
    phase_modulated_random_search,
    run_phase_modulated_nelder_mead,
    run_phase_modulated_theta_scan,
)
from src.analysis.ancilla_optimization import (  # noqa: E402
    AlphaReoptScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    ThetaScanResult,
    build_joint_operator,
    build_two_qubit_operators,
    compute_covariance_analysis,
    compute_decoupled_baseline,
    compute_interaction_robustness,
    run_theta_scan,
    scan_alpha_with_reoptimisation,
)
from src.analysis.weighted_joint_measurement import (  # noqa: E402
    AlphaReoptResultNM,
    MScalingResult,
    NScalingResult,
    run_alpha_scan_with_reoptimisation,
    run_m_scaling,
    run_n_scaling,
)
from src.visualization.ancilla_drive_plots import (  # noqa: E402
    plot_drive_2d_slice_heatmap,
    plot_drive_combined_sensitivity,
    plot_drive_cross_experiment_comparison,
    plot_drive_decoupled_baseline,
    plot_drive_fraction_below_sql,
    plot_drive_nm_expectation_variance,
    plot_drive_optimal_params,
    plot_drive_random_search_histogram,
    plot_drive_theta_scan,
)
from src.visualization.ancilla_plots import (  # noqa: E402
    plot_alpha_reoptimisation,
    plot_covariance_analysis,
    plot_decoupled_baseline,
    plot_interaction_robustness,
    plot_m_scaling,
    plot_n_scaling,
    plot_theta_scan,
    plot_weighted_alpha_scan,
)

REPORTS_DIR = PROJECT_ROOT / "reports"

BASE_DATE = "2026-05-15"


def _csv_path(name: str, date: str) -> Path:
    return REPORTS_DIR / date / "raw_data" / f"{date}-{name}.csv"


def _fig_path(name: str, date: str) -> Path:
    return REPORTS_DIR / date / "figures" / f"{date}-{name}.svg"


def generate_decoupled_baseline(force: bool = False) -> None:
    """Sections 1 & 2: Decoupled baseline and expanded T_H bound."""
    csv_p = _csv_path("decoupled-baseline", date=BASE_DATE)
    fig_p = _fig_path("decoupled-baseline", date=BASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DecoupledBaselineResult.from_csv(csv_p)
    else:
        print("[run]  Computing decoupled baseline...")
        T_H_vals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0])
        result = compute_decoupled_baseline(T_H_vals)
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    plot_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_theta_scan(force: bool = False) -> None:
    """Sections 3 & 9: θ-scan with Nelder-Mead optimisation."""
    csv_p = _csv_path("theta-scan", date=BASE_DATE)
    fig_p = _fig_path("theta-scan", date=BASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing θ-scan (may be slow)...")
        ops = build_two_qubit_operators()
        M_op = build_joint_operator(ops)
        theta_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        result = run_theta_scan(
            theta_vals,
            n_restarts=10,
            seed=42,
            maxiter=2000,
            meas_op=M_op,
        )
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    result = ThetaScanResult.from_csv(csv_p)
    plot_theta_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_interaction_robustness(force: bool = False) -> None:
    """Section 8: T_H × α interaction robustness."""
    csv_p = _csv_path("interaction-robustness", date=BASE_DATE)
    fig_p = _fig_path("interaction-robustness", date=BASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing interaction robustness...")
        T_H_vals = np.array([0.5, 1.0, 2.0])
        alpha_vals = np.array([0.0, 1.0, 2.0])
        result = compute_interaction_robustness(
            T_H_vals,
            alpha_vals,
            theta_true=1.0,
            alpha_name="xx",
        )
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    result = InteractionRobustnessResult.from_csv(csv_p)
    plot_interaction_robustness(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_alpha_reoptimisation(force: bool = False) -> None:
    """Section 7: α-scan with state re-optimisation."""
    csv_p = _csv_path("alpha-reoptimisation", date=BASE_DATE)
    fig_p = _fig_path("alpha-reoptimisation", date=BASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing α re-optimisation scan (may be slow)...")
        alpha_vals = np.array([-1.0, 0.0, 1.0])
        result = scan_alpha_with_reoptimisation(
            "xx",
            alpha_values=alpha_vals,
            n_restarts=5,
            maxiter=500,
            seed=42,
        )
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    result = AlphaReoptScanResult.from_csv(csv_p)
    plot_alpha_reoptimisation(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling with optimal weights, M=N."""
    date_str = "2026-05-18"
    csv_p = _csv_path("n-scaling", date=date_str)
    fig_p = _fig_path("n-scaling", date=date_str)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = NScalingResult.from_csv(csv_p)
    else:
        print("[run]  Computing N-scaling (may be slow)...")
        N_vals = [1, 2, 3, 4, 6, 8, 12, 16]
        result = run_n_scaling(
            N_values=N_vals,
            M=-1,
            theta_true=1.0,
            num_seeds=5,
            maxiter=200,
            seed=42,
            n_bootstrap=10000,
        )
        result.save_csv(csv_p)
        result.save_metadata_json(csv_p)
        print(f"[save] {csv_p}")

    result = NScalingResult.from_csv(csv_p)
    plot_n_scaling(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_m_scaling(force: bool = False) -> None:
    """M-scaling at fixed N=4."""
    date_str = "2026-05-18"
    csv_p = _csv_path("m-scaling", date=date_str)
    fig_p = _fig_path("m-scaling", date=date_str)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = MScalingResult.from_csv(csv_p)
    else:
        print("[run]  Computing M-scaling (may be slow)...")
        M_vals = [0, 1, 2, 3, 4, 6, 8, 12]
        result = run_m_scaling(
            M_values=M_vals,
            N=4,
            theta_true=1.0,
            num_seeds=5,
            maxiter=200,
            seed=42,
        )
        result.save_csv(csv_p)
        meta_path = csv_p.with_suffix(".meta.json")
        import json as _json

        _json.dumps(
            {"N_value": result.N_value, "improvement_01": result.improvement_01}
        )
        meta_path.write_text(
            _json.dumps(
                {"N_value": result.N_value, "improvement_01": result.improvement_01},
                indent=2,
            )
        )
        print(f"[save] {csv_p}")

    result = MScalingResult.from_csv(csv_p)
    plot_m_scaling(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_weighted_alpha_scan(force: bool = False) -> None:
    """α_{xx} scan with weight re-optimisation, N=M=4."""
    date_str = "2026-05-18"
    csv_p = _csv_path("alpha-scan-nm", date=date_str)
    fig_p = _fig_path("alpha-scan-nm", date=date_str)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = AlphaReoptResultNM.from_csv(csv_p)
    else:
        print("[run]  Computing α_{xx} scan with re-optimisation...")
        alpha_vals = np.linspace(-2.0, 2.0, 21)
        result = run_alpha_scan_with_reoptimisation(
            alpha_name="xx",
            N=4,
            M=4,
            alpha_values=alpha_vals,
            theta_true=1.0,
            num_seeds=3,
            maxiter=100,
        )
        result.save_csv(csv_p)
        result.save_metadata_json(csv_p)
        print(f"[save] {csv_p}")

    result = AlphaReoptResultNM.from_csv(csv_p)
    plot_weighted_alpha_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_covariance_analysis(force: bool = False) -> None:
    """Section 5: Covariance analysis."""
    csv_p = _csv_path("covariance-analysis", date=BASE_DATE)
    fig_p = _fig_path("covariance-analysis", date=BASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing covariance analysis...")
        result = compute_covariance_analysis()
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    result = CovarianceAnalysisResult.from_csv(csv_p)
    plot_covariance_analysis(result, fig_p)
    print(f"[fig]  {fig_p}")


# ──────────────────────────────────────────────
# Driven-Ancilla Experiments (2026-05-18 report)
# ──────────────────────────────────────────────

DRIVE_DATE = "2026-05-18"
DRIVE_THETA_VALS = [0.1, 0.5, 1.0, 2.0, 5.0]


def generate_drive_decoupled_baseline(force: bool = False) -> None:
    """Experiment 1: Decoupled baseline verification."""
    csv_p = _csv_path("drive-decoupled-baseline", date=DRIVE_DATE)
    fig_p = _fig_path("drive-decoupled-baseline", date=DRIVE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_csv(csv_p)
    else:
        print("[run]  Computing drive decoupled baseline...")
        result = compute_drive_decoupled_baseline()
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def _run_drive_2d_slice(
    theta: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a 2D slice scan for a single θ value and generate CSV + SVG."""
    tag = f"drive-2d-slice-{slice_type}-azz-theta{theta}"
    csv_p = _csv_path(tag, date=DRIVE_DATE)
    fig_p = _fig_path(tag, date=DRIVE_DATE)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_csv(csv_p)
    else:
        print(f"  [run]  Computing ({slice_type}, a_zz) slice at θ={theta}...")
        result = drive_2d_slice(
            theta=theta,
            slice_type=slice_type,
            n_drive=201,
            n_azz=201,
        )
        result.save_csv(csv_p)
        print(f"  [save] {csv_p}")

    plot_drive_2d_slice_heatmap(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_drive_2d_slice_ax_azz(force: bool = False) -> None:
    """Experiment 2a: 2D slice scans over (a_x, a_zz) at all θ values."""
    print(f"[run]  (a_x, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        _run_drive_2d_slice(theta, slice_type="ax", force=force)


def generate_drive_2d_slice_ay_azz(force: bool = False) -> None:
    """Experiment 2b: 2D slice scans over (a_y, a_zz) at all θ values."""
    print(f"[run]  (a_y, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        _run_drive_2d_slice(theta, slice_type="ay", force=force)


def generate_drive_random_search(force: bool = False) -> None:
    """Experiment 3: 4D random search at all θ values."""
    print(f"[run]  4D random search at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        tag = f"drive-random-search-theta{theta}"
        csv_p = _csv_path(tag, date=DRIVE_DATE)
        fig_p = _fig_path(tag, date=DRIVE_DATE)

        if csv_p.exists() and not force:
            print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
            result = DriveRandomSearchResult.from_csv(csv_p)
        else:
            print(f"  [run]  Running 4D random search at θ={theta} (500 samples)...")
            result = drive_random_search(
                theta=theta,
                n_samples=500,
                seed=42,
            )
            result.save_csv(csv_p)
            print(f"  [save] {csv_p}")

        plot_drive_random_search_histogram(result, fig_p)
        print(f"  [fig]  {fig_p}")


def generate_drive_theta_scan(force: bool = False) -> None:
    """Experiments 4 & 5: θ-scan with Nelder-Mead refinement."""
    csv_p = _csv_path("drive-theta-scan", date=DRIVE_DATE)
    fig_p = _fig_path("drive-theta-scan", date=DRIVE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveThetaScanResult.from_csv(csv_p)
    else:
        print("[run]  Computing drive θ-scan (may be slow)...")
        result = run_drive_theta_scan(
            theta_values=DRIVE_THETA_VALS,
            n_random=500,
            n_nm_refine=50,
            seed=42,
            maxiter=5000,
        )
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_theta_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_drive_optimal_params(force: bool = False) -> None:
    """Optimal parameter evolution vs θ."""
    csv_p = _csv_path("drive-theta-scan", date=DRIVE_DATE)
    fig_p = _fig_path("drive-optimal-params", date=DRIVE_DATE)

    result = DriveThetaScanResult.from_csv(csv_p)
    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


# ──────────────────────────────────────────────
# Phase-Modulated Ancilla Drive Experiments (2026-05-19 report)
# ──────────────────────────────────────────────

PHASE_DATE = "2026-05-19"
PHASE_THETA_VALS = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
# The report specifies 501×501 grids (vs 201×201 in the fixed-drive scan)
# for publication-quality slices.
PHASE_N_GRID = 201


# ============================================================================
# Parallel dispatch helpers
# ============================================================================


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

    import multiprocessing as _mp

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_item = {
            executor.submit(worker_fn, item): item for item in item_list
        }
        for future in concurrent.futures.as_completed(fut_to_item):
            item = fut_to_item[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item}: {exc}")
                raise


def generate_phase_decoupled_baseline(force: bool = False) -> None:
    """Phase-modulated decoupled baseline verification."""
    csv_p = _csv_path("phase-decoupled-baseline", date=PHASE_DATE)
    fig_p = _fig_path("phase-decoupled-baseline", date=PHASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_csv(csv_p)
    else:
        print("[run]  Computing phase-modulated decoupled baseline...")
        result = compute_phase_modulated_decoupled_baseline()
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def _run_phase_2d_slice(
    theta: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a phase-modulated 2D slice scan for a single θ value."""
    tag = f"phase-2d-slice-{slice_type}-azz-theta{theta}"
    csv_p = _csv_path(tag, date=PHASE_DATE)
    fig_p = _fig_path(tag, date=PHASE_DATE)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_csv(csv_p)
    else:
        print(f"  [run]  Computing phase ({slice_type}, a_zz) slice at θ={theta}...")
        result = phase_modulated_2d_slice(
            theta=theta,
            slice_type=slice_type,
            n_drive=PHASE_N_GRID,
            n_azz=PHASE_N_GRID,
        )
        result.save_csv(csv_p)
        print(f"  [save] {csv_p}")

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
    csv_p = _csv_path(tag, date=PHASE_DATE)
    fig_p = _fig_path(tag, date=PHASE_DATE)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveRandomSearchResult.from_csv(csv_p)
    else:
        print(f"  [run]  Running phase 4D random search at θ={theta} "
              f"(500 samples)...")
        result = phase_modulated_random_search(
            theta=theta,
            n_samples=500,
            seed=42,
        )
        result.save_csv(csv_p)
        print(f"  [save] {csv_p}")

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
    # Constants matching run_phase_modulated_theta_scan defaults
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
    csv_p = _csv_path("phase-theta-scan", date=PHASE_DATE)
    fig_p = _fig_path("phase-theta-scan", date=PHASE_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveThetaScanResult.from_csv(csv_p)
    else:
        n = len(PHASE_THETA_VALS)
        print(f"[run]  Computing phase θ-scan for {n} θ values (parallel)...")

        import multiprocessing as _mp

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
        sql_vals = [1.0 / 10.0] * len(theta_arr)  # SQL = 1/T_H, T_H=10

        result = DriveThetaScanResult(
            theta_values=theta_arr,
            best_params_per_theta=best_params,
            best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
            sql_values=np.array(sql_vals, dtype=float),
            expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
            variance_Jz_per_theta=np.array(var_vals, dtype=float),
        )
        result.save_csv(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_theta_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_optimal_params(force: bool = False) -> None:
    """Phase-modulated optimal parameter evolution vs θ."""
    csv_p = _csv_path("phase-theta-scan", date=PHASE_DATE)
    fig_p = _fig_path("phase-optimal-params", date=PHASE_DATE)

    result = DriveThetaScanResult.from_csv(csv_p)
    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_combined_sensitivity(force: bool = False) -> None:
    """Combined sensitivity plot + NM expectation/variance plot.

    Reads existing per-θ CSVs (2D slices, random search) and the theta-scan
    result to produce two figures:
        - phase-combined-sensitivity.svg : Δθ vs θ for all methods + SQL
        - phase-nm-expectation-variance.svg : ⟨J_z^S⟩ and Var(J_z^S) at NM optimum

    This must be run AFTER the data-generation steps
    (phase-2d-slice-*, phase-random-search, phase-theta-scan).
    """
    fig_p1 = _fig_path("phase-combined-sensitivity", date=PHASE_DATE)
    fig_p2 = _fig_path("phase-nm-expectation-variance", date=PHASE_DATE)

    # Load NM result
    theta_scan_csv = _csv_path("phase-theta-scan", date=PHASE_DATE)
    if not theta_scan_csv.exists():
        print("[skip] phase-theta-scan.csv does not exist; "
              "run 'phase-theta-scan' first")
        return
    nm_result = DriveThetaScanResult.from_csv(theta_scan_csv)

    # Use PHASE_THETA_VALS as source of truth (exact float values used for
    # filenames in the generation steps).
    theta_vals = np.array(PHASE_THETA_VALS, dtype=float)
    n_theta = len(theta_vals)

    # Align NM result arrays by index (same ordering as PHASE_THETA_VALS).
    best_nm = np.full(n_theta, np.nan)
    exp_vals = np.full(n_theta, np.nan)
    var_vals = np.full(n_theta, np.nan)

    nm_theta = nm_result.theta_values
    if len(nm_theta) >= n_theta:
        # Direct index-based alignment (theta-scan was run with same ordering)
        for i in range(n_theta):
            best_nm[i] = float(nm_result.best_delta_theta_per_theta[i])
            exp_vals[i] = float(nm_result.expectation_Jz_per_theta[i])
            var_vals[i] = float(nm_result.variance_Jz_per_theta[i])

    # Collect per-θ minima from 2D slice and random search CSVs
    best_ax = np.full(n_theta, np.nan)
    best_ay = np.full(n_theta, np.nan)
    best_rs = np.full(n_theta, np.nan)

    def _safe_grid_min(grid: np.ndarray) -> float:
        """Return the minimum finite value of *grid*, or NaN if none exist."""
        finite_vals = grid[np.isfinite(grid)]
        if finite_vals.size == 0:
            return np.nan
        return float(np.min(finite_vals))

    for i, theta in enumerate(theta_vals):
        # 2D slices
        for slice_type, best_arr in [("ax", best_ax), ("ay", best_ay)]:
            tag = f"phase-2d-slice-{slice_type}-azz-theta{theta}"
            csv_p = _csv_path(tag, date=PHASE_DATE)
            if csv_p.exists():
                result_slice = Drive2DSliceResult.from_csv(csv_p)
                best_arr[i] = _safe_grid_min(result_slice.delta_theta_grid)

        # Random search
        tag_rs = f"phase-random-search-theta{theta}"
        csv_p_rs = _csv_path(tag_rs, date=PHASE_DATE)
        if csv_p_rs.exists():
            result_rs = DriveRandomSearchResult.from_csv(csv_p_rs)
            best_rs[i] = result_rs.best_delta_theta

    # Debug: confirm how many valid θ points each method collected
    print(f"  [debug] best_ax finite: {np.sum(np.isfinite(best_ax))} / {n_theta}")
    print(f"  [debug] best_ay finite: {np.sum(np.isfinite(best_ay))} / {n_theta}")
    print(f"  [debug] best_rs finite: {np.sum(np.isfinite(best_rs))} / {n_theta}")
    print(f"  [debug] best_nm finite: {np.sum(np.isfinite(best_nm))} / {n_theta}")

    # SQL reference (constant)
    sql_vals = np.full(n_theta, 0.1)

    # Generate combined sensitivity plot
    plot_drive_combined_sensitivity(
        theta_vals, best_ax, best_ay, best_rs, best_nm, sql_vals, fig_p1,
    )
    print(f"[fig]  {fig_p1}")

    # Generate NM expectation/variance plot
    plot_drive_nm_expectation_variance(
        theta_vals, exp_vals, var_vals, fig_p2,
    )
    print(f"[fig]  {fig_p2}")


def generate_phase_fraction_below_sql(force: bool = False) -> None:
    """Fraction of parameter space below SQL vs θ for all methods.

    Reads existing per-θ CSVs (2D slices, random search) and computes
    the fraction of points whose Δθ falls below the SQL for each θ.
    """
    fig_p = _fig_path("phase-fraction-below-sql", date=PHASE_DATE)

    theta_vals = np.array(PHASE_THETA_VALS, dtype=float)
    n_theta = len(theta_vals)

    fractions_ax = np.full(n_theta, np.nan)
    fractions_ay = np.full(n_theta, np.nan)
    fractions_rs = np.full(n_theta, np.nan)

    for i, theta in enumerate(theta_vals):
        # (a_x, a_zz) slice
        tag_ax = f"phase-2d-slice-ax-azz-theta{theta}"
        csv_ax = _csv_path(tag_ax, date=PHASE_DATE)
        if csv_ax.exists():
            result = Drive2DSliceResult.from_csv(csv_ax)
            fractions_ax[i] = (
                np.sum(result.delta_theta_grid < result.sql)
                / result.delta_theta_grid.size
            )

        # (a_y, a_zz) slice
        tag_ay = f"phase-2d-slice-ay-azz-theta{theta}"
        csv_ay = _csv_path(tag_ay, date=PHASE_DATE)
        if csv_ay.exists():
            result = Drive2DSliceResult.from_csv(csv_ay)
            fractions_ay[i] = (
                np.sum(result.delta_theta_grid < result.sql)
                / result.delta_theta_grid.size
            )

        # 4D random search
        tag_rs = f"phase-random-search-theta{theta}"
        csv_rs = _csv_path(tag_rs, date=PHASE_DATE)
        if csv_rs.exists():
            result = DriveRandomSearchResult.from_csv(csv_rs)
            fractions_rs[i] = (
                np.sum(result.delta_theta_values < result.sql)
                / len(result.delta_theta_values)
            )

    plot_drive_fraction_below_sql(
        theta_vals, fractions_ax, fractions_ay, fractions_rs, fig_p,
    )
    print(f"[fig]  {fig_p}")


# ──────────────────────────────────────────────
# Cross-experiment comparison figure (2026-05-18 vs 2026-05-19)
# ──────────────────────────────────────────────


def generate_phase_cross_experiment_comparison(force: bool = False) -> None:
    """Comparison of fixed-drive (2026-05-18) vs modulated-drive (2026-05-19)
    θ-scan results.

    Loads both CSVs, interpolates the sparse 2026-05-18 data to the fine
    50-point θ grid of the 2026-05-19 scan, and produces a 2×1 figure
    showing Δθ vs θ (upper) and the ratio Δθ_19/Δθ_18 (lower).
    """
    fig_p = _fig_path("phase-cross-experiment-comparison", date=PHASE_DATE)

    # Load modulated-drive result (2026-05-19, 50 points)
    csv_19 = _csv_path("phase-theta-scan", date=PHASE_DATE)
    if not csv_19.exists():
        print("[skip] 2026-05-19-phase-theta-scan.csv does not exist; "
              "run 'phase-theta-scan' first")
        return
    result_19 = DriveThetaScanResult.from_csv(csv_19)

    # Load fixed-drive result (2026-05-18, 5 points)
    csv_18 = _csv_path("drive-theta-scan", date=DRIVE_DATE)
    if not csv_18.exists():
        print("[skip] 2026-05-18-drive-theta-scan.csv does not exist; "
              "run 'drive-theta-scan' first")
        return
    result_18 = DriveThetaScanResult.from_csv(csv_18)

    # Use the fine 50-point θ grid from the modulated-drive scan
    theta_fine = result_19.theta_values  # 50 points

    # Interpolate the sparse fixed-drive data onto the fine grid
    theta_coarse = result_18.theta_values  # 5 points
    delta_18_coarse = result_18.best_delta_theta_per_theta
    delta_18_fine = np.interp(theta_fine, theta_coarse, delta_18_coarse)

    # SQL is constant across both experiments (1/T_H = 0.1)
    sql_fine = result_19.sql_values  # already on the fine grid

    plot_drive_cross_experiment_comparison(
        theta_values=theta_fine,
        best_delta_19=result_19.best_delta_theta_per_theta,
        best_delta_18=delta_18_fine,
        sql_values=sql_fine,
        save_path=fig_p,
    )
    print(f"[fig]  {fig_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing CSVs)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset, e.g. 'decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist (individual generate functions also
    # create them via save_csv / plot helpers, but a top-level mkdir avoids
    # race conditions in concurrent runs).
    for date_str in [BASE_DATE, "2026-05-18", "2026-05-19"]:
        (REPORTS_DIR / date_str / "raw_data").mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / date_str / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "decoupled-baseline": generate_decoupled_baseline,
        "theta-scan": generate_theta_scan,
        "interaction-robustness": generate_interaction_robustness,
        "alpha-reoptimisation": generate_alpha_reoptimisation,
        "covariance-analysis": generate_covariance_analysis,
        "n-scaling": generate_n_scaling,
        "m-scaling": generate_m_scaling,
        "alpha-scan-nm": generate_weighted_alpha_scan,
        "drive-decoupled-baseline": generate_drive_decoupled_baseline,
        "drive-2d-slice-ax-azz": generate_drive_2d_slice_ax_azz,
        "drive-2d-slice-ay-azz": generate_drive_2d_slice_ay_azz,
        "drive-random-search": generate_drive_random_search,
        "drive-theta-scan": generate_drive_theta_scan,
        "drive-optimal-params": generate_drive_optimal_params,
        # Phase-modulated experiments (2026-05-19 report)
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
