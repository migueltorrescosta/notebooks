"""
Generation pipeline and CLI for the 2026-07-01 Mixed omega-Modulated Drive report.

Contains all data/figure generation functions and the CLI entry point that
were extracted from ``partial_omega_modulated_drive.py`` to reduce file length
and cyclomatic complexity.

Usage:
    uv run python reports/20260701/partial_omega_modulated_drive.py --force
    uv run python reports/20260701/partial_omega_modulated_drive.py --only decoupled-baseline
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import multiprocessing as _mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import from the (digit-prefixed) core module via importlib because Python's
# static import syntax cannot handle package components starting with a digit.
_core = importlib.import_module("reports.20260701.partial_omega_modulated_drive")
DEFAULT_PSI0 = _core.DEFAULT_PSI0
DEFAULT_T_BS = _core.DEFAULT_T_BS
DEFAULT_T_HOLD = _core.DEFAULT_T_HOLD
DRIVE_BOUNDS = _core.DRIVE_BOUNDS
_configure_environment = _core._configure_environment
_make_rs_nm_fns = _core._make_rs_nm_fns
compute_all_sensitivities = _core.compute_all_sensitivities
partial_2d_slice = _core.partial_2d_slice
partial_random_search = _core.partial_random_search

from src.analysis.ancilla_drive_results import (  # noqa: E402
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_drive_scans import (  # noqa: E402
    compute_drive_decoupled_baseline,
)
from src.analysis.ancilla_optimization import (  # noqa: E402
    build_two_qubit_operators,
)
from src.analysis.optimisation_pipeline import (  # noqa: E402
    TwoPhaseConfig,
    run_two_phase_pipeline,
)
from src.utils.parallel import parallel_map  # noqa: E402
from src.utils.paths import report_path_fn  # noqa: E402
from src.visualization.ancilla_drive_plots import (  # noqa: E402
    plot_combined_sensitivity,
    plot_drive_2d_slice_heatmap,
    plot_drive_decoupled_baseline,
    plot_drive_nm_expectation_variance,
    plot_drive_omega_scan,
    plot_drive_random_search_histogram,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
PARTIAL_DATE = "20260701"
PARTIAL_OMEGA_VALS = [round(v, 2) for v in np.linspace(0.01, 5.0, 500).tolist()]
PARTIAL_N_GRID = 201

_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, PARTIAL_DATE)

# Guard flag: set to True to suppress unused SVG generation.
# These SVGs (2D slice heatmaps, decoupled baseline bar chart, NM expectation/
# variance plot, and 500 per-omega random-search histograms) are not referenced
# by the report markdown. Set to False via --include-extra-figures to regenerate.
_SUPPRESS_EXTRA_FIGURES = True


# ============================================================================
# Plot: trio comparison (EP = CFI vs QFI)
# ============================================================================


def plot_trio_comparison(
    omega_values: np.ndarray,
    delta_ep: np.ndarray,
    delta_cfi: np.ndarray,
    delta_qfi: np.ndarray,
    sql: float,
    save_path: str | Path,
    figsize: tuple[float, float] = (9, 5),
) -> Path:
    """EP (=CFI) vs QFI trio comparison plot at the optimal parameters.

    Three lines on a single figure:
    - Delta-omega_EP (solid blue)
    - Delta-omega_CFI (dashed green, should perfectly overlap EP)
    - Delta-omega_QFI (solid orange, should be <= EP)

    The SQL is shown as a horizontal dashed reference line.

    Args:
        omega_values: Array of omega values.
        delta_ep: Delta-omega_EP at each omega.
        delta_cfi: Delta-omega_CFI at each omega.
        delta_qfi: Delta-omega_QFI at each omega.
        sql: SQL reference value.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(
        y=sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql:.4f}",
    )

    valid_ep = np.isfinite(delta_ep)
    valid_cfi = np.isfinite(delta_cfi)
    valid_qfi = np.isfinite(delta_qfi)

    if np.any(valid_ep):
        ax.plot(
            omega_values[valid_ep],
            delta_ep[valid_ep],
            "o-",
            color="C0",
            markersize=6,
            linewidth=2,
            label=r"$\Delta\omega_{\mathrm{EP}}$",
        )
    if np.any(valid_cfi):
        ax.plot(
            omega_values[valid_cfi],
            delta_cfi[valid_cfi],
            "s--",
            color="C1",
            markersize=5,
            linewidth=1.5,
            label=r"$\Delta\omega_{\mathrm{CFI}}$",
        )
    if np.any(valid_qfi):
        ax.plot(
            omega_values[valid_qfi],
            delta_qfi[valid_qfi],
            "D-",
            color="C2",
            markersize=6,
            linewidth=2,
            label=r"$\Delta\omega_{\mathrm{QFI}}$",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title("EP vs CFI vs QFI: partial $\\omega$-modulated drive")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Generator functions
# ============================================================================


def generate_decoupled_baseline(force: bool = False) -> None:
    """Decoupled baseline verification."""
    csv_p = _parquet_path("decoupled-baseline")
    fig_p = _fig_path("decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline...")
        result = compute_drive_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    if not _SUPPRESS_EXTRA_FIGURES:
        plot_drive_decoupled_baseline(result, fig_p)
        print(f"[fig]  {fig_p}")
    else:
        print(f"[skip] {fig_p.name} (extra SVG suppressed)")


def _run_partial_2d_slice(
    omega: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a 2D slice scan for a single omega value."""
    tag = f"2d-slice-{slice_type}-azz-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing ({slice_type}, a_zz) slice at omega={omega}...")
        result = partial_2d_slice(
            omega=omega,
            slice_type=slice_type,
            n_drive=PARTIAL_N_GRID,
            n_azz=PARTIAL_N_GRID,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    if not _SUPPRESS_EXTRA_FIGURES:
        plot_drive_2d_slice_heatmap(result, fig_p)
        print(f"  [fig]  {fig_p}")
    else:
        print(f"  [skip] {fig_p.name} (extra SVG suppressed)")


def generate_2d_slice_ax_azz(force: bool = False) -> None:
    """2D slice scan over (a_x, a_zz) at omega=0.2 (as specified in report)."""
    print("[run]  (a_x, a_zz) slice at omega=0.2")
    _run_partial_2d_slice(omega=0.2, slice_type="ax", force=force)


def generate_2d_slice_ay_azz(force: bool = False) -> None:
    """2D slice scan over (a_y, a_zz) at omega=0.2 (as specified in report)."""
    print("[run]  (a_y, a_zz) slice at omega=0.2")
    _run_partial_2d_slice(omega=0.2, slice_type="ay", force=force)


def _run_partial_random_search(omega: float, force: bool) -> None:
    """Run a 4D random search for a single omega value."""
    tag = f"random-search-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveRandomSearchResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Running 4D random search at omega={omega} (500 samples)...")
        result = partial_random_search(
            omega=omega,
            n_samples=500,
            seed=42,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    if not _SUPPRESS_EXTRA_FIGURES:
        plot_drive_random_search_histogram(result, fig_p)
        print(f"  [fig]  {fig_p}")
    else:
        print(f"  [skip] {fig_p.name} (extra SVG suppressed)")


def generate_random_search(force: bool = False) -> None:
    """4D random search at all omega values (parallel)."""
    n = len(PARTIAL_OMEGA_VALS)
    print(f"[run]  4D random search at {n} omega values (parallel)")
    worker = partial(_run_partial_random_search, force=force)
    parallel_map(worker, PARTIAL_OMEGA_VALS, desc="random search")
    _merge_random_search_parquets()


def _merge_random_search_parquets() -> None:
    """Merge all per-omega random-search parquets into a single aggregated file.

    Reads all ``random-search-omega*.parquet`` files from the raw_data directory,
    concatenates them into one DataFrame, saves as ``{date}-random-search.parquet``,
    then deletes the per-omega files.
    """
    raw_dir = REPORTS_DIR / PARTIAL_DATE / "raw_data"
    date_prefix = PARTIAL_DATE
    pattern = f"{date_prefix}-random-search-omega*.parquet"
    per_omega_files = sorted(raw_dir.glob(pattern))

    if not per_omega_files:
        print("[skip] No per-omega random-search parquets found to merge")
        return

    merged_path = raw_dir / f"{date_prefix}-random-search.parquet"
    dfs = [pd.read_parquet(f) for f in per_omega_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_parquet(merged_path, index=False)
    n_merged = len(merged_df)
    n_files = len(per_omega_files)

    for f in per_omega_files:
        f.unlink()

    print(f"[merge] {n_files} files -> {merged_path.name} ({n_merged} rows)")


def _run_partial_omega_scan_single(omega: float) -> dict[str, float | np.ndarray]:
    """Run random search + NM refinement for a single omega value."""
    ops = build_two_qubit_operators()
    _rs_fn, _nm_fn = _make_rs_nm_fns(omega, ops)

    config = TwoPhaseConfig(
        n_random=500,
        n_nm_refine=50,
        nm_maxiter=5000,
        seed=42,
        bounds=DRIVE_BOUNDS,
    )

    best_nm, _ = run_two_phase_pipeline(
        random_search_fn=_rs_fn,
        nm_fn=_nm_fn,
        config=config,
        seed=42 + int(omega * 1000),
    )

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
    parquet_path: Path,
    force: bool,
) -> DriveOmegaScanResult | None:
    """Check if cached parquet exists and load it."""
    if not parquet_path.exists() or force:
        return None
    print(f"[skip] {parquet_path.name} exists (use --force to overwrite)")
    return DriveOmegaScanResult.from_parquet(parquet_path)


def _run_parallel_omega_scan_core(
    omega_values: list[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Execute the parallel omega-scan computation.

    Submits ``_run_partial_omega_scan_single`` for each omega value to a
    process pool and collects results.

    Args:
        omega_values: Omega values to scan.

    Returns:
        List of result dicts sorted by omega value.
    """
    max_workers = min(8, os.cpu_count() or 1)
    print(f"  [parallel] Using {max_workers} workers for omega-scan")

    per_omega_results: list[dict[str, Any]] = []
    mp_ctx = _mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_omega = {
            executor.submit(_run_partial_omega_scan_single, omega): omega
            for omega in omega_values
        }
        for future in concurrent.futures.as_completed(fut_to_omega):
            omega = fut_to_omega[future]
            try:
                per_omega_results.append(future.result())
                print(f"  [done] omega={omega}")
            except Exception as exc:
                print(f"  [ERROR] omega={omega}: {exc}")
                raise

    per_omega_results.sort(key=lambda r: float(r["omega"]))
    return per_omega_results


def _assemble_omega_scan_result(
    per_omega_results: list[dict[str, Any]],
) -> DriveOmegaScanResult:
    """Convert a list of per-omega result dicts into a DriveOmegaScanResult.

    Args:
        per_omega_results: Sorted list of result dicts.

    Returns:
        DriveOmegaScanResult with all per-omega data assembled.
    """
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
        sql_values=np.full(len(omega_arr), 1.0 / DEFAULT_T_HOLD, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
    )


def _compute_omega_scan_core(
    omega_values: list[float] | np.ndarray | None = None,
) -> DriveOmegaScanResult:
    """Run the parallel omega-scan computation.

    Args:
        omega_values: Omega values to scan.  Defaults to ``PARTIAL_OMEGA_VALS``.
    """
    if omega_values is None:
        omega_values = PARTIAL_OMEGA_VALS
    n = len(omega_values)
    print(f"[run]  Computing omega-scan for {n} omega values (parallel)...")
    per_omega_results = _run_parallel_omega_scan_core(omega_values)
    return _assemble_omega_scan_result(per_omega_results)


def _save_and_plot_omega_scan(
    result: DriveOmegaScanResult,
    parquet_path: Path,
    fig_path: Path,
) -> None:
    """Save result to Parquet and generate the omega-scan figure."""
    result.save_parquet(parquet_path)
    print(f"[save] {parquet_path}")

    plot_drive_omega_scan(result, fig_path)
    print(f"[fig]  {fig_path}")


def generate_omega_scan(force: bool = False) -> None:
    """Omega-scan with Nelder-Mead refinement (parallel)."""
    csv_p = _parquet_path("omega-scan")
    fig_p = _fig_path("omega-scan")

    result = _check_omega_scan_cache(csv_p, force)
    if result is None:
        result = _compute_omega_scan_core(PARTIAL_OMEGA_VALS)
        _save_and_plot_omega_scan(result, csv_p, fig_p)
    else:
        plot_drive_omega_scan(result, fig_p)
        print(f"[fig]  {fig_p}")


def _safe_grid_min(grid: np.ndarray) -> float:
    """Return the minimum finite value in *grid*, or NaN if none are finite."""
    finite_vals = grid[np.isfinite(grid)]
    if finite_vals.size == 0:
        return np.nan
    return float(np.min(finite_vals))


def _extract_nm_data(
    nm_result: DriveOmegaScanResult,
    n_omega: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract NM-optimised sensitivity, expectation, and variance arrays.

    Args:
        nm_result: Omega-scan result with per-omega NM data.
        n_omega: Number of omega values.

    Returns:
        Tuple of (best_nm, exp_vals, var_vals) arrays, each of length *n_omega*.
    """
    best_nm = np.full(n_omega, np.nan)
    exp_vals = np.full(n_omega, np.nan)
    var_vals = np.full(n_omega, np.nan)

    nm_omega = nm_result.omega_values
    if len(nm_omega) >= n_omega:
        for i in range(n_omega):
            best_nm[i] = float(nm_result.best_delta_omega_per_omega[i])
            exp_vals[i] = float(nm_result.expectation_Jz_per_omega[i])
            var_vals[i] = float(nm_result.variance_Jz_per_omega[i])

    return best_nm, exp_vals, var_vals


def _load_2d_slice_best(omega_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load best sensitivity from 2D slice scans for a_x and a_y.

    Reads existing ``2d-slice-ax-azz-omega*`` and ``2d-slice-ay-azz-omega*``
    parquet files and extracts the minimum finite sensitivity per omega.

    Args:
        omega_vals: Omega values whose slice data to load.

    Returns:
        Tuple of (best_ax, best_ay) arrays, each of length ``len(omega_vals)``.
    """
    n_omega = len(omega_vals)
    best_ax = np.full(n_omega, np.nan)
    best_ay = np.full(n_omega, np.nan)

    for i, omega in enumerate(omega_vals):
        for slice_type, best_arr in [("ax", best_ax), ("ay", best_ay)]:
            tag = f"2d-slice-{slice_type}-azz-omega{omega}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                result_slice = Drive2DSliceResult.from_parquet(csv_p)
                best_arr[i] = _safe_grid_min(result_slice.delta_omega_grid)

    return best_ax, best_ay


def _load_random_search_best(omega_vals: np.ndarray) -> np.ndarray:
    """Load the best (minimum) random-search sensitivity per omega.

    Reads the aggregated ``random-search`` parquet file and groups by
    ``omega_value`` to obtain the minimum ``delta_omega`` per omega.

    Args:
        omega_vals: Omega values whose RS data to load.

    Returns:
        Array of best RS sensitivity values, length ``len(omega_vals)``.
        Entries for which no RS data exists remain NaN.
    """
    n_omega = len(omega_vals)
    best_rs = np.full(n_omega, np.nan)

    rs_agg_path = _parquet_path("random-search")
    if rs_agg_path.exists():
        rs_df = pd.read_parquet(rs_agg_path)
        best_per_omega = rs_df.groupby("omega_value", sort=True)["delta_omega"].min()
        for i, omega in enumerate(omega_vals):
            if omega in best_per_omega.index:
                best_rs[i] = best_per_omega.loc[omega]

    return best_rs


def generate_combined_sensitivity(force: bool = False) -> None:
    """Combined sensitivity plot + NM expectation/variance plot."""
    fig_p1 = _fig_path("combined-sensitivity")
    fig_p2 = _fig_path("nm-expectation-variance")

    omega_scan_pq = _parquet_path("omega-scan")
    if not omega_scan_pq.exists():
        print("[skip] omega-scan.parquet does not exist; run 'omega-scan' first")
        return
    nm_result = DriveOmegaScanResult.from_parquet(omega_scan_pq)

    omega_vals = np.array(PARTIAL_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    best_nm, exp_vals, var_vals = _extract_nm_data(nm_result, n_omega)
    best_ax, best_ay = _load_2d_slice_best(omega_vals)
    best_rs = _load_random_search_best(omega_vals)

    print(f"  [debug] best_ax finite: {np.sum(np.isfinite(best_ax))} / {n_omega}")
    print(f"  [debug] best_ay finite: {np.sum(np.isfinite(best_ay))} / {n_omega}")
    print(f"  [debug] best_rs finite: {np.sum(np.isfinite(best_rs))} / {n_omega}")
    print(f"  [debug] best_nm finite: {np.sum(np.isfinite(best_nm))} / {n_omega}")

    sql_vals = np.full(n_omega, 0.1)

    plot_combined_sensitivity(
        omega_vals,
        best_ax,
        best_ay,
        best_rs,
        best_nm,
        sql_vals,
        fig_p1,
    )
    print(f"[fig]  {fig_p1}")

    if not _SUPPRESS_EXTRA_FIGURES:
        plot_drive_nm_expectation_variance(omega_vals, exp_vals, var_vals, fig_p2)
        print(f"[fig]  {fig_p2}")
    else:
        print("[skip] nm-expectation-variance SVG (extra SVG suppressed)")


def generate_trio_comparison(force: bool = False) -> None:
    """EP (=CFI) vs QFI trio comparison plot at the optimal parameters.

    Reads the omega-scan NM optimum and computes all three sensitivities
    at each omega using the NM-optimal parameters.
    """
    fig_p = _fig_path("trio-comparison")

    omega_scan_pq = _parquet_path("omega-scan")
    if not omega_scan_pq.exists():
        print("[skip] omega-scan.parquet does not exist; run 'omega-scan' first")
        return

    nm_result = DriveOmegaScanResult.from_parquet(omega_scan_pq)

    omega_vals = np.array(PARTIAL_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    delta_ep = np.full(n_omega, np.inf, dtype=float)
    delta_cfi = np.full(n_omega, np.inf, dtype=float)
    delta_qfi = np.full(n_omega, np.inf, dtype=float)

    ops = build_two_qubit_operators()

    for i, omega in enumerate(omega_vals):
        if i >= len(nm_result.best_params_per_omega):
            continue
        ax, ay, az, azz = nm_result.best_params_per_omega[i]
        try:
            sens = compute_all_sensitivities(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_HOLD,
                omega,
                ax,
                ay,
                az,
                azz,
                ops,
            )
            delta_ep[i] = sens["delta_omega_ep"]
            delta_cfi[i] = sens["delta_omega_cfi"]
            delta_qfi[i] = sens["delta_omega_qfi"]
        except Exception as exc:
            print(f"  [WARN] omega={omega}: sensitivity computation failed: {exc}")

    sql = 1.0 / DEFAULT_T_HOLD
    plot_trio_comparison(omega_vals, delta_ep, delta_cfi, delta_qfi, sql, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def run_cli() -> None:
    """CLI entry point dispatched from the core module's ``main()``."""
    global _SUPPRESS_EXTRA_FIGURES  # noqa: PLW0603  # allow module-level flag override

    _configure_environment()
    parser = argparse.ArgumentParser(
        description="Generate 2026-07-01 report figures and Parquet data",
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
        help="Generate only one dataset, e.g. 'decoupled-baseline'",
    )
    parser.add_argument(
        "--include-extra-figures",
        action="store_true",
        default=False,
        help="Generate extra SVG figures not referenced by the report: "
        "2D slice heatmaps, decoupled baseline bar chart, "
        "NM expectation/variance plot, and 500 per-omega random-search "
        "histograms (default: suppressed)",
    )
    args = parser.parse_args()

    if args.include_extra_figures:
        _SUPPRESS_EXTRA_FIGURES = False

    (REPORTS_DIR / PARTIAL_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / PARTIAL_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "decoupled-baseline": generate_decoupled_baseline,
        "2d-slice-ax-azz": generate_2d_slice_ax_azz,
        "2d-slice-ay-azz": generate_2d_slice_ay_azz,
        "random-search": generate_random_search,
        "omega-scan": generate_omega_scan,
        "combined-sensitivity": generate_combined_sensitivity,
        "trio-comparison": generate_trio_comparison,
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
