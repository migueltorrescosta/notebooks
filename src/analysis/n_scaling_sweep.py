"""
N-scaling sweep orchestration and figure generation for multi-particle
phase-modulated drive reports.

Provides two orchestrators:

* :func:`generate_n_scaling_plots` — single-omega log-log N-scaling
  figures from an existing sweep-result Parquet (promoted from
  20260522/20260523/20260525).
* :func:`run_n_scaling_scan` — full (N, ω) grid scan with checkpoint
  recovery and figure generation (promoted from 20260611/20260612).

Also exports helper functions used by the checkpoint-recovery pipeline
(:func:`build_n_result_from_row`, :func:`build_n_result_from_dict`,
:func:`n_group_key`, :func:`make_checkpoint_name_fn`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

from src.analysis.checkpoint_recovery import load_checkpoints, run_pending_groups
from src.analysis.n_scaling_result import NScalingResult, NScalingScanResult
from src.visualization.scaling_plots import (
    plot_n_scaling_optimal_params,
    plot_n_scaling_ratio,
    plot_n_scaling_sensitivity,
    plot_n_scaling_single_omega,
)

# ============================================================================
# Checkpoint-recovery helpers (used by run_n_scaling_scan)
# ============================================================================


def build_n_result_from_row(row_dict: dict[str, Any]) -> NScalingResult:
    """Build an ``NScalingResult`` from a Pandas row dictionary.

    Uses ``.get()`` with safe defaults for optional fields so that older
    checkpoint Parquet files without those columns can still be loaded
    gracefully.
    """
    return NScalingResult(
        N=int(row_dict["N"]),
        omega=float(row_dict["omega"]),
        delta_omega_opt=float(row_dict["delta_omega_opt"]),
        sql=float(row_dict["sql"]),
        ratio=float(row_dict["ratio"]),
        a_x_opt=float(row_dict["a_x_opt"]),
        a_y_opt=float(row_dict["a_y_opt"]),
        a_z_opt=float(row_dict["a_z_opt"]),
        a_zz_opt=float(row_dict["a_zz_opt"]),
        expectation_Jz=float(row_dict.get("expectation_Jz", 0.0)),
        variance_Jz=float(row_dict.get("variance_Jz", 0.0)),
        success=bool(int(row_dict.get("success", 0))),
        nfev=int(row_dict.get("nfev", 0)),
    )


def build_n_result_from_dict(rdict: dict[str, Any]) -> NScalingResult:
    """Build an ``NScalingResult`` from a worker dict.

    Uses direct indexing (``rdict[key]``) so missing keys raise immediately.
    """
    return NScalingResult(
        N=int(rdict["N"]),
        omega=float(rdict["omega"]),
        delta_omega_opt=float(rdict["delta_omega_opt"]),
        sql=float(rdict["sql"]),
        ratio=float(rdict["ratio"]),
        a_x_opt=float(rdict["a_x_opt"]),
        a_y_opt=float(rdict["a_y_opt"]),
        a_z_opt=float(rdict["a_z_opt"]),
        a_zz_opt=float(rdict["a_zz_opt"]),
        expectation_Jz=float(rdict["expectation_Jz"]),
        variance_Jz=float(rdict["variance_Jz"]),
        success=bool(rdict["success"]),
        nfev=int(rdict["nfev"]),
    )


def n_group_key(item: tuple[int, float]) -> int:
    """Extract the integer group key (N) from a ``(N, omega)`` tuple."""
    return item[0]


def make_checkpoint_name_fn(checkpoint_dir: Path) -> Callable[[Hashable], Path]:
    """Return a function that maps ``N`` to a checkpoint file path.

    The returned callable produces paths like ``N_003.parquet`` inside
    *checkpoint_dir*.
    """

    def _name_fn(key: Hashable) -> Path:
        return checkpoint_dir / f"N_{int(key):03d}.parquet"  # type: ignore[call-overload]

    return _name_fn


# ============================================================================
# N-Scaling Orchestrator
# ============================================================================


def run_n_scaling_scan(
    *,
    force: bool = False,
    run_single_n_omega: Callable[[int, float], NScalingResult],
    n_values: list[int],
    omega_values: list[float],
    parquet_path: Path,
    checkpoint_dir: Path,
    fig_ratio_path: Path,
    fig_sensitivity_path: Path,
    fig_params_path: Path,
    t_hold: float = 10.0,
) -> None:
    """Full (N, ω) grid scan with checkpoint recovery and figures.

    Each ``(N, ω)`` pair is passed to *run_single_n_omega*, which should
    run the two-phase random-search → Nelder-Mead pipeline and return an
    ``NScalingResult``.  Results are batched by N (via
    :func:`n_group_key`) and checkpointed per-N-value to allow resumption
    if the run is interrupted.

    After all pairs are processed (or loaded from checkpoints), three
    summary figures are generated: ratio, sensitivity, and optimal
    parameters.

    Args:
        force: If True, delete existing results and re-run everything.
        run_single_n_omega: Callable ``(N, omega) -> NScalingResult``.
        n_values: List of N values to scan.
        omega_values: List of ω values to scan.
        parquet_path: Path for the merged result Parquet file.
        checkpoint_dir: Directory for per-N checkpoint Parquet files.
        fig_ratio_path: Output path for the ratio-vs-N figure.
        fig_sensitivity_path: Output path for the sensitivity-vs-N figure.
        fig_params_path: Output path for the optimal-parameters figure.
        t_hold: Holding time used by *run_single_n_omega* (for reference
            lines in figures).
    """

    def _worker_fn(args: tuple[int, float]) -> dict[str, Any]:
        N, omega = args
        print(f"  [run] N={N}, ω={omega}")
        result = run_single_n_omega(N, omega)
        return {
            "N": result.N,
            "omega": result.omega,
            "delta_omega_opt": result.delta_omega_opt,
            "sql": result.sql,
            "ratio": result.ratio,
            "a_x_opt": result.a_x_opt,
            "a_y_opt": result.a_y_opt,
            "a_z_opt": result.a_z_opt,
            "a_zz_opt": result.a_zz_opt,
            "expectation_Jz": result.expectation_Jz,
            "variance_Jz": result.variance_Jz,
            "success": int(result.success),
            "nfev": result.nfev,
        }

    if parquet_path.exists() and not force:
        print(f"[skip] {parquet_path.name} exists (use --force to overwrite)")
        summary = NScalingScanResult.from_parquet(parquet_path)
    else:
        if force:
            parquet_path.unlink(missing_ok=True)
            if checkpoint_dir.exists():
                import shutil

                shutil.rmtree(checkpoint_dir)

        completed, checkpoint_results = load_checkpoints(
            checkpoint_dir,
            build_n_result_from_row,
            ["N", "omega"],
        )

        items_to_run = [
            (N, omega)
            for N in n_values
            for omega in omega_values
            if (N, omega) not in completed
        ]
        if items_to_run:
            print(f"[run] N-scaling scan: {len(items_to_run)} remaining (N, ω) pairs")
            print(f"  (batch by N value, {min(32, os.cpu_count() or 1)} workers)")
            new_results = run_pending_groups(
                items_to_run,
                checkpoint_dir,
                _worker_fn,
                build_n_result_from_dict,
                n_group_key,
                make_checkpoint_name_fn(checkpoint_dir),
                NScalingScanResult,
            )
            checkpoint_results.extend(new_results)
        else:
            print("  [skip] all pairs already completed in checkpoints")

        # Merge all checkpoint results and save final file
        summary = NScalingScanResult(results=checkpoint_results)
        summary.save_parquet(parquet_path)
        print(f"[save] {parquet_path}")

    # Generate figures
    df = summary.to_dataframe()
    plot_n_scaling_ratio(df, fig_ratio_path)
    print(f"[fig]  {fig_ratio_path}")
    plot_n_scaling_sensitivity(df, fig_sensitivity_path, t_hold=t_hold)
    print(f"[fig]  {fig_sensitivity_path}")
    plot_n_scaling_optimal_params(df, fig_params_path)
    print(f"[fig]  {fig_params_path}")


# ============================================================================
# Single-omega N-scaling figure generation
# ============================================================================


def generate_n_scaling_plots(
    force: bool = False,
    *,
    parquet_path: Path,
    result_cls: type[Any],
    omega_fig_pairs: list[tuple[float, Path]],
    include_2n_sql: bool = False,
    t_hold: float | None = None,
    label: str = "n-scaling plots",
    plot_fn: Callable[..., Any] | None = None,
) -> None:
    """Generate N-scaling log-log figures from a sweep result.

    Loads the sweep Parquet, converts it to a DataFrame, and for each
    ``(omega, fig_path)`` in *omega_fig_pairs* produces a single-omega
    Δθ-vs-N figure.

    Args:
        force: If True, overwrite existing figure files.
        parquet_path: Path to the sweep result Parquet.
        result_cls: Sweep result class with ``from_parquet`` and
            ``to_dataframe``.
        omega_fig_pairs: List of ``(omega_value, fig_save_path)`` tuples.
            One figure is produced per tuple.
        include_2n_sql: If True, also draw the 2N-SQL reference line.
        t_hold: Holding time for reference lines.  Inferred from the
            DataFrame if ``None``.
        label: Human-readable label for console output.
        plot_fn: Optional custom plot function.  When provided, called
            as ``plot_fn(df, omega_fixed, save_path, t_hold=t_hold)``
            instead of the default
            :func:`~src.visualization.scaling_plots.plot_n_scaling_single_omega`.
    """
    if not parquet_path.exists():
        print(f"[skip] Sweep data not found at {parquet_path}; run sweep first")
        return

    result = result_cls.from_parquet(parquet_path)
    df = result.to_dataframe()

    plot_func = plot_fn or _default_plot_fn
    n_skipped = 0
    n_plotted = 0

    for omega_val, fig_p in omega_fig_pairs:
        fig_p = Path(fig_p)
        fig_p.parent.mkdir(parents=True, exist_ok=True)

        if fig_p.exists() and not force:
            print(f"[skip] {fig_p.name} exists (use --force to overwrite)")
            n_skipped += 1
            continue

        plot_func(
            df,
            omega_fixed=omega_val,
            save_path=fig_p,
            t_hold=t_hold,
            include_2n_sql=include_2n_sql,
        )
        print(f"[fig]  {fig_p}")
        n_plotted += 1

    print(f"[done] {label}: {n_plotted} plotted, {n_skipped} skipped")


def _default_plot_fn(
    df: Any,
    omega_fixed: float,
    save_path: Path,
    t_hold: float | None = None,
    include_2n_sql: bool = False,
) -> None:
    """Default plot function wrapping ``plot_n_scaling_single_omega``."""
    plot_n_scaling_single_omega(
        df,
        omega_fixed=omega_fixed,
        save_path=save_path,
        t_hold=t_hold,
        include_2n_sql=include_2n_sql,
    )
