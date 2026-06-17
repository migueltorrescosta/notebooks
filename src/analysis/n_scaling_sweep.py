"""
N-scaling figure generation for multi-particle MZI sweep results.

Provides the :func:`generate_n_scaling_plots` orchestrator that loads
a sweep-result Parquet and produces single-omega N-scaling log-log
figures via :func:`src.visualization.scaling_plots.plot_n_scaling_single_omega`.

This module was promoted from duplicated definitions in three reports
(20260522, 20260523, 20260525) and should be reused by any
future multi-MZI sweep report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from src.visualization.scaling_plots import plot_n_scaling_single_omega


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
