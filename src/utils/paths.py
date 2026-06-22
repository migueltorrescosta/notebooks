"""Shared path helpers for report data and figure files."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def parquet_path(reports_dir: Path, date: str, name: str) -> Path:
    """Return path to a raw_data Parquet file for a given report.

    Parameters
    ----------
    reports_dir : Path
        Parent directory containing report date directories (e.g., ``reports/``).
    date : str
        Report date string used as the subdirectory and filename prefix
        (e.g., ``"20260616"``).
    name : str
        Uniquely identifies the result within this report
        (e.g., ``"n-scaling-scan"``).

    Returns
    -------
    Path
        ``reports_dir / date / "raw_data" / "{date}-{name}.parquet"``
    """
    return reports_dir / date / "raw_data" / f"{date}-{name}.parquet"


def fig_path(reports_dir: Path, date: str, name: str) -> Path:
    """Return path to a figures SVG file for a given report.

    Parameters
    ----------
    reports_dir : Path
        Parent directory containing report date directories (e.g., ``reports/``).
    date : str
        Report date string used as the subdirectory and filename prefix
        (e.g., ``"20260616"``).
    name : str
        Uniquely identifies the figure within this report
        (e.g., ``"ratio-vs-n"``).

    Returns
    -------
    Path
        ``reports_dir / date / "figures" / "{date}-{name}.svg"``
    """
    return reports_dir / date / "figures" / f"{date}-{name}.svg"


def report_path_fn(
    reports_dir: Path, date: str
) -> tuple[Callable[[str], Path], Callable[[str], Path]]:
    """Return ``(parquet_path_fn, fig_path_fn)`` closures bound to *date*.

    The returned functions each take a single *name* argument and delegate
    to :func:`parquet_path` and :func:`fig_path` with *reports_dir* and
    *date* already filled in.  This is the standard pattern used across
    all report ``local.py`` files::

        _parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)
        path = _parquet_path("n-scaling-scan")   # → …/raw_data/{date}-n-scaling-scan.parquet

    Parameters
    ----------
    reports_dir : Path
        Parent directory containing report date directories (e.g., ``reports/``).
    date : str
        Report date string (e.g., ``"20260616"``).

    Returns
    -------
    tuple[Callable[[str], Path], Callable[[str], Path]]
        ``(parquet_path_fn, fig_path_fn)`` — each accepts a *name* string.
    """
    return (
        partial(parquet_path, reports_dir, date),
        partial(fig_path, reports_dir, date),
    )
