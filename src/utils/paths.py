"""Shared path helpers for report data and figure files."""

from __future__ import annotations

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
