"""
Local module for the 2026-05-09 ancilla-assisted non-Markovian metrology report.

All simulation logic has been promoted to ``src/physics/pseudomode_system.py``.
This module re-exports the public API and provides the CLI for data/figure
generation.

Usage:
    uv run python reports/20260509/local.py --force
    uv run python reports/20260509/local.py --experiment ancilla-nonmarkovian --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPORT_DATE = "20260509"

# Paths
REPORTS_DIR = Path(__file__).resolve().parent.parent
_REPORT_DIR = Path(__file__).resolve().parent


# =============================================================================
# CLI
# =============================================================================


def _generate_ancilla_nonmarkovian_raw_data(force: bool = False) -> None:
    """Generate raw data for Ancilla-Assisted-Non-Markovian report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla non-Markovian raw data generation")


def _generate_ancilla_nonmarkovian_figures(force: bool = False) -> None:
    """Generate figures for Ancilla-Assisted-Non-Markovian report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla non-Markovian figure generation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs for 2026-05-09 report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing CSVs)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["ancilla-nonmarkovian"],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    (_REPORT_DIR / "raw_data").mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "ancilla-nonmarkovian": (
            _generate_ancilla_nonmarkovian_raw_data,
            _generate_ancilla_nonmarkovian_figures,
        ),
    }

    if args.experiment:
        raw_fn, fig_fn = experiments[args.experiment]
        print(f"\n=== {args.experiment} ===")
        print("  Raw data ...")
        raw_fn(force=args.force)
        print("  Figures ...")
        fig_fn(force=args.force)
    else:
        for name, (raw_fn, fig_fn) in experiments.items():
            print(f"\n=== {name} ===")
            print("  Raw data ...")
            raw_fn(force=args.force)
            print("  Figures ...")
            fig_fn(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
