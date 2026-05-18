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
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.visualization.ancilla_plots import (  # noqa: E402
    plot_alpha_reoptimisation,
    plot_covariance_analysis,
    plot_decoupled_baseline,
    plot_interaction_robustness,
    plot_theta_scan,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DATA_DIR = REPORTS_DIR / "raw_data"
FIGURES_DIR = REPORTS_DIR / "figures"


def _csv_path(name: str) -> Path:
    return RAW_DATA_DIR / f"2026-05-15-{name}.csv"


def _fig_path(name: str) -> Path:
    return FIGURES_DIR / f"2026-05-15-{name}.svg"


def generate_decoupled_baseline(force: bool = False) -> None:
    """Sections 1 & 2: Decoupled baseline and expanded T_H bound."""
    csv_p = _csv_path("decoupled-baseline")
    fig_p = _fig_path("decoupled-baseline")

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
    csv_p = _csv_path("theta-scan")
    fig_p = _fig_path("theta-scan")

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
    csv_p = _csv_path("interaction-robustness")
    fig_p = _fig_path("interaction-robustness")

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
    csv_p = _csv_path("alpha-reoptimisation")
    fig_p = _fig_path("alpha-reoptimisation")

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


def generate_covariance_analysis(force: bool = False) -> None:
    """Section 5: Covariance analysis."""
    csv_p = _csv_path("covariance-analysis")
    fig_p = _fig_path("covariance-analysis")

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

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = {
        "decoupled-baseline": generate_decoupled_baseline,
        "theta-scan": generate_theta_scan,
        "interaction-robustness": generate_interaction_robustness,
        "alpha-reoptimisation": generate_alpha_reoptimisation,
        "covariance-analysis": generate_covariance_analysis,
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
