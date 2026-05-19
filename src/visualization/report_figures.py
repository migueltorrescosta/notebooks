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

from src.analysis.ancilla_drive_metrology import (  # noqa: E402
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
    compute_drive_decoupled_baseline,
    drive_2d_slice,
    drive_random_search,
    run_drive_theta_scan,
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
    plot_drive_decoupled_baseline,
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
RAW_DATA_DIR = REPORTS_DIR / "raw_data"
FIGURES_DIR = REPORTS_DIR / "figures"

BASE_DATE = "2026-05-15"


def _csv_path(name: str, date: str) -> Path:
    return RAW_DATA_DIR / f"{date}-{name}.csv"


def _fig_path(name: str, date: str) -> Path:
    return FIGURES_DIR / f"{date}-{name}.svg"


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
            n_drive=501,
            n_azz=501,
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
        "n-scaling": generate_n_scaling,
        "m-scaling": generate_m_scaling,
        "alpha-scan-nm": generate_weighted_alpha_scan,
        "drive-decoupled-baseline": generate_drive_decoupled_baseline,
        "drive-2d-slice-ax-azz": generate_drive_2d_slice_ax_azz,
        "drive-2d-slice-ay-azz": generate_drive_2d_slice_ay_azz,
        "drive-random-search": generate_drive_random_search,
        "drive-theta-scan": generate_drive_theta_scan,
        "drive-optimal-params": generate_drive_optimal_params,
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
