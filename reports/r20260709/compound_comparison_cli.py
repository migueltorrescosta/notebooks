"""
CLI and data-generation pipeline for the Bounded-Compound Comparison.

Orchestrates the standard cache-check → compute → save pipeline for
each experiment step: decoupled baseline, Scenario A ω-scan,
Scenario B ω-scan, compound ratio, and fixed-parameter ratio.

Usage:
    uv run python reports/r20260709/compound_comparison_cli.py --force
    uv run python reports/r20260709/compound_comparison_cli.py --force --sampling-mode sphere --radius 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reports.r20260709.compound_comparison import (
    compute_compound_ratio,
    compute_decoupled_baseline,
    compute_fixed_parameter_compound_ratio,
    run_scenario_a_omega_scan,
    run_scenario_b_omega_scan,
)
from reports.r20260709.compound_comparison_results import (
    DecoupledBaselineResult,
    ScenarioACompoundResult,
)
from src.analysis.ancilla_drive_results import DriveOmegaScanResult
from src.utils.paths import configure_environment, report_path_fn

# ============================================================================
# Constants
# ============================================================================

DEFAULT_T_HOLD: float = 10.0
SQL_REFERENCE: float = 1.0 / DEFAULT_T_HOLD
DEFAULT_SPHERE_RADIUS: float = 5.0
DEFAULT_SAMPLING_MODE: str = "sphere"
OMEGA_MIN: float = 0.01
OMEGA_MAX: float = 5.0
DEFAULT_N_OMEGA: int = 500

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260709"
_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Data Generation Steps
# ============================================================================


def generate_decoupled_baseline(force: bool = False) -> DecoupledBaselineResult | None:
    """Compute and save the decoupled baseline verification.

    Returns:
        The computed (or loaded) DecoupledBaselineResult, or None on cache hit.
    """
    tag = "decoupled-baseline"
    pq_path = _parquet_path(tag)
    if pq_path.exists() and not force:
        print(f"[skip] {pq_path.name} exists")
        return None

    domega_a, domega_b = compute_decoupled_baseline()
    result = DecoupledBaselineResult(
        scenarios=["A", "B"],
        delta_omega_values=np.array([domega_a, domega_b], dtype=float),
        sql_values=np.full(2, SQL_REFERENCE, dtype=float),
        ratio_to_sql_values=np.array(
            [domega_a / SQL_REFERENCE, domega_b / SQL_REFERENCE], dtype=float
        ),
        t_hold_value=DEFAULT_T_HOLD,
    )
    result.save_parquet(pq_path)
    print(f"[save] {pq_path}")
    print(f"  Scenario A: Δω = {domega_a:.6f} (ratio = {domega_a / SQL_REFERENCE:.4f})")
    print(f"  Scenario B: Δω = {domega_b:.6f} (ratio = {domega_b / SQL_REFERENCE:.4f})")
    return result


def generate_scenario_a_scan(
    omega_vals: list[float] | None = None,
    force: bool = False,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
    n_random: int = 500,
    n_refine: int = 50,
) -> None:
    """Run Scenario A ω-scan and save results.

    Args:
        omega_vals: ω values to scan (default: OMEGA_VALS).
        force: Re-run even if output exists.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius.
        n_random: Number of random search samples per omega.
        n_refine: Number of Nelder-Mead refinements per omega.
    """
    if omega_vals is None:
        omega_vals = [
            round(v, 2) for v in np.linspace(OMEGA_MIN, OMEGA_MAX, DEFAULT_N_OMEGA)
        ]
    tag_a = "scenario-a-omega-scan"
    pq_path_a = _parquet_path(tag_a)

    if pq_path_a.exists() and not force:
        print(f"[skip] {pq_path_a.name} exists")
    else:
        result_a = run_scenario_a_omega_scan(
            omega_vals,
            n_random=n_random,
            n_nm_refine=n_refine,
            sampling_mode=sampling_mode,
            radius=radius,
        )
        pq_path_a.parent.mkdir(parents=True, exist_ok=True)
        result_a.save_parquet(pq_path_a)
        print(f"[save] {pq_path_a}")

        # Print summary
        valid = np.isfinite(result_a.best_delta_omega_per_omega)
        if np.any(valid):
            best_idx = int(
                np.nanargmin(
                    np.where(valid, result_a.best_delta_omega_per_omega, np.inf)
                )
            )
            best_d = result_a.best_delta_omega_per_omega[best_idx]
            best_w = result_a.omega_values[best_idx]
            best_r = best_d / SQL_REFERENCE
            print(f"  Best Δω_A = {best_d:.6f} at ω = {best_w:.2f} ({best_r:.2f}× SQL)")


def generate_scenario_b_scan(
    omega_vals: list[float] | None = None,
    force: bool = False,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
    n_random: int = 500,
    n_refine: int = 50,
) -> None:
    """Run Scenario B ω-scan and save results.

    Args:
        omega_vals: ω values to scan (default: OMEGA_VALS).
        force: Re-run even if output exists.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius.
        n_random: Number of random search samples per omega.
        n_refine: Number of Nelder-Mead refinements per omega.
    """
    if omega_vals is None:
        omega_vals = [
            round(v, 2) for v in np.linspace(OMEGA_MIN, OMEGA_MAX, DEFAULT_N_OMEGA)
        ]
    tag_b = "scenario-b-omega-scan"
    pq_path_b = _parquet_path(tag_b)

    if pq_path_b.exists() and not force:
        print(f"[skip] {pq_path_b.name} exists")
    else:
        result_b = run_scenario_b_omega_scan(
            omega_vals,
            n_random=n_random,
            n_nm_refine=n_refine,
            sampling_mode=sampling_mode,
            radius=radius,
        )
        pq_path_b.parent.mkdir(parents=True, exist_ok=True)
        result_b.save_parquet(pq_path_b)
        print(f"[save] {pq_path_b}")

        # Print summary
        valid = np.isfinite(result_b.best_delta_omega_per_omega)
        if np.any(valid):
            best_idx = int(
                np.nanargmin(
                    np.where(valid, result_b.best_delta_omega_per_omega, np.inf)
                )
            )
            best_d = result_b.best_delta_omega_per_omega[best_idx]
            best_w = result_b.omega_values[best_idx]
            best_r = best_d / SQL_REFERENCE
            print(f"  Best Δω_B = {best_d:.6f} at ω = {best_w:.2f} ({best_r:.2f}× SQL)")


def generate_compound_ratio(force: bool = False) -> None:
    """Compute compound ratio from existing Scenario A and B results."""
    tag_cr = "compound-ratio"
    pq_path_cr = _parquet_path(tag_cr)

    tag_a = "scenario-a-omega-scan"
    tag_b = "scenario-b-omega-scan"
    pq_path_a = _parquet_path(tag_a)
    pq_path_b = _parquet_path(tag_b)

    if not pq_path_a.exists():
        print(f"[skip] {tag_a} not found — run generate_scenario_a_scan first")
        return
    if not pq_path_b.exists():
        print(f"[skip] {tag_b} not found — run generate_scenario_b_scan first")
        return

    if pq_path_cr.exists() and not force:
        print(f"[skip] {pq_path_cr.name} exists")
    else:
        result_a = ScenarioACompoundResult.from_parquet(pq_path_a)
        result_b = DriveOmegaScanResult.from_parquet(pq_path_b)
        cr = compute_compound_ratio(result_a, result_b)
        pq_path_cr.parent.mkdir(parents=True, exist_ok=True)
        cr.save_parquet(pq_path_cr)
        print(f"[save] {pq_path_cr}")

        # Print summary
        valid = np.isfinite(cr.compound_ratio)
        if np.any(valid):
            best_cr_idx = int(np.nanargmax(np.where(valid, cr.compound_ratio, 0.0)))
            best_cr = cr.compound_ratio[best_cr_idx]
            best_w = cr.omega_values[best_cr_idx]
            print(f"  Best compound ratio = {best_cr:.4f}× at ω = {best_w:.2f}")
            print(f"  Best R_A = {cr.ratio_A_to_sql[best_cr_idx]:.4f}× SQL")
            print(f"  Best R_B = {cr.ratio_B_to_sql[best_cr_idx]:.4f}× SQL")


def generate_fixed_parameter_ratio(force: bool = False) -> None:
    """Compute fixed-parameter compound ratio from existing Scenario A and B results."""
    tag_fpr = "fixed-parameter-ratio"
    pq_path_fpr = _parquet_path(tag_fpr)

    tag_a = "scenario-a-omega-scan"
    tag_b = "scenario-b-omega-scan"
    pq_path_a = _parquet_path(tag_a)
    pq_path_b = _parquet_path(tag_b)

    if not pq_path_a.exists():
        print(f"[skip] {tag_a} not found — run generate_scenario_a_scan first")
        return
    if not pq_path_b.exists():
        print(f"[skip] {tag_b} not found — run generate_scenario_b_scan first")
        return

    if pq_path_fpr.exists() and not force:
        print(f"[skip] {pq_path_fpr.name} exists")
    else:
        result_a = ScenarioACompoundResult.from_parquet(pq_path_a)
        result_b = DriveOmegaScanResult.from_parquet(pq_path_b)
        fpr = compute_fixed_parameter_compound_ratio(result_a, result_b)
        pq_path_fpr.parent.mkdir(parents=True, exist_ok=True)
        fpr.save_parquet(pq_path_fpr)
        print(f"[save] {pq_path_fpr}")

        # Print summary
        valid = np.isfinite(fpr.fixed_ratio)
        if np.any(valid):
            best_idx = int(np.nanargmax(np.where(valid, fpr.fixed_ratio, 0.0)))
            best_r = fpr.fixed_ratio[best_idx]
            best_w = fpr.omega_values[best_idx]
            print(f"  Best fixed-param ratio = {best_r:.4f}× at ω = {best_w:.2f}")
            print(
                f"  A's params: ({fpr.a_x_A[best_idx]:.2f}, "
                f"{fpr.a_y_A[best_idx]:.2f}, {fpr.a_z_A[best_idx]:.2f})"
            )
            print(f"  B's a_zz: {fpr.a_zz_B[best_idx]:.2f}")

        # Verify azz=0 decoupled limit
        decoupled_mask = np.abs(fpr.a_zz_B) < 1e-10
        if np.any(decoupled_mask):
            decoupled_ratios = fpr.fixed_ratio[decoupled_mask]
            print(
                f"  Decoupled limit (a_zz=0): ratios = "
                f"{np.unique(decoupled_ratios)} (expect 1.0)"
            )


# ============================================================================
# CLI Entry Point
# ============================================================================


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for data generation."""
    configure_environment()
    parser = argparse.ArgumentParser(
        description="Symmetric ω-Modulated Drive: Bounded-Compound Comparison"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if output exists"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only a specific step: decoupled-baseline, scenario-a, scenario-b, compound-ratio, fixed-parameter-ratio",
    )
    parser.add_argument(
        "--n-omega",
        type=int,
        default=DEFAULT_N_OMEGA,
        help=f"Number of ω points (default {DEFAULT_N_OMEGA})",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=OMEGA_MIN,
        help=f"Minimum ω value (default {OMEGA_MIN})",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=OMEGA_MAX,
        help=f"Maximum ω value (default {OMEGA_MAX})",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default=DEFAULT_SAMPLING_MODE,
        choices=["sphere", "cube"],
        help=f"Sampling mode: sphere (Marsaglia) or cube (legacy) (default {DEFAULT_SAMPLING_MODE})",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_SPHERE_RADIUS,
        help=f"Sphere radius R (default {DEFAULT_SPHERE_RADIUS})",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=500,
        help="Number of random search samples per omega (default 500)",
    )
    parser.add_argument(
        "--n-refine",
        type=int,
        default=50,
        help="Number of Nelder-Mead refinements per omega (default 50)",
    )
    args = parser.parse_args(argv)

    # Build ω grid from CLI args
    omega_vals = [
        round(v, 2) for v in np.linspace(args.omega_min, args.omega_max, args.n_omega)
    ]
    print(
        f"  ω grid: {len(omega_vals)} points from {omega_vals[0]} to {omega_vals[-1]}"
    )
    print(f"  Sampling: mode={args.sampling_mode}, R={args.radius}")

    # Wrap generate functions to pass omega_vals and sphere params where needed
    def _run_scenario_a() -> None:
        generate_scenario_a_scan(
            omega_vals=omega_vals,
            force=args.force,
            sampling_mode=args.sampling_mode,
            radius=args.radius,
            n_random=args.n_random,
            n_refine=args.n_refine,
        )

    def _run_scenario_b() -> None:
        generate_scenario_b_scan(
            omega_vals=omega_vals,
            force=args.force,
            sampling_mode=args.sampling_mode,
            radius=args.radius,
            n_random=args.n_random,
            n_refine=args.n_refine,
        )

    steps: dict[str, tuple[Any, dict[str, Any]]] = {
        "decoupled-baseline": (generate_decoupled_baseline, {"force": args.force}),
        "scenario-a": (_run_scenario_a, {}),
        "scenario-b": (_run_scenario_b, {}),
        "compound-ratio": (generate_compound_ratio, {"force": args.force}),
        "fixed-parameter-ratio": (
            generate_fixed_parameter_ratio,
            {"force": args.force},
        ),
    }

    def _run_step(name: str, fn: Any, kwargs: dict[str, Any]) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Step: {name}")
        print(f"{'=' * 60}")
        fn(**kwargs)

    if args.only:
        if args.only not in steps:
            print(f"Unknown step: {args.only}. Valid: {list(steps.keys())}")
            sys.exit(1)
        fn, kwargs = steps[args.only]
        _run_step(args.only, fn, kwargs)
    else:
        for name, (fn, kwargs) in steps.items():
            _run_step(name, fn, kwargs)


if __name__ == "__main__":
    main()
