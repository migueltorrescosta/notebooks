"""
Combined local module for the 2026-05-11 reports.

Re-exports all migrated symbols from src/ for the four reports:
- Ancilla-vs-System-Comparison
- Scaling-Survey-Fock-MZI
- Scaling-Survey-Collective-Spin
- Scaling-Survey-Advanced-Architectures

Usage:
    uv run python reports/20260511/local.py --force
    uv run python reports/20260511/local.py --experiment ancilla-vs-system --force

This module is not importable as reports.20260511.local (hyphens in path).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.ancilla_comparison import (
    ComparisonResult,
    analytical_fq_A_zero,
    analytical_fq_B_max,
    build_system_jz_jx,
    compute_generator_B,
    random_density_matrix,
    run_comparison,
)
from src.analysis.scaling_fit import ScalingFitResult
from src.analysis.scaling_survey import (
    CavityMziConfig,
    DDConfig,
    DistributedMziConfig,
    ModelConfig,
    SurveyConfig,
    ThermalLangevinConfig,
    TTLNoiseConfig,
    WeakValueConfig,
    cavity_enhanced_mzi,
    cavity_enhanced_mzi_with_noise,
    cavity_enhanced_sensitivity,
    combined_sensitivity,
    cpmg_filter_function,
    create_quantum_only_config,
    create_survey_model,
    create_thermal_config,
    crossover_N,
    dd_effective_coherence_time,
    dd_phase_sensitivity,
    dd_sensitivity_scaling,
    distributed_mzi_sensitivity,
    distributed_scaling_exponent,
    fit_all_exponents,
    fit_thermal_scaling_exponent,
    run_scaling_survey,
    sweep_thermal_scaling,
    thermal_sensitivity_normalized,
    ttl_path_length_noise,
    ttl_phase_noise,
    ttl_scaling_sweep,
    weak_value_mzi_sensitivity,
)
from src.physics.mzi_states import compute_fisher_information, input_state_factory

__all__ = [
    "CavityMziConfig",
    "ComparisonResult",
    "DDConfig",
    "DistributedMziConfig",
    "ModelConfig",
    "ScalingFitResult",
    "SurveyConfig",
    "TTLNoiseConfig",
    "ThermalLangevinConfig",
    "WeakValueConfig",
    "analytical_fq_A_zero",
    "analytical_fq_B_max",
    "build_system_jz_jx",
    "cavity_enhanced_mzi",
    "cavity_enhanced_mzi_with_noise",
    "cavity_enhanced_sensitivity",
    "combined_sensitivity",
    "compute_fisher_information",
    "compute_generator_B",
    "cpmg_filter_function",
    "create_quantum_only_config",
    "create_survey_model",
    "create_thermal_config",
    "crossover_N",
    "dd_effective_coherence_time",
    "dd_phase_sensitivity",
    "dd_sensitivity_scaling",
    "distributed_mzi_sensitivity",
    "distributed_scaling_exponent",
    "fit_all_exponents",
    "fit_thermal_scaling_exponent",
    "input_state_factory",
    "main",
    "random_density_matrix",
    "run_comparison",
    "run_scaling_survey",
    "sweep_thermal_scaling",
    "thermal_sensitivity_normalized",
    "ttl_path_length_noise",
    "ttl_phase_noise",
    "ttl_scaling_sweep",
    "weak_value_mzi_sensitivity",
]

REPORT_DATE = "20260511"
REPORTS_DIR = Path(__file__).resolve().parent.parent
_REPORT_DIR = REPORTS_DIR / REPORT_DATE


# =============================================================================
# CLI
# =============================================================================


def _generate_ancilla_vs_system_raw_data(force: bool = False) -> None:
    """Generate raw data for Ancilla-vs-System-Comparison report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla-vs-System raw data generation")


def _generate_ancilla_vs_system_figures(force: bool = False) -> None:
    """Generate figures for Ancilla-vs-System-Comparison report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla-vs-System figure generation")


def _generate_fock_mzi_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Fock-MZI report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Fock-MZI raw data generation")


def _generate_fock_mzi_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Fock-MZI report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Fock-MZI figure generation")


def _generate_collective_spin_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Collective-Spin report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Collective-Spin raw data generation")


def _generate_collective_spin_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Collective-Spin report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Collective-Spin figure generation")


def _generate_advanced_architectures_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Advanced-Architectures report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Advanced-Architectures raw data generation")


def _generate_advanced_architectures_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Advanced-Architectures report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Advanced-Architectures figure generation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs for 2026-05-11 reports",
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
        choices=[
            "ancilla-vs-system",
            "fock-mzi",
            "collective-spin",
            "advanced-architectures",
        ],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    (_REPORT_DIR / "raw_data").mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "ancilla-vs-system": (
            _generate_ancilla_vs_system_raw_data,
            _generate_ancilla_vs_system_figures,
        ),
        "fock-mzi": (
            _generate_fock_mzi_raw_data,
            _generate_fock_mzi_figures,
        ),
        "collective-spin": (
            _generate_collective_spin_raw_data,
            _generate_collective_spin_figures,
        ),
        "advanced-architectures": (
            _generate_advanced_architectures_raw_data,
            _generate_advanced_architectures_figures,
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
