"""
Local module for the 2026-05-12 reports.

Contains all code exclusive to these reports:
- Single-Particle MZI Holding-Time Scaling (core simulation functions,
  data generation, and figure creation)
- Ancilla-Assisted Metrology Optimization (`validate_hold_unitarity`)

Usage:
    uv run python reports/20260512/local.py --force
    uv run python reports/20260512/local.py --experiment single-particle --force

This module is **not** importable as ``reports.20260512.local`` (the directory
name contains hyphens).  Importers should use ``importlib.util.spec_from_file_location``
or add this directory to ``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.ancilla_optimization import (
    random_search_alpha,
    run_omega_scan,
    scan_alpha_single_parameter,
)
from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.single_particle_mzi import compute_sensitivity_sweep

sns.set_theme(style="whitegrid")

REPORT_DATE = "20260512"
REPORTS_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# Single-Particle MZI: Sensitivity Scaling with Holding Time
# =============================================================================
#
# Implements the exact analytical model for a single-particle (spin-1/2
# equivalent) Mach-Zehnder interferometer where the parameter ω is encoded
# via H = ω J_z during a holding time T_hold.
#
# Physical Model:
# - Hilbert space: two-mode bosonic Fock space truncated at max_photons = 1.
#   Only two basis states are physical: |1,0⟩ and |0,1⟩ (dimension 2).
# - Beam splitter: 50:50, U_BS = exp(-i(π/4)(a₀†a₁ + a₁†a₀)).
# - Holding: U_hold(T_hold) = exp(-i ω T_hold J_z), with J_z = (n₁ - n₂)/2.
# - State: |1,0⟩ → U_BS → U_hold → U_BS → measurement of J_z.
#
# Analytical result:
# ⟨J_z⟩ = -(1/2) cos(ω T_hold)
# Var(J_z) = (1/4) sin²(ω T_hold)
# ∂⟨J_z⟩/∂ω = (T_hold/2) sin(ω T_hold)
# ⇒ Δω = 1/T_hold  (independent of ω, away from sin(ω T_hold) = 0)
#
# Units: Dimensionless throughout.
# Conventions: J_z = (n₁ - n₂)/2, beam-splitter generator = a₀†a₁ + a₁†a₀.

# =============================================================================
# Re-exported from src/physics/single_particle_mzi.py
# =============================================================================
# These functions have been promoted to src/physics/single_particle_mzi.py
# and are re-exported here for backward compatibility.

# =============================================================================
# Validation
# =============================================================================

# run_validation promoted to src.physics.single_particle_mzi, imported above.


# =============================================================================
# Raw Data Generation
# =============================================================================


def generate_single_particle_raw_data(force: bool = False) -> Path:
    """Run the single-particle sensitivity sweep and save raw data Parquet.

    Uses the report's standard parameters:
        omega = 1.0
        T_hold range: 0.1 to 100, 500 log-spaced points
        Also runs multi-omega sweep at [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    Args:
        force: Re-run even if Parquet exists.

    Returns:
        Path to the saved data directory.

    """
    raw_dir = REPORTS_DIR / REPORT_DATE / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Standard sweep at omega = 1.0
    csv_path = raw_dir / f"{REPORT_DATE}-single-particle-sweep.parquet"
    if not csv_path.exists() or force:
        print(f"  Generating {csv_path.name} ...")
        df = compute_sensitivity_sweep(omega=1.0, n_points=500)
        df.to_parquet(csv_path, index=False)
    else:
        print(f"  {csv_path.name} exists (use --force to regenerate)")

    # Multi-omega sweep
    multi_csv = raw_dir / f"{REPORT_DATE}-single-particle-multi-omega-sweep.parquet"
    if not multi_csv.exists() or force:
        print(f"  Generating {multi_csv.name} ...")
        omega_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rows = []
        for omega in omega_values:
            df_t = compute_sensitivity_sweep(omega=omega, n_points=500)
            rows.append(df_t)
        multi_df = pd.concat(rows, ignore_index=True)
        multi_df.to_parquet(multi_csv, index=False)
    else:
        print(f"  {multi_csv.name} exists (use --force to regenerate)")

    return raw_dir


def generate_ancilla_raw_data(force: bool = False) -> Path:
    """Run ancilla-metrology experiments and save raw data Parquet.

    Generates:
        1. Omega-scan with Nelder–Mead optimisation
        2. Alpha single-coefficient grid scan
        3. Alpha 4D random search

    Args:
        force: Re-run even if Parquets exist.

    Returns:
        Path to the saved data directory.

    """
    raw_dir = REPORTS_DIR / REPORT_DATE / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Omega scan (moderate settings for automated generation;
    #    use higher n_restarts/maxiter for report-quality results)
    omega_csv = raw_dir / f"{REPORT_DATE}-ancilla-omega-scan.parquet"
    if not omega_csv.exists() or force:
        print(f"  Generating {omega_csv.name} (this may take several minutes) ...")
        omega_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        scan_result = run_omega_scan(
            omega_values=omega_values,
            n_restarts=3,
            maxiter=500,
        )
        scan_result.save_parquet(omega_csv)
    else:
        print(f"  {omega_csv.name} exists (use --force to regenerate)")

    # 2. Alpha single-coefficient scan
    alpha_csv = raw_dir / f"{REPORT_DATE}-ancilla-alpha-scan.parquet"
    if not alpha_csv.exists() or force:
        print(f"  Generating {alpha_csv.name} ...")
        alpha_names = ["xx", "xz", "zx", "zz"]
        scan_rows = []
        for name in alpha_names:
            result = scan_alpha_single_parameter(
                alpha_name=name,
                alpha_min=-2.0,
                alpha_max=2.0,
                n_points=21,
            )
            df = result.to_dataframe()
            df["coefficient"] = name
            scan_rows.append(df)
        pd.concat(scan_rows, ignore_index=True).to_parquet(alpha_csv, index=False)
    else:
        print(f"  {alpha_csv.name} exists (use --force to regenerate)")

    # 3. Alpha 4D random search
    random_csv = raw_dir / f"{REPORT_DATE}-ancilla-random-search.parquet"
    if not random_csv.exists() or force:
        print(f"  Generating {random_csv.name} ...")
        result = random_search_alpha(n_samples=200)
        result.save_parquet(random_csv)
    else:
        print(f"  {random_csv.name} exists (use --force to regenerate)")

    return raw_dir


# =============================================================================
# Figure Generation
# =============================================================================


def generate_single_particle_figures(force: bool = False) -> Path:
    """Generate figures for the single-particle MZI report.

    Creates:
        1. log-log Δω vs T_hold with analytical and numerical traces
        2. ⟨J_z⟩ vs T_hold oscillatory signal
        3. ∂⟨J_z⟩/∂ω analytical vs numerical comparison

    Args:
        force: Re-generate even if SVGs exist.

    Returns:
        Path to the saved figures directory.

    """
    fig_dir = REPORTS_DIR / REPORT_DATE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run the standard sweep
    df = compute_sensitivity_sweep(omega=1.0, n_points=500)
    fit_scaling_exponent(
        np.asarray(df["t_hold"]).astype(float),
        np.asarray(df["delta_omega_analytical"]).astype(float),
    )

    t_h_min = float(df["t_hold"].min())
    t_h_max = float(df["t_hold"].max())

    # ── Figure 1: log-log Δθ vs T_hold ──
    fig1_path = fig_dir / f"{REPORT_DATE}-single-particle-scaling.svg"
    if not fig1_path.exists() or force:
        print(f"  Generating {fig1_path.name} ...")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Theory reference: 1/T_hold
        t_ref = np.array([t_h_min, t_h_max])
        dt_ref = 1.0 / t_ref
        ax.loglog(
            t_ref,
            dt_ref,
            "--",
            color="gray",
            linewidth=2,
            label=r"$1/T_hold$ (theory, $\alpha=-1$)",
        )

        # Clean (non-fringe) points
        clean = df[~df["is_fringe_extremum"]]
        ax.loglog(
            clean["T_hold"],
            clean["delta_omega_analytical"],
            "o",
            color="#1f77b4",
            markersize=4,
            label=r"$\Delta\omega$ (analytical)",
        )
        ax.loglog(
            clean["T_hold"],
            clean["delta_omega_numerical"],
            "x",
            color="#ff7f0e",
            markersize=4,
            label=r"$\Delta\omega$ (numerical)",
        )

        # Fringe points
        fringe = df[df["is_fringe_extremum"]]
        if not fringe.empty:
            ax.loglog(
                fringe["T_hold"],
                fringe["delta_omega_analytical"],
                "r*",
                markersize=8,
                alpha=0.6,
                label="Fringe extremum (excluded)",
            )

        ax.set_xlabel(r"$T_hold$")
        ax.set_ylabel(r"$\Delta\omega$")
        ax.set_title("Single-Particle MZI: Sensitivity Scaling with $T_hold$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig1_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    # ── Figure 2: ⟨J_z⟩ vs T_hold ──
    fig2_path = fig_dir / f"{REPORT_DATE}-single-particle-jz-mean.svg"
    if not fig2_path.exists() or force:
        print(f"  Generating {fig2_path.name} ...")
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.semilogx(
            df["t_hold"],
            df["jz_mean"],
            "-o",
            color="#2ca02c",
            markersize=3,
            label=r"$\langle J_z \rangle$",
        )

        t_dense = np.logspace(np.log10(t_h_min), np.log10(t_h_max), 500)
        ax.semilogx(
            t_dense,
            -0.5 * np.cos(1.0 * t_dense),
            ":",
            color="gray",
            linewidth=1,
            label=r"$-\frac{1}{2}\cos(\omega T_hold)$",
        )

        ax.set_xlabel(r"$T_hold$")
        ax.set_ylabel(r"$\langle J_z \rangle$")
        ax.set_title("Single-Particle MZI: Signal Oscillation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig2_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    # ── Figure 3: ∂⟨J_z⟩/∂θ comparison ──
    fig3_path = fig_dir / f"{REPORT_DATE}-single-particle-derivative.svg"
    if not fig3_path.exists() or force:
        print(f"  Generating {fig3_path.name} ...")
        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.loglog(
            df["t_hold"],
            df["d_jz_analytical"],
            "-",
            color="#1f77b4",
            linewidth=2,
            label=r"Analytical: $\frac{T_hold}{2}\sin(\omega T_hold)$",
        )
        ax1.loglog(
            df["t_hold"],
            df["d_jz_numerical"],
            "x",
            color="#ff7f0e",
            markersize=3,
            label=r"Numerical ($\delta = 10^{-6}$)",
        )
        ax1.set_xlabel(r"$T_hold$")
        ax1.set_ylabel(r"$\partial\langle J_z \rangle / \partial\omega$")
        ax1.set_title("Derivative Comparison: Analytical vs Numerical")
        ax1.legend(loc="upper left")

        # Relative difference on second y-axis
        rel_diff = np.abs(
            (df["d_jz_analytical"] - df["d_jz_numerical"])
            / np.maximum(np.abs(df["d_jz_analytical"]), 1e-15)
        )
        ax2 = ax1.twinx()
        ax2.loglog(
            df["t_hold"],
            rel_diff,
            "--",
            color="red",
            linewidth=1,
            alpha=0.7,
            label="Relative diff",
        )
        ax2.set_ylabel("Relative difference")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(fig3_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    return fig_dir


def generate_ancilla_figures(force: bool = False) -> Path:
    """Generate figures for the ancilla-assisted metrology report.

    Creates:
        1. Δω vs ω (best-per-omega from Nelder–Mead)

    Args:
        force: Re-generate even if SVGs exist.

    Returns:
        Path to the saved figures directory.

    """
    fig_dir = REPORTS_DIR / REPORT_DATE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Sensitivity vs ω
    fig_path = fig_dir / f"{REPORT_DATE}-ancilla-omega-scan.svg"
    if not fig_path.exists() or force:
        print(f"  Generating {fig_path.name} ...")
        omega_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        scan_result = run_omega_scan(
            omega_values=omega_values,
            n_restarts=3,
            maxiter=500,
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        # Best per ω
        ax.semilogy(
            scan_result.omega_values,
            scan_result.best_per_omega,
            "-o",
            color="firebrick",
            linewidth=2,
            markersize=8,
            label="Best Δθ (Nelder–Mead)",
        )

        # SQL reference (1/T_hold with optimal T_hold from each ω)
        sql_vals = []
        for omega in omega_values:
            results = scan_result.all_results.get(omega, [])
            if results:
                best = min(results, key=lambda r: r.delta_omega_opt)
                t_h_star = best.params_opt[6]
                sql_vals.append(1.0 / t_h_star if t_h_star > 0 else float("inf"))
            else:
                sql_vals.append(float("inf"))
        ax.semilogy(
            omega_values,
            sql_vals,
            "--",
            color="gray",
            linewidth=2,
            label="SQL (1/T_hold*)",
        )

        ax.set_xlabel(r"True $\omega$")
        ax.set_ylabel(r"$\Delta\omega$")
        ax.set_title("Ancilla-Assisted Metrology: Sensitivity vs ω")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    return fig_dir


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs for 2026-05-12 reports",
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
        choices=["single-particle", "ancilla"],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist.
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "single-particle": (
            generate_single_particle_raw_data,
            generate_single_particle_figures,
        ),
        "ancilla": (
            generate_ancilla_raw_data,
            generate_ancilla_figures,
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
