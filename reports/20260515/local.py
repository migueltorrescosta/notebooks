"""
Local module for the 2026-05-15 Ancilla-Assisted Metrology Joint-Measurement report.

Contains all code exclusive to this report:
- Plotting functions (decoupled baseline, omega scan, interaction robustness,
  alpha re-optimisation, covariance analysis)
- Data generation and figure rendering for each of the five experiments

Usage:
    uv run python reports/20260515/local.py --force

This module is **not** importable as ``reports.20260515.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.ancilla_optimization import (
    AlphaReoptScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    OmegaScanResult,
    build_joint_operator,
    build_two_qubit_operators,
    compute_covariance_analysis,
    compute_decoupled_baseline,
    compute_interaction_robustness,
    run_omega_scan,
    scan_alpha_with_reoptimisation,
)

sns.set_theme(style="whitegrid")

# ============================================================================
# Plot functions
# ============================================================================

# ──────────────────────────────────────────────
# 1. Decoupled baseline (Sections 1 & 2)
# ──────────────────────────────────────────────


def plot_decoupled_baseline(
    result: DecoupledBaselineResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Plot Δω vs t_hold on log-log axes with the 1/t_hold SQL reference.

    Both quantities overlap exactly in the decoupled baseline,
    confirming SQL saturation.
    """
    if isinstance(result, (str, Path)):
        result = DecoupledBaselineResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(
        result.t_hold_values,
        result.delta_omega_values,
        marker="o",
        linestyle="-",
        color="C0",
        label=r"$\Delta\omega$ (joint / S-only)",
    )
    ax.loglog(
        result.t_hold_values,
        result.sql_values,
        marker="",
        linestyle="--",
        color="C1",
        alpha=0.7,
        label=r"SQL $= 1/t_hold$",
    )
    ax.set_xlabel(r"$t_hold$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title("Decoupled baseline: SQL saturation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 2. ω-scan (Sections 3 & 9)
# ──────────────────────────────────────────────


def plot_omega_scan(
    result: OmegaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Scatter plot of Δω vs ω with error bars (spread) and SQL reference.

    Fringe-extremum points are highlighted in red.
    """
    if isinstance(result, (str, Path)):
        result = OmegaScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    fringe = df[df["flag"] == "fringe"]
    ok = df[df["flag"] != "fringe"]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference as a shaded region
    if not df.empty:
        sql_min = df["sql"].min()
        ax.axhline(
            y=sql_min,
            color="C1",
            linestyle="--",
            alpha=0.6,
            label=rf"SQL bound = {sql_min:.4f}",
        )

    # OK points with error bars (spread)
    if not ok.empty:
        ax.errorbar(
            ok["omega"],
            ok["best_delta_omega"],
            yerr=ok["spread"],
            fmt="o",
            color="C0",
            capsize=4,
            label=r"$\Delta\omega$ (valid)",
        )

    # Fringe points highlighted
    if not fringe.empty:
        ax.scatter(
            fringe["omega"],
            fringe["best_delta_omega"],
            marker="x",
            s=80,
            color="red",
            zorder=5,
            label="fringe extremum",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(r"$\omega$-scan: sensitivity vs phase rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 3. Interaction robustness (Section 8)
# ──────────────────────────────────────────────


def plot_interaction_robustness(
    result: InteractionRobustnessResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Multi-panel plot of Δω vs t_hold for various α values.

    Left panel: joint measurement.  Right panel: S-only measurement.
    Lines are grouped by α value with distinct colours.
    """
    if isinstance(result, (str, Path)):
        result = InteractionRobustnessResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    t_hold = result.t_hold_values
    alphas = result.alpha_values
    n_a = len(alphas)
    palette = sns.color_palette("viridis", n_a)

    fig, (ax_j, ax_s) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for j, a_val in enumerate(alphas):
        label = rf"$\alpha={a_val:.1f}$"
        ax_j.loglog(
            t_hold,
            result.delta_omega_joint[:, j],
            marker="o",
            color=palette[j],
            label=label,
        )
        ax_s.loglog(
            t_hold,
            result.delta_omega_sonly[:, j],
            marker="s",
            color=palette[j],
            label=label,
        )

    # SQL reference
    sql_ref = 1.0 / t_hold
    for ax in (ax_j, ax_s):
        ax.loglog(t_hold, sql_ref, "k--", alpha=0.4, label=r"SQL $=1/t_hold$")
        ax.set_xlabel(r"$t_hold$")

    ax_j.set_ylabel(r"$\Delta\omega$")
    ax_j.set_title("Joint measurement")
    ax_s.set_title("S-only measurement")
    ax_j.legend(fontsize="small", ncol=2)
    ax_s.legend(fontsize="small", ncol=2)

    fig.suptitle(r"Interaction robustness: $\Delta\omega$ vs $t_hold$", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 4. α-scan with state re-optimisation (Section 7)
# ──────────────────────────────────────────────


def plot_alpha_reoptimisation(
    result: AlphaReoptScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Two-line plot of Δω vs α with joint and S-only measurements.

    Includes a horizontal SQL reference line.
    """
    if isinstance(result, (str, Path)):
        result = AlphaReoptScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        result.alpha_values,
        result.delta_omega_joint,
        marker="o",
        linestyle="-",
        color="C0",
        label="Joint measurement",
    )
    ax.plot(
        result.alpha_values,
        result.delta_omega_sonly,
        marker="s",
        linestyle="--",
        color="C1",
        label="S-only measurement",
    )

    # SQL reference
    sql_val = np.min(result.delta_omega_sonly)
    ax.axhline(
        y=sql_val,
        color="gray",
        linestyle=":",
        alpha=0.6,
        label=rf"SQL $\approx {sql_val:.3f}$",
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(r"$\alpha$-scan with state re-optimisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 5. Covariance analysis (Section 5)
# ──────────────────────────────────────────────


def plot_covariance_analysis(
    result: CovarianceAnalysisResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Bar chart of max |Cov(J_z^S, J_z^A)| per α coefficient.

    Bars are colour-coded by the sign of the covariance.
    """
    if isinstance(result, (str, Path)):
        result = CovarianceAnalysisResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(result.coefficient_names))
    colours = ["C0" if s > 0 else "C3" for s in result.covariance_signs]

    bars = ax.bar(
        x,
        result.max_covariances,
        color=colours,
        width=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate sign on each bar
    for i, (bar, sign) in enumerate(zip(bars, result.covariance_signs, strict=False)):
        label = f"{'+' if sign > 0 else '-'}{result.max_covariances[i]:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.003,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(result.coefficient_names)
    ax.set_ylabel(r"$\max |\mathrm{Cov}(J_z^S, J_z^A)|$")
    ax.set_title("Covariance generated by each $\\alpha$ coefficient")
    ax.set_ylim(0, max(result.max_covariances) * 1.25)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Constants and helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent

REPORT_DATE = "20260515"


def _parquet_path(name: str, date: str) -> Path:
    return REPORTS_DIR / date / "raw_data" / f"{date}-{name}.parquet"


def _fig_path(name: str, date: str) -> Path:
    return REPORTS_DIR / date / "figures" / f"{date}-{name}.svg"


# ============================================================================
# Generate functions
# ============================================================================


def generate_decoupled_baseline(force: bool = False) -> None:
    """Sections 1 & 2: Decoupled baseline and expanded t_hold bound."""
    csv_p = _parquet_path("decoupled-baseline", date=REPORT_DATE)
    fig_p = _fig_path("decoupled-baseline", date=REPORT_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing decoupled baseline...")
        t_hold_vals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0])
        result = compute_decoupled_baseline(t_hold_vals)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_omega_scan(force: bool = False) -> None:
    """Sections 3 & 9: ω-scan with Nelder-Mead optimisation."""
    csv_p = _parquet_path("omega-scan", date=REPORT_DATE)
    fig_p = _fig_path("omega-scan", date=REPORT_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing ω-scan (may be slow)...")
        ops = build_two_qubit_operators()
        M_op = build_joint_operator(ops)
        omega_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        result = run_omega_scan(
            omega_vals,
            n_restarts=10,
            seed=42,
            maxiter=2000,
            meas_op=M_op,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = OmegaScanResult.from_parquet(csv_p)
    plot_omega_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_interaction_robustness(force: bool = False) -> None:
    """Section 8: t_hold × α interaction robustness."""
    csv_p = _parquet_path("interaction-robustness", date=REPORT_DATE)
    fig_p = _fig_path("interaction-robustness", date=REPORT_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing interaction robustness...")
        t_hold_vals = np.array([0.5, 1.0, 2.0])
        alpha_vals = np.array([0.0, 1.0, 2.0])
        result = compute_interaction_robustness(
            t_hold_vals,
            alpha_vals,
            omega_true=1.0,
            alpha_name="xx",
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = InteractionRobustnessResult.from_parquet(csv_p)
    plot_interaction_robustness(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_alpha_reoptimisation(force: bool = False) -> None:
    """Section 7: α-scan with state re-optimisation."""
    csv_p = _parquet_path("alpha-reoptimisation", date=REPORT_DATE)
    fig_p = _fig_path("alpha-reoptimisation", date=REPORT_DATE)

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
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = AlphaReoptScanResult.from_parquet(csv_p)
    plot_alpha_reoptimisation(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_covariance_analysis(force: bool = False) -> None:
    """Section 5: Covariance analysis."""
    csv_p = _parquet_path("covariance-analysis", date=REPORT_DATE)
    fig_p = _fig_path("covariance-analysis", date=REPORT_DATE)

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing covariance analysis...")
        result = compute_covariance_analysis()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = CovarianceAnalysisResult.from_parquet(csv_p)
    plot_covariance_analysis(result, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and Parquet data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing Parquets)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset, e.g. 'decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist.
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "decoupled-baseline": generate_decoupled_baseline,
        "omega-scan": generate_omega_scan,
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
