"""
Plotting functions for ancilla-assisted metrology results.

Each function accepts a result dataclass (or a path to a CSV file),
generates a publication-quality figure using matplotlib + seaborn,
and saves it as SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.ancilla_optimization import (
    AlphaReoptScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    ThetaScanResult,
)
from src.analysis.weighted_joint_measurement import (
    AlphaReoptResultNM,
    MScalingResult,
    NScalingResult,
)

sns.set_theme(style="whitegrid")


# ──────────────────────────────────────────────
# 1. Decoupled baseline (Sections 1 & 2)
# ──────────────────────────────────────────────


def plot_decoupled_baseline(
    result: DecoupledBaselineResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Plot Δθ vs T_H on log-log axes with the 1/T_H SQL reference.

    Both quantities overlap exactly in the decoupled baseline,
    confirming SQL saturation.
    """
    if isinstance(result, (str, Path)):
        result = DecoupledBaselineResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(
        result.T_H_values,
        result.delta_theta_values,
        marker="o",
        linestyle="-",
        color="C0",
        label=r"$\Delta\theta$ (joint / S-only)",
    )
    ax.loglog(
        result.T_H_values,
        result.sql_values,
        marker="",
        linestyle="--",
        color="C1",
        alpha=0.7,
        label=r"SQL $= 1/T_H$",
    )
    ax.set_xlabel(r"$T_H$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title("Decoupled baseline: SQL saturation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 2. θ-scan (Sections 3 & 9)
# ──────────────────────────────────────────────


def plot_theta_scan(
    result: ThetaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Scatter plot of Δθ vs θ with error bars (spread) and SQL reference.

    Fringe-extremum points are highlighted in red.
    """
    if isinstance(result, (str, Path)):
        result = ThetaScanResult.from_csv(result)

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
            ok["theta"],
            ok["best_delta_theta"],
            yerr=ok["spread"],
            fmt="o",
            color="C0",
            capsize=4,
            label=r"$\Delta\theta$ (valid)",
        )

    # Fringe points highlighted
    if not fringe.empty:
        ax.scatter(
            fringe["theta"],
            fringe["best_delta_theta"],
            marker="x",
            s=80,
            color="red",
            zorder=5,
            label="fringe extremum",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(r"$\theta$-scan: sensitivity vs phase rate")
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
    """Multi-panel plot of Δθ vs T_H for various α values.

    Left panel: joint measurement.  Right panel: S-only measurement.
    Lines are grouped by α value with distinct colours.
    """
    if isinstance(result, (str, Path)):
        result = InteractionRobustnessResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    T_H = result.T_H_values
    alphas = result.alpha_values
    n_a = len(alphas)
    palette = sns.color_palette("viridis", n_a)

    fig, (ax_j, ax_s) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for j, a_val in enumerate(alphas):
        label = rf"$\alpha={a_val:.1f}$"
        ax_j.loglog(
            T_H,
            result.delta_theta_joint[:, j],
            marker="o",
            color=palette[j],
            label=label,
        )
        ax_s.loglog(
            T_H,
            result.delta_theta_sonly[:, j],
            marker="s",
            color=palette[j],
            label=label,
        )

    # SQL reference
    sql_ref = 1.0 / T_H
    for ax in (ax_j, ax_s):
        ax.loglog(T_H, sql_ref, "k--", alpha=0.4, label=r"SQL $=1/T_H$")
        ax.set_xlabel(r"$T_H$")

    ax_j.set_ylabel(r"$\Delta\theta$")
    ax_j.set_title("Joint measurement")
    ax_s.set_title("S-only measurement")
    ax_j.legend(fontsize="small", ncol=2)
    ax_s.legend(fontsize="small", ncol=2)

    fig.suptitle(r"Interaction robustness: $\Delta\theta$ vs $T_H$", y=1.02)
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
    """Two-line plot of Δθ vs α with joint and S-only measurements.

    Includes a horizontal SQL reference line.
    """
    if isinstance(result, (str, Path)):
        result = AlphaReoptScanResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        result.alpha_values,
        result.delta_theta_joint,
        marker="o",
        linestyle="-",
        color="C0",
        label="Joint measurement",
    )
    ax.plot(
        result.alpha_values,
        result.delta_theta_sonly,
        marker="s",
        linestyle="--",
        color="C1",
        label="S-only measurement",
    )

    # SQL reference
    sql_val = np.min(result.delta_theta_sonly)
    ax.axhline(
        y=sql_val,
        color="gray",
        linestyle=":",
        alpha=0.6,
        label=rf"SQL $\approx {sql_val:.3f}$",
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\Delta\theta$")
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
        result = CovarianceAnalysisResult.from_csv(result)

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


# ──────────────────────────────────────────────
# 6. N-scaling (N,M-general system)
# ──────────────────────────────────────────────


def plot_n_scaling(
    result: NScalingResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Log-log plot of Δθ vs N with weighted fit, SQL, and Heisenberg limits.

    Shows:
    - Mean Δθ per N with error bars (std across seeds)
    - Weighted log-log linear fit line with shaded 95% bootstrap CI
    - SQL reference: Δθ_SQL = 1/(√N · T_H)
    - Heisenberg limit reference: Δθ_HL = 1/(N · T_H)
    """
    if isinstance(result, (str, Path)):
        result = NScalingResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use an effective T_H for reference lines. Since T_H is optimised
    # per N and varies, show the SQL/HL lines using the maximum T_H
    # from the bounds (20.0), giving the most optimistic reference.
    T_H_ref = 20.0

    # Data points with error bars
    ax.errorbar(
        result.N_values,
        result.delta_theta_values,
        yerr=result.delta_theta_std,
        fmt="o",
        color="C0",
        capsize=4,
        markersize=6,
        label=r"$\Delta\theta$ (mean $\pm$ std)",
    )

    # Best Δθ per N (minimum across seeds)
    best_per_N = (
        np.min(result.delta_theta_seeds, axis=1)
        if result.delta_theta_seeds is not None
        else result.delta_theta_values
    )
    ax.scatter(
        result.N_values,
        best_per_N,
        marker="*",
        color="C3",
        s=80,
        zorder=5,
        label="Best per N",
    )

    # Weighted fit line (over the N range)
    N_fit = np.logspace(
        np.log10(max(result.N_values.min(), 1)),
        np.log10(result.N_values.max()),
        100,
    )
    log_N_fit = np.log(N_fit)
    nu = result.scaling_exponent
    c_fit = np.mean(
        np.log(result.delta_theta_values) + nu * np.log(result.N_values.astype(float))
    )
    dt_fit = np.exp(-nu * log_N_fit + c_fit)
    ax.loglog(
        N_fit,
        dt_fit,
        "--",
        color="C0",
        alpha=0.7,
        label=rf"Fit: $\nu = {nu:.3f}$",
    )

    # Shaded 95% CI for the fit (from bootstrap)
    if not np.isnan(result.scaling_exponent_ci[0]):
        nu_lo, nu_hi = result.scaling_exponent_ci
        dt_lo = np.exp(-nu_lo * log_N_fit + c_fit)
        dt_hi = np.exp(-nu_hi * log_N_fit + c_fit)
        ax.fill_between(
            N_fit,
            dt_lo,
            dt_hi,
            alpha=0.15,
            color="C0",
            label=rf"95% CI: $[{nu_lo:.3f}, {nu_hi:.3f}]$",
        )

    # SQL and HL reference lines (using T_H_ref)
    N_range = np.logspace(
        np.log10(max(result.N_values.min(), 1)),
        np.log10(result.N_values.max()),
        100,
    )
    ax.loglog(
        N_range,
        1.0 / (np.sqrt(N_range) * T_H_ref),
        ":",
        color="C2",
        alpha=0.6,
        label=r"SQL: $1/(\sqrt{N}\,T_H)$",
    )
    ax.loglog(
        N_range,
        1.0 / (N_range * T_H_ref),
        ":",
        color="C4",
        alpha=0.6,
        label=r"HL: $1/(N\,T_H)$",
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(rf"N-scaling: $\Delta\theta$ vs $N$ (M={result.M_value})")
    ax.legend(fontsize="small")
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 7. M-scaling (diminishing returns)
# ──────────────────────────────────────────────


def plot_m_scaling(
    result: MScalingResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Plot Δθ vs M showing diminishing returns from ancilla size.

    Shows:
    - Best Δθ per M
    - Optimal weight a* and b* per M (twin axis)
    - SQL reference for N=4
    """
    if isinstance(result, (str, Path)):
        result = MScalingResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.85)

    # Δθ vs M
    ax1.semilogy(
        result.M_values,
        result.delta_theta_values,
        marker="o",
        linestyle="-",
        color="C0",
        linewidth=2,
        markersize=6,
        label=r"$\Delta\theta$ (best)",
    )

    # SQL reference for N=4
    T_H_ref = 20.0
    sql = 1.0 / (np.sqrt(result.N_value) * T_H_ref)
    ax1.axhline(
        y=sql,
        color="C2",
        linestyle=":",
        alpha=0.6,
        label=rf"SQL: $1/(\sqrt{{{result.N_value}}}\,T_H)$",
    )

    # Ensure all data points are within the plotted y-range
    y_min = 0.99 * min(result.delta_theta_values)
    y_max = 1.01 * max(result.delta_theta_values)
    ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel(r"$M$ (ancilla particles)")
    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title(
        rf"M-scaling: $\Delta\theta$ vs ancilla size $M$ (N={result.N_value})"
    )

    # Annotate improvement from M=0 to M=1
    if result.improvement_01 > 0:
        improvement_pct = result.improvement_01 * 100
        ax1.annotate(
            rf"$\Delta\theta$ improvement: {improvement_pct:.1f}\%",
            xy=(1, result.delta_theta_values[1]),
            xytext=(4, result.delta_theta_values[1] * 0.7),
            arrowprops={"arrowstyle": "->", "color": "gray"},
            fontsize=10,
        )

    ax1.legend(fontsize="small")

    # Twin axis for optimal weights
    ax2 = ax1.twinx()
    ax2.plot(
        result.M_values,
        result.a_opt_values,
        marker="s",
        linestyle="--",
        color="C3",
        alpha=0.6,
        markersize=4,
        label=r"$a^*$",
    )
    ax2.plot(
        result.M_values,
        result.b_opt_values,
        marker="^",
        linestyle="--",
        color="C4",
        alpha=0.6,
        markersize=4,
        label=r"$b^*$",
    )
    ax2.set_ylabel(r"Optimal weight $a^*, b^*$")
    ax2.legend(fontsize="small", loc="lower right")

    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 8. α-scan with weight re-optimisation (N,M-general)
# ──────────────────────────────────────────────


def plot_weighted_alpha_scan(
    result: AlphaReoptResultNM | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Plot Δθ vs α for weighted joint and S-only measurements.

    Shows:
    - Weighted joint measurement Δθ vs α
    - S-only measurement Δθ vs α at same optimised parameters
    - Optimal weight angle (secondary axis)
    """
    if isinstance(result, (str, Path)):
        result = AlphaReoptResultNM.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Weighted joint measurement
    ax1.plot(
        result.alpha_values,
        result.delta_theta_weighted,
        marker="o",
        linestyle="-",
        color="C0",
        linewidth=2,
        markersize=6,
        label=r"Weighted joint: $\Delta\theta$",
    )

    # S-only measurement
    ax1.plot(
        result.alpha_values,
        result.delta_theta_sonly,
        marker="s",
        linestyle="--",
        color="C1",
        linewidth=2,
        markersize=6,
        label=r"S-only: $\Delta\theta$",
    )

    ax1.set_xlabel(rf"$\alpha_{{{result.alpha_name}}}$")
    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title(rf"$\alpha_{{{result.alpha_name}}}$ scan: N={result.N}, M={result.M}")
    ax1.legend(fontsize="small")

    # Twin axis for weight angle
    ax2 = ax1.twinx()
    ax2.plot(
        result.alpha_values,
        result.phi_opt_values,
        marker=".",
        linestyle=":",
        color="gray",
        alpha=0.5,
        markersize=3,
        label=r"$\phi^*$",
    )
    ax2.set_ylabel(r"Optimal $\phi$ (rad)")
    ax2.legend(fontsize="small", loc="lower right")

    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
