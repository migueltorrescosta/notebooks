"""
Plotting functions for driven-ancilla metrology results.

Each function accepts a result dataclass (or a path to a Parquet file),
generates a publication-quality figure using matplotlib + seaborn,
and saves it as SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
)

sns.set_theme(style="whitegrid")


# ──────────────────────────────────────────────
# 1. Decoupled baseline
# ──────────────────────────────────────────────


def plot_drive_decoupled_baseline(
    result: DriveDecoupledBaselineResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Bar chart comparing Δθ to SQL for the decoupled (all-zero) configuration."""
    if isinstance(result, (str, Path)):
        result = DriveDecoupledBaselineResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    labels = [r"$\Delta\theta$", r"SQL $= 1/T_H$"]
    values = [result.delta_theta, result.sql]
    colours = ["C0", "C1"]

    bars = ax.bar(labels, values, color=colours, width=0.5, edgecolor="black")
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{val:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"Decoupled baseline at $T_H={result.T_H_value:.0f}$")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 2. 2D slice heatmap
# ──────────────────────────────────────────────


def plot_drive_2d_slice_heatmap(
    result: Drive2DSliceResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
    vmax: float | None = None,
) -> Path:
    """Heatmap of Δθ over (a_drive, a_zz) with SQL contour."""
    if isinstance(result, (str, Path)):
        result = Drive2DSliceResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sql = result.sql
    if vmax is None:
        # Clip to 3x SQL for colour scale
        vmax = min(3.0 * sql, np.nanmax(result.delta_theta_grid))
        if not np.isfinite(vmax):
            vmax = 3.0 * sql

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        result.azz_values,
        result.drive_values,
        result.delta_theta_grid,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    _cbar = fig.colorbar(im, ax=ax, label=r"$\Delta\theta$")

    # SQL contour line
    cs = ax.contour(
        result.azz_values,
        result.drive_values,
        result.delta_theta_grid,
        levels=[sql],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt=f"SQL = {sql:.3f}", fontsize=9)

    drive_label = r"$a_x$" if result.slice_type == "ax" else r"$a_y$"
    ax.set_xlabel(r"$a_{zz}$ (interaction)")
    ax.set_ylabel(drive_label + " (drive)")
    ax.set_title(
        rf"$\Delta\theta$ vs {drive_label} and $a_{{zz}}$ at $\theta={result.theta_value:.1f}$"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 3. Random search histogram
# ──────────────────────────────────────────────


def plot_drive_random_search_histogram(
    result: DriveRandomSearchResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
    n_bins: int = 50,
) -> Path:
    """Histogram of Δθ values from 4D random search with SQL and best marked."""
    if isinstance(result, (str, Path)):
        result = DriveRandomSearchResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    finite = result.delta_theta_values[np.isfinite(result.delta_theta_values)]
    sql = result.sql

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        finite, bins=n_bins, color="C0", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax.axvline(
        x=sql,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=rf"SQL = {sql:.4f}",
    )
    ax.axvline(
        x=result.best_delta_theta,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=rf"Best = {result.best_delta_theta:.4f}",
    )

    below_sql = np.sum(finite < sql)
    if below_sql > 0:
        ax.annotate(
            rf"{below_sql} / {len(finite)} below SQL",
            xy=(result.best_delta_theta, 0.6 * ax.get_ylim()[1]),
            fontsize=10,
            color="red",
        )

    ax.set_xlabel(r"$\Delta\theta$")
    ax.set_ylabel("Count")
    ax.set_title(
        rf"4D random search at $\theta={result.theta_value:.1f}$ "
        rf"({len(finite)} finite points)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 4. θ-scan line plot
# ──────────────────────────────────────────────


def plot_drive_theta_scan(
    result: DriveThetaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 7),
) -> Path:
    """Two-panel figure: Δθ vs θ (top) and Δθ/SQL ratio (bottom)."""
    if isinstance(result, (str, Path)):
        result = DriveThetaScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δθ vs θ ──────────────────────────────────────────
    sql_vals = result.sql_values
    if len(sql_vals) == 0:
        raise ValueError(
            "sql_values is empty; cannot draw SQL reference line. "
            "Ensure the data has a populated 'sql' column."
        )

    # SQL reference line (use the first SQL value as reference)
    ax1.axhline(
        y=sql_vals[0],
        color="C1",
        linestyle="--",
        alpha=0.6,
        label=f"SQL = {sql_vals[0]:.4f}",
    )

    ax1.plot(
        result.theta_values,
        result.best_delta_theta_per_theta,
        marker="o",
        linestyle="-",
        color="C0",
        markersize=8,
        linewidth=2,
        label=r"$\Delta\theta$ (best)",
    )

    # Highlight points below SQL
    below_sql = result.best_delta_theta_per_theta < sql_vals
    if np.any(below_sql):
        ax1.scatter(
            result.theta_values[below_sql],
            result.best_delta_theta_per_theta[below_sql],
            marker="*",
            s=150,
            color="red",
            zorder=5,
            label="Below SQL",
        )

    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title(r"$\theta$-scan: driven-ancilla sensitivity")
    ax1.legend()

    # ── Lower panel: Δθ / SQL ratio ───────────────────────────────────
    ratio = result.best_delta_theta_per_theta / sql_vals

    ax2.plot(
        result.theta_values,
        ratio,
        marker="o",
        linestyle="-",
        color="C0",
        markersize=8,
        linewidth=2,
    )
    ax2.axhline(y=1.0, color="C1", linestyle="--", alpha=0.6, label="SQL")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\Delta\theta \;/\; \mathrm{SQL}$")
    ax2.legend()

    # Annotate the minimum ratio
    min_idx = np.argmin(ratio)
    min_ratio = ratio[min_idx]
    min_theta = result.theta_values[min_idx]
    ax2.annotate(
        f"Best = {min_ratio:.3f}$\\times$ at $\\theta$={min_theta:.1f}",
        xy=(min_theta, min_ratio),
        xytext=(min_theta + 0.6, min_ratio + 0.15),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 5. Optimal parameters vs θ (bar or multi-line)
# ──────────────────────────────────────────────


def plot_drive_optimal_params(
    result: DriveThetaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Plot optimal drive parameters (a_x*, a_y*, a_z*, a_zz*) vs θ."""
    if isinstance(result, (str, Path)):
        result = DriveThetaScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    params_names = [r"$a_x^*$", r"$a_y^*$", r"$a_z^*$", r"$a_{zz}^*$"]
    colours = ["C0", "C1", "C2", "C3"]
    markers = ["o", "s", "^", "D"]

    for idx in range(4):
        values = [
            params[idx] if len(params) > idx else 0.0
            for params in result.best_params_per_theta
        ]
        ax.plot(
            result.theta_values,
            values,
            marker=markers[idx],
            linestyle="-",
            color=colours[idx],
            label=params_names[idx],
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Optimal parameter value")
    ax.set_title("Optimal drive and interaction parameters vs $\\theta$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 6–9. Combined sensitivity, NM expectation/variance,
#       cross-experiment comparison, fraction below SQL
# ──────────────────────────────────────────────
#
# These plot functions have been moved to ``reports/20260519/local.py``
# as they are used exclusively by the 20260519 report.


# ──────────────────────────────────────────────
# 7. NM expectation and variance of J_z vs θ
# ──────────────────────────────────────────────


def plot_drive_nm_expectation_variance(
    theta_values: np.ndarray,
    expectation_Jz: np.ndarray,
    variance_Jz: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Side-by-side plot of ⟨J_z^S⟩ and Var(J_z^S) at the NM optimum vs θ.

    Args:
        theta_values: Array of θ values.
        expectation_Jz: ⟨J_z^S⟩ at NM optimum for each θ.
        variance_Jz: Var(J_z^S) at NM optimum for each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: expectation
    valid_exp = np.isfinite(expectation_Jz)
    if np.any(valid_exp):
        ax1.plot(
            theta_values[valid_exp],
            expectation_Jz[valid_exp],
            "o-",
            color="C0",
            markersize=7,
            linewidth=1.5,
        )
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\langle J_z^S \rangle$")
    ax1.set_title(r"Expectation $\langle J_z^S\rangle$ at NM optimum")

    # Right panel: variance
    valid_var = np.isfinite(variance_Jz)
    if np.any(valid_var):
        ax2.plot(
            theta_values[valid_var],
            variance_Jz[valid_var],
            "s-",
            color="C1",
            markersize=7,
            linewidth=1.5,
        )
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax2.set_title(r"Variance $\mathrm{Var}(J_z^S)$ at NM optimum")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 8. Cross-experiment comparison (fixed vs modulated drive)
# ──────────────────────────────────────────────


def plot_drive_cross_experiment_comparison(
    theta_values: np.ndarray,
    best_delta_19: np.ndarray,
    best_delta_18: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Compare Δθ from the fixed-drive (20260518) and modulated-drive
    (20260519) experiments in a 2×1 vertically stacked figure.

    Upper panel: Overlaid line plots of Δθ vs θ for both experiments,
    with the SQL shown as a dashed reference line.

    Lower panel: Ratio Δθ_19 / Δθ_18 vs θ. A horizontal line at y=1
    separates regimes where the fixed drive (above 1) or modulated drive
    (below 1) performs better.

    Args:
        theta_values: Common θ grid (50 points from the modulated-drive scan).
        best_delta_19: Δθ from the modulated-drive scan (20260519).
        best_delta_18: Δθ from the fixed-drive scan (20260518),
            interpolated to the same θ grid.
        sql_values: SQL reference values (constant, 0.1) at each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δθ vs θ ──────────────────────────────────────────
    sql_ref = float(sql_values[0]) if len(sql_values) > 0 else 0.1

    ax1.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=rf"SQL = {sql_ref:.4f}",
    )

    ax1.plot(
        theta_values,
        best_delta_18,
        marker="s",
        linestyle="-",
        color="C0",
        markersize=5,
        linewidth=1.8,
        label=r"Fixed drive (20260518)",
    )
    ax1.plot(
        theta_values,
        best_delta_19,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=5,
        linewidth=1.8,
        label=r"Modulated drive (20260519)",
    )

    ax1.set_ylabel(r"$\Delta\theta$")
    ax1.set_title("Cross-experiment comparison: fixed vs modulated drive")
    ax1.legend(fontsize=9)

    # ── Lower panel: ratio Δθ_19 / Δθ_18 ──────────────────────────────
    # Guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(best_delta_18) & (best_delta_18 > 0),
            best_delta_19 / best_delta_18,
            np.nan,
        )

    ax2.plot(
        theta_values,
        ratio,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=4,
        linewidth=1.5,
    )
    ax2.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="y = 1"
    )

    # Annotate the minimum ratio
    valid = np.isfinite(ratio)
    if np.any(valid):
        min_idx = np.argmin(ratio[valid])
        min_ratio = float(ratio[valid][min_idx])
        min_theta = float(theta_values[valid][min_idx])
        ax2.annotate(
            f"Best = {min_ratio:.3f}$\\times$ at $\\theta$={min_theta:.1f}",
            xy=(min_theta, min_ratio),
            xytext=(min_theta + 0.6, min_ratio + 0.15),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\Delta\theta_{19} \;/\; \Delta\theta_{18}$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 9. Fraction below SQL vs θ
# ──────────────────────────────────────────────


def plot_drive_fraction_below_sql(
    theta_values: np.ndarray,
    fractions_2d_ax: np.ndarray,
    fractions_2d_ay: np.ndarray,
    fractions_random: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of the fraction of parameter space below SQL as a function of θ.

    Args:
        theta_values: Array of θ values.
        fractions_2d_ax: Fraction below SQL from (a_x, a_zz) slices at each θ.
        fractions_2d_ay: Fraction below SQL from (a_y, a_zz) slices at each θ.
        fractions_random: Fraction below SQL from 4D random search at each θ.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        theta_values,
        fractions_2d_ax,
        "o-",
        color="C0",
        label=r"2D slice $(a_x, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        theta_values,
        fractions_2d_ay,
        "s-",
        color="C1",
        label=r"2D slice $(a_y, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        theta_values,
        fractions_random,
        "^-",
        color="C2",
        label="4D random search",
        markersize=6,
        linewidth=1.5,
    )

    # Reference lines at y=0 and y=1
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Fraction below SQL")
    ax.set_title("Robustness of SQL violation: fraction of parameter space below SQL")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
