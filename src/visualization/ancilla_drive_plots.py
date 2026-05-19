"""
Plotting functions for driven-ancilla metrology results.

Each function accepts a result dataclass (or a path to a CSV file),
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
        result = DriveDecoupledBaselineResult.from_csv(result)

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
        result = Drive2DSliceResult.from_csv(result)

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
        result = DriveRandomSearchResult.from_csv(result)

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
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of Δθ vs θ with SQL reference and optimal parameters table."""
    if isinstance(result, (str, Path)):
        result = DriveThetaScanResult.from_csv(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_table) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # SQL reference
    sql_vals = result.sql_values
    ax.axhline(
        y=sql_vals[0] if len(sql_vals) > 0 else 0.1,
        color="C1",
        linestyle="--",
        alpha=0.6,
        label=r"SQL $= 1/T_H$",
    )

    # Δθ vs θ
    ax.plot(
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
        ax.scatter(
            result.theta_values[below_sql],
            result.best_delta_theta_per_theta[below_sql],
            marker="*",
            s=150,
            color="red",
            zorder=5,
            label="Below SQL",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(r"$\theta$-scan: driven-ancilla sensitivity")
    ax.legend()

    # Table of optimal parameters
    col_labels = [
        r"$\theta$",
        r"$a_x^*$",
        r"$a_y^*$",
        r"$a_z^*$",
        r"$a_{zz}^*$",
        r"$\Delta\theta$",
    ]
    cell_data: list[list[str]] = []
    for i, theta in enumerate(result.theta_values):
        params = (
            result.best_params_per_theta[i]
            if i < len(result.best_params_per_theta)
            else (0, 0, 0, 0)
        )
        dt = (
            result.best_delta_theta_per_theta[i]
            if i < len(result.best_delta_theta_per_theta)
            else float("inf")
        )
        cell_data.append(
            [
                f"{theta:.1f}",
                f"{params[0]:.3f}",
                f"{params[1]:.3f}",
                f"{params[2]:.3f}",
                f"{params[3]:.3f}",
                f"{dt:.4f}" if np.isfinite(dt) else "inf",
            ]
        )

    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

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
        result = DriveThetaScanResult.from_csv(result)

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
