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

from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)

sns.set_theme(style="whitegrid")


def _markevery(n_pts: int, target: int = 30) -> int:
    """Compute marker interval to show roughly *target* markers on a line plot.

    For 500 points this returns 16 (≈31 visible markers); for 50 points
    it returns 1 (every point gets a marker).
    """
    return max(1, n_pts // target)


# ──────────────────────────────────────────────
# 1. Decoupled baseline
# ──────────────────────────────────────────────


def plot_drive_decoupled_baseline(
    result: DriveDecoupledBaselineResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    """Bar chart comparing Δω to SQL for the decoupled (all-zero) configuration."""
    if isinstance(result, (str, Path)):
        result = DriveDecoupledBaselineResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    labels = [r"$\Delta\omega$", r"SQL $= 1/t_hold$"]
    values = [result.delta_omega, result.sql]
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

    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(f"Decoupled baseline at $t_hold={result.t_hold_value:.0f}$")
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
    sql_value: float | None = None,
) -> Path:
    """Heatmap of Δω over (a_drive, a_zz) with SQL contour."""
    if isinstance(result, (str, Path)):
        result = Drive2DSliceResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sql = sql_value if sql_value is not None else result.sql
    if vmax is None:
        # Clip to 3x SQL for colour scale
        vmax = min(3.0 * sql, np.nanmax(result.delta_omega_grid))
        if not np.isfinite(vmax):
            vmax = 3.0 * sql

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        result.azz_values,
        result.drive_values,
        result.delta_omega_grid,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    _cbar = fig.colorbar(im, ax=ax, label=r"$\Delta\omega$")

    # SQL contour line
    cs = ax.contour(
        result.azz_values,
        result.drive_values,
        result.delta_omega_grid,
        levels=[sql],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt=f"SQL = {sql:.3f}", fontsize=9)

    if result.slice_type == "ax":
        drive_label = r"$a_x$"
    elif result.slice_type == "ay":
        drive_label = r"$a_y$"
    else:
        drive_label = r"$a_z$"

    # Minimum marker and annotation (skip when the grid is essentially flat —
    # i.e., all finite values equal to within numerical precision, so no
    # meaningful minimum exists).  We consider only finite values because
    # non-converged (inf) cells are numerical outliers, not physical structure.
    grid = result.delta_omega_grid
    finite_vals = grid[np.isfinite(grid)]
    grid_ptp = finite_vals.max() - finite_vals.min() if len(finite_vals) > 0 else 0.0
    if grid_ptp > 1e-9:
        min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
        min_drive = result.drive_values[min_idx[0]]
        min_azz = result.azz_values[min_idx[1]]
        min_val = result.delta_omega_grid[min_idx]
        ax.plot(
            min_azz,
            min_drive,
            marker="*",
            color="white",
            markersize=14,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"Min = {min_val:.4f}\n({drive_label}={min_drive:.2f}, $a_{{zz}}$={min_azz:.2f})",
            xy=(min_azz, min_drive),
            xytext=(min_azz + 0.8, min_drive + 0.6),
            arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
            fontsize=9,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "edgecolor": "white",
                "alpha": 0.7,
            },
            zorder=6,
        )
    ax.set_xlabel(r"$a_{zz}$ (interaction)")
    ax.set_ylabel(drive_label + " (drive)")
    ax.set_title(
        rf"$\Delta\omega$ vs {drive_label} and $a_{{zz}}$ at $\omega={result.omega_value:.1f}$"
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
    """Histogram of Δω values from 4D random search with SQL and best marked."""
    if isinstance(result, (str, Path)):
        result = DriveRandomSearchResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    finite = result.delta_omega_values[np.isfinite(result.delta_omega_values)]
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
        x=result.best_delta_omega,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=rf"Best = {result.best_delta_omega:.4f}",
    )

    below_sql = np.sum(finite < sql)
    if below_sql > 0:
        ax.annotate(
            rf"{below_sql} / {len(finite)} below SQL",
            xy=(result.best_delta_omega, 0.6 * ax.get_ylim()[1]),
            fontsize=10,
            color="red",
        )

    ax.set_xlabel(r"$\Delta\omega$")
    ax.set_ylabel("Count")
    ax.set_title(
        rf"4D random search at $\omega={result.omega_value:.1f}$ "
        rf"({len(finite)} finite points)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 4. ω-scan line plot
# ──────────────────────────────────────────────


def plot_drive_omega_scan(
    result: DriveOmegaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 7),
) -> Path:
    """Two-panel figure: Δω vs ω (top) and Δω/SQL ratio (bottom)."""
    if isinstance(result, (str, Path)):
        result = DriveOmegaScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_omega = len(result.omega_values)
    me = _markevery(n_omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δω vs ω ──────────────────────────────────────────
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
        result.omega_values,
        result.best_delta_omega_per_omega,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=8,
        linewidth=2,
        label=r"$\Delta\omega$ (best)",
    )

    # Highlight points below SQL
    below_sql = result.best_delta_omega_per_omega < sql_vals
    if np.any(below_sql):
        ax1.scatter(
            result.omega_values[below_sql],
            result.best_delta_omega_per_omega[below_sql],
            marker="*",
            s=150,
            color="red",
            zorder=5,
            label="Below SQL",
        )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(r"$\omega$-scan: driven-ancilla sensitivity")
    ax1.legend()

    # ── Lower panel: Δω / SQL ratio ───────────────────────────────────
    ratio = result.best_delta_omega_per_omega / sql_vals

    ax2.plot(
        result.omega_values,
        ratio,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=8,
        linewidth=2,
    )
    ax2.axhline(y=1.0, color="C1", linestyle="--", alpha=0.6, label="SQL")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega \;/\; \mathrm{SQL}$")
    ax2.legend()

    # Annotate the minimum ratio
    min_idx = np.argmin(ratio)
    min_ratio = ratio[min_idx]
    min_omega = result.omega_values[min_idx]
    omega_fmt = ".2f" if n_omega > 100 else ".1f"
    ax2.annotate(
        f"Best = {min_ratio:.3f}$\\times$ at $\\omega$={min_omega:{omega_fmt}}",
        xy=(min_omega, min_ratio),
        xytext=(min_omega + 0.6, min_ratio + 0.15),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 5. Optimal parameters vs ω (bar or multi-line)
# ──────────────────────────────────────────────


def plot_drive_optimal_params(
    result: DriveOmegaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Plot optimal drive parameters (a_x*, a_y*, a_z*, a_zz*) vs ω."""
    if isinstance(result, (str, Path)):
        result = DriveOmegaScanResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    n_omega = len(result.omega_values)
    me = _markevery(n_omega)

    params_names = [r"$a_x^*$", r"$a_y^*$", r"$a_z^*$", r"$a_{zz}^*$"]
    colours = ["C0", "C1", "C2", "C3"]
    markers = ["o", "s", "^", "D"]

    for idx in range(4):
        values = [
            params[idx] if len(params) > idx else 0.0
            for params in result.best_params_per_omega
        ]
        ax.plot(
            result.omega_values,
            values,
            marker=markers[idx],
            markevery=me,
            linestyle="-",
            color=colours[idx],
            label=params_names[idx],
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Optimal parameter value")
    ax.set_title("Optimal drive and interaction parameters vs $\\omega$")
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
# These plot functions have been moved to ``reports/20260519/phase_modulated_drive.py``
# as they are used exclusively by the 20260519 report.


# ──────────────────────────────────────────────
# 7. NM expectation and variance of J_z vs ω
# ──────────────────────────────────────────────


def plot_drive_nm_expectation_variance(
    omega_values: np.ndarray,
    expectation_Jz: np.ndarray,
    variance_Jz: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Side-by-side plot of ⟨J_z^S⟩ and Var(J_z^S) at the NM optimum vs ω.

    Args:
        omega_values: Array of ω values.
        expectation_Jz: ⟨J_z^S⟩ at NM optimum for each ω.
        variance_Jz: Var(J_z^S) at NM optimum for each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_omega = len(omega_values)
    me = _markevery(n_omega)

    # Left panel: expectation
    valid_exp = np.isfinite(expectation_Jz)
    if np.any(valid_exp):
        ax1.plot(
            omega_values[valid_exp],
            expectation_Jz[valid_exp],
            "o-",
            markevery=me,
            color="C0",
            markersize=7,
            linewidth=1.5,
        )
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\langle J_z^S \rangle$")
    ax1.set_title(r"Expectation $\langle J_z^S\rangle$ at NM optimum")

    # Right panel: variance
    valid_var = np.isfinite(variance_Jz)
    if np.any(valid_var):
        ax2.plot(
            omega_values[valid_var],
            variance_Jz[valid_var],
            "s-",
            markevery=me,
            color="C1",
            markersize=7,
            linewidth=1.5,
        )
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\mathrm{Var}(J_z^S)$")
    ax2.set_title(r"Variance $\mathrm{Var}(J_z^S)$ at NM optimum")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 7b. Combined sensitivity — 2D slices, 4D random search, NM refinement
# ──────────────────────────────────────────────


def plot_combined_sensitivity(
    omega_values: np.ndarray,
    best_ax_slice: np.ndarray,
    best_ay_slice: np.ndarray,
    best_random: np.ndarray,
    best_nm: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    sql_value: float | None = None,
    best_az_slice: np.ndarray | None = None,
) -> Path:
    """Line plot comparing Δω from 2D slices, 4D random search, NM refinement, and SQL.

    Args:
        omega_values: Array of ω values.
        best_ax_slice: Best Δω from (a_x, a_zz) slice at each ω.
        best_ay_slice: Best Δω from (a_y, a_zz) slice at each ω.
        best_random: Best Δω from 4D random search at each ω.
        best_nm: Best Δω from Nelder–Mead refinement at each ω.
        sql_values: SQL reference at each ω (constant).
        save_path: Output SVG path.
        figsize: Figure size (width, height).
        title: Plot title. If None, a default is used.
        best_az_slice: Optional best Δω from (a_z, a_zz) slice at each ω.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    n_omega = len(omega_values)
    me = _markevery(n_omega)

    # SQL reference line
    sql = (
        sql_value
        if sql_value is not None
        else (float(sql_values[0]) if len(sql_values) > 0 else 0.1)
    )
    ax.axhline(
        y=sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {sql:.4f}",
    )

    methods: list[tuple[np.ndarray, str, str, str]] = [
        (best_ax_slice, "o-", "C0", r"2D slice $(a_x, a_{zz})$"),
        (best_ay_slice, "s-", "C1", r"2D slice $(a_y, a_{zz})$"),
        (best_random, "^-", "C2", "4D random search"),
        (best_nm, "D-", "C3", "4D Nelder–Mead"),
    ]
    if best_az_slice is not None:
        methods.append(
            (best_az_slice, "v-", "C4", r"2D slice $(a_z, a_{zz})$"),
        )

    for data, fmt, colour, label in methods:
        valid = np.isfinite(data)
        if np.any(valid):
            ax.plot(
                omega_values[valid],
                data[valid],
                fmt,
                color=colour,
                label=label,
                markevery=me,
                markersize=6,
                linewidth=1.5,
                markerfacecolor=colour,
            )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    if title is None:
        title = (
            "Sensitivity vs $\\omega$: 2D slices, 4D random search, "
            "Nelder–Mead refinement"
        )
    ax.set_title(title)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 8. Cross-experiment comparison (fixed vs modulated drive)
# ──────────────────────────────────────────────


def plot_drive_cross_experiment_comparison(
    omega_values: np.ndarray,
    best_delta_19: np.ndarray,
    best_delta_18: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Compare Δω from the fixed-drive (20260518) and modulated-drive
    (20260519) experiments in a 2×1 vertically stacked figure.

    Upper panel: Overlaid line plots of Δω vs ω for both experiments,
    with the SQL shown as a dashed reference line.

    Lower panel: Ratio Δω_19 / Δω_18 vs ω. A horizontal line at y=1
    separates regimes where the fixed drive (above 1) or modulated drive
    (below 1) performs better.

    Args:
        omega_values: Common ω grid (50 points from the modulated-drive scan).
        best_delta_19: Δω from the modulated-drive scan (20260519).
        best_delta_18: Δω from the fixed-drive scan (20260518),
            interpolated to the same ω grid.
        sql_values: SQL reference values (constant, 0.1) at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_omega = len(omega_values)
    me = _markevery(n_omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δω vs ω ──────────────────────────────────────────
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
        omega_values,
        best_delta_18,
        marker="s",
        markevery=me,
        linestyle="-",
        color="C0",
        markersize=5,
        linewidth=1.8,
        label=r"Fixed drive (20260518)",
    )
    ax1.plot(
        omega_values,
        best_delta_19,
        marker="o",
        markevery=me,
        linestyle="-",
        color="C3",
        markersize=5,
        linewidth=1.8,
        label=r"Modulated drive (20260519)",
    )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title("Cross-experiment comparison: fixed vs modulated drive")
    ax1.legend(fontsize=9)

    # ── Lower panel: ratio Δω_19 / Δω_18 ──────────────────────────────
    # Guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(best_delta_18) & (best_delta_18 > 0),
            best_delta_19 / best_delta_18,
            np.nan,
        )

    ax2.plot(
        omega_values,
        ratio,
        marker="o",
        markevery=me,
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
        min_omega = float(omega_values[valid][min_idx])
        omega_fmt = ".2f" if n_omega > 100 else ".1f"
        ax2.annotate(
            f"Best = {min_ratio:.3f}$\\times$ at $\\omega$={min_omega:{omega_fmt}}",
            xy=(min_omega, min_ratio),
            xytext=(min_omega + 0.6, min_ratio + 0.15),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega_{19} \;/\; \Delta\omega_{18}$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 9. Fraction below SQL vs ω
# ──────────────────────────────────────────────


def plot_drive_fraction_below_sql(
    omega_values: np.ndarray,
    fractions_2d_ax: np.ndarray,
    fractions_2d_ay: np.ndarray,
    fractions_2d_az: np.ndarray,
    fractions_random: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of the fraction of parameter space below SQL as a function of ω.

    Args:
        omega_values: Array of ω values.
        fractions_2d_ax: Fraction below SQL from (a_x, a_zz) slices at each ω.
        fractions_2d_ay: Fraction below SQL from (a_y, a_zz) slices at each ω.
        fractions_2d_az: Fraction below SQL from (a_z, a_zz) slices at each ω.
        fractions_random: Fraction below SQL from 4D random search at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    n_omega = len(omega_values)
    me = _markevery(n_omega)

    ax.plot(
        omega_values,
        fractions_2d_ax,
        "o-",
        color="C0",
        label=r"2D slice $(a_x, a_{zz})$",
        markevery=me,
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_2d_ay,
        "s-",
        color="C1",
        label=r"2D slice $(a_y, a_{zz})$",
        markevery=me,
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_2d_az,
        "v-",
        color="C4",
        label=r"2D slice $(a_z, a_{zz})$",
        markevery=me,
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_random,
        "^-",
        color="C2",
        label="4D random search",
        markevery=me,
        markersize=6,
        linewidth=1.5,
    )

    # Reference lines at y=0 and y=1
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Fraction below SQL")
    ax.set_title("Robustness of SQL violation: fraction of parameter space below SQL")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
