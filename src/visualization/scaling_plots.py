"""
N-scaling visualisation for ancilla-drive metrology reports.

Provides three standard plots used by reports #20260611 and #20260612:
- SQL-violation ratio vs N
- Sensitivity vs N (log-log)
- Optimal parameters vs N

All functions accept a ``pd.DataFrame`` (from ``NScalingScanResult.to_dataframe()``)
to decouple visualisation from the specific result dataclass.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd

sns.set_theme(style="whitegrid")


def plot_n_scaling_ratio(
    df: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot SQL-violation ratio R(N) = SQL/Δω_opt vs N, coloured by ω.

    A horizontal line at R=1 indicates the SQL.

    Args:
        df: DataFrame with columns 'N', 'omega', 'ratio'.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")
        valid = np.isfinite(sub["ratio"])
        if np.any(valid):
            ax.plot(
                sub["N"][valid],
                sub["ratio"][valid],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$",
                markersize=6,
                linewidth=1.5,
            )

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="SQL (R=1)",
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$R(N) = \Delta\omega_{\mathrm{SQL}} / \Delta\omega_{\mathrm{opt}}$")
    ax.set_title("SQL-violation ratio vs system particle number $N$")
    ax.legend(fontsize=9, title=r"$\omega$")
    ax.set_xlim(left=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_sensitivity(
    df: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
    t_hold: float = 10.0,
) -> Path:
    """Plot Δω_opt vs N on log-log axes, coloured by ω.

    SQL and Heisenberg-limit lines are shown for reference.

    Args:
        df: DataFrame with columns 'N', 'omega', 'delta_omega_opt'.
        save_path: Output SVG path.
        figsize: Figure size.
        t_hold: Holding time for reference lines.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")
        valid = np.isfinite(sub["delta_omega_opt"])
        if np.any(valid):
            ax.loglog(
                sub["N"][valid],
                sub["delta_omega_opt"][valid],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$",
                markersize=6,
                linewidth=1.5,
            )

    N_range = np.linspace(1, 20, 100)
    sql_line = 1.0 / (np.sqrt(N_range) * t_hold)
    ax.loglog(
        N_range,
        sql_line,
        "k--",
        alpha=0.7,
        linewidth=1.5,
        label=r"SQL $1/(\sqrt{N}t_hold)$",
    )

    hl_line = 1.0 / (N_range * t_hold)
    ax.loglog(
        N_range,
        hl_line,
        "k:",
        alpha=0.5,
        linewidth=1.2,
        label=r"HL $1/(N t_hold)$",
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{opt}}$")
    ax.set_title("Optimal sensitivity vs system particle number $N$")
    ax.legend(fontsize=9, title=r"$\omega$")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_optimal_params(
    df: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Plot optimal parameters (a_x*, a_y*, a_z*, a_zz*) vs N.

    One panel per parameter, coloured by ω.

    Args:
        df: DataFrame with columns 'N', 'omega', 'a_x_opt', 'a_y_opt',
            'a_z_opt', 'a_zz_opt'.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    params = [
        ("a_x_opt", r"$a_x^*$"),
        ("a_y_opt", r"$a_y^*$"),
        ("a_z_opt", r"$a_z^*$"),
        ("a_zz_opt", r"$a_{zz}^*$"),
    ]
    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    for ax, (param_col, param_label) in zip(axes.flat, params, strict=False):
        for omega_val, colour in zip(omega_values, colours, strict=False):
            sub = df[np.isclose(df["omega"], omega_val)]
            sub = sub.sort_values("N")
            ax.plot(
                sub["N"],
                sub[param_col],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$" if ax == axes.flat[0] else None,
                markersize=5,
                linewidth=1.2,
            )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(param_label)

    for ax in axes.flat:
        ax.set_xlabel(r"$N$")
        ax.set_xlim(left=0.5)

    axes.flat[0].legend(fontsize=8, title=r"$\omega$")
    fig.suptitle("Optimal parameters vs system particle number $N$")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_ratio_comparison(
    df_multi: pd.DataFrame,
    df_fixed: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot SQL-violation ratio R(N) comparing multi-particle vs fixed ancilla.

    Shows R_multi(N) (solid) and R_fixed(N) (dashed) for each ω value,
    highlighting the dramatic improvement from J_A = N/2 over J_A = 1/2.

    Args:
        df_multi: DataFrame for multi-particle ancilla (J_A=N/2) with
            columns 'N', 'omega', 'ratio'.
        df_fixed: DataFrame for fixed ancilla (J_A=1/2) with
            columns 'N', 'omega', 'ratio'.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if df_multi.empty:
        print("[skip] No multi-particle data to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df_multi["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        # Multi-particle (solid)
        sub_m = df_multi[np.isclose(df_multi["omega"], omega_val)]
        sub_m = sub_m.sort_values("N")
        valid_m = np.isfinite(sub_m["ratio"])
        if np.any(valid_m):
            ax.plot(
                sub_m["N"][valid_m],
                sub_m["ratio"][valid_m],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$  (multi)",
                markersize=6,
                linewidth=1.8,
            )

        # Fixed (dashed)
        sub_f = df_fixed[np.isclose(df_fixed["omega"], omega_val)]
        sub_f = sub_f.sort_values("N")
        valid_f = np.isfinite(sub_f["ratio"])
        if np.any(valid_f):
            ax.plot(
                sub_f["N"][valid_f],
                sub_f["ratio"][valid_f],
                "s--",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$  (fixed)",
                markersize=4,
                linewidth=1.0,
                alpha=0.6,
            )

    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="SQL (R=1)",
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$R(N) = \Delta\omega_{\mathrm{SQL}} / \Delta\omega_{\mathrm{opt}}$")
    ax.set_title("SQL-violation ratio: multi-particle vs fixed ancilla")
    ax.legend(fontsize=8, title=r"Protocol", ncol=2)
    ax.set_xlim(left=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_single_omega(
    df: pd.DataFrame,
    omega_fixed: float,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
    t_hold: float | None = None,
    include_2n_sql: bool = False,
    title: str | None = None,
    xlabel: str = r"$N$ (particles per subsystem)",
    ylabel: str = r"$\Delta\omega$",
) -> Path:
    r"""Plot Δω_opt vs N for a single ω value, on log-log axes.

    SQL and Heisenberg-limit reference lines are shown.  The
    *t_hold* parameter is inferred from ``df['t_hold']`` (the unique
    value) when not given explicitly.

    Args:
        df: DataFrame with columns 'omega', 'N', 'delta_omega_opt'.
        omega_fixed: Only this ω value is plotted.
        save_path: Output SVG path.
        figsize: Figure size in inches.
        t_hold: Holding time for reference lines.  Inferred from
            ``df['t_hold']`` if ``None`` (the unique value is used).
        include_2n_sql: If ``True``, also draw the 2N-SQL reference
            line.
        title: Plot title.  Defaults to
            ``f"N-Scaling at $\omega={omega_fixed:.2f}$"``.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Path to the saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to the requested ω
    sub = df[np.isclose(df["omega"], omega_fixed)]
    if sub.empty:
        print(f"[skip] No data for ω={omega_fixed} to plot.")
        return save_path

    sub = sub.sort_values("N")

    # Infer t_hold if not provided
    if t_hold is None:
        t_hold = float(sub["t_hold"].iloc[0])

    fig, ax = plt.subplots(figsize=figsize)

    # Reference lines
    n_min = float(sub["N"].min())
    n_max = float(sub["N"].max())
    N_dense = np.logspace(np.log10(max(n_min, 1)), np.log10(n_max), 100)
    sql_line = 1.0 / (np.sqrt(N_dense) * t_hold)
    hl_line = 1.0 / (N_dense * t_hold)

    ax.loglog(N_dense, sql_line, "--", color="gray", alpha=0.7, label="SQL")
    if include_2n_sql:
        sql_2n_line = 1.0 / (np.sqrt(2 * N_dense) * t_hold)
        ax.loglog(N_dense, sql_2n_line, "-.", color="gray", alpha=0.5, label="2N-SQL")
    ax.loglog(N_dense, hl_line, ":", color="gray", alpha=0.5, label="HL")

    # Data
    valid = np.isfinite(sub["delta_omega_opt"]) & (sub["delta_omega_opt"] > 0)
    if np.any(valid):
        ax.loglog(
            sub["N"][valid],
            sub["delta_omega_opt"][valid],
            "o-",
            color="C0",
            markersize=8,
            linewidth=1.8,
            label=rf"$\Delta\omega_{{\mathrm{{opt}}}}(\omega={omega_fixed:.2f})$",
        )

    if title is None:
        title = rf"N-Scaling at $\omega={omega_fixed:.2f}$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
