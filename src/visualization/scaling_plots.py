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
    T_hold: float = 10.0,
) -> Path:
    """Plot Δω_opt vs N on log-log axes, coloured by ω.

    SQL and Heisenberg-limit lines are shown for reference.

    Args:
        df: DataFrame with columns 'N', 'omega', 'delta_omega_opt'.
        save_path: Output SVG path.
        figsize: Figure size.
        T_hold: Holding time for reference lines.

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
    sql_line = 1.0 / (np.sqrt(N_range) * T_hold)
    ax.loglog(
        N_range,
        sql_line,
        "k--",
        alpha=0.7,
        linewidth=1.5,
        label=r"SQL $1/(\sqrt{N}T_H)$",
    )

    hl_line = 1.0 / (N_range * T_hold)
    ax.loglog(
        N_range,
        hl_line,
        "k:",
        alpha=0.5,
        linewidth=1.2,
        label=r"HL $1/(N T_H)$",
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
