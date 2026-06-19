"""
Figure generation for the 20260616 combined protocol report.

Each function loads Parquet data via pandas, generates a full-resolution SVG,
and saves it to the report's figures directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_dir(save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)


def plot_n1_omega_scan(
    parquet_path: str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 8),
) -> Path:
    """Two-panel figure: Δω vs ω (top) and R = SQL/Δω (bottom) for N=1.

    Args:
        parquet_path: Path to the N=1 ω scan Parquet file.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("omega")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    sql_val = float(df["sql"].iloc[0])
    ax1.axhline(
        y=sql_val,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=rf"SQL = {sql_val:.4f}",
    )
    ax1.plot(
        df["omega"],
        df["delta_omega_opt"],
        "o-",
        color="C0",
        markersize=6,
        linewidth=1.8,
        label=r"$\Delta\omega_{\mathrm{opt}}$",
    )
    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(r"$N=1$: Optimal sensitivity vs $\omega$ (combined protocol)")
    ax1.legend()

    ax2.plot(
        df["omega"],
        df["ratio"],
        "o-",
        color="C3",
        markersize=6,
        linewidth=1.8,
    )
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="SQL (R=1)")
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$R = \Delta\omega_{\mathrm{SQL}}/\Delta\omega_{\mathrm{opt}}$")
    ax2.set_title(r"SQL-violation ratio $R$ vs $\omega$")

    best_idx = int(df["ratio"].idxmax())
    best_row = df.loc[best_idx]
    ax2.annotate(
        rf"Best: $R={best_row['ratio']:.3f}$ at $\omega={best_row['omega']:.1f}$",
        xy=(best_row["omega"], best_row["ratio"]),
        xytext=(best_row["omega"] + 0.8, best_row["ratio"] + 0.3),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")
    return save_path


def plot_n1_2d_slice(
    parquet_path: str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (9, 7),
) -> Path:
    """Heatmap of log₁₀(Δω) over (α_xx, α_zz) at N=1.

    Args:
        parquet_path: Path to the 2D slice Parquet file.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    df = pd.read_parquet(parquet_path)
    omega_val = float(df["omega_value"].iloc[0])
    a_x = float(df["a_x_fixed"].iloc[0])
    a_y = float(df["a_y_fixed"].iloc[0])
    a_z = float(df["a_z_fixed"].iloc[0])
    sql_val = float(df["sql"].iloc[0])

    alpha_xx_vals = sorted(df["alpha_xx"].unique())
    alpha_zz_vals = sorted(df["alpha_zz"].unique())
    n_xx = len(alpha_xx_vals)
    n_zz = len(alpha_zz_vals)
    grid = np.full((n_xx, n_zz), np.nan, dtype=float)
    for _, row in df.iterrows():
        i = alpha_xx_vals.index(row["alpha_xx"])
        j = alpha_zz_vals.index(row["alpha_zz"])
        grid[i, j] = row["delta_omega"]

    fig, ax = plt.subplots(figsize=figsize)

    grid_log = np.log10(np.maximum(grid, 1e-10))
    im = ax.pcolormesh(
        alpha_zz_vals,
        alpha_xx_vals,
        grid_log,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label=r"$\log_{10}(\Delta\omega)$")

    cs = ax.contour(
        alpha_zz_vals,
        alpha_xx_vals,
        grid,
        levels=[sql_val],
        colors="red",
        linewidths=1.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt=f"SQL = {sql_val:.3f}", fontsize=9)

    ax.set_xlabel(r"$\alpha_{zz}$")
    ax.set_ylabel(r"$\alpha_{xx}$")
    ax.set_title(
        rf"$\Delta\omega$ landscape at $N=1$, $\omega={omega_val:.1f}$\n"
        rf"(fixed $a_x={a_x:.1f}, a_y={a_y:.1f}, a_z={a_z:.1f}$,"
        rf" $\alpha_{{xz}}=\alpha_{{zx}}=0$)"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")
    return save_path


def plot_combined_n_scaling_sensitivity(
    parquet_path: str | Path,
    save_path: str | Path,
    label_prefix: str = "Combined",
    figsize: tuple[float, float] = (10, 6),
    t_hold: float = 10.0,
) -> Path:
    """Log-log plot of Δω_opt vs N, coloured by ω.

    Args:
        parquet_path: Path to the N-scaling Parquet file.
        save_path: Output SVG path.
        label_prefix: Prefix for legend labels.
        figsize: Figure size.
        t_hold: Holding time for SQL/HL reference.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(f"[skip] Empty data in {parquet_path}")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)].sort_values("N")
        valid = np.isfinite(sub["delta_omega_opt"]) & (sub["delta_omega_opt"] > 0)
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

    N_range = np.linspace(1, max(df["N"]), 100)
    sql_line = 1.0 / (np.sqrt(N_range) * t_hold)
    hl_line = 1.0 / (N_range * t_hold)

    ax.loglog(
        N_range,
        sql_line,
        "k--",
        alpha=0.7,
        linewidth=1.5,
        label=r"SQL $1/(\sqrt{N}t_H)$",
    )
    ax.loglog(N_range, hl_line, "k:", alpha=0.5, linewidth=1.2, label=r"HL $1/(N t_H)$")

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{opt}}$")
    ax.set_title(f"{label_prefix} protocol: optimal sensitivity vs $N$")
    ax.legend(fontsize=9, title=r"$\omega$")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")
    return save_path


def plot_combined_n_scaling_ratio(
    parquet_path: str | Path,
    save_path: str | Path,
    label_prefix: str = "Combined",
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot R(N) = SQL/Δω_opt vs N, coloured by ω.

    Args:
        parquet_path: Path to the N-scaling Parquet file.
        save_path: Output SVG path.
        label_prefix: Prefix for plot title.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(f"[skip] Empty data in {parquet_path}")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)].sort_values("N")
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
    ax.set_title(f"{label_prefix} protocol: SQL-violation ratio vs $N$")
    ax.legend(fontsize=9, title=r"$\omega$")
    ax.set_xlim(left=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")
    return save_path


def plot_combined_optimal_params_vs_n(
    parquet_path: str | Path,
    save_path: str | Path,
    label_prefix: str = "Combined",
    figsize: tuple[float, float] = (14, 10),
) -> Path:
    """Plot optimal parameters vs N for the combined protocol.

    Args:
        parquet_path: Path to the N-scaling Parquet file.
        save_path: Output SVG path.
        label_prefix: Prefix for plot title.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(f"[skip] Empty data in {parquet_path}")
        return save_path

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    param_cols = [
        ("a_x_opt", r"$a_x^*$"),
        ("a_y_opt", r"$a_y^*$"),
        ("a_z_opt", r"$a_z^*$"),
        ("alpha_xx_opt", r"$\alpha_{xx}^*$"),
        ("alpha_xz_opt", r"$\alpha_{xz}^*$"),
        ("alpha_zx_opt", r"$\alpha_{zx}^*$"),
        ("alpha_zz_opt", r"$\alpha_{zz}^*$"),
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(param_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (param_col, param_label) in zip(axes, param_cols, strict=False):
        for omega_val, colour in zip(omega_values, colours, strict=False):
            sub = df[np.isclose(df["omega"], omega_val)].sort_values("N")
            ax.plot(
                sub["N"],
                sub[param_col],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$" if ax == axes[0] else None,
                markersize=5,
                linewidth=1.2,
            )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(param_label)
        ax.set_xlabel(r"$N$")

    for ax in axes[len(param_cols) :]:
        ax.set_visible(False)

    axes[0].legend(fontsize=8, title=r"$\omega$", loc="upper right")
    fig.suptitle(f"{label_prefix} protocol: optimal parameters vs $N$")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {save_path}")
    return save_path
