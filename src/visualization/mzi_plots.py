r"""Shared MZI plot functions for report modules.

Provides the standard overlay and scaling plots used by MZI scaling reports:

- :func:`plot_delta_omega_overlay` — :math:`\Delta\omega` vs :math:`\omega` for
  multiple resource values on a single panel.
- :func:`plot_scaling` — log-log plot of best :math:`\Delta\omega` vs resource
  parameter with analytical QFI bounds and scaling-exponent fits.
- :func:`maybe_plot_delta_omega_overlays` — orchestration for the overlay figure
  across multiple state types.
- :func:`maybe_plot_scaling_comparison` — orchestration for the combined scaling
  figure.
- :func:`generate_plots` — combined plot-generation entry point.

All functions operate on
:class:`~src.analysis.sensitivity_metrics.MziSensitivityData` (or subclasses).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.scaling_fit import fit_scaling_exponent
from src.analysis.sensitivity_metrics import (
    MziSensitivityData,
    analyse_best_worst_sensitivity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# Ensure we pick up the user's rcParams (seaborn style is set in each report).
mpl.use("Agg")


def plot_delta_omega_overlay(
    data: MziSensitivityData,
    selected_R: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    r"""Overlay :math:`\Delta\omega_C` and :math:`\Delta\omega_Q` vs
    :math:`\omega` for multiple resource values on a single panel.

    Each resource value gets a unique colour from the *viridis* colormap.
    Solid lines show :math:`\Delta\omega_C`, dashed horizontal lines show the
    corresponding QFI bound.  The y-axis uses a log scale so that different
    resource values are clearly separated.

    Args:
        data: Sensitivity data containing all resource values.
        selected_R: Which resource values to include (auto-selected if
            ``None`` by sampling up to 7 values evenly).
        save_path: Output SVG path. Auto-generated from ``data.state_type``
            if ``None``.
        title: Optional title override. If ``None``, uses
            ``"{state_type} — Phase Sensitivity vs ω"``.

    Returns:
        Path to saved SVG.
    """
    if selected_R is None:
        n_R = len(data.resource_values)
        step = max(1, n_R // 7)
        selected_R = [float(data.resource_values[i]) for i in range(0, n_R, step)]

    if save_path is None:
        save_path = Path(f"{data.state_type}_delta_omega_comparison.svg")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"{data.state_type.upper()} — Phase Sensitivity vs $\\omega$"

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.15, 0.85, len(selected_R)))

    for idx, R_val in enumerate(selected_R):
        match = np.where(np.isclose(data.resource_values, R_val, rtol=1e-10))[0]
        if len(match) == 0:
            continue
        r_idx = match[0]
        omega = data.omega_values
        dt_c = data.delta_omega_c_grid[r_idx, :]
        dt_q = data.delta_omega_q_per_R[r_idx]

        c_finite = np.isfinite(dt_c)
        if np.any(c_finite):
            ax.semilogy(
                omega[c_finite],
                dt_c[c_finite],
                color=colors[idx],
                linewidth=1.5,
                label=rf"R={R_val}  $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        # QFI bound (dashed horizontal)
        ax.axhline(
            y=float(dt_q),
            color=colors[idx],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling(
    data_list: Sequence[MziSensitivityData | None],
    labels: Sequence[str],
    save_path: str | Path | None = None,
    N_min_fit: float = 4.0,
    xlabel: str = "Resource parameter $R$",
    title: str = "Phase Sensitivity Scaling in Standard MZI",
) -> Path:
    """Log-log plot of best :math:`\\Delta\\omega` vs resource parameter with
    analytical QFI bounds and scaling-exponent fits.

    Overlays multiple state types on a single figure.  Reference lines
    for :math:`\\propto 1/R` (Heisenberg) and :math:`\\propto 1/\\sqrt{R}`
    (SQL) are included.

    Args:
        data_list: List of sensitivity data (entries may be ``None``).
        labels: Display labels for each data entry.
        save_path: Output SVG path. Defaults to ``"scaling_comparison.svg"``.
        N_min_fit: Minimum resource value for exponent fits.
        xlabel: X-axis label.
        title: Figure title.

    Returns:
        Path to saved SVG.

    Raises:
        ValueError: If all entries in *data_list* are ``None``.
    """
    if all(d is None for d in data_list):
        raise ValueError("At least one data set must be provided")

    if save_path is None:
        save_path = Path("scaling_comparison.svg")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    t_hold = 10.0  # Default; overridden if any data has a different value

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference lines — use a generic resource axis
    R_ref = np.logspace(0, 1.5, 50)
    ax.plot(
        R_ref,
        1.0 / (t_hold * R_ref),
        "k--",
        alpha=0.4,
        label=r"$\propto 1/R$ (Heisenberg)",
    )
    ax.plot(
        R_ref,
        1.0 / (t_hold * np.sqrt(R_ref)),
        "k:",
        alpha=0.4,
        label=r"$\propto 1/\sqrt{R}$ (SQL)",
    )

    colours = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (data, label) in enumerate(zip(data_list, labels, strict=False)):
        if data is None:
            continue
        colour = colours[i % len(colours)]
        marker = markers[i % len(markers)]

        # QFI bound
        ax.loglog(
            data.resource_values,
            data.delta_omega_q_per_R,
            f"{colour}--",
            alpha=0.5,
            label=f"{label} QFI bound",
        )

        # Best Δω_C at each resource value
        analysis = analyse_best_worst_sensitivity(
            data.resource_values,
            data.omega_values,
            data.delta_omega_c_grid,
        )
        R_vals = analysis["resource_values"]
        best_dt_c = analysis["best_sensitivity"]
        finite = np.isfinite(best_dt_c)
        if np.any(finite):
            ax.loglog(
                R_vals[finite],
                best_dt_c[finite],
                f"{colour}{marker}-",
                label=rf"{label} best $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        # Fit exponent
        best_c_finite = np.array(best_dt_c, dtype=float)
        fit_result = fit_scaling_exponent(
            np.array(R_vals, dtype=float), best_c_finite, min_N=int(N_min_fit)
        )
        if fit_result.valid:
            N_fit = fit_result.N_values
            delta_fit = fit_result.C * N_fit**fit_result.alpha
            ax.loglog(
                N_fit,
                delta_fit,
                f"{colour}--",
                alpha=0.7,
                linewidth=1.5,
                label=f"{label}: "
                rf"$\alpha = {fit_result.alpha:.3f}$",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def maybe_plot_delta_omega_overlays(
    results: Mapping[str, MziSensitivityData],
    state_configs: Sequence[tuple[str, Any, str]],
    force: bool,
    only: str | None,
    fig_path_fn: Callable[[str], Path],
) -> None:
    r"""Plot :math:`\Delta\omega` overlay figures for each state type.

    Args:
        results: Mapping of state type to sensitivity data.
        state_configs: List of ``(state_type, resource_range, label)`` tuples.
        force: Re-generate even if SVG exists.
        only: If set, only plot for the specified state type.
        fig_path_fn: Callable that accepts a filename and returns a Path.
    """
    for st, _r_range, _label in state_configs:
        if only is not None and st != only:
            continue
        data = results.get(st)
        if data is None:
            continue
        overlay_path = fig_path_fn(f"{st}_delta_omega_comparison")
        if not overlay_path.exists() or force:
            plot_delta_omega_overlay(data, save_path=overlay_path)
            print(f"  Plotted {overlay_path}")


def maybe_plot_scaling_comparison(
    results: Mapping[str, MziSensitivityData],
    force: bool,
    fig_path_fn: Callable[[str], Path],
    data_keys: Sequence[str] | None = None,
    data_labels: Sequence[str] | None = None,
    xlabel: str = "Resource parameter $R$",
    title: str = "Phase Sensitivity Scaling in Standard MZI",
) -> None:
    """Plot combined scaling comparison from multiple state types.

    Args:
        results: Mapping of state type to sensitivity data.
        force: Re-generate even if SVG exists.
        fig_path_fn: Callable that accepts a filename and returns a Path.
        data_keys: Keys into ``results`` to include. Defaults to all keys.
        data_labels: Display labels corresponding to *data_keys*.
        xlabel: X-axis label.
        title: Figure title.
    """
    path = fig_path_fn("scaling_comparison")
    if not path.exists() or force:
        if data_keys is None:
            data_keys = list(results.keys())
        if data_labels is None:
            data_labels = [k.upper() for k in data_keys]
        plot_scaling(
            [results.get(k) for k in data_keys],
            data_labels,
            save_path=path,
            xlabel=xlabel,
            title=title,
        )
        print(f"  Plotted {path}")


def generate_plots(
    results: Mapping[str, MziSensitivityData],
    state_configs: Sequence[tuple[str, Any, str]],
    force: bool,
    only: str | None,
    fig_path_fn: Callable[[str], Path],
    scaling_data_keys: Sequence[str] | None = None,
    scaling_data_labels: Sequence[str] | None = None,
    xlabel: str = "Resource parameter $R$",
    scaling_title: str = "Phase Sensitivity Scaling in Standard MZI",
) -> None:
    """Generate all standard plots from computed sensitivity data.

    Args:
        results: Mapping of state type to sensitivity data.
        state_configs: List of ``(state_type, resource_range, label)`` tuples.
        force: Re-generate plots even if SVG files exist.
        only: If set, only plot for the specified state type.
        fig_path_fn: Callable ``(filename: str) -> Path``.
        scaling_data_keys: Keys into ``results`` for the scaling plot.
        scaling_data_labels: Display labels for *scaling_data_keys*.
        xlabel: X-axis label for the scaling plot.
        scaling_title: Title for the scaling plot.
    """
    maybe_plot_delta_omega_overlays(results, state_configs, force, only, fig_path_fn)
    maybe_plot_scaling_comparison(
        results,
        force,
        fig_path_fn,
        data_keys=scaling_data_keys,
        data_labels=scaling_data_labels,
        xlabel=xlabel,
        title=scaling_title,
    )
