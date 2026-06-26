"""
Heatmap visualisations for coupling-sweep results across (N, ω).

Provides two promoted functions — :func:`plot_ratio_heatmap` and
:func:`plot_alpha_opt_heatmap` — that were previously duplicated in
three XX-coupling reports (20260522, 20260523, 20260525).

Both functions accept flat arrays rather than report-specific dataclass
instances, making them reusable across any sweep result format.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _infer_alpha_colour_range(
    alpha_map: np.ndarray,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Auto-infer colour-bar range for an alpha heatmap.

    Symmetric cmaps (RdBu, etc.) use ``±max|α|``; others use ``[0, max(α)]``.
    When all values are non-finite, falls back to ``(0, 1)``.
    User-provided *vmin*/*vmax* are always respected when not ``None``.

    Args:
        alpha_map: 2D grid of alpha values.
        cmap: Matplotlib colormap name.
        vmin: User-provided minimum (kept if not ``None``).
        vmax: User-provided maximum (kept if not ``None``).

    Returns:
        Tuple ``(vmin, vmax)``.
    """
    finite_alpha = alpha_map[np.isfinite(alpha_map)]
    if len(finite_alpha) == 0:
        return (vmin or 0.0, vmax or 1.0)

    symmetric_cmaps = {"RdBu", "RdBu_r", "bwr", "bwr_r", "coolwarm", "coolwarm_r"}
    if cmap in symmetric_cmaps:
        abs_max = float(np.nanmax(np.abs(finite_alpha))) or 1.0
        vmin_auto: float = -abs_max
        vmax_auto: float = abs_max
    else:
        vmax_auto = float(np.nanmax(finite_alpha)) or 1.0
        vmin_auto = 0.0

    return (
        vmin if vmin is not None else vmin_auto,
        vmax if vmax is not None else vmax_auto,
    )


def plot_ratio_heatmap(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    ratio_values: np.ndarray,
    save_path: str | Path,
    *,
    title: str = "Sensitivity Ratio",
    title_suffix: str = "",
    cbar_label: str = r"$\Delta\omega_{\mathrm{opt}} / \Delta\omega_{\mathrm{SQL}}$",
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of Δω_opt / SQL ratio across (ω, N).

    Constructs a 2D grid from the flat *ratio_values* array and renders it
    as a ``pcolormesh`` heatmap with a colour bar. A dashed horizontal
    line at ``y = 1.0`` marks the SQL boundary.

    Args:
        omega_values: Flat array of ω values.
        N_values: Flat array of N values.
        ratio_values: Flat array of ratio values (same length as above).
        save_path: Output SVG path.
        title: Base title for the plot.
        title_suffix: Optional suffix appended to the title.
        cbar_label: Colour-bar label.
        figsize: Figure size ``(width, height)``.

    Returns:
        Path to the saved SVG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(omega_values)
    N_vals = np.unique(N_values)
    ratio_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(omega_values, omega) & (N_values == N_val)
            if np.any(mask):
                ratio_map[j, i] = float(ratio_values[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(0.0, float(np.nanmin(ratio_map)))
    finite_mask = ratio_map < 10
    if np.any(finite_mask):
        vmax = max(2.0, float(np.nanmax(ratio_map[finite_mask])))
    else:
        vmax = 2.0

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        ratio_map,
        shading="nearest",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    full_title = title
    if title_suffix:
        full_title += f"\n{title_suffix}"
    ax.set_title(full_title)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alpha_opt_heatmap(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    alpha_values: np.ndarray,
    save_path: str | Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = r"$\alpha^*$",
    title: str = r"Optimal $\alpha$",
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Plot a heatmap of optimal α values across (ω, N).

    Constructs a 2D grid from the flat *alpha_values* array and renders it
    as a ``pcolormesh`` heatmap with a colour bar.

    When *vmin* and *vmax* are both ``None``, the colour range is inferred
    automatically: for symmetric cmaps (``"RdBu"``, ``"RdBu_r"``, etc.) the
    range is ``±max|α|``; otherwise ``[0, max(α)]``.

    Args:
        omega_values: Flat array of ω values.
        N_values: Flat array of N values.
        alpha_values: Flat array of optimal α values.
        save_path: Output SVG path.
        cmap: Matplotlib colormap name.
        vmin: Colour-bar minimum. Auto-inferred if ``None``.
        vmax: Colour-bar maximum. Auto-inferred if ``None``.
        cbar_label: Colour-bar label.
        title: Plot title.
        figsize: Figure size ``(width, height)``.

    Returns:
        Path to the saved SVG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_vals = np.unique(omega_values)
    N_vals = np.unique(N_values)
    alpha_map = np.full((len(N_vals), len(omega_vals)), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        for j, N_val in enumerate(N_vals):
            mask = np.isclose(omega_values, omega) & (N_values == N_val)
            if np.any(mask):
                alpha_map[j, i] = float(alpha_values[mask][0])

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = _infer_alpha_colour_range(alpha_map, cmap, vmin, vmax)

    im = ax.pcolormesh(
        omega_vals,
        N_vals,
        alpha_map,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$N$ (particles per subsystem)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path
