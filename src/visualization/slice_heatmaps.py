"""
3D slice visualisation for heatmap infrastructure.

Provides functions for taking cross-sections (slices) through 3D data
grids and rendering them as 2D heatmaps. Useful for visualizing parameter
sweeps where a third dimension (e.g., time, coupling strength) modulates
the heatmap structure.

Functions:
    plot_slice_heatmap: Render a single 2D slice from 3D data.
    plot_slice_panel: Render multiple slices in a grid layout.
    plot_slice_animation: Create an animated GIF of slices through the
        third dimension.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _build_3d_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted unique axis arrays and reshape data into a 3D grid.

    Args:
        x_values: Flat array of x-axis coordinates.
        y_values: Flat array of y-axis coordinates.
        z_values: Flat array of z-axis (slice dimension) coordinates.
        data: Flat array of values (same length as coordinate arrays).

    Returns:
        Tuple ``(x_unique, y_unique, z_unique, grid)`` where ``grid``
        has shape ``(len(z_unique), len(y_unique), len(x_unique))``.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    if not (len(x_values) == len(y_values) == len(z_values) == len(data)):
        raise ValueError(
            f"All input arrays must have the same length, got "
            f"x={len(x_values)}, y={len(y_values)}, z={len(z_values)}, "
            f"data={len(data)}"
        )

    x_unique = np.unique(x_values)
    y_unique = np.unique(y_values)
    z_unique = np.unique(z_values)

    grid = np.full((len(z_unique), len(y_unique), len(x_unique)), np.nan)

    for k, z_val in enumerate(z_unique):
        for j, y_val in enumerate(y_unique):
            for i, x_val in enumerate(x_unique):
                mask = (
                    np.isclose(x_values, x_val)
                    & np.isclose(y_values, y_val)
                    & np.isclose(z_values, z_val)
                )
                if np.any(mask):
                    grid[k, j, i] = float(data[mask][0])

    return x_unique, y_unique, z_unique, grid


def plot_slice_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    data: np.ndarray,
    z_slice: float,
    save_path: str | Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    title: str = "",
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Render a single 2D slice from 3D data as a heatmap.

    Extracts the cross-section at ``z = z_slice`` and plots it as a
    ``pcolormesh`` heatmap with a colour bar.

    Args:
        x_values: Flat array of x-axis coordinates.
        y_values: Flat array of y-axis coordinates.
        z_values: Flat array of z-axis (slice dimension) coordinates.
        data: Flat array of values (same length as coordinate arrays).
        z_slice: Value along the z-axis at which to take the slice.
            Must be close to one of the unique z values (within tolerance
            of ``1e-10 * range(z)``).
        save_path: Output file path (SVG, PNG, etc.).
        cmap: Matplotlib colormap name.
        vmin: Colour-bar minimum. Auto-inferred if ``None``.
        vmax: Colour-bar maximum. Auto-inferred if ``None``.
        cbar_label: Colour-bar label text.
        title: Plot title. Defaults to showing the slice position.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size ``(width, height)``.

    Returns:
        Path to the saved file.

    Raises:
        ValueError: If z_slice is not close to any unique z value.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_unique, y_unique, z_unique, grid = _build_3d_grid(
        x_values, y_values, z_values, data
    )

    z_range = z_unique[-1] - z_unique[0] if len(z_unique) > 1 else 1.0
    z_tol = max(1e-10 * abs(z_range), 1e-15)
    z_idx = int(np.argmin(np.abs(z_unique - z_slice)))
    if abs(z_unique[z_idx] - z_slice) > z_tol:
        raise ValueError(
            f"z_slice={z_slice} is not close to any unique z value "
            f"(closest: {z_unique[z_idx]})"
        )

    slice_data = grid[z_idx]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        x_unique, y_unique, slice_data, shading="nearest", cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Slice at z = {z_slice:.4g}")

    fig.tight_layout()
    fig.savefig(save_path, format=save_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_slice_panel(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    data: np.ndarray,
    z_slices: list[float],
    save_path: str | Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    title: str = "",
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
    ncols: int = 3,
    figsize_per_subplot: tuple[float, float] = (4, 3.5),
) -> Path:
    """Render multiple slices in a grid panel layout.

    Each slice is shown as a subplot with shared colour scaling across
    all panels when *vmin*/*vmax* are ``None``.

    Args:
        x_values: Flat array of x-axis coordinates.
        y_values: Flat array of y-axis coordinates.
        z_values: Flat array of z-axis (slice dimension) coordinates.
        data: Flat array of values (same length as coordinate arrays).
        z_slices: List of z values at which to take slices.
        save_path: Output file path.
        cmap: Matplotlib colormap name.
        vmin: Colour-bar minimum. Auto-inferred if ``None``.
        vmax: Colour-bar maximum. Auto-inferred if ``None``.
        cbar_label: Colour-bar label text.
        title: Overall figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ncols: Number of columns in the panel grid.
        figsize_per_subplot: Size ``(width, height)`` for each subplot.

    Returns:
        Path to the saved file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_unique, y_unique, z_unique, grid = _build_3d_grid(
        x_values, y_values, z_values, data
    )

    n_slices = len(z_slices)
    nrows = int(np.ceil(n_slices / ncols))

    # Determine shared colour range if not provided
    if vmin is None or vmax is None:
        slice_grids = []
        z_range = z_unique[-1] - z_unique[0] if len(z_unique) > 1 else 1.0
        z_tol = max(1e-10 * abs(z_range), 1e-15)
        for zs in z_slices:
            idx = int(np.argmin(np.abs(z_unique - zs)))
            if abs(z_unique[idx] - zs) <= z_tol:
                slice_grids.append(grid[idx])
        if slice_grids:
            combined = np.concatenate([g.ravel() for g in slice_grids])
            finite = combined[np.isfinite(combined)]
            if len(finite) > 0:
                if vmin is None:
                    vmin = float(np.min(finite))
                if vmax is None:
                    vmax = float(np.max(finite))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    z_range = z_unique[-1] - z_unique[0] if len(z_unique) > 1 else 1.0
    z_tol = max(1e-10 * abs(z_range), 1e-15)

    for idx, zs in enumerate(z_slices):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        z_idx = int(np.argmin(np.abs(z_unique - zs)))
        if abs(z_unique[z_idx] - zs) > z_tol:
            ax.set_visible(False)
            continue
        im = ax.pcolormesh(
            x_unique, y_unique, grid[z_idx],
            shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"z = {zs:.4g}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    # Hide unused subplots
    for idx in range(n_slices, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Add shared colour bar
    if n_slices > 0:
        z_idx_first = int(np.argmin(np.abs(z_unique - z_slices[0])))
        if abs(z_unique[z_idx_first] - z_slices[0]) <= z_tol:
            fig.colorbar(
                im, ax=axes.ravel().tolist(), label=cbar_label,
                shrink=0.8, pad=0.02,
            )

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    fig.savefig(save_path, format=save_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_slice_animation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    data: np.ndarray,
    save_path: str | Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    xlabel: str = r"$x$",
    ylabel: str = r"$y$",
    z_label: str = r"$z$",
    title: str = "",
    figsize: tuple[float, float] = (8, 6),
    fps: int = 5,
    dpi: int = 100,
) -> Path:
    """Create an animated GIF cycling through z-slices.

    Each frame shows the 2D heatmap at a different z value, with a
    colour bar and title indicating the current slice position.

    Args:
        x_values: Flat array of x-axis coordinates.
        y_values: Flat array of y-axis coordinates.
        z_values: Flat array of z-axis (slice dimension) coordinates.
        data: Flat array of values (same length as coordinate arrays).
        save_path: Output file path (should end in ``.gif``).
        cmap: Matplotlib colormap name.
        vmin: Colour-bar minimum. Auto-inferred if ``None``.
        vmax: Colour-bar maximum. Auto-inferred if ``None``.
        cbar_label: Colour-bar label text.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        z_label: Label for the animated dimension (shown in title).
        title: Base plot title.
        figsize: Figure size ``(width, height)``.
        fps: Frames per second.
        dpi: Output resolution.

    Returns:
        Path to the saved GIF file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_unique, y_unique, z_unique, grid = _build_3d_grid(
        x_values, y_values, z_values, data
    )

    n_frames = len(z_unique)

    # Determine shared colour range
    if vmin is None or vmax is None:
        finite = grid[np.isfinite(grid)]
        if len(finite) > 0:
            if vmin is None:
                vmin = float(np.min(finite))
            if vmax is None:
                vmax = float(np.max(finite))
        else:
            vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        x_unique, y_unique, grid[0],
        shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_text = ax.set_title(f"{title} ({z_label} = {z_unique[0]:.4g})")

    fig.tight_layout()

    def _update(frame: int) -> list:
        im.set_array(grid[frame].ravel())
        z_val = z_unique[frame]
        if title:
            title_text.set_text(f"{title} ({z_label} = {z_val:.4g})")
        else:
            title_text.set_text(f"{z_label} = {z_val:.4g}")
        return [im, title_text]

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(
        str(save_path),
        writer=PillowWriter(fps=fps),
        dpi=dpi,
    )
    plt.close(fig)
    return save_path
