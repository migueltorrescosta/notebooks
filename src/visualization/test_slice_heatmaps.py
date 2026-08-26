"""Tests for 3D slice visualisation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.visualization.slice_heatmaps import (
    _build_3d_grid,
    plot_slice_animation,
    plot_slice_heatmap,
    plot_slice_panel,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_flat_3d_data(
    nx: int = 3, ny: int = 3, nz: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create flat coordinate and data arrays for a small 3D grid."""
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    zs = np.linspace(0, 1, nz)
    x_flat = np.repeat(xs, ny * nz)
    y_flat = np.tile(np.repeat(ys, nz), nx)
    z_flat = np.tile(zs, nx * ny)
    data = x_flat + y_flat + z_flat
    return x_flat, y_flat, z_flat, data


# ── Tests for _build_3d_grid ─────────────────────────────────────────────────


class TestBuild3dGrid:
    def test_basic_shape(self) -> None:
        """Grid shape matches (nz, ny, nx)."""
        x, y, z, d = _make_flat_3d_data(3, 4, 5)
        _xu, _yu, _zu, grid = _build_3d_grid(x, y, z, d)
        assert grid.shape == (5, 4, 3)

    def test_unique_axes(self) -> None:
        """Returned axis arrays are sorted unique values."""
        x, y, z, d = _make_flat_3d_data(3, 3, 3)
        xu, yu, zu, _grid = _build_3d_grid(x, y, z, d)
        assert len(xu) == 3
        assert len(yu) == 3
        assert len(zu) == 3
        assert np.all(np.diff(xu) > 0)
        assert np.all(np.diff(yu) > 0)
        assert np.all(np.diff(zu) > 0)

    def test_values_populated(self) -> None:
        """Non-NaN values appear in the grid at expected locations."""
        x, y, z, d = _make_flat_3d_data(2, 2, 2)
        _xu, _yu, _zu, grid = _build_3d_grid(x, y, z, d)
        assert np.sum(np.isfinite(grid)) == 8

    def test_inconsistent_lengths_raises(self) -> None:
        """Mismatched array lengths raise ValueError."""
        x = np.array([0, 1])
        y = np.array([0])
        z = np.array([0, 1])
        d = np.array([1, 2])
        with pytest.raises(ValueError, match="same length"):
            _build_3d_grid(x, y, z, d)

    def test_sparse_data_leaves_nans(self) -> None:
        """Missing combinations produce NaN in the grid."""
        x = np.array([0, 0, 1])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 0])
        d = np.array([1.0, 2.0, 3.0])
        _xu, _yu, _zu, grid = _build_3d_grid(x, y, z, d)
        # (x=1, y=1, z=0) is missing → NaN
        assert np.isnan(grid[0, 1, 1])


# ── Tests for plot_slice_heatmap ──────────────────────────────────────────────


class TestPlotSliceHeatmap:
    def test_creates_file(self, tmp_path: Path) -> None:
        """Output file exists after plotting."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "slice.svg"
        result = plot_slice_heatmap(x, y, z, d, z_slice=0.0, save_path=p)
        assert result.exists()
        assert result.suffix == ".svg"

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title is used instead of default."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "titled.svg"
        result = plot_slice_heatmap(
            x, y, z, d, z_slice=0.5, save_path=p, title="My Plot"
        )
        assert result.exists()

    def test_invalid_z_slice_raises(self, tmp_path: Path) -> None:
        """z_slice far from any unique z raises ValueError."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "bad.svg"
        with pytest.raises(ValueError, match="not close"):
            plot_slice_heatmap(x, y, z, d, z_slice=999.0, save_path=p)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "subdir" / "deep" / "slice.svg"
        result = plot_slice_heatmap(x, y, z, d, z_slice=0.0, save_path=p)
        assert result.exists()

    def test_png_format(self, tmp_path: Path) -> None:
        """PNG format is supported."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "slice.png"
        result = plot_slice_heatmap(x, y, z, d, z_slice=0.0, save_path=p)
        assert result.suffix == ".png"

    def test_custom_labels(self, tmp_path: Path) -> None:
        """Custom axis labels are accepted."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "labels.svg"
        result = plot_slice_heatmap(
            x, y, z, d, z_slice=0.0, save_path=p,
            xlabel="Freq", ylabel="N", cbar_label="Value",
        )
        assert result.exists()


# ── Tests for plot_slice_panel ────────────────────────────────────────────────


class TestPlotSlicePanel:
    def test_creates_file(self, tmp_path: Path) -> None:
        """Output file exists after plotting."""
        x, y, z, d = _make_flat_3d_data(3, 3, 4)
        p = tmp_path / "panel.svg"
        result = plot_slice_panel(
            x, y, z, d, z_slices=[0.0, 0.33, 0.67, 1.0], save_path=p,
        )
        assert result.exists()

    def test_single_slice(self, tmp_path: Path) -> None:
        """Panel with one slice works."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "single.svg"
        result = plot_slice_panel(x, y, z, d, z_slices=[0.0], save_path=p)
        assert result.exists()

    def test_custom_ncols(self, tmp_path: Path) -> None:
        """Custom column count is respected."""
        x, y, z, d = _make_flat_3d_data(3, 3, 6)
        p = tmp_path / "ncols.svg"
        result = plot_slice_panel(
            x, y, z, d,
            z_slices=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            save_path=p, ncols=2,
        )
        assert result.exists()

    def test_title(self, tmp_path: Path) -> None:
        """Overall title is applied."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "titled.svg"
        result = plot_slice_panel(
            x, y, z, d, z_slices=[0.0, 1.0], save_path=p, title="Panel Title"
        )
        assert result.exists()

    def test_shared_colorbar_range(self, tmp_path: Path) -> None:
        """Panel with explicit vmin/vmax uses shared range."""
        x, y, z, d = _make_flat_3d_data()
        p = tmp_path / "shared.svg"
        result = plot_slice_panel(
            x, y, z, d, z_slices=[0.0, 0.5, 1.0], save_path=p,
            vmin=0.0, vmax=3.0,
        )
        assert result.exists()


# ── Tests for plot_slice_animation ────────────────────────────────────────────


class TestPlotSliceAnimation:
    def test_creates_gif(self, tmp_path: Path) -> None:
        """Output GIF file exists after animation."""
        x, y, z, d = _make_flat_3d_data(3, 3, 4)
        p = tmp_path / "anim.gif"
        result = plot_slice_animation(x, y, z, d, save_path=p, fps=2)
        assert result.exists()
        assert result.suffix == ".gif"

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title appears in animation frames."""
        x, y, z, d = _make_flat_3d_data(2, 2, 3)
        p = tmp_path / "titled.gif"
        result = plot_slice_animation(
            x, y, z, d, save_path=p, title="Evolution", fps=2,
        )
        assert result.exists()

    def test_single_frame(self, tmp_path: Path) -> None:
        """Animation with a single z value works."""
        x, y, z, d = _make_flat_3d_data(2, 2, 1)
        p = tmp_path / "single.gif"
        result = plot_slice_animation(x, y, z, d, save_path=p, fps=1)
        assert result.exists()

    def test_custom_dpi(self, tmp_path: Path) -> None:
        """Custom DPI is accepted."""
        x, y, z, d = _make_flat_3d_data(2, 2, 2)
        p = tmp_path / "highres.gif"
        result = plot_slice_animation(
            x, y, z, d, save_path=p, dpi=150, fps=2,
        )
        assert result.exists()
