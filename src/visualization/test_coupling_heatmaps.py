"""Tests for coupling-heatmap visualisation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.visualization.coupling_heatmaps import (
    _infer_alpha_colour_range,
    plot_alpha_opt_heatmap,
    plot_ratio_heatmap,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Tests for _infer_alpha_colour_range ───────────────────────────────────────


class TestInferAlphaColourRange:
    def test_symmetric_cmap_rdbu(self) -> None:
        """Symmetric cmap gives ±max|alpha| range."""
        alpha_map = np.array([[1.0, -2.0], [0.5, -1.5]])
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "RdBu", None, None)
        assert vmin == pytest.approx(-2.0)
        assert vmax == pytest.approx(2.0)

    def test_nonsymmetric_cmap_viridis(self) -> None:
        """Non-symmetric cmap gives [0, max(alpha)] range."""
        alpha_map = np.array([[1.0, 2.0], [0.5, 1.5]])
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "viridis", None, None)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(2.0)

    def test_all_nan_falls_back(self) -> None:
        """All-NaN map falls back to (0, 1)."""
        alpha_map = np.full((3, 3), np.nan)
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "viridis", None, None)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(1.0)

    def test_user_vmin_vmax_respected(self) -> None:
        """User-provided vmin/vmax override auto-inferred values."""
        alpha_map = np.array([[1.0, 2.0], [0.5, 1.5]])
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "viridis", 0.5, 1.8)
        assert vmin == pytest.approx(0.5)
        assert vmax == pytest.approx(1.8)

    def test_all_nan_with_user_bounds(self) -> None:
        """All-NaN with user vmin returns user bounds, not fallback."""
        alpha_map = np.full((3, 3), np.nan)
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "RdBu", -1.0, 2.0)
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(2.0)

    def test_coolwarm_is_symmetric(self) -> None:
        """coolwarm and its variants are treated as symmetric."""
        alpha_map = np.array([[3.0, -1.0]])
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "coolwarm", None, None)
        assert vmin == pytest.approx(-3.0)
        assert vmax == pytest.approx(3.0)

    def test_symmetric_all_zero(self) -> None:
        """All-zero map with symmetric cmap: 0.0 or 1.0 → 1.0, so range is ±1."""
        alpha_map = np.zeros((3, 3))
        vmin, vmax = _infer_alpha_colour_range(alpha_map, "RdBu", None, None)
        assert vmin == pytest.approx(-1.0)
        assert vmax == pytest.approx(1.0)


# ── Tests for plot_ratio_heatmap ──────────────────────────────────────────────


def test_plot_ratio_heatmap_creates_svg(tmp_path: Path) -> None:
    omega = np.array([0.1, 0.2, 0.3])
    N_vals = np.array([1, 2, 3])
    # Build flat arrays
    omega_flat = np.repeat(omega, 3)
    N_flat = np.tile(N_vals, 3)
    ratio_flat = np.array([1.0, 1.5, 2.0, 0.8, 1.2, 1.8, 0.5, 0.9, 1.1])

    p = tmp_path / "ratio_hm.svg"
    result = plot_ratio_heatmap(omega_flat, N_flat, ratio_flat, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_ratio_heatmap_with_title_suffix(tmp_path: Path) -> None:
    omega = np.array([0.1, 0.5])
    N_vals = np.array([1, 5])
    omega_flat = np.repeat(omega, 2)
    N_flat = np.tile(N_vals, 2)
    ratio_flat = np.array([1.0, 2.0, 0.5, 1.5])

    p = tmp_path / "ratio_suffix.svg"
    result = plot_ratio_heatmap(
        omega_flat,
        N_flat,
        ratio_flat,
        p,
        title="Test",
        title_suffix="(extra)",
    )
    assert result.exists()


def test_plot_ratio_heatmap_with_large_ratios(tmp_path: Path) -> None:
    """Large ratio values (>10) are clamped for the colour range."""
    omega = np.array([0.1, 0.2])
    N_vals = np.array([1, 2])
    omega_flat = np.repeat(omega, 2)
    N_flat = np.tile(N_vals, 2)
    ratio_flat = np.array([1.0, 20.0, 0.5, 30.0])

    p = tmp_path / "ratio_large.svg"
    result = plot_ratio_heatmap(omega_flat, N_flat, ratio_flat, p)
    assert result.exists()


# ── Tests for plot_alpha_opt_heatmap ──────────────────────────────────────────


def test_plot_alpha_opt_heatmap_creates_svg(tmp_path: Path) -> None:
    omega = np.array([0.1, 0.2, 0.3])
    N_vals = np.array([1, 2, 3])
    omega_flat = np.repeat(omega, 3)
    N_flat = np.tile(N_vals, 3)
    alpha_flat = np.array([0.1, 0.5, 1.0, 0.2, 0.6, 1.1, 0.3, 0.7, 1.2])

    p = tmp_path / "alpha_hm.svg"
    result = plot_alpha_opt_heatmap(omega_flat, N_flat, alpha_flat, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_alpha_opt_heatmap_with_user_vmin_vmax(tmp_path: Path) -> None:
    omega = np.array([0.1, 0.5])
    N_vals = np.array([1, 5])
    omega_flat = np.repeat(omega, 2)
    N_flat = np.tile(N_vals, 2)
    alpha_flat = np.array([-1.0, 1.0, -0.5, 0.5])

    p = tmp_path / "alpha_bounded.svg"
    result = plot_alpha_opt_heatmap(
        omega_flat,
        N_flat,
        alpha_flat,
        p,
        vmin=-2.0,
        vmax=2.0,
    )
    assert result.exists()


def test_plot_alpha_opt_heatmap_symmetric_cmap(tmp_path: Path) -> None:
    """Symmetric cmap (RdBu) auto-infers symmetric colour range."""
    omega = np.array([0.1, 0.5])
    N_vals = np.array([1, 5])
    omega_flat = np.repeat(omega, 2)
    N_flat = np.tile(N_vals, 2)
    alpha_flat = np.array([-1.0, 2.0, -0.5, 1.5])

    p = tmp_path / "alpha_sym.svg"
    result = plot_alpha_opt_heatmap(
        omega_flat,
        N_flat,
        alpha_flat,
        p,
        cmap="RdBu",
    )
    assert result.exists()
