"""Tests for ancilla_drive_plots module.

Verifies that each plot function creates the expected SVG file
on disk when given minimal valid input data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.visualization.ancilla_drive_plots import (
    plot_drive_2d_slice_heatmap,
    plot_drive_cross_experiment_comparison,
    plot_drive_decoupled_baseline,
    plot_drive_fraction_below_sql,
    plot_drive_nm_expectation_variance,
    plot_drive_omega_scan,
    plot_drive_optimal_params,
    plot_drive_random_search_histogram,
)


@pytest.fixture
def tmp_svg() -> Generator[Path, None, None]:
    """Yield a temporary SVG path and clean up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        path = Path(f.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class TestPlotDriveDecoupledBaseline:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        result = DriveDecoupledBaselineResult(
            t_hold_value=10.0,
            delta_omega=0.12,
            sql=0.1,
        )
        out = plot_drive_decoupled_baseline(result, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDrive2DSliceHeatmap:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        result = Drive2DSliceResult(
            drive_values=np.linspace(-2, 2, 10),
            azz_values=np.linspace(-2, 2, 8),
            delta_omega_grid=np.random.default_rng(42).uniform(0.05, 0.3, (10, 8)),
            omega_value=1.0,
            slice_type="ax",
            sql=0.1,
        )
        out = plot_drive_2d_slice_heatmap(result, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDriveRandomSearchHistogram:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        rng = np.random.default_rng(42)
        result = DriveRandomSearchResult(
            samples=rng.uniform(-2, 2, (50, 4)),
            delta_omega_values=rng.uniform(0.05, 0.3, 50),
            best_params=(1.0, 0.5, -0.3, 0.8),
            best_delta_omega=0.06,
            omega_value=1.0,
            sql=0.1,
            t_hold=10.0,
        )
        out = plot_drive_random_search_histogram(result, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDriveOmegaScan:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        result = DriveOmegaScanResult(
            omega_values=np.array([0.5, 1.0, 1.5]),
            best_params_per_omega=[
                (1.0, 0.0, 0.0, 0.5),
                (0.8, 0.2, 0.1, 0.6),
                (0.6, 0.4, 0.2, 0.7),
            ],
            best_delta_omega_per_omega=np.array([0.08, 0.07, 0.09]),
            sql_values=np.full(3, 0.1),
            expectation_Jz_per_omega=np.array([0.3, 0.4, 0.2]),
            variance_Jz_per_omega=np.array([0.25, 0.24, 0.26]),
        )
        out = plot_drive_omega_scan(result, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0

    def test_empty_sql_raises_valueerror(self) -> None:
        result = DriveOmegaScanResult(
            omega_values=np.array([0.5]),
            best_params_per_omega=[(0.0, 0.0, 0.0, 0.0)],
            best_delta_omega_per_omega=np.array([0.1]),
            sql_values=np.array([]),
        )
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="sql_values is empty"):
                plot_drive_omega_scan(result, path)
        finally:
            path.unlink(missing_ok=True)


class TestPlotDriveOptimalParams:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        result = DriveOmegaScanResult(
            omega_values=np.array([0.5, 1.0, 1.5]),
            best_params_per_omega=[
                (1.0, 0.0, 0.0, 0.5),
                (0.8, 0.2, 0.1, 0.6),
                (0.6, 0.4, 0.2, 0.7),
            ],
            best_delta_omega_per_omega=np.array([0.08, 0.07, 0.09]),
            sql_values=np.full(3, 0.1),
        )
        out = plot_drive_optimal_params(result, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDriveNmExpectationVariance:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        omega = np.array([0.5, 1.0, 1.5])
        exp_jz = np.array([0.3, 0.4, 0.2])
        var_jz = np.array([0.25, 0.24, 0.26])
        out = plot_drive_nm_expectation_variance(omega, exp_jz, var_jz, tmp_svg)
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDriveCrossExperimentComparison:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        omega = np.array([0.5, 1.0, 1.5])
        delta_19 = np.array([0.08, 0.07, 0.09])
        delta_18 = np.array([0.12, 0.11, 0.13])
        sql = np.full(3, 0.1)
        out = plot_drive_cross_experiment_comparison(
            omega, delta_19, delta_18, sql, tmp_svg
        )
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0


class TestPlotDriveFractionBelowSQL:
    def test_creates_svg_file(self, tmp_svg: Path) -> None:
        omega = np.array([0.5, 1.0, 1.5])
        frac_ax = np.array([0.3, 0.5, 0.2])
        frac_ay = np.array([0.4, 0.6, 0.3])
        frac_random = np.array([0.1, 0.2, 0.05])
        out = plot_drive_fraction_below_sql(
            omega, frac_ax, frac_ay, frac_random, tmp_svg
        )
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0
