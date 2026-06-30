"""Tests for shared MZI plot functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.analysis.sensitivity_metrics import MziSensitivityData
from src.visualization.mzi_plots import (
    generate_plots,
    maybe_plot_delta_omega_overlays,
    maybe_plot_scaling_comparison,
    plot_delta_omega_overlay,
    plot_scaling,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _fig_path_fn(tmp_path: Path) -> Callable[[str], Path]:
    """Create a fig_path_fn bound to *tmp_path* for use across multiple tests."""

    def inner(name: str) -> Path:
        return tmp_path / f"{name}.svg"

    return inner


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_dummy_resource_values(n_R: int = 7) -> np.ndarray:
    base = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=float)
    if n_R <= len(base):
        return base[:n_R]
    # Extend with powers of 2
    extra = np.array([2.0**k for k in range(7, 7 + n_R - len(base))], dtype=float)
    return np.concatenate([base, extra])


def _make_dummy_omega(n_omega: int = 50) -> np.ndarray:
    return np.linspace(0.1, 5.0, n_omega)


def _make_sensitivity_data(
    state_type: str = "test",
    n_R: int = 5,
    n_omega: int = 20,
) -> MziSensitivityData:
    resource_values = _make_dummy_resource_values(n_R)
    omega_values = _make_dummy_omega(n_omega)
    # Build a grid where Δω_C = SQL/R (sensitivity improves with R)
    sql = 1.0 / (10.0 * resource_values)  # per R
    delta_omega_c_grid = np.ones((n_R, n_omega))
    for i in range(n_R):
        # Sensitivity varies with ω, best at ω=2.5
        delta_omega_c_grid[i, :] = sql[i] * (1.0 + 0.1 * (omega_values - 2.5) ** 2)
    delta_omega_q_per_R = sql / 2.0  # QFI bound is 2× better

    # Add the other required grids with plausible dummy data
    expectation_grid = np.zeros((n_R, n_omega))
    variance_grid = np.ones((n_R, n_omega)) * 0.25
    derivative_grid = np.ones((n_R, n_omega)) * 0.1
    delta_omega_ep_grid = np.sqrt(variance_grid) / np.abs(derivative_grid)
    fisher_classical_grid = 1.0 / delta_omega_c_grid**2

    return MziSensitivityData(
        state_type=state_type,
        resource_type="N",
        resource_values=resource_values,
        omega_values=omega_values,
        expectation_grid=expectation_grid,
        variance_grid=variance_grid,
        derivative_grid=derivative_grid,
        delta_omega_ep_grid=delta_omega_ep_grid,
        delta_omega_q_per_R=delta_omega_q_per_R,
        fisher_classical_grid=fisher_classical_grid,
        delta_omega_c_grid=delta_omega_c_grid,
    )


# ── Tests for plot_delta_omega_overlay ────────────────────────────────────────


class TestPlotDeltaOmegaOverlay:
    def test_basic_overlay_creates_svg(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        p = tmp_path / "overlay.svg"
        result = plot_delta_omega_overlay(data, save_path=p)
        assert result.exists()
        assert result.suffix == ".svg"

    def test_with_selected_r(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data(n_R=10)
        p = tmp_path / "overlay_selected.svg"
        result = plot_delta_omega_overlay(
            data, selected_R=[1.0, 4.0, 16.0], save_path=p
        )
        assert result.exists()

    def test_with_missing_selected_r(self, tmp_path: Path) -> None:
        """A selected_R value not in data.resource_values is silently skipped."""
        data = _make_sensitivity_data(n_R=3)
        p = tmp_path / "overlay_missing.svg"
        result = plot_delta_omega_overlay(data, selected_R=[999.0], save_path=p)
        assert result.exists()

    def test_default_save_path(self) -> None:
        """When save_path is None, filename is derived from data.state_type."""
        data = _make_sensitivity_data(state_type="test_state")
        result = plot_delta_omega_overlay(data, save_path=None)
        assert result.name == "test_state_delta_omega_comparison.svg"
        result.unlink()

    def test_with_title(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        p = tmp_path / "overlay_title.svg"
        result = plot_delta_omega_overlay(data, save_path=p, title="Custom title")
        assert result.exists()

    def test_auto_select_r_single(self, tmp_path: Path) -> None:
        """When only 1-2 R values, auto-selection picks all."""
        data = _make_sensitivity_data(n_R=1)
        p = tmp_path / "overlay_single_r.svg"
        result = plot_delta_omega_overlay(data, save_path=p)
        assert result.exists()

    def test_handles_non_finite_c_values(self, tmp_path: Path) -> None:
        """Non-finite Δω_C values do not crash the plot."""
        data = _make_sensitivity_data(n_R=3, n_omega=5)
        data.delta_omega_c_grid[0, 2:4] = np.inf
        data.delta_omega_c_grid[1, 1] = np.nan
        p = tmp_path / "overlay_nonfinite.svg"
        result = plot_delta_omega_overlay(data, save_path=p)
        assert result.exists()


# ── Tests for plot_scaling ────────────────────────────────────────────────────


class TestPlotScaling:
    def test_single_dataset_creates_svg(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        p = tmp_path / "scaling.svg"
        result = plot_scaling([data], ["test"], save_path=p)
        assert result.exists()

    def test_multiple_datasets(self, tmp_path: Path) -> None:
        d1 = _make_sensitivity_data(state_type="a", n_R=5)
        d2 = _make_sensitivity_data(state_type="b", n_R=5)
        p = tmp_path / "scaling_multi.svg"
        result = plot_scaling([d1, d2], ["A", "B"], save_path=p)
        assert result.exists()

    def test_with_none_entries(self, tmp_path: Path) -> None:
        """None entries in data_list are silently skipped."""
        data = _make_sensitivity_data()
        p = tmp_path / "scaling_none.svg"
        result = plot_scaling([None, data, None], ["a", "b", "c"], save_path=p)
        assert result.exists()

    def test_all_none_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            plot_scaling([None, None], ["a", "b"])

    def test_default_save_path(self) -> None:
        data = _make_sensitivity_data()
        result = plot_scaling([data], ["test"], save_path=None)
        assert result.name == "scaling_comparison.svg"
        result.unlink()

    def test_fit_exponent_appears(self, tmp_path: Path) -> None:
        """Scaling fit line is plotted and visible."""
        data = _make_sensitivity_data()
        p = tmp_path / "scaling_fit.svg"
        result = plot_scaling([data], ["test"], save_path=p, N_min_fit=2.0)
        assert result.exists()

    def test_custom_labels_label_count_mismatch_ok(self, tmp_path: Path) -> None:
        """label count can differ from data count (strict=False)."""
        data = _make_sensitivity_data()
        p = tmp_path / "scaling_mismatch.svg"
        # More labels than datasets — no error with strict=False
        result = plot_scaling([data], ["a", "b"], save_path=p)
        assert result.exists()

    def test_non_default_xlabel_title(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        p = tmp_path / "scaling_custom.svg"
        result = plot_scaling(
            [data],
            ["test"],
            save_path=p,
            xlabel="Custom X",
            title="Custom Title",
        )
        assert result.exists()


# ── Tests for maybe_plot_delta_omega_overlays ─────────────────────────────────


class TestMaybePlotDeltaOmegaOverlays:
    def test_no_force_skips_existing(self, tmp_path: Path) -> None:
        """When force=False and SVG exists, it is not regenerated."""
        data = _make_sensitivity_data()
        results = {"test": data}
        state_configs = [("test", None, "Test")]
        existing = tmp_path / "test_delta_omega_comparison.svg"
        existing.write_text("dummy")

        fpf = _fig_path_fn(tmp_path)
        maybe_plot_delta_omega_overlays(results, state_configs, False, None, fpf)
        assert existing.read_text() == "dummy"

    def test_force_regenerates(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        results = {"test": data}
        state_configs = [("test", None, "Test")]

        fpf = _fig_path_fn(tmp_path)
        maybe_plot_delta_omega_overlays(results, state_configs, True, None, fpf)
        assert (tmp_path / "test_delta_omega_comparison.svg").exists()

    def test_only_parameter(self, tmp_path: Path) -> None:
        """only='foo' skips 'test'."""
        data = _make_sensitivity_data()
        results = {"test": data}
        state_configs = [("test", None, "Test")]

        fpf = _fig_path_fn(tmp_path)
        maybe_plot_delta_omega_overlays(results, state_configs, True, "foo", fpf)
        assert not (tmp_path / "test_delta_omega_comparison.svg").exists()

    def test_missing_data_in_results(self, tmp_path: Path) -> None:
        """If state_type is not in results, it is silently skipped."""
        results: dict[str, MziSensitivityData] = {}
        state_configs = [("nonexistent", None, "N/A")]

        fpf = _fig_path_fn(tmp_path)
        # Should not raise
        maybe_plot_delta_omega_overlays(results, state_configs, True, None, fpf)


# ── Tests for maybe_plot_scaling_comparison ───────────────────────────────────


class TestMaybePlotScalingComparison:
    def test_creates_svg(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        results = {"test": data}
        fpf = _fig_path_fn(tmp_path)
        maybe_plot_scaling_comparison(results, True, fpf)
        assert (tmp_path / "scaling_comparison.svg").exists()

    def test_no_force_skips_existing(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        results = {"test": data}
        existing = tmp_path / "scaling_comparison.svg"
        existing.write_text("dummy")

        fpf = _fig_path_fn(tmp_path)
        maybe_plot_scaling_comparison(results, False, fpf)
        assert existing.read_text() == "dummy"

    def test_with_custom_keys_labels(self, tmp_path: Path) -> None:
        d1 = _make_sensitivity_data(state_type="a")
        d2 = _make_sensitivity_data(state_type="b")
        results = {"a": d1, "b": d2}
        fpf = _fig_path_fn(tmp_path)
        maybe_plot_scaling_comparison(
            results, True, fpf, data_keys=["a"], data_labels=["State A"]
        )
        assert (tmp_path / "scaling_comparison.svg").exists()


# ── Tests for generate_plots (integration) ────────────────────────────────────


class TestGeneratePlots:
    def test_generates_all_plots(self, tmp_path: Path) -> None:
        data = _make_sensitivity_data()
        results = {"test": data}
        state_configs = [("test", None, "Test")]

        fpf = _fig_path_fn(tmp_path)
        generate_plots(results, state_configs, True, None, fpf)

        assert (tmp_path / "test_delta_omega_comparison.svg").exists()
        assert (tmp_path / "scaling_comparison.svg").exists()

    def test_with_scaling_keys(self, tmp_path: Path) -> None:
        d1 = _make_sensitivity_data(state_type="a")
        d2 = _make_sensitivity_data(state_type="b")
        results = {"a": d1, "b": d2}
        state_configs = [("a", None, "A"), ("b", None, "B")]

        fpf = _fig_path_fn(tmp_path)
        generate_plots(
            results,
            state_configs,
            True,
            None,
            fpf,
            scaling_data_keys=["a", "b"],
            scaling_data_labels=["A", "B"],
            xlabel="Custom X",
            scaling_title="Custom Title",
        )

        assert (tmp_path / "scaling_comparison.svg").exists()

    def test_only_parameter_filters(self, tmp_path: Path) -> None:
        d1 = _make_sensitivity_data(state_type="a")
        results = {"a": d1}
        state_configs = [("a", None, "A")]
        fpf = _fig_path_fn(tmp_path)
        generate_plots(results, state_configs, True, "a", fpf)
        assert (tmp_path / "a_delta_omega_comparison.svg").exists()
