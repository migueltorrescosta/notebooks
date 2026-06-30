"""
Tests for the 2026-05-09 ancilla-assisted non-Markovian metrology report.

Verifies the sweep functions, result dataclass (NonMarkovianSweepData),
plotting helpers, and CLI pipeline defined in non_markovian_ancilla.py.

Note: Unit tests for the underlying pseudomode physics (operators,
Hamiltonian, evolution, QFI) live in src/physics/test_pseudomode_system.py.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest

from src.utils.serialization import assert_roundtrip_fields

# Load non_markovian_ancilla.py via importlib
_report_local = importlib.import_module("reports.20260509.non_markovian_ancilla")


# =============================================================================
# Tests: Module-level CLI
# =============================================================================


class TestModuleLoading:
    """Test that the module loads correctly with all expected attributes."""

    def test_module_has_main(self) -> None:
        assert hasattr(_report_local, "main")
        assert callable(_report_local.main)

    def test_module_has_report_date(self) -> None:
        assert _report_local.REPORT_DATE == "20260509"

    def test_module_has_expected_classes(self) -> None:
        expected_classes = [
            "NonMarkovianSweepData",
        ]
        for cls_name in expected_classes:
            assert hasattr(_report_local, cls_name), f"Missing class: {cls_name}"

    def test_module_can_generate_raw_data(self) -> None:
        """Verify the CLI generator functions are callable."""
        assert callable(_report_local._generate_ancilla_nonmarkovian_raw_data)
        assert callable(_report_local._generate_ancilla_nonmarkovian_figures)


# =============================================================================
# Tests: NonMarkovianSweepData
# =============================================================================


class TestNonMarkovianSweepData:
    """Dataclass serialization and deserialization."""

    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("sweep_type", "eq"),
        ("N", "eq"),
        ("K", "eq"),
        ("alpha", "eq"),
        ("g_sp", "eq"),
        ("omega_0", "eq"),
        ("tau", "eq"),
        ("dt", "eq"),
        ("lam", "eq"),
        ("T_decay", "eq"),
        ("theta", "eq"),
        ("sweep_values", "allclose"),
        ("ratio_with", "allclose"),
        ("ratio_without", "allclose"),
        ("qfi_with", "allclose"),
        ("qfi_without", "allclose"),
        ("qfi_initial", "allclose"),
        ("pm_occupancy", "allclose"),
    ]

    def _make_dummy_data(self) -> Any:  # type: ignore[explicit-any]
        """Create dummy sweep data for serialization tests."""
        n = 10
        return _report_local.NonMarkovianSweepData(
            sweep_type="ancilla",
            sweep_values=np.linspace(0, np.pi, n),
            ratio_with=np.linspace(0.5, 0.9, n),
            ratio_without=np.linspace(0.3, 0.4, n),
            qfi_with=np.linspace(2.0, 3.0, n),
            qfi_without=np.linspace(1.0, 1.5, n),
            qfi_initial=np.full(n, 4.0),
            pm_occupancy=np.linspace(0.0, 0.5, n),
            N=5,
            K=3,
            alpha=1.0,
            g_sp=0.5,
            omega_0=0.0,
            tau=0.1,
            dt=0.01,
            lam=1.0,
            T_decay=2.0,
            theta=0.0,
        )

    def test_to_dataframe_shape(self) -> None:
        data = self._make_dummy_data()
        df = data.to_dataframe()
        assert len(df) == len(data.sweep_values)
        assert df["sweep_type"].iloc[0] == "ancilla"

    def test_roundtrip_parquet(self, tmp_path: Path) -> None:
        """Verify all fields survive a Parquet roundtrip."""
        data = self._make_dummy_data()
        path = tmp_path / "test_sweep.parquet"
        data.save_parquet(path)
        assert path.exists()

        loaded = _report_local.NonMarkovianSweepData.from_parquet(path)
        assert_roundtrip_fields(loaded, data, self._FIELD_SPECS)

    def test_from_parquet_fails_on_missing_columns(self, tmp_path: Path) -> None:
        """Fail-fast when required columns are missing."""
        import pandas as pd

        df = pd.DataFrame({"sweep_type": ["ancilla"], "ratio_with": [0.5]})
        bad_path = tmp_path / "bad.parquet"
        df.to_parquet(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _report_local.NonMarkovianSweepData.from_parquet(bad_path)


# =============================================================================
# Tests: Sweep Functions (small N, K for speed)
# =============================================================================


class TestSweepFunctions:
    """Verify sweep functions run and return correct shapes."""

    _fast_config_kw: ClassVar[dict] = {
        "N": 2,
        "K": 2,
        "alpha": 0.5,
        "tau": 0.1,
        "g_sp": 0.3,
        "omega_0": 0.0,
        "lam": 0.5,
        "T_decay": 0.5,
        "dt": 0.05,
    }

    def test_ancilla_sweep_shape(self) -> None:
        theta_grid = np.linspace(0.0, np.pi / 2, 5)
        base = _report_local.PseudomodeConfig(**self._fast_config_kw)
        data = _report_local.ancilla_sweep(theta_grid, base)
        assert data.sweep_type == "ancilla"
        assert len(data.sweep_values) == 5
        assert len(data.ratio_with) == 5
        assert data.N == 2
        assert data.K == 2

    def test_memory_sweep_shape(self) -> None:
        lam_grid = np.logspace(np.log10(0.1), np.log10(5.0), 5)
        base = _report_local.PseudomodeConfig(**self._fast_config_kw)
        data = _report_local.memory_sweep(lam_grid, theta=0.5, base_config=base)
        assert data.sweep_type == "memory"
        assert len(data.sweep_values) == 5
        assert len(data.ratio_with) == 5
        assert data.theta == 0.5

    def test_time_sweep_shape(self) -> None:
        T_grid = np.linspace(0.0, 2.0, 5)
        base = _report_local.PseudomodeConfig(**self._fast_config_kw)
        data = _report_local.time_sweep(T_grid, theta=0.5, lam=0.5, base_config=base)
        assert data.sweep_type == "time"
        assert len(data.sweep_values) == 5
        assert len(data.ratio_with) == 5
        assert data.lam == 0.5

    def test_find_optimal_theta(self) -> None:
        n = 10
        data = _report_local.NonMarkovianSweepData(
            sweep_type="ancilla",
            sweep_values=np.linspace(0, np.pi, n),
            ratio_with=np.sin(np.linspace(0, np.pi, n)) ** 2,  # peak at pi/2
            ratio_without=np.full(n, 0.3),
            qfi_with=np.full(n, 3.0),
            qfi_without=np.full(n, 1.0),
            qfi_initial=np.full(n, 4.0),
            pm_occupancy=np.zeros(n),
            N=2,
            K=2,
            alpha=1.0,
            g_sp=0.5,
            omega_0=0.0,
            tau=0.1,
            dt=0.01,
            lam=1.0,
            T_decay=2.0,
            theta=0.0,
        )
        theta_opt = _report_local.find_optimal_theta(data)
        # The peak of sin^2 in [0, pi] should be at pi/2
        assert np.isclose(theta_opt, np.pi / 2, atol=0.2)

    def test_theta_zero_recovers_no_ancilla_baseline(self) -> None:
        """At theta=0, R_with should be ~ R_without."""
        theta_grid = np.array([0.0, 0.01])
        base = _report_local.PseudomodeConfig(**self._fast_config_kw)
        data = _report_local.ancilla_sweep(theta_grid, base)
        # At theta=0, the ancilla is not entangled, so with and without should match
        assert np.isclose(data.ratio_with[0], data.ratio_without[0], rtol=1e-5)

    def test_sweep_values_match_theta_grid(self) -> None:
        """Ancilla sweep should produce sweep_values == theta_grid."""
        theta_grid = np.array([0.0, np.pi / 4, np.pi / 2])
        base = _report_local.PseudomodeConfig(**self._fast_config_kw)
        data = _report_local.ancilla_sweep(theta_grid, base)
        np.testing.assert_allclose(data.sweep_values, theta_grid)


# =============================================================================
# Tests: Plot Functions
# =============================================================================


class TestPlotFunctions:
    """Verify plot functions run without errors."""

    def _make_dummy_data(self) -> Any:  # type: ignore[explicit-any]
        n = 10
        return _report_local.NonMarkovianSweepData(
            sweep_type="ancilla",
            sweep_values=np.linspace(0, np.pi, n),
            ratio_with=np.linspace(0.5, 0.9, n),
            ratio_without=np.linspace(0.3, 0.4, n),
            qfi_with=np.linspace(2.0, 3.0, n),
            qfi_without=np.linspace(1.0, 1.5, n),
            qfi_initial=np.full(n, 4.0),
            pm_occupancy=np.linspace(0.0, 0.5, n),
            N=5,
            K=3,
            alpha=1.0,
            g_sp=0.5,
            omega_0=0.0,
            tau=0.1,
            dt=0.01,
            lam=1.0,
            T_decay=2.0,
            theta=0.5,
        )

    def test_plot_ancilla_sweep(self, tmp_path: Path) -> None:
        data = self._make_dummy_data()
        path = tmp_path / "test-ancilla-sweep.svg"
        result = _report_local.plot_ancilla_sweep(data, save_path=path)
        assert result.exists()

    def test_plot_memory_sweep(self, tmp_path: Path) -> None:
        data = self._make_dummy_data()
        path = tmp_path / "test-memory-sweep.svg"
        result = _report_local.plot_memory_sweep(data, save_path=path)
        assert result.exists()

    def test_plot_time_sweep(self, tmp_path: Path) -> None:
        data_by_theta = {
            0.0: self._make_dummy_data(),
            0.5: self._make_dummy_data(),
        }
        path = tmp_path / "test-time-sweep.svg"
        result = _report_local.plot_time_sweep(data_by_theta, lam=1.0, save_path=path)
        assert result.exists()
