"""Smoke tests for :mod:`src.analysis.dynamical_decoupling`.

Tests the main simulation functions at single parameter points
and validates error handling in configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from .dynamical_decoupling import (
    DDConfig,
    cpmg_filter_function,
    dd_effective_coherence_time,
    dd_phase_sensitivity,
    dd_sensitivity_scaling,
)


class TestDDConfig:
    """Validates DDConfig.__post_init__ validation logic."""

    def test_valid_config(self) -> None:
        """Happy path: all valid parameters."""
        config = DDConfig(n_pulses=4, sequence="CPMG", tau=0.5, pulse_axis="x")
        assert config.n_pulses == 4
        assert config.sequence == "CPMG"

    def test_negative_pulses_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Number of pulses must be non-negative"):
            DDConfig(n_pulses=-1)

    def test_invalid_sequence_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Sequence must be 'CPMG' or 'XY8'"):
            DDConfig(sequence="unknown")

    def test_non_positive_tau_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Inter-pulse delay tau must be positive"):
            DDConfig(tau=0.0)

    def test_invalid_pulse_axis_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Pulse axis must be 'x' or 'y'"):
            DDConfig(pulse_axis="z")


class TestCpmgFilterFunction:
    """Smoke tests for the CPMG filter function."""

    def test_output_shape(self) -> None:
        """Returns array of same shape as input omega."""
        omega = np.linspace(0, 10, 100)
        F = cpmg_filter_function(omega, n_pulses=4, tau=0.5)
        assert F.shape == omega.shape

    def test_zero_at_dc(self) -> None:
        """F(0) should be 0 for n_pulses > 0 (DC suppressed)."""
        omega = np.linspace(0, 10, 1000)
        F = cpmg_filter_function(omega, n_pulses=4, tau=0.5)
        assert F[0] == 0.0

    def test_all_non_negative(self) -> None:
        """Filter function values are non-negative everywhere."""
        omega = np.linspace(0.01, 10, 100)
        F = cpmg_filter_function(omega, n_pulses=4, tau=0.5)
        assert np.all(F >= 0)

    def test_zero_pulses_returns_ones(self) -> None:
        """No DD means no filtering — all frequencies pass with unit weight."""
        omega = np.linspace(0, 10, 100)
        F = cpmg_filter_function(omega, n_pulses=0, tau=0.5)
        assert np.allclose(F, 1.0)

    def test_negative_n_pulses_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Number of pulses must be non-negative"):
            cpmg_filter_function(np.array([1.0]), n_pulses=-1, tau=0.5)

    def test_non_positive_tau_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Inter-pulse delay tau must be positive"):
            cpmg_filter_function(np.array([1.0]), n_pulses=4, tau=0.0)


class TestDDEffectiveCoherenceTime:
    """Smoke tests for DD coherence time extension."""

    def test_cpmg_improves_coherence(self) -> None:
        """CPMG extends coherence beyond bare T_2_0."""
        T_dd = dd_effective_coherence_time(T_2_0=1.0, n_pulses=8, sequence="CPMG")
        assert T_dd > 1.0

    def test_xy8_improves_more_than_cpmg(self) -> None:
        """XY-8 gives better coherence than CPMG for same pulse count."""
        T_cpmg = dd_effective_coherence_time(1.0, 8, "CPMG")
        T_xy8 = dd_effective_coherence_time(1.0, 8, "XY8")
        assert T_xy8 > T_cpmg

    def test_zero_pulses_returns_bare_time(self) -> None:
        """No DD → coherence time unchanged."""
        T_dd = dd_effective_coherence_time(T_2_0=1.0, n_pulses=0)
        assert T_dd == 1.0

    def test_more_pulses_improves_coherence(self) -> None:
        """More pulses → longer effective coherence."""
        T_few = dd_effective_coherence_time(1.0, 2)
        T_many = dd_effective_coherence_time(1.0, 16)
        assert T_many > T_few

    def test_non_positive_T_2_0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Bare coherence time must be positive"):
            dd_effective_coherence_time(-1.0, 4)

    def test_negative_pulses_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Number of pulses must be non-negative"):
            dd_effective_coherence_time(1.0, -1)

    def test_invalid_sequence_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Sequence must be 'CPMG' or 'XY8'"):
            dd_effective_coherence_time(1.0, 4, sequence="unknown")


class TestDDPhaseSensitivity:
    """Smoke tests for DD-enhanced phase sensitivity."""

    def test_sensitivity_positive(self) -> None:
        """Δφ is positive for valid inputs."""
        delta = dd_phase_sensitivity(N=100, phi_phase=np.pi / 4, T_dd=1.0, n_pulses=4)
        assert delta > 0

    def test_more_pulses_better_sensitivity(self) -> None:
        """Higher pulse count → lower Δφ (better sensitivity)."""
        d1 = dd_phase_sensitivity(100, 0.0, 1.0, 0, 1.0)
        d2 = dd_phase_sensitivity(100, 0.0, 1.0, 8, 1.0)
        assert d2 < d1

    def test_larger_n_improves_sensitivity(self) -> None:
        """More photons → lower Δφ."""
        d_small = dd_phase_sensitivity(10, 0.0, 1.0, 4)
        d_large = dd_phase_sensitivity(100, 0.0, 1.0, 4)
        assert d_large < d_small

    def test_xy8_better_than_cpmg(self) -> None:
        """XY-8 gives better sensitivity than CPMG."""
        d_cpmg = dd_phase_sensitivity(100, 0.0, 1.0, 8, 1.0, "CPMG")
        d_xy8 = dd_phase_sensitivity(100, 0.0, 1.0, 8, 1.0, "XY8")
        assert d_xy8 < d_cpmg

    def test_negative_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Mean photon number N must be positive"):
            dd_phase_sensitivity(-1, 0.0, 1.0, 4)

    def test_negative_n_pulses_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Number of pulses must be non-negative"):
            dd_phase_sensitivity(100, 0.0, 1.0, -1)

    def test_non_positive_T_dd_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match="Total evolution time T_dd must be positive"
        ):
            dd_phase_sensitivity(100, 0.0, -1.0, 4)

    def test_non_positive_T_2_0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Bare coherence time must be positive"):
            dd_phase_sensitivity(100, 0.0, 1.0, 4, T_2_0=0.0)


class TestDDSensitivityScaling:
    """Smoke tests for DD scaling analysis."""

    def test_fitted_alpha_near_negative_half(self) -> None:
        """Scaling exponent α ≈ -0.5 (SQL) regardless of pulse count."""
        N_vals = np.logspace(1, 4, 10)
        result = dd_sensitivity_scaling(N_vals, n_pulses=4, T_dd=1.0)
        assert np.isclose(result["fitted_alpha"], -0.5, atol=0.05)
        assert result["prefactor_C"] > 0
        assert result["expected_alpha"] == -0.5

    def test_more_pulses_improves_prefactor(self) -> None:
        """Prefactor C improves (decreases) with more pulses."""
        N_vals = np.logspace(1, 4, 10)
        r_few = dd_sensitivity_scaling(N_vals, n_pulses=0, T_dd=1.0)
        r_many = dd_sensitivity_scaling(N_vals, n_pulses=8, T_dd=1.0)
        assert r_many["prefactor_C"] < r_few["prefactor_C"]

    def test_result_contains_all_keys(self) -> None:
        """Result dict has all expected fields."""
        N_vals = np.logspace(1, 4, 5)
        result = dd_sensitivity_scaling(N_vals, n_pulses=4, T_dd=1.0)
        for key in ("N", "delta_phi", "fitted_alpha", "expected_alpha", "prefactor_C"):
            assert key in result

    def test_empty_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="N_values array must not be empty"):
            dd_sensitivity_scaling(np.array([]), n_pulses=4, T_dd=1.0)

    def test_non_positive_n_values_raises_value_error(self) -> None:
        N_vals = np.array([-1, 2, 4])
        with pytest.raises(ValueError, match="All N_values must be positive"):
            dd_sensitivity_scaling(N_vals, n_pulses=4, T_dd=1.0)

    def test_negative_n_pulses_raises_value_error(self) -> None:
        N_vals = np.array([1, 2, 4])
        with pytest.raises(ValueError, match="Number of pulses must be non-negative"):
            dd_sensitivity_scaling(N_vals, n_pulses=-1, T_dd=1.0)

    def test_non_positive_T_dd_raises_value_error(self) -> None:
        N_vals = np.array([1, 2, 4])
        with pytest.raises(
            ValueError,
            match="Total evolution time T_dd must be positive",
        ):
            dd_sensitivity_scaling(N_vals, n_pulses=4, T_dd=-1.0)
