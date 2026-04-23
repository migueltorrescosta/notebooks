"""Unit tests for Sensitivity Analysis physics module."""

import numpy as np
import pytest

from src.sensitivity_analysis import (
    compute_rabi_frequency,
    sensitivity,
    compute_sensitivity_grid,
    compute_observable,
)


class TestRabiFrequency:
    def test_rabi_frequency_positive(self) -> None:
        """Rabi frequency should be positive."""
        omega = compute_rabi_frequency(4, 0, 0.0, 0.0, 1.0, 1.0)
        assert omega >= 0

    def test_rabi_frequency_zero(self) -> None:
        """Should be zero when coefficients cancel."""
        # At resonance: x_coeff = -j_s, z_coeff = δ_s
        # At ω = 0 when both are 0
        omega = compute_rabi_frequency(4, 2, 0.0, 0.0, 0.0, 0.0)
        assert omega == pytest.approx(0.0)


class TestSensitivity:
    def test_sensitivity_dict(self) -> None:
        """Should return dictionary with all keys."""
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert "omega_k" in result
        assert "sensitivity_to_j" in result
        assert "sensitivity_to_delta" in result

    def test_sensitivity_bounds(self) -> None:
        """Sensitivities should be bounded."""
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 10.0)
        # sin²(ωt) ≤ 1, divided by ω², so bounded
        assert abs(result["sensitivity_to_j"]) <= 1.0
        assert abs(result["sensitivity_to_delta"]) <= 1.0


class TestGrid:
    def test_grid_shapes(self) -> None:
        """Grids should have correct shapes."""
        alpha_x = np.linspace(-5, 5, 11)
        alpha_z = np.linspace(-5, 5, 11)
        result = compute_sensitivity_grid(4, 0, 0.0, 0.0, alpha_x, alpha_z, 1.0)
        assert result["omega_k"].shape == (11, 11)
        assert result["sensitivity_to_j"].shape == (11, 11)


class TestObservable:
    def test_observable_in_range(self) -> None:
        """⟨σ_z⟩ should be in [-1, 1]."""
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert -1.0 <= obs <= 1.0

    def test_observable_at_t0(self) -> None:
        """At t=0, ⟨σ_z⟩ = 1."""
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 0.0)
        assert obs == pytest.approx(1.0, abs=0.01)


class TestValidation:
    def test_sensitivity_values(self) -> None:
        """Sensitivities should be calculated."""
        result = sensitivity(4, 0, 1.0, 1.0, 1.0, 1.0, 0.5)
        # Just check that sensitivities are calculated (finite values)
        assert np.isfinite(result["sensitivity_to_j"])
        assert np.isfinite(result["sensitivity_to_delta"])
