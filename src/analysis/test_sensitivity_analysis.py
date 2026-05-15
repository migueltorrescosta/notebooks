"""
Tests for Sensitivity Analysis physics module.

Covers:
- Rabi frequency computation: positivity, cancellation
- Sensitivity output: key presence, boundedness, finite values
- Sensitivity grid: output shape correctness
- Observable computation: range bounds, initial condition
- Input validation: k out of range
"""

from __future__ import annotations

import numpy as np
import pytest

from .sensitivity_analysis import (
    compute_observable,
    compute_rabi_frequency,
    compute_sensitivity_grid,
    sensitivity,
)


class TestRabiFrequency:
    def test_given_nonzero_coefficients_then_rabi_frequency_is_positive(
        self,
    ) -> None:
        omega = compute_rabi_frequency(4, 0, 0.0, 0.0, 1.0, 1.0)
        assert omega >= 0

    def test_given_canceling_coefficients_then_rabi_frequency_is_zero(
        self,
    ) -> None:
        omega = compute_rabi_frequency(4, 2, 0.0, 0.0, 0.0, 0.0)
        assert omega == pytest.approx(0.0)


class TestSensitivity:
    def test_returns_dict_with_all_keys(self) -> None:
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert "omega_k" in result
        assert "sensitivity_to_j" in result
        assert "sensitivity_to_delta" in result

    def test_given_large_evolution_time_then_sensitivity_is_bounded(
        self,
    ) -> None:
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 10.0)
        assert abs(result["sensitivity_to_j"]) <= 1.0
        assert abs(result["sensitivity_to_delta"]) <= 1.0


class TestGrid:
    def test_sensitivity_grid_has_correct_shape(self) -> None:
        alpha_x = np.linspace(-5, 5, 11)
        alpha_z = np.linspace(-5, 5, 11)
        result = compute_sensitivity_grid(4, 0, 0.0, 0.0, alpha_x, alpha_z, 1.0)
        assert result["omega_k"].shape == (11, 11)
        assert result["sensitivity_to_j"].shape == (11, 11)


class TestObservable:
    def test_observable_is_bounded_between_minus_one_and_one(self) -> None:
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert -1.0 <= obs <= 1.0

    def test_given_zero_time_then_observable_is_one(self) -> None:
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 0.0)
        assert obs == pytest.approx(1.0, abs=0.01)


class TestValidation:
    def test_given_nonzero_params_then_sensitivities_are_finite(self) -> None:
        result = sensitivity(4, 0, 1.0, 1.0, 1.0, 1.0, 0.5)
        assert np.isfinite(result["sensitivity_to_j"])
        assert np.isfinite(result["sensitivity_to_delta"])


class TestInputValidation:
    def test_k_greater_than_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_rabi_frequency(4, 5, 0.0, 0.0, 0.0, 0.0)

    def test_negative_k_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_rabi_frequency(4, -1, 0.0, 0.0, 0.0, 0.0)

    def test_negative_k_in_sensitivity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            sensitivity(4, -1, 0.0, 0.0, 0.0, 0.0, 1.0)

    def test_k_greater_than_n_in_sensitivity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            sensitivity(4, 5, 0.0, 0.0, 0.0, 0.0, 1.0)
