"""Unit tests for Sensitivity Analysis physics module."""

import numpy as np
import pytest

from .sensitivity_analysis import (
    compute_observable,
    compute_rabi_frequency,
    compute_sensitivity_grid,
    sensitivity,
)


class TestRabiFrequency:
    def test_rabi_frequency_should_be_positive(self) -> None:
        omega = compute_rabi_frequency(4, 0, 0.0, 0.0, 1.0, 1.0)
        assert omega >= 0, "Expected omega >= 0"

    def test_rabi_frequency_should_be_zero_when_coefficients_cancel(self) -> None:
        # At resonance: x_coeff = -j_s, z_coeff = δ_s
        # At ω = 0 when both are 0
        omega = compute_rabi_frequency(4, 2, 0.0, 0.0, 0.0, 0.0)
        assert omega == pytest.approx(0.0), "Expected omega == pytest.approx(0.0)"


class TestSensitivity:
    def test_sensitivity_should_return_dict_with_all_keys(self) -> None:
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert "omega_k" in result, 'Expected "omega_k" in result'
        assert "sensitivity_to_j" in result, 'Expected "sensitivity_to_j" in result'
        assert "sensitivity_to_delta" in result, (
            'Expected "sensitivity_to_delta" in result'
        )

    def test_sensitivity_should_be_bounded(self) -> None:
        result = sensitivity(4, 0, 0.0, 0.0, 1.0, 1.0, 10.0)
        # sin²(ωt) ≤ 1, divided by ω², so bounded
        assert abs(result["sensitivity_to_j"]) <= 1.0, (
            'Expected abs(result["sensitivity_to_j"]) <= 1.0'
        )
        assert abs(result["sensitivity_to_delta"]) <= 1.0, (
            'Expected abs(result["sensitivity_to_delta"]) <= 1.0'
        )


class TestGrid:
    def test_grids_should_have_correct_shapes(self) -> None:
        alpha_x = np.linspace(-5, 5, 11)
        alpha_z = np.linspace(-5, 5, 11)
        result = compute_sensitivity_grid(4, 0, 0.0, 0.0, alpha_x, alpha_z, 1.0)
        assert result["omega_k"].shape == (11, 11), (
            'Expected result["omega_k"].shape == (11, 11)'
        )
        assert result["sensitivity_to_j"].shape == (11, 11), (
            'Expected result["sensitivity_to_j"].shape == (11, 11)'
        )


class TestObservable:
    def test_sigma_z_should_be_in_minus_1_to_1(self) -> None:
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert -1.0 <= obs <= 1.0, "Expected -1.0 <= obs <= 1.0"

    def test_at_t0_sigma_z_should_equal_1(self) -> None:
        obs = compute_observable(4, 0, 0.0, 0.0, 1.0, 1.0, 0.0)
        assert obs == pytest.approx(1.0, abs=0.01), (
            "Expected obs == pytest.approx(1.0, abs=0.01)"
        )


class TestValidation:
    def test_sensitivities_should_be_calculated(self) -> None:
        result = sensitivity(4, 0, 1.0, 1.0, 1.0, 1.0, 0.5)
        # Just check that sensitivities are calculated (finite values)
        assert np.isfinite(result["sensitivity_to_j"]), (
            'Expected result["sensitivity_to_j"] to be finite'
        )
        assert np.isfinite(result["sensitivity_to_delta"]), (
            'Expected result["sensitivity_to_delta"] to be finite'
        )
