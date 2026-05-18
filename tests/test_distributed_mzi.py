"""Tests for the distributed array interferometer model.

Tests verify:
1. DistributedMziConfig validation
2. Classical averaging: Δφ = 1/√(M·N)
3. Entangled scaling: Δφ = 1/(M·N)
4. Correlated noise degrades sensitivity
5. Scaling exponents match expectations
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.distributed_mzi import (
    DistributedMziConfig,
    compute_distributed_scaling,
    distributed_mzi_sensitivity,
    distributed_scaling_exponent,
    effective_scaling_at_N,
)


class TestDistributedMziConfig:
    """Test DistributedMziConfig validation."""

    def test_given_default_values_then_be_reasonable(self) -> None:
        config = DistributedMziConfig()
        assert config.M == 2
        assert config.entangled is False
        assert config.correlation_noise == 0.0

    def test_given_negative_m_raises_then_raise_valueerror(self) -> None:
        with pytest.raises(ValueError):
            DistributedMziConfig(M=-1)

    def test_given_invalid_correlation_raises_then_raise_valueerror_if_outside_0_1(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            DistributedMziConfig(correlation_noise=-0.1)
        with pytest.raises(ValueError):
            DistributedMziConfig(correlation_noise=1.5)


class TestDistributedMziSensitivity:
    """Test sensitivity computation."""

    def test_given_classical_averaging_sql_then_be_1_over_sqrt_m_times_n(self) -> None:
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(4 * 100)
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_given_classical_averaging_regime_then_identify_classical_averaging(
        self,
    ) -> None:
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert "Classical averaging" in result["regime"], (
            f"Unexpected regime: {result['regime']}"
        )

    def test_given_entangled_heisenberg_then_be_1_over_m_times_n(self) -> None:
        config = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / (4 * 100)
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_given_entangled_regime_then_identify_heisenberg_regime(self) -> None:
        config = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert "Heisenberg" in result["regime"] or "Collective" in result["regime"], (
            f"Unexpected regime: {result['regime']}"
        )

    def test_given_fully_correlated_classical_then_remove_m_benefit_delta_phi_equal_1_over_sqrt_n(
        self,
    ) -> None:
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=1.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(100)  # Single-sensor SQL
        assert result["delta_phi"] == pytest.approx(expected, rel=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_given_positive_n_raises_then_raise_valueerror_if_non_positive(self) -> None:
        config = DistributedMziConfig()
        with pytest.raises(ValueError):
            distributed_mzi_sensitivity(0, 0.0, config)

    def test_given_single_sensor_then_give_sql_for_classical_and_hl_for_entangled(
        self,
    ) -> None:
        config_classical = DistributedMziConfig(M=1, entangled=False)
        config_entangled = DistributedMziConfig(M=1, entangled=True)

        result_c = distributed_mzi_sensitivity(100, 0.0, config_classical)
        result_e = distributed_mzi_sensitivity(100, 0.0, config_entangled)

        # Classical: SQL = 1/√N
        assert result_c["delta_phi"] == pytest.approx(0.1, rel=1e-6), (
            f"Classical M=1 should give SQL, got {result_c['delta_phi']}"
        )
        # Entangled: HL = 1/N (NOON state with N particles)
        assert result_e["delta_phi"] == pytest.approx(0.01, rel=1e-6), (
            f"Entangled M=1 should give HL, got {result_e['delta_phi']}"
        )
        # Entangled beats classical for M=1 (NOON beats coherent)
        assert result_e["delta_phi"] < result_c["delta_phi"], (
            f"Entangled ({result_e['delta_phi']}) should beat "
            f"classical ({result_c['delta_phi']})"
        )


class TestDistributedScalingExponent:
    """Test scaling exponent predictions."""

    def test_given_classical_exponent_then_give_alpha_equal_minus_0_5(self) -> None:
        config = DistributedMziConfig(entangled=False)
        assert distributed_scaling_exponent(config) == -0.5, (
            "Expected distributed_scaling_exponent(config) == -0.5"
        )

    def test_given_entangled_exponent_then_give_alpha_equal_minus_1_0(self) -> None:
        config = DistributedMziConfig(entangled=True)
        assert distributed_scaling_exponent(config) == -1.0, (
            "Expected distributed_scaling_exponent(config) == -1.0"
        )

    def test_given_effective_scaling_finite_then_be_finite_and_reasonable(self) -> None:
        config = DistributedMziConfig(M=4, entangled=False)
        alpha = effective_scaling_at_N(100, config)
        assert np.isfinite(alpha), f"Alpha should be finite, got {alpha}"
        assert alpha < 0, f"Alpha should be negative for SQL, got {alpha}"


class TestComputeDistributedScaling:
    """Test distributed scaling grid computation."""

    def test_given_grid_shape_then_have_correct_shape(self) -> None:
        M_values = [1, 2, 4]
        N_values = [10, 100]
        config = DistributedMziConfig(entangled=False)
        result = compute_distributed_scaling(M_values, N_values, config)
        assert result["delta_phi_grid"].shape == (3, 2), (
            f"Expected (3, 2), got {result['delta_phi_grid'].shape}"
        )

    def test_given_more_sensors_improves_then_always_improve_or_match_sensitivity(
        self,
    ) -> None:
        M_values = [1, 2, 4]
        N_values = [100]
        config = DistributedMziConfig(entangled=False)
        result = compute_distributed_scaling(M_values, N_values, config)
        grid = result["delta_phi_grid"]
        # As M increases, Δφ should decrease
        assert grid[0, 0] > grid[1, 0], (
            f"M=1 ({grid[0, 0]:.6e}) should be worse than M=2 ({grid[1, 0]:.6e})"
        )
        assert grid[1, 0] > grid[2, 0], (
            f"M=2 ({grid[1, 0]:.6e}) should be worse than M=4 ({grid[2, 0]:.6e})"
        )
