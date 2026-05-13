"""Tests for the thermal Langevin noise model.

Tests verify:
1. High-frequency limit matches SQL
2. Low-frequency limit approaches a constant floor
3. Quadrature sum with quantum noise gives expected crossover
4. Scaling sweep returns valid exponents
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.scaling_fit import ScalingFitResult
from src.physics.thermal_langevin import (
    ThermalLangevinConfig,
    combined_sensitivity,
    create_quantum_only_config,
    create_thermal_config,
    create_thermal_dominated_config,
    crossover_N,
    fit_thermal_scaling_exponent,
    mechanical_susceptibility,
    sql_sensitivity,
    sweep_thermal_scaling,
    thermal_sensitivity_at_N,
    thermal_sensitivity_normalized,
)


class TestThermalLangevinBasics:
    """Basic tests for thermal Langevin noise model."""

    def test_sql_sensitivity_scales_correctly(self) -> None:
        """SQL should scale as 1/√N."""
        assert sql_sensitivity(4) == pytest.approx(0.5)  # 1/2
        assert sql_sensitivity(100) == pytest.approx(0.1)  # 1/10
        assert sql_sensitivity(1) == pytest.approx(1.0), (
            "Expected sql_sensitivity(1) == pytest.approx(1.0)"
        )

    def test_thermal_sensitivity_normalized(self) -> None:
        """Normalized thermal sensitivity follows the formula."""
        config = create_thermal_config(thermal_strength=0.5, thermal_exponent=0.0)

        # Constant thermal_exponent=0 means thermal is constant
        assert thermal_sensitivity_normalized(1, config) == pytest.approx(0.5), (
            "Expected thermal_sensitivity_normalized(1, config) == pytest.approx(0.5)"
        )
        assert thermal_sensitivity_normalized(10, config) == pytest.approx(0.5), (
            "Expected thermal_sensitivity_normalized(10, config) == pytest.approx(0.5)"
        )
        assert thermal_sensitivity_normalized(100, config) == pytest.approx(0.5), (
            "Expected thermal_sensitivity_normalized(100, config) == pytest.approx(0.5)"
        )

    def test_thermal_sensitivity_with_exponent(self) -> None:
        """With non-zero exponent, thermal scales with N."""
        config = ThermalLangevinConfig(
            thermal_strength=1.0,
            thermal_exponent=-0.25,
            use_normalized=True,
        )

        # N=1: 1 * 1^(-0.25) = 1
        # N=16: 1 * 16^(-0.25) = 1/2
        assert thermal_sensitivity_normalized(1, config) == pytest.approx(1.0), (
            "Expected thermal_sensitivity_normalized(1, config) == pytest.approx(1.0)"
        )
        assert thermal_sensitivity_normalized(16, config) == pytest.approx(0.5), (
            "Expected thermal_sensitivity_normalized(16, config) == pytest.approx(0.5)"
        )

    def test_combined_sensitivity_quadrature_sum(self) -> None:
        """Combined sensitivity should be sqrt(quantum² + thermal²)."""
        # Test: quantum=3, thermal=4 → combined=5
        # We need N and config where this works out
        # quantum = 1/sqrt(N) = 3 → N = 1/9
        # thermal = 4
        config = create_thermal_config(thermal_strength=4.0, thermal_exponent=0.0)
        N = 1.0 / 9.0

        combined = combined_sensitivity(N, config)

        # Check: sqrt((1/sqrt(1/9))² + 4²) = sqrt(3² + 4²) = sqrt(25) = 5
        quantum = 1.0 / np.sqrt(N)
        thermal = 4.0
        expected = np.sqrt(quantum**2 + thermal**2)
        assert combined == pytest.approx(expected), (
            "Expected combined == pytest.approx(expected)"
        )

    def test_mechanical_susceptibility(self) -> None:
        """Test susceptibility function works for edge cases."""
        # Test scalar input
        chi = mechanical_susceptibility(omega=0.0, m=1.0, omega_m=1.0, gamma=0.1)

        # At DC: χ = 1/mω_m² = 1/(1*1) = 1
        assert np.real(chi) == pytest.approx(1.0, abs=0.01), (
            "Expected np.real(chi) == pytest.approx(1.0, abs=0.01)"
        )


class TestThermalScalingLimits:
    """Tests for scaling limits (SQL vs thermal floor)."""

    def test_quantum_only_gives_sql_exponent(self) -> None:
        """When quantum noise dominates entirely, α should be -0.5 (SQL)."""
        config = create_quantum_only_config()

        N_values = [4, 8, 16, 32, 64]
        result = fit_thermal_scaling_exponent(N_values, config, min_N=4)

        assert isinstance(result, ScalingFitResult), (
            "Expected result to be instance of ScalingFitResult"
        )
        assert result.valid, "Condition failed: result.valid"

        # Should be very close to SQL (α = -0.5)
        assert -0.55 < result.alpha < -0.45, "Expected -0.55 < result.alpha < -0.45"

    def test_thermal_dominated_gives_alpha_zero(self) -> None:
        """When thermal noise dominates, α should be near 0."""
        # Thermal with very strong thermal (constant floor)
        config = create_thermal_config(thermal_strength=1000.0, thermal_exponent=0.0)

        N_values = [4, 8, 16, 32, 64]
        result = fit_thermal_scaling_exponent(N_values, config, min_N=4)

        # Since thermal is constant, combined ≈ thermal ≈ constant
        # So log(Δφ) ≈ constant, slope α ≈ 0
        assert (
            result.alpha > -0.1
        )  # Close to 0 (maybe slightly negative due to small quantum contribution at small N)


class TestCrossoverBehavior:
    """Tests for crossover between quantum and thermal regimes."""

    def test_crossover_N_analytical(self) -> None:
        """Crossover finder should locate N where thermal = quantum."""
        # thermal_strength = 0.1 means:
        # At N=1: quantum=1, thermal=0.1 → quantum dominates
        # At N=100: quantum=0.1, thermal=0.1 → crossover
        config = create_thermal_config(thermal_strength=0.1, thermal_exponent=0.0)

        # Analytical crossover should be 1 / thermal_strength² = 1 / 0.01 = 100
        N_cross = crossover_N(config)

        assert N_cross == pytest.approx(100.0), (
            "Expected N_cross == pytest.approx(100.0)"
        )

    def test_crossover_behavior_visible_in_sweep(self) -> None:
        """Verify the crossover is visible in sensitivity curves."""
        config = create_thermal_config(thermal_strength=0.1, thermal_exponent=0.0)

        # Below crossover: N=10 → quantum=1/√10 ≈ 0.316 > thermal=0.1
        # Actually wait: quantum DECREASES with N, thermal CONSTANT
        # So at SMALL N: quantum is LARGE (dominates), thermal is SMALL
        # At LARGE N: quantum is SMALL, thermal is LARGE (dominates)

        N_low = 4.0
        N_high = 10000.0

        delta_low = thermal_sensitivity_at_N(N_low, config)
        delta_high = thermal_sensitivity_at_N(N_high, config)

        # At low N: quantum dominates (1/2=0.5) > thermal (0.1)
        # At high N: thermal dominates (0.1) > quantum (0.01)

        # Check: log-log slope changes from -0.5 to 0
        # Let's check the scaling
        # values actually:
        # At N=4: combined ~ 0.51 (dominated by quantum 0.5
        # At N=10000: combined ~ 0.10005 (dominated by thermal 0.1

        assert delta_low > delta_high  # Sensitivity improves (decreases) with N)
        # But the rate of improvement slows

        # Actually:
        # Below crossover (N < 100): quantum slope ~-0.5
        # Above crossover (N > 100): thermal takes over, slope ~0

        # Let's verify the scaling
        N_below = [4, 8, 16, 32, 64]
        N_above = [200, 400, 800, 1600, 3200]

        result_below = fit_thermal_scaling_exponent(N_below, config, min_N=4)
        result_above = fit_thermal_scaling_exponent(N_above, config, min_N=4)

        # Below crossover: quantum-like slope ≈ -0.5
        assert result_below.alpha < -0.4, "Expected result_below.alpha < -0.4"

        # Above crossover: flatter (closer to 0)
        # Actually, above crossover the slope might still be slightly negative
        # because quantum contributes a little
        # small contribution. Let's just check it's flatter.
        assert result_above.alpha > result_below.alpha, (
            "Expected result_above.alpha > result_below.alpha"
        )


class TestThermalScalingExponents:
    """Tests for scaling exponent extraction."""

    def test_sweep_thermal_scaling_returns_arrays(self) -> None:
        """Sweep should return N and delta_phi arrays."""
        config = create_thermal_config(thermal_strength=0.1, thermal_exponent=0.0)
        N_values = [2, 4, 8, 16, 32]

        N_arr, delta_arr = sweep_thermal_scaling(N_values, config)

        assert len(N_arr) == len(N_values), "Expected len(N_arr) == len(N_values)"
        assert len(delta_arr) == len(N_values), (
            "Expected len(delta_arr) == len(N_values)"
        )
        assert np.all(np.isfinite(delta_arr)), (
            "All values should satisfy np.isfinite(delta_arr)"
        )
        assert np.all(delta_arr > 0), "Expected np.all(delta_arr > 0)"

    def test_fit_result_has_valid_metrics(self) -> None:
        """Fit results should have error estimates and R²."""
        config = create_quantum_only_config()
        N_values = [4, 8, 16, 32, 64]

        result = fit_thermal_scaling_exponent(N_values, config, min_N=4)

        assert isinstance(result.alpha, float), (
            "Expected result.alpha to be instance of float"
        )
        assert isinstance(result.alpha_err, float), (
            "Expected result.alpha_err to be instance of float"
        )
        assert isinstance(result.R_squared, float), (
            "Expected result.R_squared to be instance of float"
        )
        assert 0 <= result.R_squared <= 1.001, "Expected 0 <= result.R_squared <= 1.001"
        assert result.alpha_err >= 0, "Expected result.alpha_err >= 0"

    def test_monotonic_improvement(self) -> None:
        """Sensitivity should improve (decrease) or stay flat with increasing N."""
        config = create_thermal_config(thermal_strength=0.01, thermal_exponent=0.0)

        N_values = [1, 10, 100, 1000, 10000]
        deltas = [thermal_sensitivity_at_N(N, config) for N in N_values]

        # Should be monotonically decreasing or flat
        for i in range(len(deltas) - 1):
            assert deltas[i + 1] <= deltas[i] + 1e-10, (
                "Expected deltas[i + 1] <= deltas[i] + 1e-10"
            )


class TestConvenienceFunctions:
    """Tests for convenience configuration functions."""

    def test_create_quantum_only(self) -> None:
        """Quantum-only config should have tiny thermal strength."""
        config = create_quantum_only_config()
        assert config.thermal_strength < 1e-5, "Expected config.thermal_strength < 1e-5"
        assert config.use_normalized, "Condition failed: config.use_normalized"

    def test_create_thermal_dominated(self) -> None:
        """Thermal-dominated config should have large thermal strength."""
        config = create_thermal_dominated_config()
        assert config.thermal_strength == 10.0, (
            "Expected config.thermal_strength == 10.0"
        )

    def test_create_thermal_config(self) -> None:
        """Custom config creation should work."""
        config = create_thermal_config(thermal_strength=0.5, thermal_exponent=-0.3)
        assert config.thermal_strength == 0.5, "Expected config.thermal_strength == 0.5"
        assert config.thermal_exponent == -0.3, (
            "Expected config.thermal_exponent == -0.3"
        )
        assert config.use_normalized, "Condition failed: config.use_normalized"
