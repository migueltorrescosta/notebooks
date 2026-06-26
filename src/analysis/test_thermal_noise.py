"""Smoke tests for :mod:`src.analysis.thermal_noise`.

Tests the main simulation functions at single parameter points
and validates error handling in configuration.
"""

from __future__ import annotations

import numpy as np

from .thermal_noise import (
    ThermalLangevinConfig,
    combined_sensitivity,
    create_quantum_only_config,
    create_thermal_config,
    create_thermal_dominated_config,
    crossover_N,
    fit_thermal_scaling_exponent,
    force_psd_thermal,
    mechanical_susceptibility,
    sweep_thermal_scaling,
    thermal_floor_approximation,
    thermal_sensitivity_at_N,
    thermal_sensitivity_normalized,
)


class TestThermalSensitivityNormalized:
    """Smoke tests for normalized thermal sensitivity."""

    def test_basic_computation(self) -> None:
        """Δφ_thermal = thermal_strength * N^thermal_exponent."""
        config = ThermalLangevinConfig(thermal_strength=0.5, thermal_exponent=0.0)
        delta = thermal_sensitivity_normalized(100.0, config)
        assert delta == 0.5

    def test_power_law_scaling(self) -> None:
        """With exponent -0.25, sensitivity follows N^(-0.25)."""
        config = ThermalLangevinConfig(thermal_strength=1.0, thermal_exponent=-0.25)
        d1 = thermal_sensitivity_normalized(10.0, config)
        d2 = thermal_sensitivity_normalized(100.0, config)
        assert d2 < d1
        # Δφ(100) = 100^(-0.25) = 10^(-0.5) ≈ 0.316
        assert np.isclose(d2, 100.0 ** (-0.25), atol=1e-10)


class TestCombinedSensitivity:
    """Smoke tests for combined thermal+quantum sensitivity."""

    def test_quantum_only_regime(self) -> None:
        """With negligible thermal noise, Δφ ≈ 1/√N."""
        config = create_quantum_only_config()
        delta = combined_sensitivity(100.0, config)
        assert np.isclose(delta, 0.1, atol=1e-5)

    def test_thermal_dominated_regime(self) -> None:
        """With strong thermal noise, Δφ ≈ thermal_strength."""
        config = create_thermal_dominated_config()
        delta = combined_sensitivity(100.0, config)
        assert np.isclose(delta, 10.0, atol=0.5)

    def test_crossover_region(self) -> None:
        """When thermal ≈ quantum, combined is √2× each."""
        config = ThermalLangevinConfig(thermal_strength=1.0, thermal_exponent=-0.5)
        delta = combined_sensitivity(1.0, config)
        expected = np.sqrt(2.0)
        assert np.isclose(delta, expected, atol=1e-5)

    def test_larger_n_gives_better_sensitivity(self) -> None:
        """Combined sensitivity improves (decreases) with N."""
        config = ThermalLangevinConfig(thermal_strength=0.1, thermal_exponent=0.0)
        d_small = combined_sensitivity(1.0, config)
        d_large = combined_sensitivity(100.0, config)
        assert d_large < d_small


class TestThermalSensitivityAtN:
    """Smoke tests for the main survey entry point."""

    def test_delegates_to_combined(self) -> None:
        """Returns same result as combined_sensitivity."""
        config = ThermalLangevinConfig(thermal_strength=0.1, thermal_exponent=0.0)
        direct = combined_sensitivity(10.0, config)
        via_fn = thermal_sensitivity_at_N(10.0, config)
        assert np.isclose(via_fn, direct, atol=1e-10)

    def test_positive_at_large_n(self) -> None:
        """Δφ > 0 for large N."""
        config = ThermalLangevinConfig(thermal_strength=0.01, thermal_exponent=0.0)
        delta = thermal_sensitivity_at_N(10000.0, config)
        assert delta > 0


class TestSweepThermalScaling:
    """Smoke tests for thermal scaling sweep."""

    def test_returns_correct_shape(self) -> None:
        """Output arrays match input N_values length."""
        N_vals = [1, 2, 4, 8, 16]
        config = ThermalLangevinConfig(thermal_strength=0.1, thermal_exponent=0.0)
        N_arr, delta_arr = sweep_thermal_scaling(N_vals, config)
        assert len(N_arr) == len(N_vals)
        assert len(delta_arr) == len(N_vals)

    def test_monotonic_decreasing(self) -> None:
        """Sensitivity decreases (improves) as N increases."""
        N_vals = [1, 2, 4, 8, 16]
        config = ThermalLangevinConfig(thermal_strength=0.1, thermal_exponent=0.0)
        _, delta_arr = sweep_thermal_scaling(N_vals, config)
        assert all(delta_arr[i] > delta_arr[i + 1] for i in range(len(delta_arr) - 1))


class TestFitThermalScalingExponent:
    """Smoke tests for fitting thermal scaling exponent."""

    def test_quantum_only_exponent(self) -> None:
        """Pure SQL gives α ≈ -0.5."""
        N_vals = [1, 2, 4, 8, 16, 32, 64, 128]
        config = create_quantum_only_config()
        result = fit_thermal_scaling_exponent(N_vals, config)
        assert np.isclose(result.alpha, -0.5, atol=0.05)

    def test_thermal_dominated_exponent(self) -> None:
        """Strong constant thermal floor gives α ≈ 0."""
        N_vals = [1, 10, 100, 1000]
        config = create_thermal_dominated_config()
        result = fit_thermal_scaling_exponent(N_vals, config)
        assert np.isclose(result.alpha, 0.0, atol=0.05)


class TestCombinedSensitivityNonNormalized:
    """Smoke tests for non-normalized combined sensitivity."""

    def test_non_normalized_returns_same_as_normalized(self) -> None:
        """Current implementation treats both paths identically."""
        config_norm = ThermalLangevinConfig(
            thermal_strength=0.1,
            thermal_exponent=0.0,
            use_normalized=True,
        )
        config_nonorm = ThermalLangevinConfig(
            thermal_strength=0.1,
            thermal_exponent=0.0,
            use_normalized=False,
        )
        d_norm = combined_sensitivity(100.0, config_norm)
        d_nonorm = combined_sensitivity(100.0, config_nonorm)
        assert np.isclose(d_norm, d_nonorm, atol=1e-10)


class TestCrossoverN:
    """Smoke tests for crossover particle number."""

    def test_constant_thermal_floor(self) -> None:
        """For constant floor, N_cross = 1/thermal_strength²."""
        config = ThermalLangevinConfig(thermal_strength=0.1, thermal_exponent=0.0)
        N_cross = crossover_N(config)
        assert np.isclose(N_cross, 100.0, atol=0.01)

    def test_very_small_thermal_no_crossover(self) -> None:
        """Negligible thermal → crossover at infinity."""
        config = create_quantum_only_config()
        N_cross = crossover_N(config)
        # With strength=1e-10 and exponent=0.0, crossover exists but is huge
        assert N_cross > 1e15

    def test_thermal_decays_as_fast_as_quantum_no_crossover(self) -> None:
        """When thermal exponent == -0.5, no finite crossover."""
        config = ThermalLangevinConfig(thermal_strength=1.0, thermal_exponent=-0.5)
        N_cross = crossover_N(config)
        assert N_cross == np.inf

    def test_negative_exponent_slower_than_sql(self) -> None:
        """thermal_exponent > -0.5 gives finite crossover."""
        config = ThermalLangevinConfig(thermal_strength=0.5, thermal_exponent=-0.25)
        N_cross = crossover_N(config)
        assert np.isfinite(N_cross)
        assert N_cross > 0

    def test_exponent_at_sql_rate_no_finite_crossover(self) -> None:
        """thermal_exponent = -0.5 → no finite crossover (thermal = quantum at all N)."""
        config = ThermalLangevinConfig(thermal_strength=1.0, thermal_exponent=-0.5)
        N_cross = crossover_N(config)
        assert N_cross == np.inf

    def test_non_normalized_bisection_matches_analytical(self) -> None:
        """Non-normalized bisection gives same result as analytical formula."""
        config = ThermalLangevinConfig(
            thermal_strength=0.1,
            thermal_exponent=0.0,
            use_normalized=False,
        )
        N_cross = crossover_N(config)
        # For alpha=0: N_cross = 1/S² = 1/0.01 = 100
        # Bisection with finite iterations gives approximation
        assert np.isclose(N_cross, 100.0, atol=0.1)


class TestMechanicalSusceptibility:
    """Smoke tests for mechanical susceptibility."""

    def test_scalar_input(self) -> None:
        """Returns complex scalar for scalar frequency."""
        chi = mechanical_susceptibility(1.0, m=1.0, omega_m=10.0, gamma=0.1)
        assert isinstance(chi, complex)

    def test_array_input(self) -> None:
        """Returns complex array for array frequency."""
        omega = np.array([1.0, 10.0, 100.0])
        chi = mechanical_susceptibility(omega, m=1.0, omega_m=10.0, gamma=0.1)
        assert isinstance(chi, np.ndarray)
        assert chi.shape == omega.shape

    def test_resonance_peak(self) -> None:
        """Susceptibility peaks near resonance ω = ω_m."""
        omega = np.array([9.9, 10.0, 10.1])
        chi = mechanical_susceptibility(omega, m=1.0, omega_m=10.0, gamma=0.1)
        chi_abs = np.abs(chi)
        assert chi_abs[1] > chi_abs[0]
        assert chi_abs[1] > chi_abs[2]


class TestForcePSDThermal:
    """Smoke tests for thermal force PSD."""

    def test_basic_computation(self) -> None:
        """S_F = 2 * gamma * k_B * temp."""
        S = force_psd_thermal(temp=300.0, gamma=0.1, k_B=1.0)
        assert np.isclose(S, 60.0, atol=1e-10)

    def test_zero_temperature(self) -> None:
        """Zero temperature → zero force PSD."""
        S = force_psd_thermal(temp=0.0, gamma=0.1)
        assert S == 0.0

    def test_default_k_B(self) -> None:
        """Default k_B = 1.0."""
        S = force_psd_thermal(temp=1.0, gamma=1.0)
        assert np.isclose(S, 2.0, atol=1e-10)


class TestThermalFloorApproximation:
    """Smoke tests for thermal floor approximation."""

    def test_returns_strength(self) -> None:
        """Floor equals thermal_strength for normalized config."""
        config = ThermalLangevinConfig(thermal_strength=0.5, thermal_exponent=0.0)
        floor = thermal_floor_approximation(config)
        assert floor == 0.5


class TestConfigFactories:
    """Smoke tests for convenience configuration factories."""

    def test_create_thermal_config(self) -> None:
        """Factory creates config with specified strength."""
        config = create_thermal_config(thermal_strength=0.2, thermal_exponent=-0.1)
        assert config.thermal_strength == 0.2
        assert config.thermal_exponent == -0.1
        assert config.use_normalized is True

    def test_create_quantum_only_config(self) -> None:
        """Quantum-only config has negligible thermal strength."""
        config = create_quantum_only_config()
        assert config.thermal_strength == 1e-10
        assert config.use_normalized is True

    def test_create_thermal_dominated_config(self) -> None:
        """Thermal-dominated config has high thermal strength."""
        config = create_thermal_dominated_config()
        assert config.thermal_strength == 10.0
        assert config.use_normalized is True
