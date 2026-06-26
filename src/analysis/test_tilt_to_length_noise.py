"""Smoke tests for :mod:`src.analysis.tilt_to_length_noise`.

Tests the main simulation functions at single parameter points
and validates error handling in configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from .tilt_to_length_noise import (
    TTLNoiseConfig,
    ttl_limited_sensitivity,
    ttl_path_length_noise,
    ttl_phase_noise,
    ttl_scaling_sweep,
    ttl_sensitivity_floor,
)


class TestTTLNoiseConfig:
    """Validates TTLNoiseConfig parameter constraints."""

    def test_valid_config(self) -> None:
        """Happy path: all valid parameters."""
        config = TTLNoiseConfig(
            theta_rms=1e-6, L=1.0, wavelength=1e-6, beam_offset=1e-3
        )
        assert config.theta_rms == 1e-6

    def test_non_positive_theta_rms_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="RMS angular jitter"):
            ttl_path_length_noise(TTLNoiseConfig(theta_rms=0.0))

    def test_non_positive_L_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Arm length L must be positive"):
            ttl_path_length_noise(TTLNoiseConfig(L=0.0))

    def test_non_positive_wavelength_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            ttl_path_length_noise(TTLNoiseConfig(wavelength=0.0))

    def test_non_positive_beam_offset_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Beam offset must be positive"):
            ttl_path_length_noise(TTLNoiseConfig(beam_offset=0.0))

    def test_nan_parameter_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ttl_path_length_noise(TTLNoiseConfig(theta_rms=float("nan")))

    def test_inf_parameter_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ttl_path_length_noise(TTLNoiseConfig(theta_rms=float("inf")))


class TestTtlPathLengthNoise:
    """Smoke tests for TTL path length noise."""

    def test_basic_computation(self) -> None:
        """δL = θ_rms · beam_offset."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        delta_L = ttl_path_length_noise(config)
        assert np.isclose(delta_L, 1e-9, atol=1e-12)

    def test_different_units(self) -> None:
        """Proportional to both θ_rms and beam_offset."""
        config = TTLNoiseConfig(theta_rms=2e-6, beam_offset=3e-3)
        delta_L = ttl_path_length_noise(config)
        assert np.isclose(delta_L, 6e-9, atol=1e-12)


class TestTtlPhaseNoise:
    """Smoke tests for TTL phase noise."""

    def test_basic_computation(self) -> None:
        """δφ = 2π · δL / λ."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        phase = ttl_phase_noise(config)
        expected = 2.0 * np.pi * 1e-9 / 1e-6
        assert np.isclose(phase, expected, atol=1e-10)

    def test_non_zero_positive(self) -> None:
        """Phase noise is positive for valid config."""
        config = TTLNoiseConfig()
        phase = ttl_phase_noise(config)
        assert phase > 0


class TestTtlSensitivityFloor:
    """Smoke tests for TTL sensitivity floor."""

    def test_equals_phase_noise(self) -> None:
        """Sensitivity floor equals TTL phase noise."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        floor = ttl_sensitivity_floor(config)
        phase = ttl_phase_noise(config)
        assert np.isclose(floor, phase, atol=1e-10)


class TestTtlLimitedSensitivity:
    """Smoke tests for TTL-limited combined sensitivity."""

    def test_quantum_dominated_regime(self) -> None:
        """At small N, quantum noise dominates over TTL."""
        config = TTLNoiseConfig(theta_rms=1e-9, beam_offset=1e-6, wavelength=1e-6)
        # SQL at N=1 is 1.0, TTL floor is ~6e-3, so quantum dominates
        total = ttl_limited_sensitivity(N=1, quantum_sensitivity=1.0, config=config)
        assert np.isclose(total, np.sqrt(1.0 + ttl_phase_noise(config) ** 2), atol=1e-5)

    def test_ttl_dominated_regime(self) -> None:
        """At very large N, TTL floor dominates."""
        config = TTLNoiseConfig()
        total = ttl_limited_sensitivity(
            N=1e10,
            quantum_sensitivity=1e-10,
            config=config,
        )
        ttl_floor = ttl_phase_noise(config)
        assert np.isclose(total, ttl_floor, atol=1e-5)

    def test_negative_n_raises_value_error(self) -> None:
        config = TTLNoiseConfig()
        with pytest.raises(ValueError, match="Particle number N must be non-negative"):
            ttl_limited_sensitivity(N=-1, quantum_sensitivity=1.0, config=config)

    def test_negative_quantum_sensitivity_raises_value_error(self) -> None:
        config = TTLNoiseConfig()
        with pytest.raises(
            ValueError, match="Quantum sensitivity must be non-negative"
        ):
            ttl_limited_sensitivity(N=1, quantum_sensitivity=-0.1, config=config)


class TestTtlScalingSweep:
    """Smoke tests for TTL scaling sweep."""

    def test_sql_case_returns_expected_keys(self) -> None:
        """Result dict contains all expected keys."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        N = np.logspace(0, 6, 20)
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        expected_keys = {
            "N",
            "delta_phi",
            "delta_phi_quantum",
            "delta_phi_ttl",
            "alpha_fitted",
        }
        assert expected_keys.issubset(result.keys())

    def test_hl_case(self) -> None:
        """Heisenberg-limited scaling also works."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        N = np.logspace(0, 6, 20)
        result = ttl_scaling_sweep(N, config, quantum_scaling="hl")
        assert result["alpha_fitted"] is not None

    def test_small_n_quantum_dominated(self) -> None:
        """At small N, delta_phi ≈ quantum contribution."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        N = np.array([1, 2, 4])
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        # For these params, TTL floor is ~6e-3 rad and SQL at N=1 is 1.0
        assert np.all(result["delta_phi_quantum"] > result["delta_phi_ttl"])

    def test_large_n_ttl_dominated(self) -> None:
        """At large N, delta_phi ≈ TTL floor."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        N = np.logspace(6, 8, 5)
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        ttl_floor = ttl_phase_noise(config)
        assert np.allclose(result["delta_phi"], ttl_floor, atol=1e-4)

    def test_non_positive_n_raises_value_error(self) -> None:
        config = TTLNoiseConfig()
        with pytest.raises(ValueError, match="All N values must be positive"):
            ttl_scaling_sweep(np.array([-1, 1]), config)

    def test_invalid_quantum_scaling_raises_value_error(self) -> None:
        config = TTLNoiseConfig()
        with pytest.raises(ValueError, match="Quantum scaling must be 'sql' or 'hl'"):
            ttl_scaling_sweep(np.array([1, 2]), config, quantum_scaling="unknown")

    def test_fitted_alpha_exists(self) -> None:
        """Fitted alpha is a float or None (not missing)."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        N = np.logspace(0, 6, 20)
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        # At minimum we have 20 points and the fit should succeed
        assert result["alpha_fitted"] is not None
        assert np.isfinite(result["alpha_fitted"])

    def test_few_points_fit_is_none(self) -> None:
        """With fewer than 3 N values, alpha_fitted is None."""
        config = TTLNoiseConfig()
        result = ttl_scaling_sweep(np.array([1, 2]), config)
        assert result["alpha_fitted"] is None

    def test_degenerate_n_points_fit_handled_gracefully(self) -> None:
        """When curve_fit fails (e.g., duplicate N values), alpha is None."""
        config = TTLNoiseConfig()
        # Identical N values create degenerate data that may cause curve_fit failure
        result = ttl_scaling_sweep(np.array([1, 1, 1]), config)
        # Either alpha is None (fit failed) or a float (fit somehow succeeded)
        assert result["alpha_fitted"] is None or np.isfinite(result["alpha_fitted"])
