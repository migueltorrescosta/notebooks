"""Tests for the tilt-to-length coupling noise model.

Tests verify:
1. Path length noise: δL = θ_rms · x_offset
2. Phase noise: δφ = 2π · δL / λ
3. Sensitivity floor equals TTL phase noise
4. Quadrature sum with quantum noise
5. Scaling sweep (α → 0 at large N)
6. Config validation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.tilt_to_length_noise import (
    TTLNoiseConfig,
    ttl_limited_sensitivity,
    ttl_path_length_noise,
    ttl_phase_noise,
    ttl_scaling_sweep,
    ttl_sensitivity_floor,
)


class TestTTLPathLengthNoise:
    """Test path length noise computation."""

    def test_basic_formula(self) -> None:
        """δL = θ_rms · x_offset: 1 μrad × 1 mm = 1 nm."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        result = ttl_path_length_noise(config)
        expected = 1e-9
        assert abs(result - expected) < 1e-12, f"Expected {expected}, got {result}"


class TestTTLPhaseNoise:
    """Test phase noise computation."""

    def test_phase_consistency(self) -> None:
        """δφ = 2π · δL / λ."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
        delta_L = ttl_path_length_noise(config)
        phi = ttl_phase_noise(config)
        expected = 2.0 * np.pi * delta_L / config.wavelength
        assert abs(phi - expected) < 1e-12, (
            f"Phase noise {phi} does not match 2πδL/λ = {expected}"
        )


class TestTTLSensitivityFloor:
    """Test sensitivity floor."""

    def test_floor_equals_phase_noise(self) -> None:
        """Sensitivity floor should equal TTL phase noise."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        floor = ttl_sensitivity_floor(config)
        phase = ttl_phase_noise(config)
        assert abs(floor - phase) < 1e-12, (
            f"Floor {floor} differs from phase noise {phase}"
        )


class TestTTLLimitedSensitivity:
    """Test combined sensitivity."""

    def test_quadrature_sum_large_quantum(self) -> None:
        """When quantum ≫ TTL, total ≈ quantum."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        large_quantum = 100.0
        total = ttl_limited_sensitivity(10.0, large_quantum, config)
        assert abs(total - large_quantum) / large_quantum < 0.01, (
            "Large quantum noise regime failed"
        )

    def test_quadrature_sum_large_ttl(self) -> None:
        """When TTL ≫ quantum, total ≈ TTL."""
        config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        small_quantum = 1e-10
        phi_ttl = ttl_phase_noise(config)
        total = ttl_limited_sensitivity(10.0, small_quantum, config)
        assert abs(total - phi_ttl) / phi_ttl < 0.01, "TTL-dominated regime failed"


class TestTTLScalingSweep:
    """Test scaling sweep."""

    def test_sql_low_N(self) -> None:
        """At low N (tiny TTL), α ≈ -0.5 (SQL regime)."""
        config = TTLNoiseConfig(theta_rms=1e-12, beam_offset=1e-6, wavelength=1e-6)
        N = np.logspace(0, 3, 20)
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        assert result["alpha_fitted"] is not None
        assert abs(result["alpha_fitted"] - (-0.5)) < 0.05, (
            f"Expected α ≈ -0.5, got {result['alpha_fitted']}"
        )

    def test_hl_low_N(self) -> None:
        """At low N (tiny TTL with HL), α ≈ -1.0."""
        config = TTLNoiseConfig(theta_rms=1e-12, beam_offset=1e-6, wavelength=1e-6)
        N = np.logspace(0, 3, 20)
        result = ttl_scaling_sweep(N, config, quantum_scaling="hl")
        assert result["alpha_fitted"] is not None
        assert abs(result["alpha_fitted"] - (-1.0)) < 0.05, (
            f"Expected α ≈ -1.0, got {result['alpha_fitted']}"
        )

    def test_returns_expected_keys(self) -> None:
        """Scaling sweep returns the expected dict structure."""
        config = TTLNoiseConfig()
        N = np.logspace(0, 6, 10)
        result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        expected_keys = {
            "N",
            "delta_phi",
            "delta_phi_quantum",
            "delta_phi_ttl",
            "alpha_fitted",
        }
        assert set(result.keys()) == expected_keys


class TestTTLConfigValidation:
    """Test config validation."""

    def test_zero_theta_raises(self) -> None:
        """Zero angular jitter should raise ValueError."""
        config = TTLNoiseConfig(theta_rms=0.0)
        with pytest.raises(ValueError, match="positive"):
            ttl_phase_noise(config)

    def test_negative_wavelength_raises(self) -> None:
        """Negative wavelength should raise ValueError."""
        config = TTLNoiseConfig(wavelength=-1e-6)
        with pytest.raises(ValueError, match="positive"):
            ttl_phase_noise(config)

    def test_config_validation_passes(self) -> None:
        """Valid config should not raise."""
        config = TTLNoiseConfig()
        ttl_phase_noise(config)  # Should not raise
