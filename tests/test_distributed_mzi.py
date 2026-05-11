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

    def test_default_values(self) -> None:
        """Default config should be reasonable."""
        config = DistributedMziConfig()
        assert config.M == 2
        assert config.entangled is False
        assert config.correlation_noise == 0.0

    def test_negative_M_raises(self) -> None:
        """Negative M should raise ValueError."""
        with pytest.raises(ValueError):
            DistributedMziConfig(M=-1)

    def test_invalid_correlation_raises(self) -> None:
        """Correlation outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            DistributedMziConfig(correlation_noise=-0.1)
        with pytest.raises(ValueError):
            DistributedMziConfig(correlation_noise=1.5)


class TestDistributedMziSensitivity:
    """Test sensitivity computation."""

    def test_classical_averaging_sql(self) -> None:
        """Uncorrelated classical: Δφ = 1/√(M·N)."""
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(4 * 100)
        assert np.isclose(result["delta_phi"], expected, rtol=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_classical_averaging_regime(self) -> None:
        """Should identify classical averaging regime."""
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert "Classical averaging" in result["regime"], (
            f"Unexpected regime: {result['regime']}"
        )

    def test_entangled_heisenberg(self) -> None:
        """Entangled sensors: Δφ = 1/(M·N)."""
        config = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / (4 * 100)
        assert np.isclose(result["delta_phi"], expected, rtol=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_entangled_regime(self) -> None:
        """Should identify Heisenberg regime."""
        config = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert "Heisenberg" in result["regime"] or "Collective" in result["regime"], (
            f"Unexpected regime: {result['regime']}"
        )

    def test_fully_correlated_classical(self) -> None:
        """Full correlation removes M benefit: Δφ = 1/√N."""
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=1.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(100)  # Single-sensor SQL
        assert np.isclose(result["delta_phi"], expected, rtol=1e-6), (
            f"Expected {expected}, got {result['delta_phi']}"
        )

    def test_positive_N_raises(self) -> None:
        """Non-positive N should raise ValueError."""
        config = DistributedMziConfig()
        with pytest.raises(ValueError):
            distributed_mzi_sensitivity(0, 0.0, config)

    def test_single_sensor(self) -> None:
        """M=1 classical = SQL; M=1 entangled = HL (NOON with N photons)."""
        config_classical = DistributedMziConfig(M=1, entangled=False)
        config_entangled = DistributedMziConfig(M=1, entangled=True)

        result_c = distributed_mzi_sensitivity(100, 0.0, config_classical)
        result_e = distributed_mzi_sensitivity(100, 0.0, config_entangled)

        # Classical: SQL = 1/√N
        assert np.isclose(result_c["delta_phi"], 0.1, rtol=1e-6), (
            f"Classical M=1 should give SQL, got {result_c['delta_phi']}"
        )
        # Entangled: HL = 1/N (NOON state with N particles)
        assert np.isclose(result_e["delta_phi"], 0.01, rtol=1e-6), (
            f"Entangled M=1 should give HL, got {result_e['delta_phi']}"
        )
        # Entangled beats classical for M=1 (NOON beats coherent)
        assert result_e["delta_phi"] < result_c["delta_phi"], (
            f"Entangled ({result_e['delta_phi']}) should beat "
            f"classical ({result_c['delta_phi']})"
        )


class TestDistributedScalingExponent:
    """Test scaling exponent predictions."""

    def test_classical_exponent(self) -> None:
        """Classical should give α = -0.5."""
        config = DistributedMziConfig(entangled=False)
        assert distributed_scaling_exponent(config) == -0.5

    def test_entangled_exponent(self) -> None:
        """Entangled should give α = -1.0."""
        config = DistributedMziConfig(entangled=True)
        assert distributed_scaling_exponent(config) == -1.0

    def test_effective_scaling_finite(self) -> None:
        """Effective scaling should be finite and reasonable."""
        config = DistributedMziConfig(M=4, entangled=False)
        alpha = effective_scaling_at_N(100, config)
        assert np.isfinite(alpha), f"Alpha should be finite, got {alpha}"
        assert alpha < 0, f"Alpha should be negative for SQL, got {alpha}"


class TestComputeDistributedScaling:
    """Test distributed scaling grid computation."""

    def test_grid_shape(self) -> None:
        """Grid should have correct shape."""
        M_values = [1, 2, 4]
        N_values = [10, 100]
        config = DistributedMziConfig(entangled=False)
        result = compute_distributed_scaling(M_values, N_values, config)
        assert result["delta_phi_grid"].shape == (3, 2), (
            f"Expected (3, 2), got {result['delta_phi_grid'].shape}"
        )

    def test_more_sensors_improves(self) -> None:
        """More sensors should always improve (or match) sensitivity."""
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
