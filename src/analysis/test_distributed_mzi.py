"""Smoke tests for :mod:`src.analysis.distributed_mzi`.

Tests the main simulation functions at single parameter points
and validates error handling in configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.noise_channels import NoiseConfig

from .distributed_mzi import (
    DistributedMziConfig,
    compute_distributed_scaling,
    distributed_mzi_sensitivity,
    distributed_scaling_exponent,
    effective_scaling_at_N,
)


class TestDistributedMziConfig:
    """Validates DistributedMziConfig parameter constraints."""

    def test_valid_config(self) -> None:
        """Happy path: valid parameters."""
        config = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.0)
        assert config.M == 4

    def test_m_below_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Number of sensors M must be >= 1"):
            DistributedMziConfig(M=0)

    def test_negative_correlation_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="correlation_noise must be in"):
            DistributedMziConfig(correlation_noise=-0.1)

    def test_correlation_above_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="correlation_noise must be in"):
            DistributedMziConfig(correlation_noise=1.5)


class TestDistributedMziSensitivityClassical:
    """Smoke tests for classical (unentangled) distributed MZI."""

    def test_single_sensor_sql(self) -> None:
        """Single sensor gives SQL: Δφ = 1/√N."""
        config = DistributedMziConfig(M=1, entangled=False)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / np.sqrt(100)
        assert np.isclose(result["delta_phi"], expected, atol=0.001)

    def test_classical_averaging_improvement(self) -> None:
        """M independent sensors improve by √M."""
        config_1 = DistributedMziConfig(M=1, entangled=False)
        config_4 = DistributedMziConfig(M=4, entangled=False)
        r1 = distributed_mzi_sensitivity(100, 0.0, config_1)
        r4 = distributed_mzi_sensitivity(100, 0.0, config_4)
        expected_ratio = np.sqrt(4)
        assert np.isclose(r1["delta_phi"] / r4["delta_phi"], expected_ratio, atol=0.001)

    def test_fully_correlated_classical_no_improvement(self) -> None:
        """With c=1, adding more sensors does not help."""
        config_1 = DistributedMziConfig(M=1, entangled=False, correlation_noise=1.0)
        config_4 = DistributedMziConfig(M=4, entangled=False, correlation_noise=1.0)
        r1 = distributed_mzi_sensitivity(100, 0.0, config_1)
        r4 = distributed_mzi_sensitivity(100, 0.0, config_4)
        assert np.isclose(r1["delta_phi"], r4["delta_phi"], atol=0.01)

    def test_partially_correlated_classical(self) -> None:
        """With c=0.5, some benefit from multiple sensors."""
        config_1 = DistributedMziConfig(M=1, entangled=False, correlation_noise=0.5)
        config_4 = DistributedMziConfig(M=4, entangled=False, correlation_noise=0.5)
        r1 = distributed_mzi_sensitivity(100, 0.0, config_1)
        r4 = distributed_mzi_sensitivity(100, 0.0, config_4)
        # 4 sensors still helps, but less than √4
        assert r4["delta_phi"] < r1["delta_phi"]


class TestDistributedMziSensitivityEntangled:
    """Smoke tests for entangled distributed MZI."""

    def test_entangled_single_sensor_sql(self) -> None:
        """Single entangled sensor: Δφ = 1/N (Heisenberg)."""
        config = DistributedMziConfig(M=1, entangled=True)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / 100
        assert np.isclose(result["delta_phi"], expected, atol=0.001)

    def test_entangled_collective_heisenberg(self) -> None:
        """M entangled sensors give Δφ = 1/(M·N) (collective Heisenberg)."""
        config = DistributedMziConfig(M=4, entangled=True)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        expected = 1.0 / (4 * 100)
        assert np.isclose(result["delta_phi"], expected, atol=0.001)

    def test_entangled_with_correlation(self) -> None:
        """Correlated noise degrades entangled sensitivity."""
        config_clean = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.0)
        config_noisy = DistributedMziConfig(M=4, entangled=True, correlation_noise=0.5)
        r_clean = distributed_mzi_sensitivity(100, 0.0, config_clean)
        r_noisy = distributed_mzi_sensitivity(100, 0.0, config_noisy)
        assert r_noisy["delta_phi"] > r_clean["delta_phi"]

    def test_entangled_fully_correlated(self) -> None:
        """Fully correlated entangled: no collective benefit."""
        config = DistributedMziConfig(M=4, entangled=True, correlation_noise=1.0)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        # With c=1, only N² term remains
        assert result["delta_phi"] > 0

    def test_entangled_single_sensor_with_correlation(self) -> None:
        """M=1 entangled with c>0: correlation has no effect (fractional)."""
        config = DistributedMziConfig(M=1, entangled=True, correlation_noise=0.5)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert result["delta_phi"] > 0


class TestDistributedMziSensitivityNoiseConfig:
    """Smoke tests for distributed MZI with noise configuration."""

    def test_detection_efficiency_degradation(self) -> None:
        """Lower detection efficiency → worse sensitivity."""
        noise_ideal = NoiseConfig(eta=1.0)
        noise_bad = NoiseConfig(eta=0.5)
        config = DistributedMziConfig(M=2, entangled=False)
        r_ideal = distributed_mzi_sensitivity(100, 0.0, config, noise_ideal)
        r_bad = distributed_mzi_sensitivity(100, 0.0, config, noise_bad)
        assert r_bad["delta_phi"] > r_ideal["delta_phi"]

    def test_loss_degradation(self) -> None:
        """Loss degrades sensitivity."""
        noise_ideal = NoiseConfig(gamma_1=0.0)
        noise_lossy = NoiseConfig(gamma_1=0.5)
        config = DistributedMziConfig(M=2, entangled=False)
        r_ideal = distributed_mzi_sensitivity(100, 0.0, config, noise_ideal)
        r_lossy = distributed_mzi_sensitivity(100, 0.0, config, noise_lossy)
        assert r_lossy["delta_phi"] > r_ideal["delta_phi"]

    def test_dephasing_degradation(self) -> None:
        """Dephasing degrades sensitivity."""
        noise_ideal = NoiseConfig(gamma_phi=0.0)
        noise_deph = NoiseConfig(gamma_phi=0.1)
        config = DistributedMziConfig(M=2, entangled=False)
        r_ideal = distributed_mzi_sensitivity(100, 0.0, config, noise_ideal)
        r_deph = distributed_mzi_sensitivity(100, 0.0, config, noise_deph)
        assert r_deph["delta_phi"] > r_ideal["delta_phi"]

    def test_default_noise_config(self) -> None:
        """No noise_config parameter uses default NoiseConfig()."""
        config = DistributedMziConfig(M=2, entangled=False)
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert result["delta_phi"] > 0

    def test_non_positive_n_raises_value_error(self) -> None:
        config = DistributedMziConfig()
        with pytest.raises(ValueError, match="Photon number per sensor must be > 0"):
            distributed_mzi_sensitivity(0, 0.0, config)


class TestDistributedMziSensitivityRegimeLabels:
    """Smoke tests for operating regime labels."""

    @pytest.mark.parametrize(
        ("entangled", "correlation", "expected_keyword"),
        [
            (False, 0.0, "Classical averaging"),
            (True, 0.0, "Collective Heisenberg"),
            (False, 0.7, "Partially correlated"),
            (True, 0.7, "Partially correlated"),
            (False, 1.0, "Correlated noise"),
            (True, 1.0, "Correlated noise"),
        ],
    )
    def test_regime_labels(
        self,
        entangled: bool,
        correlation: float,
        expected_keyword: str,
    ) -> None:
        """Regime labels describe the operating conditions."""
        config = DistributedMziConfig(
            M=4,
            entangled=entangled,
            correlation_noise=correlation,
        )
        result = distributed_mzi_sensitivity(100, 0.0, config)
        assert expected_keyword in result["regime"]


class TestDistributedScalingExponent:
    """Smoke tests for expected scaling exponents."""

    def test_classical_exponent(self) -> None:
        """Classical averaging: α = -0.5."""
        config = DistributedMziConfig(M=4, entangled=False)
        alpha = distributed_scaling_exponent(config)
        assert alpha == -0.5

    def test_entangled_exponent(self) -> None:
        """Entangled: α = -1.0."""
        config = DistributedMziConfig(M=4, entangled=True)
        alpha = distributed_scaling_exponent(config)
        assert alpha == -1.0


class TestEffectiveScalingAtN:
    """Smoke tests for effective local scaling exponent."""

    def test_classical_near_sql(self) -> None:
        """Classical sensors give α ≈ -0.5."""
        config = DistributedMziConfig(M=2, entangled=False)
        alpha = effective_scaling_at_N(100, config)
        assert np.isclose(alpha, -0.5, atol=0.05)

    def test_entangled_near_heisenberg(self) -> None:
        """Entangled sensors give α ≈ -1.0."""
        config = DistributedMziConfig(M=2, entangled=True)
        alpha = effective_scaling_at_N(100, config)
        assert np.isclose(alpha, -1.0, atol=0.05)

    def test_low_n_effective_exponent_finite(self) -> None:
        """Effective scaling exponent is finite even at low N."""
        config = DistributedMziConfig(M=2, entangled=False, correlation_noise=0.9)
        alpha = effective_scaling_at_N(5, config)
        assert np.isfinite(alpha)


class TestComputeDistributedScaling:
    """Smoke tests for the scaling grid computation."""

    def test_grid_shape(self) -> None:
        """Output grids have shape (n_M, n_N)."""
        M_vals = [1, 2, 4]
        N_vals = [10, 100]
        config = DistributedMziConfig(entangled=False)
        result = compute_distributed_scaling(M_vals, N_vals, config)
        assert result["delta_phi_grid"].shape == (3, 2)
        assert result["M_grid"].shape == (3, 2)
        assert result["N_grid"].shape == (3, 2)

    def test_classical_scaling_factors(self) -> None:
        """Classical: more sensors → better (larger) scaling factor."""
        M_vals = [1, 2, 4]
        N_vals = [100]
        config = DistributedMziConfig(entangled=False)
        result = compute_distributed_scaling(M_vals, N_vals, config)
        factors = result["scaling_factors"][:, 0]
        assert factors[0] < factors[1] < factors[2]

    def test_regimes_returned(self) -> None:
        """Regime labels are returned for each grid point."""
        M_vals = [1, 2]
        N_vals = [10, 100]
        config = DistributedMziConfig(entangled=False, correlation_noise=0.5)
        result = compute_distributed_scaling(M_vals, N_vals, config)
        assert len(result["regimes"]) == 2  # one row per M
        assert len(result["regimes"][0]) == 2  # one entry per N

    def test_contains_all_keys(self) -> None:
        """Result dict has all expected keys."""
        M_vals = [1, 2]
        N_vals = [10, 100]
        config = DistributedMziConfig(entangled=True)
        result = compute_distributed_scaling(M_vals, N_vals, config)
        expected_keys = {
            "M_grid",
            "N_grid",
            "delta_phi_grid",
            "scaling_factors",
            "regimes",
        }
        assert expected_keys.issubset(result.keys())
