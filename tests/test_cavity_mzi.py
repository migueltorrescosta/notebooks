"""Tests for the cavity-enhanced MZI module.

Tests verify:
1. CavityMziConfig validation
2. Unitary evolution preserves norm
3. Finesse = 1 reproduces standard MZI
4. Sensitivity function produces finite positive values
5. Noisy evolution preserves trace
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.cavity_mzi import (
    CavityMziConfig,
    cavity_enhanced_mzi,
    cavity_enhanced_sensitivity,
    cavity_enhanced_mzi_with_noise,
)
from src.physics.mzi_simulation import fock_state


class TestCavityMziConfig:
    """Test CavityMziConfig validation."""

    def test_default_values(self) -> None:
        """Default config should be reasonable."""
        config = CavityMziConfig()
        assert config.F == 10.0
        assert np.isclose(config.theta, np.pi / 4)


class TestCavityEnhancedMzi:
    """Test cavity-enhanced MZI unitary evolution."""

    def test_norm_preservation(self) -> None:
        """Output state should be normalized."""
        config = CavityMziConfig(F=5.0)
        state = fock_state(1, 0, max_photons=5)
        out = cavity_enhanced_mzi(state, np.pi / 4, config, max_photons=5)
        norm = np.sum(np.abs(out) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10), f"Norm = {norm}"

    def test_finesse_one_matches_standard(self) -> None:
        """Finesse = 1 should give same as single-pass MZI."""
        from src.physics.mzi_simulation import (
            beam_splitter_unitary,
            phase_shift_unitary,
        )

        config = CavityMziConfig(F=1.0)
        phi = np.pi / 4
        max_photons = 5

        # Cavity-enhanced with F=1
        state = fock_state(1, 0, max_photons)
        cavity_out = cavity_enhanced_mzi(state, phi, config, max_photons)

        # Standard MZI: BS -> Phase(phi) -> BS
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        phase = phase_shift_unitary(phi, max_photons)
        mzi_out = bs @ phase @ bs @ state

        assert np.allclose(cavity_out, mzi_out, atol=1e-10), (
            "F=1 cavity should match standard MZI"
        )

    def test_finesse_lt_one_raises(self) -> None:
        """Finesse < 1 should raise ValueError."""
        config = CavityMziConfig(F=0.5)
        state = fock_state(1, 0, max_photons=3)
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            cavity_enhanced_mzi(state, 0.0, config, max_photons=3)


class TestCavityEnhancedSensitivity:
    """Test sensitivity computation."""

    def test_sensitivity_positive(self) -> None:
        """Sensitivity should be positive and finite (small N for memory)."""
        config = CavityMziConfig(F=10.0)
        delta = cavity_enhanced_sensitivity(4, np.pi / 4, config)
        assert np.isfinite(delta), f"Delta should be finite, got {delta}"
        assert delta > 0, f"Delta should be positive, got {delta}"

    def test_sensitivity_improves_with_finesse(self) -> None:
        """Higher finesse should give better (lower) sensitivity.

        Uses a small phase φ = 0.01 to avoid phase-wrapping effects
        (F·φ must be well below 2π for monotonic improvement).
        """
        small_phi = 0.01
        config_low = CavityMziConfig(F=2.0)
        config_high = CavityMziConfig(F=10.0)
        delta_low = cavity_enhanced_sensitivity(4, small_phi, config_low)
        delta_high = cavity_enhanced_sensitivity(4, small_phi, config_high)
        assert delta_high < delta_low, (
            f"Higher finesse should give lower Δφ: "
            f"F=2 → {delta_low:.6e}, F=10 → {delta_high:.6e}"
        )

    def test_noon_state_better_than_coherent(self) -> None:
        """NOON input should give lower Δφ than coherent (at fixed N)."""
        config = CavityMziConfig(F=5.0)
        delta_coherent = cavity_enhanced_sensitivity(
            4, np.pi / 4, config, state_type="coherent"
        )
        delta_noon = cavity_enhanced_sensitivity(
            4, np.pi / 4, config, state_type="noon"
        )
        assert delta_noon < delta_coherent, (
            f"NOON should beat coherent: NOON Δφ={delta_noon:.6e}, "
            f"coherent Δφ={delta_coherent:.6e}"
        )

    def test_invalid_state_type_raises(self) -> None:
        """Unknown state_type should raise ValueError."""
        config = CavityMziConfig()
        with pytest.raises(ValueError, match="Unknown state_type"):
            cavity_enhanced_sensitivity(10, 0.0, config, state_type="invalid")

    def test_non_positive_N_raises(self) -> None:
        """Non-positive N should raise ValueError."""
        config = CavityMziConfig()
        with pytest.raises(ValueError, match="positive"):
            cavity_enhanced_sensitivity(0, 0.0, config)


class TestCavityEnhancedMziWithNoise:
    """Test noisy cavity-enhanced MZI."""

    def test_trace_preservation(self) -> None:
        """Noisy evolution should preserve trace."""
        config = CavityMziConfig(F=3.0)
        state = fock_state(1, 0, max_photons=5)
        rho = cavity_enhanced_mzi_with_noise(
            state,
            phi=np.pi / 4,
            noise_gamma_1=0.1,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.05,
            config=config,
            max_photons=5,
        )
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6), (
            f"Trace should be 1, got {np.trace(rho)}"
        )

    def test_no_noise_matches_unitary(self) -> None:
        """Zero noise should match unitary evolution (up to numerical)."""
        config = CavityMziConfig(F=3.0)
        state = fock_state(1, 0, max_photons=5)

        rho_noisy = cavity_enhanced_mzi_with_noise(
            state,
            phi=np.pi / 4,
            noise_gamma_1=0.0,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.0,
            config=config,
            max_photons=5,
        )
        state_unitary = cavity_enhanced_mzi(state, np.pi / 4, config, max_photons=5)
        rho_unitary = np.outer(state_unitary, state_unitary.conj())

        assert np.allclose(rho_noisy, rho_unitary, atol=1e-6), (
            "Zero noise should match unitary evolution"
        )
