"""Smoke tests for :mod:`src.analysis.cavity_mzi`.

Tests the main simulation functions at single parameter points
and validates error handling in configuration.

Uses small Hilbert spaces (max_photons ≤ 3) for fast execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.mzi_states import input_state_factory

from .cavity_mzi import (
    CavityMziConfig,
    cavity_enhanced_mzi,
    cavity_enhanced_mzi_with_noise,
    cavity_enhanced_sensitivity,
)


class TestCavityMziConfig:
    """Validates CavityMziConfig parameter constraints."""

    def test_valid_config(self) -> None:
        """Happy path: valid parameters."""
        config = CavityMziConfig(F=10.0, theta=np.pi / 4, phi_bs=0.0)
        assert config.F == 10.0

    def test_finesse_below_one_raises_value_error(self) -> None:
        """Finesse must be >= 1 (cavity with fewer than 1 pass is unphysical)."""
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            cavity_enhanced_mzi(
                np.zeros(9),
                phi_phase=0.0,
                config=CavityMziConfig(F=0.5),
                max_photons=2,
            )


class TestCavityEnhancedMzi:
    """Smoke tests for unitary cavity-enhanced MZI evolution."""

    def _make_input_state(self, max_photons: int = 2) -> np.ndarray:
        """Create a |1,0⟩ Fock state for testing."""
        return input_state_factory("fock", N=1, max_photons=max_photons)

    def test_output_normalized(self) -> None:
        """Output state is normalized."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=10.0)
        final = cavity_enhanced_mzi(
            state, phi_phase=np.pi / 4, config=config, max_photons=2
        )
        assert np.isclose(np.sum(np.abs(final) ** 2), 1.0, atol=1e-10)

    def test_f_equals_one_matches_standard_mzi(self) -> None:
        """F=1 reproduces the standard MZI (single phase pass)."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=1.0)
        final = cavity_enhanced_mzi(
            state, phi_phase=np.pi / 3, config=config, max_photons=2
        )
        assert np.isclose(np.sum(np.abs(final) ** 2), 1.0, atol=1e-10)

    def test_larger_f_gives_different_output(self) -> None:
        """Different finesse → different output state."""
        state = self._make_input_state(max_photons=2)
        config_1 = CavityMziConfig(F=1.0)
        config_5 = CavityMziConfig(F=5.0)
        out_1 = cavity_enhanced_mzi(state, np.pi / 4, config_1, max_photons=2)
        out_5 = cavity_enhanced_mzi(state, np.pi / 4, config_5, max_photons=2)
        assert not np.allclose(out_1, out_5, atol=1e-10)

    def test_zero_phase_normalized(self) -> None:
        """With φ=0, output remains normalized."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=10.0)
        final = cavity_enhanced_mzi(state, phi_phase=0.0, config=config, max_photons=2)
        assert np.isclose(np.sum(np.abs(final) ** 2), 1.0, atol=1e-10)


class TestCavityEnhancedMziWithNoise:
    """Smoke tests for noisy cavity-enhanced MZI."""

    def _make_input_state(self, max_photons: int = 2) -> np.ndarray:
        """Create a |1,0⟩ Fock state for testing."""
        return input_state_factory("fock", N=1, max_photons=max_photons)

    def test_output_density_matrix_shape(self) -> None:
        """Output density matrix has correct dimensions."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=5.0)
        rho = cavity_enhanced_mzi_with_noise(
            state,
            phi_phase=np.pi / 4,
            noise_gamma_1=0.1,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.05,
            config=config,
            max_photons=2,
        )
        expected_dim = (2 + 1) ** 2  # (max_photons+1)²
        assert rho.shape == (expected_dim, expected_dim)

    def test_trace_approximately_one(self) -> None:
        """Density matrix trace is close to 1 (small loss)."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=5.0)
        rho = cavity_enhanced_mzi_with_noise(
            state,
            phi_phase=np.pi / 4,
            noise_gamma_1=0.01,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.01,
            config=config,
            max_photons=2,
        )
        assert np.isclose(np.trace(rho), 1.0, atol=0.01)

    def test_no_noise_gives_pure_state(self) -> None:
        """With zero noise rates, the output approximates a pure state."""
        state = self._make_input_state(max_photons=2)
        config = CavityMziConfig(F=1.0)
        rho = cavity_enhanced_mzi_with_noise(
            state,
            phi_phase=np.pi / 4,
            noise_gamma_1=0.0,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.0,
            config=config,
            max_photons=2,
        )
        purity = np.trace(rho @ rho).real
        assert np.isclose(purity, 1.0, atol=0.01)

    def test_finesse_below_one_raises_value_error(self) -> None:
        state = input_state_factory("fock", N=1, max_photons=2)
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            cavity_enhanced_mzi_with_noise(
                state,
                phi_phase=0.0,
                noise_gamma_1=0.0,
                noise_gamma_2=0.0,
                noise_gamma_phi=0.0,
                config=CavityMziConfig(F=0.5),
                max_photons=2,
            )


class TestCavityEnhancedSensitivity:
    """Smoke tests for cavity-enhanced sensitivity computation."""

    def test_coherent_state_sensitivity_positive(self) -> None:
        """Δφ > 0 for coherent state with cavity."""
        config = CavityMziConfig(F=5.0)
        delta = cavity_enhanced_sensitivity(4, np.pi / 4, config, state_type="coherent")
        assert delta > 0
        assert np.isfinite(delta)

    def test_noon_state_sensitivity_better_than_coherent(self) -> None:
        """NOON state gives better sensitivity than coherent at same N."""
        config = CavityMziConfig(F=5.0)
        delta_coherent = cavity_enhanced_sensitivity(
            2, np.pi / 4, config, state_type="coherent"
        )
        delta_noon = cavity_enhanced_sensitivity(
            2, np.pi / 4, config, state_type="noon"
        )
        assert delta_noon < delta_coherent

    def test_higher_finesse_improves_sensitivity(self) -> None:
        """Higher finesse ℱ gives better (lower) Δφ."""
        delta_low = cavity_enhanced_sensitivity(
            4,
            np.pi / 4,
            CavityMziConfig(F=1.0),
            state_type="coherent",
        )
        delta_high = cavity_enhanced_sensitivity(
            4,
            np.pi / 4,
            CavityMziConfig(F=10.0),
            state_type="coherent",
        )
        assert delta_high < delta_low

    def test_negative_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Mean photon number must be positive"):
            cavity_enhanced_sensitivity(-1, np.pi / 4, CavityMziConfig())

    def test_finesse_below_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            cavity_enhanced_sensitivity(4, np.pi / 4, CavityMziConfig(F=0.5))

    def test_unknown_state_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown state_type"):
            cavity_enhanced_sensitivity(
                4,
                np.pi / 4,
                CavityMziConfig(F=5.0),
                state_type="invalid",
            )
