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
import qutip

from src.physics.cavity_mzi import (
    CavityMziConfig,
    cavity_enhanced_mzi,
    cavity_enhanced_mzi_with_noise,
    cavity_enhanced_sensitivity,
)


def _make_fock_10(max_photons: int) -> np.ndarray:
    """Construct |1,0⟩ state via QuTiP."""
    dim = max_photons + 1
    return qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()


class TestCavityMziConfig:
    """Test CavityMziConfig validation."""

    def test_given_default_values_then_be_reasonable(self) -> None:
        config = CavityMziConfig()
        assert config.F == 10.0
        assert config.theta == pytest.approx(np.pi / 4)


class TestCavityEnhancedMzi:
    """Test cavity-enhanced MZI unitary evolution."""

    def test_given_norm_preservation_then_normalize_output_state(self) -> None:
        config = CavityMziConfig(F=5.0)
        state = _make_fock_10(max_photons=5)
        out = cavity_enhanced_mzi(state, np.pi / 4, config, max_photons=5)
        norm = np.sum(np.abs(out) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10), f"Norm = {norm}"

    def test_given_finesse_one_matches_standard_then_match_single_pass_mzi(
        self,
    ) -> None:
        from src.physics.mzi_simulation import (
            beam_splitter_unitary,
            phase_shift_unitary,
        )

        config = CavityMziConfig(F=1.0)
        phi = np.pi / 4
        max_photons = 5

        # Cavity-enhanced with F=1
        state = _make_fock_10(max_photons)
        cavity_out = cavity_enhanced_mzi(state, phi, config, max_photons)

        # Standard MZI: BS -> Phase(phi) -> BS
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        phase = phase_shift_unitary(phi, max_photons)
        mzi_out = bs @ phase @ bs @ state

        assert cavity_out == pytest.approx(mzi_out, abs=1e-10), (
            "F=1 cavity should match standard MZI"
        )

    def test_given_finesse_lt_one_raises_then_raise_valueerror(self) -> None:
        config = CavityMziConfig(F=0.5)
        state = _make_fock_10(max_photons=3)
        with pytest.raises(ValueError, match="Cavity finesse must be >= 1"):
            cavity_enhanced_mzi(state, 0.0, config, max_photons=3)


class TestCavityEnhancedSensitivity:
    """Test sensitivity computation."""

    def test_given_sensitivity_positive_then_be_positive_and_finite(self) -> None:
        config = CavityMziConfig(F=10.0)
        delta = cavity_enhanced_sensitivity(4, np.pi / 4, config)
        assert np.isfinite(delta), f"Delta should be finite, got {delta}"
        assert delta > 0, f"Delta should be positive, got {delta}"

    def test_given_sensitivity_improves_with_finesse_then_improve_with_higher_finesse(
        self,
    ) -> None:
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

    def test_given_noon_state_better_than_coherent_then_give_lower_delta_phi_than_coherent(
        self,
    ) -> None:
        config = CavityMziConfig(F=5.0)
        delta_coherent = cavity_enhanced_sensitivity(
            4,
            np.pi / 4,
            config,
            state_type="coherent",
        )
        delta_noon = cavity_enhanced_sensitivity(
            4,
            np.pi / 4,
            config,
            state_type="noon",
        )
        assert delta_noon < delta_coherent, (
            f"NOON should beat coherent: NOON Δφ={delta_noon:.6e}, "
            f"coherent Δφ={delta_coherent:.6e}"
        )

    def test_given_invalid_state_type_raises_then_raise_valueerror(self) -> None:
        config = CavityMziConfig()
        with pytest.raises(ValueError, match="Unknown state_type"):
            cavity_enhanced_sensitivity(10, 0.0, config, state_type="invalid")

    def test_given_non_positive_n_raises_then_raise_valueerror(self) -> None:
        config = CavityMziConfig()
        with pytest.raises(ValueError, match="positive"):
            cavity_enhanced_sensitivity(0, 0.0, config)


class TestCavityEnhancedMziWithNoise:
    """Test noisy cavity-enhanced MZI."""

    def test_given_trace_preservation_then_preserve_trace_under_noisy_evolution(
        self,
    ) -> None:
        config = CavityMziConfig(F=3.0)
        state = _make_fock_10(max_photons=5)
        rho = cavity_enhanced_mzi_with_noise(
            state,
            phi=np.pi / 4,
            noise_gamma_1=0.1,
            noise_gamma_2=0.0,
            noise_gamma_phi=0.05,
            config=config,
            max_photons=5,
        )
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-6), (
            f"Trace should be 1, got {np.trace(rho)}"
        )

    def test_given_no_noise_matches_unitary_then_match_unitary_evolution(self) -> None:
        config = CavityMziConfig(F=3.0)
        state = _make_fock_10(max_photons=5)

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

        assert rho_noisy == pytest.approx(rho_unitary, abs=1e-6), (
            "Zero noise should match unitary evolution"
        )
