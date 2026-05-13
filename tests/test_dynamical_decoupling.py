"""Tests for the dynamical decoupling module.

Tests verify:
1. DDConfig validation
2. CPMG filter function: zero-pulse = unit, non-negative, DC suppression
3. Effective coherence time scaling with pulses
4. Phase sensitivity SQL scaling (α = -0.5 preserved)
5. Prefactor C improves (decreases) with more pulses
6. XY-8 outperforms CPMG
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.dynamical_decoupling import (
    DDConfig,
    cpmg_filter_function,
    dd_effective_coherence_time,
    dd_phase_sensitivity,
    dd_sensitivity_scaling,
)


class TestDDConfig:
    """Test DDConfig validation."""

    def test_default_values(self) -> None:
        """Default config should be reasonable."""
        config = DDConfig()
        assert config.n_pulses == 0, "Expected config.n_pulses == 0"
        assert config.sequence == "CPMG", 'Expected config.sequence == "CPMG"'
        assert config.tau == 0.1, "Expected config.tau == 0.1"

    def test_negative_pulses_raises(self) -> None:
        """Negative pulses should raise ValueError."""
        with pytest.raises(ValueError):
            DDConfig(n_pulses=-1)

    def test_unknown_sequence_raises(self) -> None:
        """Unknown sequence should raise ValueError."""
        with pytest.raises(ValueError):
            DDConfig(sequence="UNKNOWN")

    def test_zero_tau_raises(self) -> None:
        """Zero tau should raise ValueError."""
        with pytest.raises(ValueError):
            DDConfig(tau=0.0)

    def test_invalid_pulse_axis_raises(self) -> None:
        """Invalid pulse axis should raise ValueError."""
        with pytest.raises(ValueError):
            DDConfig(pulse_axis="z")


class TestCpmgFilterFunction:
    """Test CPMG filter function."""

    def test_zero_pulses_is_unit(self) -> None:
        """Zero pulses: filter ≡ 1."""
        omega = np.linspace(-10, 10, 100)
        F = cpmg_filter_function(omega, n_pulses=0, tau=1.0)
        assert pytest.approx(1.0, abs=1e-10) == F, "Zero-pulse filter should be 1"

    def test_non_negative(self) -> None:
        """Filter function should be non-negative."""
        omega = np.linspace(-20, 20, 5000)
        for n_pulses in [1, 2, 4, 8]:
            F = cpmg_filter_function(omega, n_pulses=n_pulses, tau=0.5)
            assert np.all(F >= -1e-12), (
                f"Negative values at n_pulses={n_pulses}: min={np.min(F):.2e}"
            )

    def test_dc_suppression(self) -> None:
        """DC (ω = 0) should be fully suppressed for n_pulses > 0."""
        for n_pulses in [1, 2, 4, 8]:
            F = cpmg_filter_function(np.array([0.0]), n_pulses=n_pulses, tau=0.5)
            assert F[0] == pytest.approx(0.0, abs=1e-15), (
                f"DC not suppressed for n_pulses={n_pulses}"
            )

    def test_single_frequency(self) -> None:
        """Single frequency (1-element array) should be handled correctly."""
        F = cpmg_filter_function(np.array([0.0]), n_pulses=4, tau=0.5)
        assert isinstance(F, np.ndarray), f"Expected ndarray, got {type(F)}"
        assert F.shape == (1,), f"Expected shape (1,), got {F.shape}"
        assert F[0] == pytest.approx(0.0, abs=1e-15), (
            "Expected F[0] == pytest.approx(0.0, abs=1e-15)"
        )


class TestDDEffectiveCoherenceTime:
    """Test effective coherence time computation."""

    def test_zero_pulses_gives_bare(self) -> None:
        """Zero pulses should return bare coherence time."""
        T_0 = dd_effective_coherence_time(T_2_0=1.0, n_pulses=0)
        assert pytest.approx(1.0) == T_0, "Expected T_0 == pytest.approx(1.0)"

    def test_improves_with_pulses(self) -> None:
        """DD should always improve or maintain coherence time (monotonic in n)."""
        T_2_0 = 1.0

        # Zero pulses: no improvement
        T_0 = dd_effective_coherence_time(T_2_0, 0, "CPMG")
        assert pytest.approx(T_2_0) == T_0, (
            "Zero pulses should give bare coherence time"
        )

        for sequence in ["CPMG", "XY8"]:
            previous_T = T_2_0  # Reset per sequence
            for n in [1, 2, 4, 8, 16]:
                T_dd = dd_effective_coherence_time(T_2_0, n, sequence)
                assert T_dd >= previous_T, (
                    f"{sequence} n={n}: {T_dd:.4f} < {previous_T:.4f}"
                )
                assert T_dd >= T_2_0, f"{sequence} n={n}: {T_dd:.4f} < bare {T_2_0:.4f}"
                previous_T = T_dd

    def test_xy8_better_than_cpmg(self) -> None:
        """XY-8 should outperform CPMG for same pulse count (n >= 2)."""
        for n in [2, 4, 8, 16]:
            T_cpmg = dd_effective_coherence_time(1.0, n, "CPMG")
            T_xy8 = dd_effective_coherence_time(1.0, n, "XY8")
            assert T_xy8 > T_cpmg, f"n={n}: XY-8 ({T_xy8:.4f}) ≤ CPMG ({T_cpmg:.4f})"


class TestDDPhaseSensitivity:
    """Test phase sensitivity computation."""

    def test_positive(self) -> None:
        """Sensitivity should always be positive."""
        for n_pulses in [0, 1, 4, 8]:
            for N in [1, 10, 100]:
                d = dd_phase_sensitivity(N, np.pi / 4, T=1.0, n_pulses=n_pulses)
                assert d > 0, f"N={N}, n={n_pulses}: Δφ={d:.6e}"

    def test_more_pulses_improves(self) -> None:
        """More pulses → lower Δφ."""
        for N in [10, 100, 1000]:
            d0 = dd_phase_sensitivity(N, 0.0, T=1.0, n_pulses=0)
            d8 = dd_phase_sensitivity(N, 0.0, T=1.0, n_pulses=8)
            assert d8 < d0, f"N={N}: 0 pulses {d0:.6e}, 8 pulses {d8:.6e}"

    def test_more_photons_improves(self) -> None:
        """More photons → lower Δφ (SQL scaling)."""
        for n_pulses in [0, 4, 8]:
            d10 = dd_phase_sensitivity(10, 0.0, T=1.0, n_pulses=n_pulses)
            d100 = dd_phase_sensitivity(100, 0.0, T=1.0, n_pulses=n_pulses)
            assert d100 < d10, f"n={n_pulses}: N=10 {d10:.6e}, N=100 {d100:.6e}"


class TestDDSensitivityScaling:
    """Test scaling analysis."""

    def test_sql_scaling_preserved(self) -> None:
        """DD should preserve SQL exponent α = -0.5."""
        N_values = np.logspace(1, 4, 20)
        for n_pulses in [0, 4, 8]:
            result = dd_sensitivity_scaling(N_values, n_pulses, T=1.0, T_2_0=1.0)
            alpha = result["fitted_alpha"]
            assert alpha == pytest.approx(-0.5, abs=0.02), (
                f"n={n_pulses}: α should be -0.5, got {alpha:.4f}"
            )

    def test_prefactor_improves_with_pulses(self) -> None:
        """More pulses → smaller prefactor C."""
        N_values = np.logspace(1, 4, 10)
        result_0 = dd_sensitivity_scaling(N_values, 0, T=1.0)
        result_4 = dd_sensitivity_scaling(N_values, 4, T=1.0)
        result_8 = dd_sensitivity_scaling(N_values, 8, T=1.0)

        assert result_8["prefactor_C"] < result_4["prefactor_C"], (
            f"8 pulses C={result_8['prefactor_C']:.4f} ≥ 4 pulses C={result_4['prefactor_C']:.4f}"
        )
        assert result_4["prefactor_C"] < result_0["prefactor_C"], (
            f"4 pulses C={result_4['prefactor_C']:.4f} ≥ 0 pulses C={result_0['prefactor_C']:.4f}"
        )
