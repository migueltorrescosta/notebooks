"""Tests for the general-J coherent spin state construction."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.coherent_spin_state import coherent_spin_state


class TestCoherentSpinState:
    """Tests for coherent_spin_state."""

    @pytest.mark.parametrize(
        ("J", "theta", "phi"),
        [
            (0.5, 0.0, 0.0),
            (0.5, np.pi, 0.0),
            (1.0, np.pi / 3, np.pi / 4),
            (1.5, np.pi / 2, np.pi),
            (2.0, 0.7, 1.3),
            (2.5, 1.2, 4.5),
        ],
        ids=["J=0.5,theta=0", "J=0.5,theta=pi", "J=1.0", "J=1.5", "J=2.0", "J=2.5"],
    )
    def test_css_normalised(self, J: float, theta: float, phi: float) -> None:
        state = coherent_spin_state(J, theta, phi)
        assert np.isclose(np.linalg.norm(state), 1.0), (
            f"CSS not normalised: norm={np.linalg.norm(state)}"
        )

    @pytest.mark.parametrize(
        ("J", "theta"),
        [(0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (2.5, 0.0)],
        ids=["J=0.5", "J=1.0", "J=2.0", "J=2.5"],
    )
    def test_css_theta_zero(self, J: float, theta: float) -> None:
        """At theta=0, CSS should be |J, J⟩ (top Dicke state)."""
        state = coherent_spin_state(J, theta, 0.0)
        expected = np.zeros(int(2 * J + 1), dtype=complex)
        expected[0] = 1.0
        assert np.allclose(state, expected, atol=1e-12), (
            f"CSS at theta=0 should be |J,J⟩ for J={J}"
        )

    @pytest.mark.parametrize(
        ("J", "theta"),
        [(0.5, np.pi), (1.0, np.pi), (2.0, np.pi)],
        ids=["J=0.5", "J=1.0", "J=2.0"],
    )
    def test_css_theta_pi(self, J: float, theta: float) -> None:
        """At theta=pi, CSS should be |J, -J⟩ (bottom Dicke state)."""
        state = coherent_spin_state(J, theta, 0.0)
        expected = np.zeros(int(2 * J + 1), dtype=complex)
        expected[-1] = 1.0
        assert np.allclose(state, expected, atol=1e-12), (
            f"CSS at theta=pi should be |J,-J⟩ for J={J}"
        )

    def test_css_raises_for_negative_J(self) -> None:
        with pytest.raises(ValueError):
            coherent_spin_state(-1.0, 0.0, 0.0)

    @pytest.mark.parametrize("J", [0.0])
    def test_css_J_zero(self, J: float) -> None:
        """J=0 should give the trivial [1] state."""
        state = coherent_spin_state(J, 0.0, 0.0)
        assert state.shape == (1,)
        assert np.isclose(state[0], 1.0)

    def test_css_dimensions(self) -> None:
        """Dimension should be 2J+1."""
        for J in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            state = coherent_spin_state(J, 0.5, 1.0)
            assert state.shape == (int(2 * J + 1),), (
                f"Wrong dimension for J={J}: {state.shape}"
            )

    def test_css_special_cases(self) -> None:
        """Verify specific known CSS values."""
        # J=0.5: CSS at (pi/2, 0) should be (|up⟩ + |down⟩)/sqrt(2) in z-basis
        state = coherent_spin_state(0.5, np.pi / 2, 0.0)
        expected_amp = 1.0 / np.sqrt(2)
        assert np.isclose(abs(state[0]), expected_amp, atol=1e-12)
        assert np.isclose(abs(state[1]), expected_amp, atol=1e-12)

        # J=1: CSS at (pi/2, 0) should have binomial coefficients sqrt(2)/2, 1, sqrt(2)/2
        state = coherent_spin_state(1.0, np.pi / 2, 0.0)
        # |1,1⟩: cos(pi/4)^2 = 0.5, sqrt(comb(2,0)) = 1 → coeff = 0.5
        # |1,0⟩: cos(pi/4)*sin(pi/4) = 0.5, sqrt(comb(2,1)) = sqrt(2) → coeff = 0.5*sqrt(2)
        # |1,-1⟩: sin(pi/4)^2 = 0.5, sqrt(comb(2,2)) = 1 → coeff = 0.5
        expected = np.array([0.5, np.sqrt(2) / 2, 0.5])
        assert np.allclose(state, expected, atol=1e-12), (
            f"J=1 CSS(pi/2,0) mismatch: {state}"
        )
