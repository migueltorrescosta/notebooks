"""Tests for spin_squeezing.py - One-axis twisting Hamiltonian."""

from __future__ import annotations

import numpy as np

from src.algorithms.spin_squeezing import (
    coherent_spin_state,
    one_axis_twist,
    squeezing_parameter,
    optimal_squeezing_time,
    generate_squeezed_state,
)


class TestCoherentSpinState:
    """Test suite for coherent_spin_state function."""

    def test_css_is_x_polarized(self) -> None:
        """Test CSS is the |J, -J>_x state (pointing along x)."""
        from src.physics.dicke_basis import jx_operator

        for N in [1, 2, 3, 4, 5, 10]:
            css = coherent_spin_state(N)
            rho = np.outer(css, css.conj())
            J_x = jx_operator(N)
            jx = np.real(np.trace(rho @ J_x))
            assert np.isclose(jx, -N / 2), (
                f"N={N}: <J_x> = {jx}, expected -N/2 = {-N / 2}"
            )

    def test_css_has_maximum_polarization(self) -> None:
        """Test CSS has maximum spin magnitude N/2."""
        from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator

        for N in [1, 2, 5, 10]:
            css = coherent_spin_state(N)
            rho = np.outer(css, css.conj())
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            jx = np.real(np.trace(rho @ J_x))
            jy = np.real(np.trace(rho @ J_y))
            jz = np.real(np.trace(rho @ J_z))

            j_mag = np.sqrt(jx**2 + jy**2 + jz**2)
            assert np.isclose(j_mag, N / 2, atol=1e-6), (
                f"N={N}: |J| = {j_mag}, expected {N / 2}"
            )


class TestOneAxisTwist:
    """Test suite for one_axis_twist function."""

    def test_oat_is_unitary(self) -> None:
        """Test OAT evolution is unitary."""
        for N in [2, 5, 10]:
            css = coherent_spin_state(N)
            for t in [0.1, 0.5, 1.0]:
                evolved = one_axis_twist(css, N, chi=1.0, t=t)
                # Check normalization is preserved
                assert np.isclose(np.linalg.norm(evolved), 1.0, atol=1e-6), (
                    f"N={N}, t={t}: norm changed"
                )

    def test_oat_evolution_changes_state(self) -> None:
        """Test OAT evolution changes CSS (non-trivial for x-polarized CSS)."""
        N = 4
        css = coherent_spin_state(N)  # |J, -J>_x

        # OAT should change the state (not a global phase)
        evolved = one_axis_twist(css, N, chi=1.0, t=0.5)

        # State should be different from original (not just a global phase)
        overlap = np.abs(np.vdot(css, evolved)) ** 2
        assert overlap < 0.99, f"Overlap should be < 0.99, got {overlap}"


class TestSqueezingParameter:
    """Test suite for squeezing_parameter function."""

    def test_css_squeezing_is_one(self) -> None:
        """Test CSS has squeezing parameter ξ = 1."""
        for N in [1, 2, 4, 8, 16]:
            css = coherent_spin_state(N)
            xi = squeezing_parameter(css, N)
            # Numerical tolerance: should be exactly 1.0 for CSS
            assert np.isclose(xi, 1.0, atol=1e-6), f"N={N}: ξ = {xi}, expected 1.0"

    def test_squeezing_parameter_bounded(self) -> None:
        """Test squeezing parameter is non-negative."""
        for N in [2, 5, 10]:
            for t in [0.0, 0.1, 0.5, 1.0]:
                state = generate_squeezed_state(N, chi=1.0, t=t)
                xi = squeezing_parameter(state, N)
                assert xi >= 0, f"N={N}, t={t}: ξ = {xi} is negative"


class TestOptimalSqueezingTime:
    """Test suite for optimal_squeezing_time function."""

    def test_optimal_time_formula(self) -> None:
        """Test optimal time formula t_opt = (6/N)^(1/3) / χ."""
        for N in [2, 4, 8, 16]:
            t_opt = optimal_squeezing_time(N, chi=1.0)
            expected = (6 / N) ** (1 / 3)
            assert np.isclose(t_opt, expected, rtol=1e-10), (
                f"N={N}: t_opt = {t_opt}, expected {expected}"
            )

    def test_optimal_time_scaling(self) -> None:
        """Test optimal time scales as N^(-1/3)."""
        # t_opt ∝ N^(-1/3), so t_opt1/t_opt2 ≈ (N1/N2)^(-1/3)
        N1, N2 = 10, 40
        t1 = optimal_squeezing_time(N1, chi=1.0)
        t2 = optimal_squeezing_time(N2, chi=1.0)

        ratio_actual = t1 / t2
        ratio_expected = (N1 / N2) ** (-1 / 3)

        assert np.isclose(ratio_actual, ratio_expected, rtol=1e-6), (
            f"t_opt scaling: {ratio_actual:.4f} vs {(N1 / N2) ** (-1 / 3):.4f}"
        )

    def test_chi_scaling(self) -> None:
        """Test optimal time scales as 1/χ."""
        chi_values = [0.5, 1.0, 2.0]
        N = 10

        for chi in chi_values:
            t_opt = optimal_squeezing_time(N, chi=chi)
            # With chi = 1, t_opt = (6/N)^(1/3)
            # With chi, scaled by 1/chi
            expected = (6 / N) ** (1 / 3) / chi
            assert np.isclose(t_opt, expected, rtol=1e-10), (
                f"N={N}, chi={chi}: t_opt = {t_opt}, expected {expected}"
            )


class TestGenerateSqueezedState:
    """Test suite for generate_squeezed_state function."""

    def test_t_zero_gives_css(self) -> None:
        """Test t=0 returns CSS."""
        for N in [2, 5, 10]:
            state = generate_squeezed_state(N, chi=1.0, t=0.0)
            css = coherent_spin_state(N)
            # Should be identical (up to global phase)
            assert np.allclose(state, css, atol=1e-6), f"N={N}: state != CSS at t=0"

    def test_normalization_preserved(self) -> None:
        """Test generated state is normalized."""
        for N in [2, 5, 10]:
            for t in [0.1, 0.5, 1.0]:
                state = generate_squeezed_state(N, chi=1.0, t=t)
                norm = np.linalg.norm(state)
                assert np.isclose(norm, 1.0, atol=1e-6), f"N={N}, t={t}: norm = {norm}"

    def test_state_dimension(self) -> None:
        """Test state has correct dimension N+1."""
        for N in [2, 5, 10]:
            state = generate_squeezed_state(N, chi=1.0, t=0.5)
            assert state.shape[0] == N + 1, (
                f"N={N}: dimension {state.shape[0]} vs {N + 1}"
            )


class TestPhysicalInvariants:
    """Test physical invariants."""

    def test_probability_conservation(self) -> None:
        """Test total probability is conserved."""
        for N in [2, 5, 10]:
            for t in [0.1, 0.5, 1.0]:
                state = generate_squeezed_state(N, chi=1.0, t=t)
                probs = np.abs(state) ** 2
                assert np.isclose(np.sum(probs), 1.0, atol=1e-6), (
                    f"N={N}, t={t}: sum probs = {np.sum(probs)}"
                )

    def test_unitary_evolution(self) -> None:
        """Test that OAT is unitary (norm preserved for any initial state)."""
        for N in [2, 5, 10]:
            # Random initial state
            rng = np.random.default_rng(42)
            psi = rng.random(N + 1) + 1j * rng.random(N + 1)
            psi = psi / np.linalg.norm(psi)

            for t in [0.1, 0.5]:
                evolved = one_axis_twist(psi, N, chi=1.0, t=t)
                assert np.isclose(
                    np.linalg.norm(evolved), np.linalg.norm(psi), atol=1e-6
                ), (
                    f"N={N}: norm changed from {np.linalg.norm(psi)} to {np.linalg.norm(evolved)}"
                )
