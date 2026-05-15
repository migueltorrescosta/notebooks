"""Tests for spin_squeezing.py - one-axis twisting (OAT) spin squeezing."""

from __future__ import annotations

import numpy as np
import pytest

from .spin_squeezing import (
    coherent_spin_state,
    generate_squeezed_state,
    one_axis_twist,
    optimal_squeezing_time,
    squeezing_parameter,
)


class TestCoherentSpinState:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 10])
    def test_given_css_then_points_along_negative_x(self, N: int) -> None:
        from src.physics.dicke_basis import jx_operator

        css = coherent_spin_state(N)
        rho = np.outer(css, css.conj())
        J_x = jx_operator(N)
        jx = np.real(np.trace(rho @ J_x))
        assert jx == pytest.approx(-N / 2)

    @pytest.mark.parametrize("N", [1, 2, 5, 10])
    def test_given_css_then_spin_magnitude_is_N_over_2(self, N: int) -> None:
        from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator

        css = coherent_spin_state(N)
        rho = np.outer(css, css.conj())
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J_z = jz_operator(N)

        jx = np.real(np.trace(rho @ J_x))
        jy = np.real(np.trace(rho @ J_y))
        jz = np.real(np.trace(rho @ J_z))

        j_mag = np.sqrt(jx**2 + jy**2 + jz**2)
        assert j_mag == pytest.approx(N / 2, abs=1e-6)


class TestOneAxisTwist:
    @pytest.mark.parametrize("t", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_oat_then_norm_is_preserved(self, N: int, t: float) -> None:
        css = coherent_spin_state(N)
        evolved = one_axis_twist(css, N, chi=1.0, t=t)
        assert np.linalg.norm(evolved) == pytest.approx(1.0, abs=1e-6)

    def test_given_oat_on_css_then_state_differs(self) -> None:
        N = 4
        css = coherent_spin_state(N)
        evolved = one_axis_twist(css, N, chi=1.0, t=0.5)
        overlap = np.abs(np.vdot(css, evolved)) ** 2
        assert overlap < 0.99


class TestSqueezingParameter:
    @pytest.mark.parametrize("N", [1, 2, 4, 8, 16])
    def test_given_css_then_squeezing_parameter_is_one(self, N: int) -> None:
        css = coherent_spin_state(N)
        xi = squeezing_parameter(css, N)
        assert xi == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_squeezed_state_then_xi_is_non_negative(
        self, N: int, t: float
    ) -> None:
        state = generate_squeezed_state(N, chi=1.0, t=t)
        xi = squeezing_parameter(state, N)
        assert xi >= 0


class TestOptimalSqueezingTime:
    @pytest.mark.parametrize(
        ("N", "expected"),
        [
            (2, (6.0 / 2.0) ** (1.0 / 3.0)),
            (4, (6.0 / 4.0) ** (1.0 / 3.0)),
            (8, (6.0 / 8.0) ** (1.0 / 3.0)),
            (16, (6.0 / 16.0) ** (1.0 / 3.0)),
        ],
        ids=["N=2", "N=4", "N=8", "N=16"],
    )
    def test_given_oat_then_optimal_time_matches_formula(
        self, N: int, expected: float
    ) -> None:
        t_opt = optimal_squeezing_time(N, chi=1.0)
        assert t_opt == pytest.approx(expected, rel=1e-10)

    def test_given_oat_then_optimal_time_scales_as_N_neg_one_third(self) -> None:
        N1, N2 = 10, 40
        t1 = optimal_squeezing_time(N1, chi=1.0)
        t2 = optimal_squeezing_time(N2, chi=1.0)
        ratio_actual = t1 / t2
        ratio_expected = (N1 / N2) ** (-1.0 / 3.0)
        assert ratio_actual == pytest.approx(ratio_expected, rel=1e-6)

    @pytest.mark.parametrize(
        "chi", [0.5, 1.0, 2.0], ids=["chi=0.5", "chi=1.0", "chi=2.0"]
    )
    def test_given_oat_then_optimal_time_scales_as_one_over_chi(
        self, chi: float
    ) -> None:
        N = 10
        t_opt = optimal_squeezing_time(N, chi=chi)
        expected = (6.0 / N) ** (1.0 / 3.0) / chi
        assert t_opt == pytest.approx(expected, rel=1e-10)


class TestGenerateSqueezedState:
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_zero_time_then_returns_css(self, N: int) -> None:
        state = generate_squeezed_state(N, chi=1.0, t=0.0)
        css = coherent_spin_state(N)
        assert state == pytest.approx(css, abs=1e-6)

    @pytest.mark.parametrize("t", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_squeezed_state_then_norm_is_one(self, N: int, t: float) -> None:
        state = generate_squeezed_state(N, chi=1.0, t=t)
        norm = np.linalg.norm(state)
        assert norm == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_squeezed_state_then_dimension_is_N_plus_one(self, N: int) -> None:
        state = generate_squeezed_state(N, chi=1.0, t=0.5)
        assert state.shape[0] == N + 1


class TestPhysicalInvariants:
    @pytest.mark.parametrize("t", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_total_probability_is_conserved(self, N: int, t: float) -> None:
        state = generate_squeezed_state(N, chi=1.0, t=t)
        probs = np.abs(state) ** 2
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("t", [0.1, 0.5])
    @pytest.mark.parametrize("N", [2, 5, 10])
    def test_given_oat_on_arbitrary_state_then_norm_preserved(
        self, N: int, t: float
    ) -> None:
        rng = np.random.default_rng(42)
        psi = rng.random(N + 1) + 1j * rng.random(N + 1)
        psi = psi / np.linalg.norm(psi)
        evolved = one_axis_twist(psi, N, chi=1.0, t=t)
        assert np.linalg.norm(evolved) == pytest.approx(np.linalg.norm(psi), abs=1e-6)
