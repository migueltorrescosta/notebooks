"""
Tests for Kerr-nonlinear Mach-Zehnder Interferometer module.
"""

import numpy as np
import pytest
import qutip

from .kerr_mzi import (
    compute_kerr_interference_fringe,
    compute_kerr_output_probabilities,
    compute_kerr_phase_sensitivity,
    kerr_mzi,
    kerr_phase_shift_unitary,
)
from .mzi_simulation import (
    noon_state,
    phase_shift_unitary,
)


class TestKerrPhaseShiftUnitary:
    """Test combined phase + Kerr unitary construction."""

    def test_kerr_phase_unitary_should_be_diagonal_in_fock_basis(self) -> None:
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert pytest.approx(np.diag(np.diag(U))) == U, (
            "Expected U == pytest.approx(np.diag(np.diag(U)))"
        )

    def test_u_dagger_u_i(self) -> None:
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10), (
            "Kerr phase shift unitary must satisfy U U† = I"
        )

    def test_matrix_dimension_should_be_max_photons_1_2(self) -> None:
        for mp in [0, 1, 2, 3]:
            U = kerr_phase_shift_unitary(0.5, 0.1, 1.0, max_photons=mp)
            expected_dim = (mp + 1) ** 2
            assert U.shape == (expected_dim, expected_dim), (
                "Expected U.shape == (expected_dim, expected_dim)"
            )

    def test_when_chi_0_should_match_phase_shift_unitary(self) -> None:
        for phi in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            U_kerr = kerr_phase_shift_unitary(phi, 0.0, 1.0, max_photons=2)
            U_std = phase_shift_unitary(phi, max_photons=2)
            assert U_kerr == pytest.approx(U_std), (
                "Expected U_kerr == pytest.approx(U_std)"
            )

    def test_when_chi_0_t_should_not_affect_the_unitary(self) -> None:
        U_T1 = kerr_phase_shift_unitary(1.0, 0.0, 0.0, max_photons=2)
        U_T2 = kerr_phase_shift_unitary(1.0, 0.0, 10.0, max_photons=2)
        assert pytest.approx(U_T2) == U_T1, "Expected U_T1 == pytest.approx(U_T2)"

    def test_each_diagonal_element_should_be_exp_i_phi_n2_chi_t_n1_2_n2_2(self) -> None:
        phi, chi, T = 0.5, 0.3, 2.0
        max_photons = 3
        U = kerr_phase_shift_unitary(phi, chi, T, max_photons=max_photons)
        dim = max_photons + 1
        kerr_coeff = chi * T
        for n1 in range(dim):
            for n2 in range(dim):
                idx = n1 * dim + n2
                expected = np.exp(1j * (phi * n2 + kerr_coeff * (n1**2 + n2**2)))
                assert U[idx, idx] == pytest.approx(expected), (
                    "Expected U[idx, idx] == pytest.approx(expected)"
                )

    def test_same_chi_t_product_should_give_same_unitary(self) -> None:
        U_a = kerr_phase_shift_unitary(0.5, 0.2, 3.0, max_photons=2)
        U_b = kerr_phase_shift_unitary(0.5, 0.6, 1.0, max_photons=2)
        assert U_a == pytest.approx(U_b), "Expected U_a == pytest.approx(U_b)"

    def test_negative_chi_should_raise_valueerror(self) -> None:
        try:
            kerr_phase_shift_unitary(1.0, -0.1, 1.0, 2)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_negative_t_should_raise_valueerror(self) -> None:
        try:
            kerr_phase_shift_unitary(1.0, 0.1, -1.0, 2)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_negative_max_photons_should_raise_valueerror(self) -> None:
        try:
            kerr_phase_shift_unitary(1.0, 0.1, 1.0, -1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_when_phi_0_and_chi_0_unitary_should_be_identity(self) -> None:
        U = kerr_phase_shift_unitary(0.0, 0.0, 1.0, max_photons=3)
        assert pytest.approx(np.eye((3 + 1) ** 2)) == U, (
            "Expected U == pytest.approx(np.eye((3 + 1) ** 2))"
        )

    def test_when_phi_0_but_chi_0_unitary_should_not_be_identity(self) -> None:
        U = kerr_phase_shift_unitary(0.0, 0.5, 1.0, max_photons=2)
        assert pytest.approx(np.eye(9)) != U, "Expected U != pytest.approx(np.eye(9))"


class TestKerrMzi:
    """Test full Kerr MZI circuit."""

    def test_norm_preserved_without_kerr(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=1.0, chi=0.0, T=0.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_norm_preserved_with_kerr(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=1.0, chi=0.5, T=1.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_norm_preserved_for_fock_state_input(self) -> None:
        dim = 3 + 1
        state = qutip.tensor(qutip.fock(dim, 2), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(state, phi=0.5, chi=0.3, T=2.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_kerr_should_change_the_output_probabilities(self) -> None:
        state = noon_state(3, max_photons=3)
        final_lin = kerr_mzi(state, phi=1.0, chi=0.0, T=0.0, max_photons=3)
        final_kerr = kerr_mzi(state, phi=1.0, chi=0.5, T=1.0, max_photons=3)
        P0_lin, _ = compute_kerr_output_probabilities(final_lin, 3)
        P0_kerr, _ = compute_kerr_output_probabilities(final_kerr, 3)
        assert P0_lin != pytest.approx(P0_kerr, abs=1e-6), (
            "Expected P0_lin != pytest.approx(P0_kerr, abs=1e-6)"
        )

    def test_with_theta_0_bs_is_identity_mzi_reduces_to_phase_kerr(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=0.5, chi=0.3, T=1.0, max_photons=3, theta=0.0)
        # Phase + Kerr only (no mode mixing)
        U = kerr_phase_shift_unitary(0.5, 0.3, 1.0, 3)
        expected = U @ state
        assert final == pytest.approx(expected), (
            "Expected final == pytest.approx(expected)"
        )

    def test_with_phi_0_chi_0_and_theta_pi_4_mzi_should_output_same_as_input_bs2_bs1_psi(
        self,
    ) -> None:
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi=0.0, chi=0.0, T=0.0, max_photons=2)
        # With phi=chi=0, the state just goes BS1 -> BS2 = identity
        P0, _P1 = compute_kerr_output_probabilities(final, 2)
        # Should produce same output distribution as NOON state (balanced)
        assert pytest.approx(0.5, abs=1e-10) == P0, (
            "Expected P0 == pytest.approx(0.5, abs=1e-10)"
        )

    def test_vacuum_input_should_always_give_0_5_0_5_output(self) -> None:
        dim = 2 + 1
        vac = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full().ravel()
        for phi, chi, T in [(0.0, 0.0, 0.0), (0.5, 0.1, 1.0), (np.pi, 2.0, 0.5)]:
            final = kerr_mzi(vac, phi=phi, chi=chi, T=T, max_photons=2)
            P0, P1 = compute_kerr_output_probabilities(final, 2)
            assert pytest.approx(0.5) == P0 and np.isclose(P1, 0.5), (
                "Expected P0 == pytest.approx(0.5) and np.isclose(P1, 0.5)"
            )

    def test_non_normalized_state_should_raise_valueerror(self) -> None:
        bad_state = np.array([0.5, 0.5, 0.0, 0.0])
        try:
            kerr_mzi(bad_state, phi=0.0, chi=0.0, T=0.0, max_photons=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_state_with_wrong_dimension_should_raise_valueerror(self) -> None:
        bad_state = np.array([1.0, 0.0])  # dim=2, expected 4 for max_photons=1
        try:
            kerr_mzi(bad_state, phi=0.0, chi=0.0, T=0.0, max_photons=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestKerrOutputProbabilities:
    """Test output probability computation."""

    def test_probabilities_should_sum_to_1_for_any_state(self) -> None:
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi=0.5, chi=0.2, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1, "Expected P0 + P1 == pytest.approx(1.0)"

    def test_probabilities_should_be_non_negative(self) -> None:
        state = noon_state(3, max_photons=3)
        for phi in np.linspace(0, 2 * np.pi, 8):
            final = kerr_mzi(state, phi=phi, chi=0.3, T=1.0, max_photons=3)
            P0, P1 = compute_kerr_output_probabilities(final, 3)
            assert P0 >= 0 and P1 >= 0, "Expected P0 >= 0; Expected P1 >= 0"

    def test_vacuum_input_gives_p0_p1_0_5(self) -> None:
        dim = 2 + 1
        vac = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(vac, phi=1.0, chi=0.5, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(0.5) == P0 and np.isclose(P1, 0.5), (
            "Expected P0 == pytest.approx(0.5) and np.isclose(P1, 0.5)"
        )

    def test_single_photon_input_should_give_valid_probabilities(self) -> None:
        dim = 2 + 1
        state = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(state, phi=0.5, chi=0.1, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1, "Expected P0 + P1 == pytest.approx(1.0)"


class TestKerrPhaseSensitivity:
    """Test QFI computation for Kerr MZI."""

    def test_without_kerr_noon_qfi_n_2_heisenberg_limit(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            F = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
            assert pytest.approx(N**2) == F, f"N={N}: F_Q={F}, expected {N**2}"

    def test_kerr_preserves_qfi_since_generator_n2_is_diagonal(self) -> None:
        for N in [1, 2, 3]:
            F_no_kerr = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
            F_kerr = compute_kerr_phase_sensitivity(N, 0.5, 1.0)
            assert F_no_kerr == pytest.approx(F_kerr), "QFI should be same with Kerr"

    def test_qfi_should_not_depend_on_chi_for_different_values(self) -> None:
        F_base = compute_kerr_phase_sensitivity(4, 0.0, 0.0)
        for chi in [0.1, 0.5, 1.0, 2.0]:
            F = compute_kerr_phase_sensitivity(4, chi, 1.0)
            assert pytest.approx(F_base) == F, "Expected F == pytest.approx(F_base)"

    def test_when_max_photons_none_should_default_to_n(self) -> None:
        F_explicit = compute_kerr_phase_sensitivity(3, 0.0, 0.0, max_photons=3)
        F_default = compute_kerr_phase_sensitivity(3, 0.0, 0.0)
        assert F_explicit == pytest.approx(F_default), (
            "Expected F_explicit == pytest.approx(F_default)"
        )

    def test_verify_f_q_n_2_for_noon_states(self) -> None:
        F_1 = compute_kerr_phase_sensitivity(1, 0.0, 0.0)
        F_2 = compute_kerr_phase_sensitivity(2, 0.0, 0.0)
        F_10 = compute_kerr_phase_sensitivity(10, 0.0, 0.0)
        assert pytest.approx(1.0) == F_1, "Expected F_1 == pytest.approx(1.0)"
        assert pytest.approx(4.0) == F_2, "Expected F_2 == pytest.approx(4.0)"
        assert pytest.approx(100.0) == F_10, "Expected F_10 == pytest.approx(100.0)"


class TestKerrInterferenceFringe:
    """Test interference fringe computation."""

    def test_fringe_should_return_array_matching_phase_range(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.1,
            T=1.0,
            max_photons=3,
            N=3,
        )
        assert fringe.shape == (50,), "Expected fringe.shape == (50,)"

    def test_probabilities_should_be_in_0_1(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.1,
            T=1.0,
            max_photons=3,
            N=3,
        )
        assert np.all(fringe >= 0) and np.all(fringe <= 1), (
            "Expected np.all(fringe >= 0) and np.all(fringe <= 1)"
        )

    def test_should_work_with_default_noon_state(self) -> None:
        phases = np.linspace(0, np.pi, 20)
        fringe_default = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=3,
            N=3,
        )
        fringe_explicit = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=3,
            initial_state=noon_state(3, max_photons=3),
        )
        assert fringe_default == pytest.approx(fringe_explicit), (
            "Expected fringe_default == pytest.approx(fringe_explicit)"
        )

    def test_interference_fringe_should_be_2pi_periodic_without_kerr(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 100)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=2,
            N=2,
        )
        # Check periodicity: should be close at 0 and 2pi
        assert fringe[0] == pytest.approx(fringe[-1], abs=1e-10), (
            "Expected fringe[0] == pytest.approx(fringe[-1], abs=1e-10)"
        )
