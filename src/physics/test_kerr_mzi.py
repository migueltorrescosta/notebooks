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
    def test_given_kerr_unitary_then_diagonal_in_fock_basis(self) -> None:
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert pytest.approx(np.diag(np.diag(U))) == U

    def test_given_kerr_unitary_then_is_unitary(self) -> None:
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)

    @pytest.mark.parametrize("max_photons", [0, 1, 2, 3], ids=["0", "1", "2", "3"])
    def test_given_max_photons_n_then_dimension_is_n_plus_1_squared(
        self, max_photons: int
    ) -> None:
        U = kerr_phase_shift_unitary(0.5, 0.1, 1.0, max_photons=max_photons)
        expected_dim = (max_photons + 1) ** 2
        assert U.shape == (expected_dim, expected_dim)

    @pytest.mark.parametrize(
        "phi_phase", [0.0, np.pi / 4, np.pi / 2, np.pi], ids=["0", "pi/4", "pi/2", "pi"]
    )
    def test_given_zero_kerr_then_matches_phase_shift_unitary(
        self, phi_phase: float
    ) -> None:
        U_kerr = kerr_phase_shift_unitary(phi_phase, 0.0, 1.0, max_photons=2)
        U_std = phase_shift_unitary(phi_phase, max_photons=2)
        assert U_kerr == pytest.approx(U_std)

    def test_given_zero_kerr_then_interaction_time_does_not_affect_unitary(
        self,
    ) -> None:
        U_T1 = kerr_phase_shift_unitary(1.0, 0.0, 0.0, max_photons=2)
        U_T2 = kerr_phase_shift_unitary(1.0, 0.0, 10.0, max_photons=2)
        assert pytest.approx(U_T2) == U_T1

    def test_given_kerr_unitary_then_diagonal_matches_analytical_formula(self) -> None:
        phi_phase, K, T_kerr = 0.5, 0.3, 2.0
        max_photons = 3
        U = kerr_phase_shift_unitary(phi_phase, K, T_kerr, max_photons=max_photons)
        dim = max_photons + 1
        kerr_coeff = K * T_kerr
        for n1 in range(dim):
            for n2 in range(dim):
                idx = n1 * dim + n2
                expected = np.exp(1j * (phi_phase * n2 + kerr_coeff * (n1**2 + n2**2)))
                assert U[idx, idx] == pytest.approx(expected)

    def test_given_same_K_t_product_then_same_unitary(self) -> None:
        U_a = kerr_phase_shift_unitary(0.5, 0.2, 3.0, max_photons=2)
        U_b = kerr_phase_shift_unitary(0.5, 0.6, 1.0, max_photons=2)
        assert U_a == pytest.approx(U_b)

    def test_given_negative_K_then_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            kerr_phase_shift_unitary(1.0, -0.1, 1.0, 2)

    def test_given_negative_t_then_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            kerr_phase_shift_unitary(1.0, 0.1, -1.0, 2)

    def test_given_negative_max_photons_then_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            kerr_phase_shift_unitary(1.0, 0.1, 1.0, -1)

    def test_given_zero_phi_phase_and_K_then_unitary_is_identity(self) -> None:
        U = kerr_phase_shift_unitary(0.0, 0.0, 1.0, max_photons=3)
        assert pytest.approx(np.eye((3 + 1) ** 2)) == U

    def test_given_zero_phi_phase_nonzero_K_then_unitary_is_not_identity(self) -> None:
        U = kerr_phase_shift_unitary(0.0, 0.5, 1.0, max_photons=2)
        assert pytest.approx(np.eye(9)) != U


class TestKerrMzi:
    def test_given_no_kerr_then_norm_preserved(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi_phase=1.0, K=0.0, T_kerr=0.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0)

    def test_given_kerr_then_norm_preserved(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi_phase=1.0, K=0.5, T_kerr=1.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0)

    def test_given_fock_state_input_then_norm_preserved(self) -> None:
        dim = 3 + 1
        state = qutip.tensor(qutip.fock(dim, 2), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(state, phi_phase=0.5, K=0.3, T_kerr=2.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0)

    def test_given_kerr_then_output_probabilities_differ_from_linear(self) -> None:
        state = noon_state(3, max_photons=3)
        final_lin = kerr_mzi(state, phi_phase=1.0, K=0.0, T_kerr=0.0, max_photons=3)
        final_kerr = kerr_mzi(state, phi_phase=1.0, K=0.5, T_kerr=1.0, max_photons=3)
        P0_lin, _ = compute_kerr_output_probabilities(final_lin, 3)
        P0_kerr, _ = compute_kerr_output_probabilities(final_kerr, 3)
        assert P0_lin != pytest.approx(P0_kerr, abs=1e-6)

    def test_given_theta_zero_then_mzi_reduces_to_phase_kerr(self) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(
            state, phi_phase=0.5, K=0.3, T_kerr=1.0, max_photons=3, theta=0.0
        )
        U = kerr_phase_shift_unitary(0.5, 0.3, 1.0, 3)
        expected = U @ state
        assert final == pytest.approx(expected)

    def test_given_zero_phi_phase_K_then_fringe_balanced(self) -> None:
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi_phase=0.0, K=0.0, T_kerr=0.0, max_photons=2)
        P0, _P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(0.5, abs=1e-10) == P0

    @pytest.mark.parametrize(
        ("phi_phase", "K", "T_kerr"),
        [(0.0, 0.0, 0.0), (0.5, 0.1, 1.0), (np.pi, 2.0, 0.5)],
        ids=["zero", "moderate", "large_kerr"],
    )
    def test_given_vacuum_input_then_output_probabilities_balanced(
        self, phi_phase: float, K: float, T_kerr: float
    ) -> None:
        dim = 2 + 1
        vac = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(vac, phi_phase=phi_phase, K=K, T_kerr=T_kerr, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(0.5) == P0
        assert pytest.approx(0.5) == P1

    def test_given_non_normalized_state_then_raises_valueerror(self) -> None:
        bad_state = np.array([0.5, 0.5, 0.0, 0.0])
        with pytest.raises(ValueError):
            kerr_mzi(bad_state, phi_phase=0.0, K=0.0, T_kerr=0.0, max_photons=1)

    def test_given_wrong_dimension_then_raises_valueerror(self) -> None:
        bad_state = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            kerr_mzi(bad_state, phi_phase=0.0, K=0.0, T_kerr=0.0, max_photons=1)


class TestKerrOutputProbabilities:
    def test_given_any_state_then_probabilities_sum_to_one(self) -> None:
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi_phase=0.5, K=0.2, T_kerr=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1

    @pytest.mark.parametrize(
        "phi_phase",
        [
            0.0,
            np.pi * 0.25,
            np.pi * 0.5,
            np.pi * 0.75,
            np.pi,
            np.pi * 1.25,
            np.pi * 1.5,
            np.pi * 1.75,
        ],
        ids=["0", "pi/4", "pi/2", "3pi/4", "pi", "5pi/4", "3pi/2", "7pi/4"],
    )
    def test_given_any_phase_then_probabilities_non_negative(
        self, phi_phase: float
    ) -> None:
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi_phase=phi_phase, K=0.3, T_kerr=1.0, max_photons=3)
        P0, P1 = compute_kerr_output_probabilities(final, 3)
        assert P0 >= 0
        assert P1 >= 0

    def test_given_vacuum_input_then_probabilities_balanced(self) -> None:
        dim = 2 + 1
        vac = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(vac, phi_phase=1.0, K=0.5, T_kerr=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(0.5) == P0
        assert pytest.approx(0.5) == P1

    def test_given_single_photon_input_then_probabilities_sum_to_one(self) -> None:
        dim = 2 + 1
        state = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()
        final = kerr_mzi(state, phi_phase=0.5, K=0.1, T_kerr=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1


class TestKerrPhaseSensitivity:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5], ids=["1", "2", "3", "4", "5"])
    def test_given_no_kerr_then_qfi_equals_n_squared(self, N: int) -> None:
        F = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
        assert pytest.approx(N**2) == F

    @pytest.mark.parametrize("N", [1, 2, 3], ids=["1", "2", "3"])
    def test_given_kerr_then_qfi_preserved(self, N: int) -> None:
        F_no_kerr = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
        F_kerr = compute_kerr_phase_sensitivity(N, 0.5, 1.0)
        assert F_no_kerr == pytest.approx(F_kerr)

    @pytest.mark.parametrize(
        "K", [0.1, 0.5, 1.0, 2.0], ids=["0.1", "0.5", "1.0", "2.0"]
    )
    def test_given_kerr_then_qfi_independent_of_K(self, K: float) -> None:
        F_base = compute_kerr_phase_sensitivity(4, 0.0, 0.0)
        F = compute_kerr_phase_sensitivity(4, K, 1.0)
        assert pytest.approx(F_base) == F

    def test_given_default_max_photons_then_matches_explicit(self) -> None:
        F_explicit = compute_kerr_phase_sensitivity(3, 0.0, 0.0, max_photons=3)
        F_default = compute_kerr_phase_sensitivity(3, 0.0, 0.0)
        assert F_explicit == pytest.approx(F_default)

    def test_given_noon_states_then_qfi_scales_as_n_squared(self) -> None:
        F_1 = compute_kerr_phase_sensitivity(1, 0.0, 0.0)
        F_2 = compute_kerr_phase_sensitivity(2, 0.0, 0.0)
        F_10 = compute_kerr_phase_sensitivity(10, 0.0, 0.0)
        assert pytest.approx(1.0) == F_1
        assert pytest.approx(4.0) == F_2
        assert pytest.approx(100.0) == F_10


class TestKerrInterferenceFringe:
    def test_given_phase_array_then_fringe_matches_shape(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases, K=0.1, T_kerr=1.0, max_photons=3, N=3
        )
        assert fringe.shape == (50,)

    def test_given_kerr_fringe_then_probabilities_bounded(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases, K=0.1, T_kerr=1.0, max_photons=3, N=3
        )
        assert np.all(fringe >= 0) and np.all(fringe <= 1)

    def test_given_default_state_then_matches_explicit_noon_state(self) -> None:
        phases = np.linspace(0, np.pi, 20)
        fringe_default = compute_kerr_interference_fringe(
            phases, K=0.0, T_kerr=1.0, max_photons=3, N=3
        )
        fringe_explicit = compute_kerr_interference_fringe(
            phases,
            K=0.0,
            T_kerr=1.0,
            max_photons=3,
            initial_state=noon_state(3, max_photons=3),
        )
        assert fringe_default == pytest.approx(fringe_explicit)

    def test_given_no_kerr_then_fringe_2pi_periodic(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 100)
        fringe = compute_kerr_interference_fringe(
            phases, K=0.0, T_kerr=1.0, max_photons=2, N=2
        )
        assert fringe[0] == pytest.approx(fringe[-1], abs=1e-10)
