import numpy as np
import pytest
import qutip

from src.physics.mzi_simulation import noon_state
from src.utils.validators import validate_state_mzi

from .mzi_states import (
    compute_fisher_information,
    compute_jz_expectation,
    compute_jz_variance,
    input_state_factory,
    single_photon_split_state,
    twin_fock_state,
    two_mode_jz_operator,
    validate_noon,
    validate_twin_fock,
)


class TestTwinFockState:
    @pytest.mark.parametrize("N", [2, 4, 6, 8, 10])
    def test_twin_fock_normalized(self, N: int) -> None:
        state = twin_fock_state(N)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0)

    def test_twin_fock_requires_even_n(self) -> None:
        with pytest.raises(ValueError):
            twin_fock_state(N=3)

    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_twin_fock_j_z_zero(self, N: int) -> None:
        state = twin_fock_state(N)
        jz_mean = np.real(compute_jz_expectation(state, N))
        assert jz_mean == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_twin_fock_variance_matches_formula(self, N: int) -> None:
        state = twin_fock_state(N)
        jz_var = compute_jz_variance(state, N)
        expected_var = N * (N + 2) / 12
        assert jz_var == pytest.approx(expected_var, rel=1e-6)

    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_twin_fock_fisher_information_matches_formula(self, N: int) -> None:
        state = twin_fock_state(N)
        f_q = compute_fisher_information(state, N)
        expected_fq = N * (N + 2) / 3
        assert f_q == pytest.approx(expected_fq, rel=1e-6)

    def test_twin_fock_includes_all_n_1_permutations(self) -> None:
        N = 4
        state = twin_fock_state(N)
        nonzero_amplitudes = np.sum(np.abs(state) > 1e-10)
        assert nonzero_amplitudes == N + 1


class TestNOONState:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_normalized(self, N: int) -> None:
        state = noon_state(N, max_photons=N)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_equal_overlap_with_extremal_fock_states(self, N: int) -> None:
        state = noon_state(N, max_photons=N)

        fock_n0 = np.zeros((N + 1) ** 2, dtype=complex)
        fock_n0[N * (N + 1)] = 1.0

        fock_0n = np.zeros((N + 1) ** 2, dtype=complex)
        fock_0n[N] = 1.0

        overlap_n0 = np.abs(np.conj(state) @ fock_n0) ** 2
        overlap_0n = np.abs(np.conj(state) @ fock_0n) ** 2

        assert overlap_n0 == pytest.approx(0.5, abs=1e-10)
        assert overlap_0n == pytest.approx(0.5, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_j_z_zero(self, N: int) -> None:
        state = noon_state(N, max_photons=N)
        jz_mean = np.real(compute_jz_expectation(state, N))
        assert jz_mean == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_variance_n_squared_over_4(self, N: int) -> None:
        state = noon_state(N, max_photons=N)
        jz_var = compute_jz_variance(state, N)
        expected_var = N**2 / 4
        assert jz_var == pytest.approx(expected_var, rel=1e-6)

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
    def test_noon_fisher_information_heisenberg_limit(self, N: int) -> None:
        state = noon_state(N, max_photons=N)
        f_q = compute_fisher_information(state, N)
        assert f_q == pytest.approx(N**2, rel=1e-6)


class TestStateOrthogonality:
    def test_noon_orthogonal_to_intermediate_fock_states(self) -> None:
        N = 4
        state = noon_state(N, max_photons=N)

        middle_fock = np.zeros((N + 1) ** 2, dtype=complex)
        middle_fock[(N - 1) * (N + 1) + 1] = 1.0

        overlap = np.abs(np.conj(state) @ middle_fock) ** 2
        assert overlap == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("alpha1", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("alpha2", [0.0, 0.5, 1.0])
    def test_two_mode_coherent_state_normalized_when_max_photons_adequate(
        self, alpha1: float, alpha2: float
    ) -> None:
        dim = 16
        state = (
            qutip.tensor(qutip.coherent(dim, alpha1), qutip.coherent(dim, alpha2))
            .full()
            .ravel()
        )
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0, rel=1e-6)


class TestInputStateFactory:
    def test_factory_creates_twin_fock(self) -> None:
        state = input_state_factory("twin_fock", N=4)
        assert validate_state_mzi(state)

    def test_factory_creates_noon(self) -> None:
        state = input_state_factory("noon", N=3)
        assert validate_state_mzi(state)

    def test_factory_creates_coherent_state(self) -> None:
        state = input_state_factory("coherent", N=0, alpha1=1.0, alpha2=0.0)
        assert validate_state_mzi(state)

    def test_factory_creates_fock_state(self) -> None:
        state = input_state_factory("fock", N=3)
        assert validate_state_mzi(state)

    def test_factory_creates_single_photon_split(self) -> None:
        state = input_state_factory("single_photon_split", N=2)
        assert validate_state_mzi(state)

    def test_factory_creates_css(self) -> None:
        state = input_state_factory("css", N=1)
        assert validate_state_mzi(state)

    def test_factory_raises_for_unknown_state_type(self) -> None:
        with pytest.raises(ValueError):
            input_state_factory("unknown_type", N=1)


class TestValidation:
    def test_twin_fock_validation_passes(self) -> None:
        N = 4
        state = twin_fock_state(N)
        assert validate_twin_fock(N, state, N)

    def test_noon_validation_passes(self) -> None:
        N = 3
        state = noon_state(N, max_photons=N)
        assert validate_noon(N, state, N)


class TestJ_zOperator:
    def test_j_z_operator_hermitian(self) -> None:
        jz = two_mode_jz_operator(max_photons=3)
        assert jz == pytest.approx(jz.conj().T)

    def test_j_z_eigenvalues_n1_minus_n2_over_2(self) -> None:
        jz = two_mode_jz_operator(max_photons=2)

        for n1 in range(3):
            for n2 in range(3):
                idx = n1 * 3 + n2
                expected = (n1 - n2) / 2
                assert jz[idx, idx] == pytest.approx(expected)


class TestSinglePhotonSplit:
    def test_single_photon_split_normalized(self) -> None:
        state = single_photon_split_state(N=3)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0)

    def test_single_photon_split_requires_n_ge_2(self) -> None:
        with pytest.raises(ValueError):
            single_photon_split_state(N=1)

    def test_single_photon_split_symmetric(self) -> None:
        state = single_photon_split_state(N=4)
        jz_mean = np.real(compute_jz_expectation(state, 4))
        assert jz_mean == pytest.approx(0.0, abs=1e-10)
